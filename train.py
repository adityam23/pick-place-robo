from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, NOISE_CONFIG # must import first

import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt

from training.utils import *

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='pick_place',
                    help='Dataset name under data/ (also prefixes the checkpoint dir)')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for checkpoint directory name (e.g. _act)')
parser.add_argument('--num_epochs', type=int, default=None,
                    help='Training epochs. Overrides config.')
parser.add_argument('--eval_every', type=int, default=0,
                    help='Run a short sim evaluation every N epochs (0 to disable)')
parser.add_argument('--eval_episodes', type=int, default=3,
                    help='Episodes per periodic evaluation')
parser.add_argument('--policy_class', type=str, default=None,
                    help='Policy class: ACT, Diffusion, CNNMLP')
parser.add_argument('--chunk_size', type=int, default=None,
                    help='Action chunk size (num_queries). Overrides config.')
parser.add_argument('--temporal_agg', action='store_true',
                    help='Enable temporal aggregation during evaluation (ACT only)')
parser.add_argument('--lr', type=float, default=None,
                    help='Learning rate. Overrides config.')
parser.add_argument('--backbone', type=str, default=None,
                    help='Vision backbone: resnet18, mobilenetv2')
parser.add_argument('--cameras', type=str, default=None,
                    help='Comma-separated camera names (e.g. front,wrist)')
parser.add_argument('--diffusion_steps', type=int, default=None,
                    help='Number of diffusion steps (training)')
parser.add_argument('--ddim_steps', type=int, default=None,
                    help='Number of DDIM inference steps')
parser.add_argument('--noise_level', type=float, default=0.0,
                    help='Observation noise augmentation level in [0, 1] (0 disables)')
args = parser.parse_args()
task = args.task

# configs
task_cfg = dict(TASK_CONFIG)
train_cfg = dict(TRAIN_CONFIG)
policy_config = dict(POLICY_CONFIG)

# Apply overrides
if args.num_epochs is not None:
    train_cfg['num_epochs'] = args.num_epochs
if args.chunk_size is not None:
    policy_config['num_queries'] = args.chunk_size
if args.temporal_agg:
    policy_config['temporal_agg'] = True
if args.policy_class is not None:
    policy_config['policy_class'] = args.policy_class
if args.lr is not None:
    policy_config['lr'] = args.lr
if args.backbone is not None:
    policy_config['backbone'] = args.backbone
if args.diffusion_steps is not None:
    policy_config['diffusion_steps'] = args.diffusion_steps
if args.ddim_steps is not None:
    policy_config['ddim_steps'] = args.ddim_steps
if args.cameras is not None:
    camera_names = [c.strip() for c in args.cameras.split(',')]
    policy_config['camera_names'] = camera_names
    task_cfg['camera_names'] = camera_names

data_dir = os.path.join(task_cfg['dataset_dir'], task)
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task + args.suffix)
# Evaluate at the resolution the policy was trained on
task_cfg['cam_width'], task_cfg['cam_height'] = dataset_image_size(data_dir, task_cfg['camera_names'][0])

run_config = {
    'dataset': data_dir,
    'policy': policy_config,
    'task': {k: task_cfg[k] for k in ('episode_len', 'cam_width', 'cam_height', 'camera_names')},
    'noise_level': args.noise_level,
    'training': {'num_epochs': train_cfg['num_epochs'], 'seed': train_cfg['seed']},
}

# device
device = os.environ['DEVICE']


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


def run_checkpoint_eval(policy, stats, epoch):
    """Short sim evaluation with a video, to watch progress during training."""
    from sim.evaluate_sim import evaluate

    print(f"\n--- Checkpoint evaluation at epoch {epoch} ---")
    evaluate(policy, stats, run_config, device, args.eval_episodes,
             seed_start=epoch, video_path=os.path.join(checkpoint_dir, 'eval_latest.mp4'),
             video_episodes=args.eval_episodes)
    policy.train()


def save_best(best_ckpt_info):
    """Overwrite policy_best.ckpt — the only checkpoint a run keeps on disk."""
    best_epoch, best_val_loss, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(checkpoint_dir, 'policy_best.ckpt'))
    run_config['training'].update(best_epoch=best_epoch, best_val_loss=round(float(best_val_loss), 5))
    save_run_config(checkpoint_dir, run_config)


def train_bc(train_dataloader, val_dataloader, policy_config, stats):
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    eval_every = args.eval_every

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(train_cfg['num_epochs']):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch > 0 and epoch % 200 == 0:
            # Checkpoint the best-so-far weights so a crash doesn't lose the run
            save_best(best_ckpt_info)
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

        if eval_every > 0 and epoch > 0 and epoch % eval_every == 0:
            run_checkpoint_eval(policy, stats, epoch)

    save_best(best_ckpt_info)
    plot_history(train_history, validation_history, train_cfg['num_epochs'], checkpoint_dir, train_cfg['seed'])
    best_epoch, best_val_loss, _ = best_ckpt_info
    print(f"Best checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.5f}")


if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    # number of training episodes
    num_episodes = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(
        data_dir, num_episodes, task_cfg['camera_names'],
        train_cfg['batch_size_train'], train_cfg['batch_size_val'],
        noise_config=NOISE_CONFIG, noise_level=args.noise_level)
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    save_run_config(checkpoint_dir, run_config)

    # train
    train_bc(train_dataloader, val_dataloader, policy_config, stats)
