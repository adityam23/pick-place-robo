#!/bin/bash
# Run training experiments with different chunk sizes + temporal aggregation
# All use the same 50-episode dataset, just vary inference strategy

set -e

echo "=== Experiment 1: chunk_size=50, temporal_agg ==="
uv run python train.py --task pick_place --chunk_size 50 --temporal_agg \
    --suffix _chunk50_tagg --eval_every 500 --eval_episodes 5 2>&1 | \
    tee training_log_chunk50_tagg.txt

echo ""
echo "=== Experiment 2: chunk_size=30, temporal_agg ==="
uv run python train.py --task pick_place --chunk_size 30 --temporal_agg \
    --suffix _chunk30_tagg --eval_every 500 --eval_episodes 5 2>&1 | \
    tee training_log_chunk30_tagg.txt

echo ""
echo "=== Experiment 3: chunk_size=100, temporal_agg ==="
uv run python train.py --task pick_place --chunk_size 100 --temporal_agg \
    --suffix _chunk100_tagg --eval_every 500 --eval_episodes 5 2>&1 | \
    tee training_log_chunk100_tagg.txt

echo ""
echo "=== All experiments complete ==="
echo "Checkpoints saved to:"
echo "  checkpoints/pick_place_chunk50_tagg/"
echo "  checkpoints/pick_place_chunk30_tagg/"
echo "  checkpoints/pick_place_chunk100_tagg/"
echo "  checkpoints/pick_place/  (baseline, no temporal_agg, chunk=100)"
