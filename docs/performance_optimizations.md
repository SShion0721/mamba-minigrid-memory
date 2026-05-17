# Training Performance Optimizations

This note records the hot-path optimizations applied after profiling the
slot-memory gated-attention PPO run.

## Applied

- Dynamic sequence packing for sequence PPO updates. Non-learned-position
  sequence models now trim each minibatch to the longest real sequence in that
  minibatch instead of always padding to `context_len`.
- Fast mask path for full-valid minibatches. PPO skips model `valid_mask` and
  update `loss_mask` tensors when every token is active, avoiding unnecessary
  padding attention bias and masked loss reductions.
- Episodic memory stats no longer synchronize CUDA once per sequence step.
  Sequence memory diagnostics are detached and copied to CPU once per forward.
- Gated attention caches reusable ALiBi and causal attention bias tensors by
  sequence length, device, and dtype.
- Slot-token fusion uses a direct single-query scaled-dot-product attention
  path instead of the heavier `nn.MultiheadAttention` wrapper call.
- Bootstrap value computation now runs under the same AMP autocast setting as
  rollout and PPO updates.
- PPO update start lines are emitted before the first minibatch so long updates
  no longer look silent when the tqdm postfix is truncated.

## Still Worth Considering

- Keep `batch_chunks` conservative on local GPUs. Large values increase memory
  pressure and can reduce throughput even when they fit.
- For `gated_attention + alibi`, additive ALiBi masks can still be slower than
  pure causal SDPA. Use `--gated-attention-pos none` for a speed ablation.
- The Python MiniGrid env loop is still serial. If rollout becomes the bottleneck
  after update optimizations, vectorized or subprocess envs are the next target.
