# Mamba-MiniGrid-Memory-Agent

用 Mamba/Mamba3 替代 LSTM，训练一个在 MiniGrid Memory 部分可观察任务中具备记忆能力的 PPO agent，并与 MLP、LSTM、Attention baseline 对比。

核心问题：

> 在 MiniGrid Memory 中，选择性状态空间模型能否作为 PPO agent 的高效记忆模块？

## 任务设定

MiniGrid Memory 的 agent 会先看到一个起始物体，随后穿过走廊，在岔路口必须选择与起始物体匹配的一侧。当前观测是局部 `7x7x3` compact semantic grid，不是 RGB 图像。

虽然官方环境暴露 `Discrete(7)` 动作空间，但 Memory 任务只需要导航动作：

```text
0 left
1 right
2 forward
```

`pickup/drop/toggle/done` 对该任务没有帮助，保留它们会让 PPO 浪费大量样本，甚至在 greedy eval 时坍缩到原地无效动作。因此本项目默认使用 action mask：`--valid-actions 0,1,2`。

## 当前网络

默认 SOTA-oriented 配置：

```text
MiniGrid compact obs: [7, 7, 3]
  + direction
  + prev_action
  + prev_reward
  + episode_start
        |
object/color/state embeddings
        |
Hybrid spatial encoder
  depthwise conv residual blocks
  learned saliency pooling
  direction-conditioned summary
        |
trajectory tokens: [B, T, d_model]
        |
Mamba / Mamba2 / Mamba3 temporal backbone
        |
LayerNorm
        |
actor head  -> masked action logits
critic head -> value
```

旧的 spatial Transformer 仍可通过 `--spatial-encoder transformer` 启用，用于消融实验。默认改为 `hybrid`，因为 `7x7` compact grid 是小型符号地图，不是自然图像；局部卷积更高效地建模墙、走廊和岔路结构，saliency pooling 更直接地提取可见物体线索。

## Mamba 为什么适合

Mamba 来自选择性状态空间模型。直观上，每一步 token 是输入 `u_t`，Mamba 维护隐状态 `h_t`：

```text
h_t = A_t h_{t-1} + B_t u_t
y_t = C_t h_t
```

其中 `A_t/B_t/C_t` 会依赖当前输入动态变化。对 Memory 任务来说，这个隐状态可以学习保存：

- 起始房间看到的是 key 还是 ball
- 当前处于走廊还是岔路口
- 最近动作如何改变了相对位置
- 最后应该走上方还是下方分支

因此它更像“学习出来的状态估计器 + 控制器”，而不是手写 PID。

## 快速训练

Windows + `mamba_env` + Mamba3/Triton 已配置时：

```powershell
cd E:\Desktop\mamba-minigrid-memory
.\scripts\train_overnight_mamba.ps1 `
  -MambaVariant mamba3 `
  -NoFallback `
  -RunName overnight_mamba3_s17random_hybrid_masked_seed42 `
  -TotalSteps 100000000
```

更稳的 curriculum 路线：

```powershell
.\scripts\train_overnight_mamba.ps1 `
  -EnvId MiniGrid-MemoryS11-v0 `
  -RunName mamba3_s11_hybrid_masked_seed42 `
  -MambaVariant mamba3 `
  -NoFallback `
  -TotalSteps 10000000
```

然后再挑战 `MiniGrid-MemoryS17Random-v0`。

## 重要参数

```text
--model mamba
--mamba-variant mamba3
--spatial-encoder hybrid
--valid-actions 0,1,2
--context-len 128
--chunk-len 128
--num-steps 256
--ent-coef 0.01
--ent-coef-final 0.001
```

`ent-coef` 会线性退火到 `ent-coef-final`：前期鼓励探索，后期让 greedy policy 更稳定。

## 评估和可视化

```powershell
micromamba run -n mamba_env python src\eval.py `
  runs\overnight_mamba3_s17random_hybrid_masked_seed42\model_latest.pt `
  --episodes 100

micromamba run -n mamba_env python src\visualize.py `
  runs\overnight_mamba3_s17random_hybrid_masked_seed42\model_latest.pt `
  --episodes 3
```

默认评估是 greedy。需要随机采样策略时加 `--stochastic`。

## 代码结构

```text
src/envs.py                  MiniGrid wrapper：compact obs + direction + side inputs
src/models.py                MLP/LSTM/Mamba/Attention actor-critic
src/ppo.py                   rollout buffer、GAE、PPO update
src/train_mamba_ppo.py       统一训练入口
src/eval.py                  checkpoint 评估
src/visualize.py             rollout 打印和视频保存
scripts/train_overnight_mamba.ps1
scripts/setup_mamba3_triton_windows.ps1
```

## 实验建议

推荐按顺序做：

```text
1. S11: MLP vs LSTM vs Mamba3
2. S13: LSTM vs Mamba3
3. S17Random: curriculum 后的泛化
4. spatial_encoder: transformer vs hybrid
5. context_len: 64 / 128 / 256
```

关键指标：

```text
eval/success_rate
eval/mean_return
eval/mean_length
charts/SPS
达到 80% success rate 所需环境步数
```
