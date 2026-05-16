# Mamba-MiniGrid-Memory-Agent

用 Mamba/Mamba3、Gated Attention (GTrXL)、GRU 等序列模型训练 MiniGrid Memory 部分可观察任务中的 PPO agent，并与 MLP、LSTM 等 baseline 对比。

核心问题：

> 在 MiniGrid Memory 中，选择性状态空间模型和门控 Transformer 能否作为 PPO agent 的高效记忆模块？

## 任务设定

MiniGrid Memory 的 agent 会先看到一个起始物体，随后穿过走廊，在岔路口必须选择与起始物体匹配的一侧。当前观测是局部 `7x7x3` compact semantic grid，不是 RGB 图像。

变体难度递增：

| 环境 | 特点 |
|---|---|
| `MiniGrid-MemoryS11-v0` | 最小固定走廊，最容易 |
| `MiniGrid-MemoryS13-v0` | 中等固定走廊 |
| `MiniGrid-MemoryS13Random-v0` | 中等随机走廊 |
| `MiniGrid-MemoryS17Random-v0` | 最大随机走廊，最难 |

虽然官方环境暴露 `Discrete(7)` 动作空间，但 Memory 任务只需要导航动作：

```text
0 left
1 right
2 forward
```

`pickup/drop/toggle/done` 对该任务没有帮助，保留它们会让 PPO 浪费大量样本。因此本项目默认使用 action mask：`--valid-actions 0,1,2`。

## 可用模型

| 模型 | `--model` | 说明 |
|---|---|---|
| MLP | `mlp` | Feedforward baseline，无记忆 |
| LSTM | `lstm` | 循环 baseline |
| GRU | `gru` | 轻量循环 baseline |
| Attention | `attention` | 因果 Transformer (标准残差) |
| Gated Attention | `gated_attention` | GTrXL 风格 (GRU 门控残差 + FlashAttention) |
| Mamba | `mamba` | SSM backbone，variant 可选 mamba/mamba2/mamba3 |

## 当前网络架构

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
  optional TNL3 slot attention (--slot-count N)
  direction-conditioned summary
        |
trajectory tokens: [B, T, d_model]
        |
  ┌─────────────────────────────────┐
  │ Mamba / GatedAttention / LSTM   │  ← temporal backbone
  │ GRU / Attention / MLP           │
  └─────────────────────────────────┘
        |
LayerNorm
        |
actor head  -> masked action logits
critic head -> value
```

旧的 spatial Transformer 仍可通过 `--spatial-encoder transformer` 启用，用于消融实验。默认 `hybrid` — 局部卷积更高效地建模墙、走廊和岔路结构，saliency pooling 更直接地提取可见物体线索。

## 为什么 Gated Attention / Mamba 适合这个任务

**Gated Attention (GTrXL):** 来自 Parisotto et al. "Stabilizing Transformers for RL"。用 GRU 风格的三门（reset/update/candidate）替代标准残差连接，解决 Transformer 在 RL 中的训练不稳定问题。update gate 初始化为 near-identity，让 PPO 早期梯度更平稳。

**Mamba:** 来自选择性状态空间模型。每一步 token 是输入 `u_t`，Mamba 维护隐状态 `h_t`：

```text
h_t = A_t h_{t-1} + B_t u_t
y_t = C_t h_t
```

其中 `A_t/B_t/C_t` 动态依赖当前输入。隐状态可以学习保存：起始物体是什么、当前在走廊还是岔路口、该走哪条分支。

## 快速训练

### Kaggle (推荐 T4 x2, 15GB+)

```bash
python src/train_mamba_ppo.py \
    --model gated_attention \
    --env-id MiniGrid-MemoryS17Random-v0 \
    --num-envs 128 \
    --num-steps 128 \
    --context-len 128 \
    --chunk-len 128 \
    --batch-chunks 32 \
    --d-model 256 \
    --attention-layers 2 \
    --attention-heads 8 \
    --lr 1e-4 \
    --total-steps 2000000
```

### Windows 本地 (RTX 3060 6GB+)

```powershell
python src/train_mamba_ppo.py `
    --model gated_attention `
    --env-id MiniGrid-MemoryS17Random-v0 `
    --num-envs 64 `
    --num-steps 128 `
    --context-len 128 `
    --chunk-len 64 `
    --batch-chunks 16 `
    --d-model 128 `
    --attention-layers 2 `
    --attention-heads 4 `
    --lr 2.5e-4 `
    --total-steps 2000000
```

### 断点续训

```bash
python src/train_mamba_ppo.py \
    --resume-from runs/<run_name>/model_latest.pt \
    ...  # 其余参数与首次训练一致
```

### 脚本方式 (mamba3 overnight)

```powershell
.\scripts\train_overnight_mamba.ps1 `
    -MambaVariant mamba3 `
    -EnvId MiniGrid-MemoryS17Random-v0 `
    -TotalSteps 100000000
```

## 重要参数

```text
--model gated_attention       序列模型选择
--mamba-variant mamba3        mamba/mamba2/mamba3
--spatial-encoder hybrid      hybrid 或 transformer
--valid-actions 0,1,2         动作掩码
--context-len 128             上下文窗口长度
--chunk-len 128               PPO 序列 chunk 长度
--num-steps 128               每次 rollout 步数
--num-envs 128                并行环境数
--batch-chunks 32             每次 PPO update 的 chunk 数
--d-model 128/256/512         模型维度
--ent-coef 0.01               entropy 系数起始值
--ent-coef-final 0.001        线性退火终值
--slot-count 4                TNL3 slot 数量 (0 = 关闭)
--spinning-penalty 0.0        原地打转惩罚
--spinning-threshold 10       容忍步数
--amp bf16                    混合精度 (none / bf16 / fp16)
--torch-compile               torch.compile 加速
```

`ent-coef` 会线性退火到 `ent-coef-final`：前期鼓励探索，后期让 greedy policy 更稳定。

## 评估和可视化

```bash
python src/eval.py runs/<run_name>/model_latest.pt --episodes 100
python src/visualize.py runs/<run_name>/model_latest.pt --episodes 3
```

默认评估是 greedy。需要随机采样加 `--stochastic`。加载旧 checkpoint 有 key 不匹配时加 `--allow-legacy-load`。

## 代码结构

```text
src/envs.py                          MiniGrid wrapper：compact obs + direction + side inputs
src/models.py                        MLP/LSTM/GRU/Mamba/Attention/GatedAttention actor-critic
src/ppo.py                           rollout buffer、GAE、PPO update、stateful context
src/train_mamba_ppo.py               统一训练入口
src/train_mlp_baseline.py            MLP 单独训练
src/train_lstm_baseline.py           LSTM 单独训练
src/train_attention_baseline.py      Attention 单独训练
src/train_gru_baseline.py            GRU 单独训练
src/impala.py                        Impala/V-trace trainer (实验性)
src/r2d2.py                          R2D2 recurrent replay trainer (实验性)
src/eval.py                          checkpoint 评估
src/visualize.py                     rollout 打印和视频保存
scripts/train_overnight_mamba.ps1    Windows overnight 训练脚本
scripts/setup_mamba3_triton_windows.ps1  Mamba3/Triton 环境配置
tests/                               测试文件
```

## 实验建议

推荐按顺序做：

```text
1. S11: MLP vs LSTM vs GRU vs Mamba3 vs GatedAttention
2. S13: LSTM vs Mamba3 vs GatedAttention
3. S17Random: curriculum (S11→S13→S13Random→S17Random) 后的泛化
4. spatial_encoder: transformer vs hybrid
5. context_len: 64 / 128 / 256
6. slot attention: --slot-count 0/4/8 消融
7. GatedAttention vs Attention (GRU 门控消融)
```

关键指标：

```text
eval/success_rate       成功率 (>80% 为通过线)
eval/mean_return        平均 return
eval/mean_length        平均 episode 长度
charts/SPS              每秒环境步数
达到 80% success rate 所需环境步数
```
