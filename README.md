# Mamba-MiniGrid-Memory-Agent

用 Mamba 替代 LSTM，训练一个在 MiniGrid 部分可观察记忆任务中具备记忆能力的 PPO agent，并和无记忆 PPO、PPO-LSTM 做对比。

核心问题：

> 在 MiniGrid Memory 任务里，Mamba 是否比 LSTM 更适合作为 PPO agent 的记忆模块？

MiniGrid Memory 的关键难点是：agent 开局能看到目标物体，随后经过走廊，在岔路口必须记住之前看到的目标并选择匹配物体。默认观测使用紧凑的 `7x7x3` compact encoding，而不是 RGB 像素，这让实验更聚焦在“记忆与决策”而不是视觉识别。

## Agent 对比

| Agent | 记忆模块 | 训练入口 | 作用 |
| --- | --- | --- | --- |
| PPO-MLP | 无 | `src/train_mlp_baseline.py` 或 `--model mlp` | baseline，验证无记忆策略的上限 |
| PPO-LSTM | LSTM | `src/train_lstm_baseline.py` 或 `--model lstm` | 传统 recurrent baseline |
| PPO-Attention | Causal Transformer | `src/train_attention_baseline.py` 或 `--model attention` | 注意力序列 baseline |
| PPO-Mamba | Mamba SSM | `src/train_mamba_ppo.py` 或 `--model mamba` | 主实验模型 |

## 当前实现

```text
MiniGrid raw obs
  image:     7 x 7 x 3 semantic grid
  direction: scalar agent heading
  mission:   fixed MemoryEnv text, omitted by default
        |
object/color/state embeddings per cell
        |
49 grid tokens + learnable positions + direction CLS token
        |
spatial Transformer encoder
        |
frame embedding + prev_action + prev_reward + episode_start
        |
MLP / LSTM / temporal Mamba / causal temporal Attention
        |
actor head  -> 7 discrete actions
critic head -> value
```

Mamba、Attention 和 LSTM 都按连续轨迹 chunk 训练，不随机打散单步样本。在线决策时，模型接收最近 `context_len` 步上下文，并使用最后一个 token 输出当前动作和值函数。

默认推荐结构是：

```text
spatial attention per frame + temporal Mamba over trajectory
```

原因是 MiniGrid 的一帧不是自然图像，而是 49 个离散语义格子。空间 attention 更适合建模“目标物体、障碍、岔路口、候选物体”的关系；时间 Mamba 再负责跨走廊保留起点目标信息。

## 安装

推荐在 WSL2 Ubuntu 或 Linux CUDA 环境中编译最新 Mamba。先安装与你驱动匹配的 CUDA PyTorch，再安装本项目依赖。

```bash
python -m venv venv
source venv/bin/activate

# 按你的 CUDA/driver 从 PyTorch 官方页面选择命令
# 示例仅作占位，请以本机 CUDA 为准：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 让 mamba-ssm / causal-conv1d 使用当前 CUDA PyTorch 编译
pip install -r requirements.txt --no-build-isolation
```

截至 2026-05-11，PyPI 上 `mamba-ssm` 最新为 `2.3.1`，`causal-conv1d` 最新为 `1.6.1`；官方 GitHub release 最新为 `v2.3.2.post1`。官方 `state-spaces/mamba` 仓库说明：PyTorch 需要先装好，Mamba 安装应使用 `--no-build-isolation`；如果要追 GitHub 最新 release 或尝试 Mamba-3，需要从 source install。

Windows 原生环境可以跑本项目的 MLP/LSTM/Attention；最新 `mamba-ssm` 原生编译依赖 Triton/Linux 生态，建议用 WSL2/Linux 跑最新版 Mamba。当前仓库里的旧 `venv` 使用 `mamba_ssm 1.0.1`，可以作为 Windows 原生可跑的回滚环境。

```bash
# 可选：追 state-spaces/mamba 最新 source/release，而不是 PyPI 包
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
  git+https://github.com/state-spaces/mamba.git@v2.3.2.post1 \
  --no-build-isolation
```

## 快速运行

先用小任务和小步数做 smoke test：

```bash
python src/train_mamba_ppo.py --model mlp --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 4096 --num-envs 4 --num-steps 64 --eval-interval 4096

python src/train_lstm_baseline.py --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 4096 --num-envs 4 --num-steps 64 --context-len 32 --chunk-len 32

python src/train_attention_baseline.py --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 4096 --num-envs 4 --num-steps 64 --context-len 32 --chunk-len 32

python src/train_mamba_ppo.py --model mamba --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 4096 --num-envs 4 --num-steps 64 --context-len 32 --chunk-len 32
```

推荐的第一版正式配置：

```bash
python src/train_mamba_ppo.py --model mamba \
  --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 1000000 \
  --num-envs 16 \
  --num-steps 128 \
  --context-len 64 \
  --chunk-len 64 \
  --batch-chunks 8 \
  --d-model 128 \
  --spatial-layers 2 \
  --spatial-heads 4 \
  --mamba-variant mamba \
  --mamba-layers 2 \
  --d-state 16
```

如果你已经源码编译了新版 `mamba-ssm`，可以直接切 Mamba-2：

```bash
python src/train_mamba_ppo.py --model mamba --mamba-variant mamba2 \
  --env-id MiniGrid-MemoryS13-v0 \
  --total-steps 2000000 \
  --num-envs 16 \
  --num-steps 128 \
  --context-len 64 \
  --chunk-len 64 \
  --spatial-layers 2 \
  --spatial-heads 4 \
  --d-state 64
```

Attention baseline：

```bash
python src/train_attention_baseline.py \
  --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 1000000 \
  --num-envs 16 \
  --num-steps 128 \
  --context-len 64 \
  --chunk-len 64 \
  --batch-chunks 8 \
  --d-model 128 \
  --spatial-layers 2 \
  --attention-layers 2 \
  --attention-heads 4
```

同等配置跑 LSTM：

```bash
python src/train_lstm_baseline.py \
  --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 1000000 \
  --num-envs 16 \
  --num-steps 128 \
  --context-len 64 \
  --chunk-len 64 \
  --batch-chunks 8 \
  --d-model 128 \
  --lstm-layers 1
```

无记忆 baseline：

```bash
python src/train_mlp_baseline.py \
  --env-id MiniGrid-MemoryS11-v0 \
  --total-steps 1000000 \
  --num-envs 16 \
  --num-steps 128
```

## 评估与可视化

```bash
python src/eval.py runs/<run_name>/model_final.pt --episodes 100

python src/eval.py runs/<run_name>/model_final.pt \
  --env-id MiniGrid-MemoryS13Random-v0 --episodes 100

python src/visualize.py runs/<run_name>/model_final.pt --episodes 3
```

`eval.py` 默认使用 greedy action。需要随机采样策略时加 `--stochastic`。

## 实验路线

| 实验 | 环境 | 对比 | 目的 |
| --- | --- | --- | --- |
| A | `MiniGrid-MemoryS11-v0` | MLP / LSTM / Mamba | 验证任务、训练和记忆模块是否有效 |
| B | `MiniGrid-MemoryS13-v0` | LSTM / Mamba | 比较更长走廊下的 sample efficiency |
| C | train S11, eval S13Random | LSTM / Mamba | 测泛化能力 |
| D | S13Random 或 S17Random | context_len 16/32/64/128 | 测 Mamba 对上下文长度的敏感性 |

建议记录：

- `success_rate`
- `mean_return`
- `mean_episode_length`
- `SPS`
- 达到 80% success rate 所需环境步数
- 泛化差距：训练环境成功率 - 测试环境成功率

## 代码结构

```text
src/
  envs.py                  # MiniGrid compact obs + prev action/reward wrapper
  models.py                # spatial encoder + MLP/LSTM/Mamba/Attention policies
  ppo.py                   # RolloutBuffer + PPO update
  train_mamba_ppo.py       # 主训练入口，支持 --model mlp|lstm|attention|mamba
  train_attention_baseline.py # causal attention baseline 便捷入口
  train_lstm_baseline.py   # LSTM baseline 便捷入口
  train_mlp_baseline.py    # no-memory baseline 便捷入口
  eval.py                  # checkpoint 评估
  visualize.py             # greedy rollout + MP4 保存
```

## 本轮代码审查与修复

这次扩充时修掉了几个会直接影响实验可信度的问题：

- 环境 wrapper 现在保留 MiniGrid 的 `direction` 输入；旧版只保留 image，会丢掉朝向信息。
- 每帧 observation 现在使用 49-cell spatial attention encoder，而不是简单 flatten MLP。
- 新增 temporal causal-attention policy，可与 Mamba 在同一 PPO 管线下比较。
- Mamba action selection 现在会把当前 observation 放进 context window；旧实现第一步和每一步决策都滞后一格。
- PPO GAE 现在使用 rollout 末尾真实 bootstrap value；旧 trainer 会重新用 `buffer.values[-1]` 近似，优势估计不准确。
- Mamba/LSTM 更新现在按连续 trajectory chunks 批量训练，不随机打散单步样本，并优先避免 chunk 跨 episode 边界。
- `mamba_ssm` 改为延迟失败：没装 Mamba 时仍然可以 import 并运行 MLP/LSTM baseline。
- checkpoint 统一保存 `config_dict`，避免 dataclass pickle 破坏长期兼容性。
- eval 默认 greedy，并支持用 `--env-id` 在更长或 random Memory 环境上做泛化测试。
