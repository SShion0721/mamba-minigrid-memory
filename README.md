# Mamba-MiniGrid-Memory-Agent

用 Mamba/Mamba3、Gated Attention (GTrXL)、GRU 等序列模型训练 MiniGrid Memory 部分可观察任务中的 PPO agent，并与 MLP、LSTM、Attention 等 baseline 对比。

当前研究主线：

> 结构化对象槽 + episode-local cue memory，能否让 Mamba3 / GatedAttention / GRU 在 MiniGrid Memory 中更可靠地保存并调用起点 cue？

## 任务设定

MiniGrid Memory 的 agent 会先看到一个起始物体，随后穿过走廊，在岔路口选择与起始物体匹配的一侧。当前观测是局部 `7x7x3` compact semantic grid，不是 RGB 图像。

| 环境 | 特点 | 建议用途 |
|---|---|---|
| `MiniGrid-MemoryS11-v0` | 最小固定走廊 | smoke、调 bug、快速验证 |
| `MiniGrid-MemoryS13-v0` | 中等固定走廊 | 验证结构能否稳定学会 |
| `MiniGrid-MemoryS13Random-v0` | 中等随机走廊 | 主消融环境 |
| `MiniGrid-MemoryS17Random-v0` | 最大随机走廊 | 最终泛化目标 |

虽然官方环境暴露 `Discrete(7)` 动作空间，但 Memory 任务只需要导航动作：

```text
0 left
1 right
2 forward
```

因此本项目默认使用 action mask：`--valid-actions 0,1,2`。不要先打开其它动作做主实验，否则 PPO 会把大量样本浪费在 `pickup/drop/toggle/done` 上。

## 当前最好结构

默认主线已经切到 v2 slot-memory 结构：

```text
7x7x3 compact semantic grid
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
  iterative slot attention (--slot-extractor iterative)
        |
step-level slot fusion (--temporal-token-mode fuse)
        |
episode-local cue memory (--memory-kind episodic_cue)
  write: 只从 compact obs 中第一次看到的 object/color cue 写入
  read: 每步用 fused/global token 查询
  gate: learned gate 控制 retrieved cue 融合强度
        |
temporal backbone
  GRU / GatedAttention(ALiBi) / Mamba3
        |
actor head  -> masked action logits
critic head -> value
cue head    -> auxiliary cue recall
```

默认关键开关：

```text
--slot-count 4
--slot-extractor iterative
--slot-iters 3
--slot-mlp-ratio 2.0
--temporal-token-mode fuse
--memory-kind episodic_cue
--memory-slots 16
--memory-topk 4
--memory-write-window 12
--aux-recall-coef 0.05
```

保留的消融开关：

```text
--slot-extractor query_pool
--temporal-token-mode flatten
--memory-kind none
--aux-recall-coef 0.0
--slot-count 0
```

## 可用模型

| 模型 | `--model` | 建议定位 |
|---|---|---|
| MLP | `mlp` | 无记忆下限，不作为主线 |
| LSTM | `lstm` | 传统 recurrent baseline |
| GRU | `gru` | 低复杂度 recurrent baseline，先跑它确认结构有效 |
| Attention | `attention` | 标准 causal Transformer 消融 |
| Gated Attention | `gated_attention` | 主力 Transformer baseline，推荐 `--gated-attention-pos alibi` |
| Mamba | `mamba` | SSM backbone，主线用 `--mamba-variant mamba3` |

## 快速开始

所有命令建议在 `mamba_env` 中运行：

```powershell
micromamba run -n mamba_env python -m pytest tests -q
```

### 1. 最小 smoke

```powershell
micromamba run -n mamba_env python src\train_mamba_ppo.py `
    --model gru `
    --env-id MiniGrid-MemoryS11-v0 `
    --total-steps 200000 `
    --num-envs 16 `
    --num-steps 128 `
    --context-len 128 `
    --chunk-len 64 `
    --batch-chunks 8 `
    --d-model 128 `
    --eval-interval 20000 `
    --run-name smoke_slot_memory_gru
```

### 2. 主实验 GRU

```powershell
micromamba run -n mamba_env python src\train_mamba_ppo.py `
    --model gru `
    --env-id MiniGrid-MemoryS13Random-v0 `
    --total-steps 5000000 `
    --num-envs 32 `
    --num-steps 128 `
    --context-len 128 `
    --chunk-len 64 `
    --batch-chunks 8 `
    --d-model 128 `
    --lr 2.5e-4 `
    --run-name slot_memory_gru_s13random_seed42
```

### 3. 主实验 GatedAttention(ALiBi)

```powershell
micromamba run -n mamba_env python src\train_mamba_ppo.py `
    --model gated_attention `
    --gated-attention-pos alibi `
    --env-id MiniGrid-MemoryS13Random-v0 `
    --total-steps 10000000 `
    --num-envs 64 `
    --num-steps 128 `
    --context-len 128 `
    --chunk-len 64 `
    --batch-chunks 16 `
    --d-model 128 `
    --attention-layers 2 `
    --attention-heads 4 `
    --lr 2.5e-4 `
    --run-name slot_memory_gated_alibi_s13random_seed42
```

### 4. 主实验 Mamba3

```powershell
micromamba run -n mamba_env python src\train_mamba_ppo.py `
    --model mamba `
    --mamba-variant mamba3 `
    --env-id MiniGrid-MemoryS13Random-v0 `
    --total-steps 10000000 `
    --num-envs 64 `
    --num-steps 128 `
    --context-len 128 `
    --chunk-len 64 `
    --batch-chunks 16 `
    --d-model 128 `
    --mamba-layers 2 `
    --lr 1e-4 `
    --amp bf16 `
    --no-stateful-rollout `
    --run-name slot_memory_mamba3_s13random_seed42
```

### 5. Curriculum 到 S17Random

```powershell
micromamba run -n mamba_env python src\train_mamba_ppo.py `
    --model gated_attention `
    --gated-attention-pos alibi `
    --curriculum `
    --total-steps 30000000 `
    --num-envs 64 `
    --num-steps 128 `
    --context-len 128 `
    --chunk-len 64 `
    --batch-chunks 16 `
    --d-model 128 `
    --lr 2.5e-4 `
    --run-name slot_memory_gated_alibi_curriculum_seed42
```

默认 curriculum 顺序：

```text
S11 -> S13 -> S13Random -> S17Random
```

默认晋级线：

```text
--curriculum-thresholds 0.90,0.85,0.80
--curriculum-patience 3
```

## 消融实验

一键生成结构消融命令：

```powershell
.\scripts\run_slot_memory_ablation.ps1 `
    -EnvId MiniGrid-MemoryS13Random-v0 `
    -Models "gru,gated_attention,mamba3" `
    -Seeds "42,43,44" `
    -TotalSteps 5000000
```

默认只是 dry run 打印命令。真正执行加 `-Execute`：

```powershell
.\scripts\run_slot_memory_ablation.ps1 `
    -EnvId MiniGrid-MemoryS13Random-v0 `
    -Models "gru,gated_attention,mamba3" `
    -Seeds "42,43,44" `
    -TotalSteps 5000000 `
    -Execute
```

脚本覆盖以下结构：

| 名称 | 结构 |
|---|---|
| `query_pool_flatten_nomem` | 旧 query pooling + flatten token + no memory |
| `query_pool_fuse_nomem` | query pooling + step fusion + no memory |
| `iterative_fuse_nomem` | iterative slots + step fusion + no memory |
| `iterative_fuse_memory` | iterative slots + step fusion + cue memory |
| `iterative_fuse_memory_aux` | iterative slots + step fusion + cue memory + aux recall |

## 超参数怎么调

调参优先级建议固定为：

```text
1. 先固定 PPO，比较结构消融
2. 再比较 GRU / GatedAttention / Mamba3
3. 再调 context_len、chunk_len、num_envs、batch_chunks
4. 最后微调 lr、entropy、模型宽度
```

### 结构与记忆参数

| 参数 | 推荐值 | 什么时候改 |
|---|---:|---|
| `--valid-actions` | `0,1,2` | 主实验不要改，打开其它动作只做负向诊断 |
| `--spatial-encoder` | `hybrid` | `transformer` 只做 spatial encoder 消融 |
| `--spatial-layers` | `2` | `transformer` encoder 才主要受影响；速度紧张用 `1` |
| `--spatial-heads` | `4` | `d_model=256` 可试 `8` |

| 参数 | 推荐值 | 什么时候改 |
|---|---:|---|
| `--slot-count` | `4` | `0` 做无 slot 消融；`2` 提速；`8` 只在 S17Random 仍明显欠拟合时试 |
| `--slot-extractor` | `iterative` | `query_pool` 只做旧结构消融 |
| `--slot-iters` | `3` | 显存/速度紧张用 `2`；slot 指标差再试 `4` |
| `--slot-mlp-ratio` | `2.0` | 一般不动；大模型可试 `3.0` |
| `--temporal-token-mode` | `fuse` | `flatten` 只做消融；主线不要让 actor/critic 间接读 slot |
| `--memory-kind` | `episodic_cue` | `none` 是关键消融 |
| `--memory-slots` | `16` | S11/S13 可试 `8`；S17Random 可试 `32`，但先看收益 |
| `--memory-topk` | `4` | 读太散用 `1` 或 `2`；retrieval entropy 太低且不稳定再试 `8` |
| `--memory-write-window` | `12` | 必须覆盖 episode 早期看到 cue 的窗口；S11 可 `8`，随机长走廊可 `12` 或 `16` |
| `--aux-recall-coef` | `0.05` | `0.0` 做消融；aux loss 压主任务就降到 `0.01`；cue recall 长期低再试 `0.1` |

看 TensorBoard 指标时：

| 指标 | 解释 | 调整 |
|---|---|---|
| `loss/aux_recall_acc` | cue 能否被 hidden 表示预测出来 | 低于随机太久，先检查 memory 写入和 `aux_recall_coef` |
| `memory/gate_mean` | retrieved cue 融入强度 | 长期接近 `0` 表示 memory 没被用上 |
| `memory/retrieval_entropy` | 检索分布是否过散 | 很高说明读不准，降低 `memory_topk` 或增强 aux |
| `memory/write_rate` | 初期 cue 写入比例 | 长期为 `0` 说明 cue target 提取或 write window 有问题 |

### Backbone 参数

| 参数 | GRU | GatedAttention | Mamba3 |
|---|---:|---:|---:|
| `--d-model` | `128` | `128` 本地，`256` 大显存 | `128` 本地，`256` 大显存 |
| 层数 | `--gru-layers 1` | `--attention-layers 2` | `--mamba-layers 2` |
| heads | 不适用 | `--attention-heads 4`，大模型用 `8` | 不适用 |
| 位置编码 | 不适用 | `--gated-attention-pos alibi` | `--mamba-rope-fraction 0.5` |
| 学习率 | `2.5e-4` | `2.5e-4` 或 `1e-4` | `1e-4` 起步 |

Mamba3 额外参数：

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `--d-state` | `16` | 先不扩大，避免训练变慢 |
| `--d-conv` | `4` | 默认足够 |
| `--expand` | `2` | 默认足够 |
| `--mamba-headdim` | `64` | 通常与 `d_model=128/256` 搭配稳定 |
| `--mamba-ngroups` | `1` | 先不动 |
| `--mamba-chunk-size` | `64` | 与 `chunk_len=64` 对齐 |
| `--mamba-rope-fraction` | `0.5` | 先不动 |

LSTM 只作为 baseline，默认 `--lstm-layers 1`。如果 LSTM 明显欠拟合，可以试 `2` 层，但优先把同样预算给 GRU / GatedAttention / Mamba3 主线。

### PPO 与序列训练参数

| 参数 | 推荐值 | 调整规则 |
|---|---:|---|
| `--total-steps` | S11 `1M`，S13 `3M-10M`，S13Random `10M-30M`，S17Random `30M+` | smoke 可以小很多，正式结论不要只看早期曲线 |
| `--num-envs` | 本地 `16/32/64`，服务器 `128` | SPS 低但显存够就加；显存爆就降 |
| `--num-steps` | `128` | 先固定；短任务可 `64`，S17Random 可试 `256` |
| `--context-len` | `128` | 必须覆盖从看到 cue 到做决策的距离；S17Random 可试 `256` |
| `--chunk-len` | `64` | 必须 `<= num_steps`；稳定后可和 `context_len` 一起升到 `128` |
| `--batch-chunks` | 本地 `8/16`，服务器 `32` | 相当于序列 mini-batch 大小，显存爆就降 |
| `--batch-size` | `256` | 主要给非序列/旧路径使用；slot-memory 主线优先调 `batch-chunks` |
| `--n-epochs` | `4` | KL 过大或训练震荡降到 `2`；样本利用不足试 `6` |
| `--lr` | GRU/Gated `2.5e-4`，Mamba3 `1e-4` | collapse、KL 飙升、success 回落就降一档 |
| `--gamma` | `0.99` | MiniGrid Memory 先不动 |
| `--gae-lambda` | `0.95` | 方差大可试 `0.90`，通常不需要 |
| `--clip-coef` | `0.2` | policy ratio clipping 默认稳妥值 |
| `--clip-vloss` | 默认关闭 | 只在 value loss 抖动很大时做 PPO2-style 兼容实验 |
| `--vf-coef` | `0.5` | value loss 明显压 policy 再降到 `0.25` |
| `--ent-coef` | `0.01` | 探索不足升到 `0.02`；策略长期随机降到 `0.005` |
| `--ent-coef-final` | `0.001` | 后期 greedy 不稳可降到 `0.0005` |
| `--max-grad-norm` | `0.5` | 梯度尖峰多可降到 `0.25` |
| `--target-kl` | 默认不设 | 不稳定时试 `0.03` 或 `0.05` |
| `--dropout` | `0.0` | RL 中先不加；明显过拟合再试 `0.05` |
| `--spinning-penalty` | `0.0` | 只在明显原地转圈时试小值，不作为主实验默认 |
| `--spinning-threshold` | `10` | 配合 spinning penalty 使用 |

这些训练稳定性选项默认已经打开，主实验不要关闭：

```text
learning-rate annealing
advantage normalization
stateful rollout
```

对应的关闭开关分别是 `--no-anneal-lr`、`--no-norm-adv`、`--no-stateful-rollout`。`clip_vloss` 默认关闭；需要复现 PPO2 / CleanRL 风格时显式加 `--clip-vloss`。

### 运行与工程参数

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `--amp` | CUDA 上 `bf16`，不稳定就 `none` | `fp16` 更容易数值不稳 |
| `--compile` | 最后长跑再开 | 先不用它调 bug |
| `--eval-interval` | `20000` | 长跑可 `50000` |
| `--eval-episodes` | smoke `10`，正式 `30/100` | 报告结果建议 `100` |
| `--save-interval` | `100000` | 长跑保留 latest/final |
| `--log-interval` | `10` | 调试可降到 `1` |
| `--seed` | `42,43,44` | 主结论至少 3 seeds |
| `--run-name` | 自动生成或手动命名 | 正式实验建议写清 backbone、结构、环境和 seed |
| `--resume-from` | 空 | 从 `runs/<run_name>/model_latest.pt` 续训 |
| `--allow-legacy-load` | 默认不开 | 只有旧 checkpoint 兼容加载时显式打开 |

## 推荐实验顺序

第一组先证明结构价值，固定 `--model gru`：

```text
query_pool + flatten + no_memory
query_pool + fuse + no_memory
iterative + fuse + no_memory
iterative + fuse + episodic_cue_memory
iterative + fuse + episodic_cue_memory + aux_recall
```

第二组再证明 backbone 差异，固定最佳结构：

```text
GRU
GatedAttention + ALiBi
Mamba3
可选 Mamba2
```

第三组做难度迁移：

```text
S11 -> S13 -> S13Random -> S17Random
```

主指标：

```text
eval/success_rate       greedy 成功率，80% 是主目标线
eval/mean_return        平均 return
eval/mean_length        平均 episode 长度
charts/SPS              每秒环境步数
loss/aux_recall_acc     cue recall accuracy
memory/gate_mean        memory gate activation
memory/retrieval_entropy
memory/write_rate
达到 80% success 所需环境步数
```

## 评估和可视化

```powershell
micromamba run -n mamba_env python src\eval.py runs\<run_name>\model_latest.pt --episodes 100
micromamba run -n mamba_env python src\visualize.py runs\<run_name>\model_latest.pt --episodes 3
```

默认评估是 greedy。需要随机采样时加 `--stochastic`。旧 checkpoint 有字段不匹配时才加 `--allow-legacy-load`。

## Checkpoint 规则

新 checkpoint 会保存：

```text
schema version
最终 config
包版本
CUDA / Mamba variant
git sha
```

默认严格加载。旧实验只作参考，不作为严格对比结论。

## 代码结构

```text
src/envs.py                         MiniGrid wrapper：compact obs + direction + side inputs
src/cue.py                          从 compact obs 提取 object/color cue target
src/models.py                       MLP/LSTM/GRU/Mamba/Attention/GatedAttention actor-critic
src/ppo.py                          rollout buffer、GAE、PPO update、stateful context、aux recall loss
src/train_mamba_ppo.py              统一训练入口、curriculum、checkpoint、memory diagnostics
src/eval.py                         checkpoint 评估
src/visualize.py                    rollout 打印和视频保存
scripts/run_slot_memory_ablation.ps1  slot-memory 消融脚本
scripts/train_overnight_mamba.ps1   Windows overnight 训练脚本
tests/                              单元测试和 smoke 测试
```

