# HF 第一阶段：实现说明与服务器验证指南

本阶段基于 `qwen3-internvit` 分支提交 `ab0fc63fb9019ccabd61017eada584b8f09c4cd9`。新代码发布在 `hf-phase1` 分支。原训练代码保留，完成下述验收前继续使用原训练入口。

## 1. 这次完成了什么

当前主链路为 `stage1_runtime.py → models/qwen25_fork/build_model.py`；PT 经 `stage2lab.adapter` 调用它，SFT 和 `stage3lab` DPO 也使用它。模型继承 Qwen2.5-VL 的多模态接口，语言部分为 Qwen3 + 自定义 MRoPE，视觉部分为 InternViT + Qwen2.5 PatchMerger。Qwen2.5 checkpoint 只用于配置，未加载其语言或视觉权重。

| 新文件 | 职责 |
|---|---|
| `hybrid_vlm/configuration_hybrid.py` | 保存真实 Qwen3、InternViT、MRoPE、网格及分组件精度配置 |
| `hybrid_vlm/modeling_hybrid.py` | 仅凭 config 构造模型，支持 HF 保存/加载与生成 |
| `hybrid_vlm/processing_hybrid.py` | Processor 保存/加载，保留原图像切块、视频帧、占位符展开 |
| `hybrid_vlm/image_processing_hybrid.py` | 将 448 像素切块及最多 2×2 切块参数序列化 |
| `hybrid_vlm/qwen3_mrope.py`、`visual_intern.py` | 保留原 MRoPE、patch 排列及 merger 计算 |
| `tools/convert_hf_checkpoint.py` | 严格转换 PT / SFT / DPO checkpoint |
| `tools/verify_hf_export.py` | 旧模型与导出模型逐项数值对照 |
| `tools/infer_hf.py` | 仅依赖安装包和导出目录的离线推理 |
| `tools/eval_hf_sft.py` | 复用原 SFT 数据及 loss 逻辑做验证集评估 |
| `tests/hf/test_roundtrip.py` | 不需要下载真实权重的小型模型测试 |

导出目录包含 config、完整 safetensors 权重（可能分片）、tokenizer、image processor、processor 和 generation config。加载不再需要三个原始 checkpoint 路径。`EXPORT_COMPLETE.json` 记录来源、精度及完整导出标记；它不是服务器数值验证通过证明。

**两个必须保留的兼容细节：**

- SFT 的语言模型、lm_head、merger 是 FP32，冻结 InternViT 是 BF16。默认 `from_pretrained()` 保留这种组合；显式传 `dtype=torch.bfloat16` 会主动降低 FP32 权重精度，不能用它作无损验收。
- 旧构建器创建新 Qwen3 backbone 后复制权重，同时复用原 Qwen3 的 lm_head。这使 embedding 和 lm_head 成为独立参数，尽管原始 Qwen3 配置可能写着共享。新配置记录实际共享关系，避免 HF 自动绑权重破坏已训练参数。

本阶段不改变训练目标、优化器、DPO 数据或 loss，也不迁移训练入口。HF 导出是模型资产，不含 optimizer、scheduler、随机数状态，不能替代旧训练 resume checkpoint。参数冻结策略也需要训练入口重新明确设置，不能把 eval() 当成冻结参数。这里只保证所覆盖接口的兼容；没有承诺任意 pipeline、Trainer、量化、device_map、vLLM 或远端 trust_remote_code 能直接使用。

## 2. 在服务器取代码

从 GitHub 下载更新用 `fetch` / `pull`，`push` 是上传。建议用独立 worktree，避免干扰正在使用的旧目录：

```bash
cd /mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training
git fetch origin hf-phase1
git worktree add ../multimodal_training_hf_phase1 origin/hf-phase1
cd ../multimodal_training_hf_phase1
git rev-parse HEAD
```

这会得到独立工作目录，原目录中的 checkpoints 和 runs 不必复制。若目标目录已经存在，使用另一个目录名；不要删除正在运行的训练目录。

先激活原来能运行模型的 Python/CUDA 环境，再创建可丢弃的测试环境，继承原环境的 CUDA PyTorch：

```bash
python -m venv --system-site-packages .venv-hf-phase1
source .venv-hf-phase1/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install 'transformers==4.57.6' 'timm==1.0.22' 'einops==0.8.2' 'pytest>=8,<9'
python -m pip install -e . --no-deps
python -c "import hybrid_vlm, torch; print('IMPORT PASS'); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -m pytest tests/hf -q
```

要求 Python ≥ 3.10、PyTorch ≥ 2.6，并保留匹配的 CUDA torch/torchvision 安装。`--no-deps` 避免安装项目时更换服务器 PyTorch；若缺 safetensors/Pillow，补装该缺失项。不要照搬本地测试机的 CPU torch。首次安装依赖需要联网，后面的独立加载不需要联网。

本地已在 Python 3.12、torch 2.6.0+cpu、torchvision 0.21.0+cpu、transformers 4.57.6、timm 1.0.22 上通过 5 个测试，覆盖完整小模型导出/重载及多模态验证。未在此环境运行真实权重、A40 或真实数据集评估。

## 3. 首先转换 SFT checkpoint

SFT checkpoint 已包含 merger，所以无需再加载 PT checkpoint。冻结视觉权重从原始 InternViT checkpoint 补齐。原始路径必须对应实际训练使用的版本，不能换成名称相似的其它模型。

设置一次以下变量。`HF_PROJECT` 是新工作目录；`LEGACY_PROJECT` 是原目录：

```bash
export HF_PROJECT="$(pwd)"
export LEGACY_PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training
export Q25="$LEGACY_PROJECT/checkpoints/Qwen2.5-VL-3B-Instruct"
export VIT="$LEGACY_PROJECT/checkpoints/InternViT-300M-448px-V2_5"
export LLM="$LEGACY_PROJECT/checkpoints/Qwen3-0.6B-Base"
export SFT="$LEGACY_PROJECT/runs/sft_150k_ddp_6gpu/best_model.pt"
export HF_EXPORT="$LEGACY_PROJECT/runs/hf_phase1_sft_v1"
mkdir -p "$HF_PROJECT/validation_logs"
set -o pipefail
python tools/convert_hf_checkpoint.py \
  --stage sft --checkpoint "$SFT" \
  --qwen25 "$Q25" --vit "$VIT" --llm "$LLM" \
  --output "$HF_EXPORT" \
  2>&1 | tee validation_logs/convert_sft.log
```

转换在 CPU 上执行，不占用训练 GPU；会同时持有多个模型/权重副本，请预留充足主机内存和数 GB 以上磁盘空间。耗时取决于磁盘、CPU 和 checkpoint 大小。工具拒绝覆盖已有输出目录；失败后保留不完整目录用于排查，重试时选择新目录名。

成功要求：进程退出码为 0，出现 `EXPORT PASS`，且导出目录存在 `EXPORT_COMPLETE.json`。转换会验证所有应有键、形状、有限数值和逐 tensor 精度/值，任何缺键都终止。不能用 `strict=False` 跳过这些错误。

PT 使用 `--stage pt`，checkpoint 为原 merger 字典或 `{'merger': ...}`。DPO 使用 `--stage dpo`，checkpoint 为完整 state dict 或 `{'state_dict': ...}`。两者都换成独立输出目录。支持格式来自当前仓库；其它包装格式或带 `module.` 前缀会明确报错，不会猜测转换。

## 4. 必做：真实权重新旧数值对照

选一张真实验证图片，替换下方路径；选择空闲 GPU，例如 GPU 0。这一步用普通 Python 单进程，不用 torchrun，也不占 6 张卡：

```bash
export VERIFY_IMAGE=/absolute/path/to/a/validation_image.jpg
CUDA_VISIBLE_DEVICES=0 python tools/verify_hf_export.py \
  --stage sft --checkpoint "$SFT" \
  --qwen25 "$Q25" --vit "$VIT" --llm "$LLM" \
  --export "$HF_EXPORT" --device cuda --precision preserve \
  --suite full --image "$VERIFY_IMAGE" --max-new-tokens 8 \
  --report validation_logs/sft_preserve.json \
  2>&1 | tee validation_logs/sft_preserve.log
```

这会同时加载原模型和新模型，并使用相同的 BF16 autocast，检查：

1. HF 加载没有缺失、多余、形状不匹配权重，全部权重 dtype 和值逐项完全一致。
2. 纯文本、单切块、4 切块、多图、不等长 padding batch、视频、图像与视频混合，以及你的真实图片。
3. 处理后的 token、像素、grid、视频时间字段，以及 MRoPE position 和 rope delta。
4. logits 和带 labels 的 loss（默认 atol/rtol 各 1e-5；不会偷偷放宽）。
5. 每个用例的贪心与 2-beam token ID；纯文本及单图的 KV cache 与无 cache 生成一致性。

期望报告 `status: PASS`，所有 cases 为 PASS，并以退出码 0 结束。无须要求生成内容一定“好”：这里验证封装等价，生成质量由原 checkpoint 决定。若报告失败，保留 JSON 和完整日志，先查首个差异，不要调大容差使其通过。

再做一次统一 FP32 对照，以区分封装差异和混合精度运行差异。这一步将原模型、新模型都转 FP32，不会改写磁盘权重：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/verify_hf_export.py \
  --stage sft --checkpoint "$SFT" \
  --qwen25 "$Q25" --vit "$VIT" --llm "$LLM" \
  --export "$HF_EXPORT" --device cuda --precision float32 \
  --suite basic --image "$VERIFY_IMAGE" --max-new-tokens 8 \
  --report validation_logs/sft_fp32.json \
  2>&1 | tee validation_logs/sft_fp32.log
```

显存紧张时，先用 `--suite basic --max-new-tokens 2` 定位；最终 full 验收仍需补齐。也可在 CPU 上运行 FP32 数值验证，但真实模型会慢很多。验证器会记录失败原因并返回非零退出码。

## 5. 必做：离线、新进程、脱离旧源码目录加载

以下推理脚本只 import 安装好的 `hybrid_vlm`，不 import `models.qwen25_fork` 或 stage runtime。使用新空目录和空 HF 缓存；原始权重路径不作为参数传入：

```bash
export OFFLINE_CHECK_DIR="$(mktemp -d)"
cp "$HF_PROJECT/tools/infer_hf.py" "$OFFLINE_CHECK_DIR/infer_hf.py"
cd "$OFFLINE_CHECK_DIR"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$OFFLINE_CHECK_DIR/empty_cache" \
  CUDA_VISIBLE_DEVICES=0 python infer_hf.py \
  --model "$HF_EXPORT" --image "$VERIFY_IMAGE" \
  --text '请描述这张图片。' --max-new-tokens 64 \
  2>&1 | tee "$HF_PROJECT/validation_logs/offline_infer.log"
cd "$HF_PROJECT"
```

这里仍然需要安装包代码；`-e .` 是开发模式，包源码位于新工作目录。若要把模型交付到另一台机器，先构建 wheel：

```bash
python -m pip wheel . --no-deps -w dist
```

另一台机器安装生成的 wheel 和匹配依赖，再复制导出模型目录即可，不需要整个旧仓库。第一阶段使用本地 AutoClass 注册，代码中必须先 `import hybrid_vlm`；没有实现“完全不安装包、仅靠 Hub remote code 加载”。

## 6. 必做：原验证集 loss 对照

数值用例通过后，用你之前 SFT 的同一 JSON、图片目录和固定 val indices 评估，保持 max_tokens 相同。不要重新随机划分验证集。

```bash
export VAL_JSON=/absolute/path/to/original_sft_annotations.json
export IMAGE_ROOT=/absolute/path/to/images
export VAL_INDICES=/absolute/path/to/original_val_indices.json
export PT=/absolute/path/to/original_pt_merger.pt

CUDA_VISIBLE_DEVICES=0 python "$LEGACY_PROJECT/scripts/sft/eval_sft.py" \
  --json "$VAL_JSON" --image-root "$IMAGE_ROOT" --indices "$VAL_INDICES" \
  --pt-checkpoint "$PT" --sft-checkpoint "$SFT" --max-tokens 4096 --limit 32 \
  2>&1 | tee validation_logs/legacy_val32.log

CUDA_VISIBLE_DEVICES=0 python tools/eval_hf_sft.py \
  --model "$HF_EXPORT" --json "$VAL_JSON" --image-root "$IMAGE_ROOT" \
  --indices "$VAL_INDICES" --max-tokens 4096 --limit 32 \
  --report validation_logs/hf_val32.json \
  2>&1 | tee validation_logs/hf_val32.log
```

先对照 32 条，之后两边都去掉 `--limit 32` 跑完整固定验证集。两边 `records`、`supervised_tokens` 必须完全一致；token_loss 应在浮点误差范围一致，建议以绝对差 ≤1e-5 作为初始验收线。更大的差异需要排查。旧评估器参数仍要求 PT checkpoint；新导出评估不需要它。

这一步沿用原 SFTCollator 和 token 加权 loss，不引入新的数据筛选、prompt 或标签规则。第一阶段不保证准确率提升，也不预期模型质量变化；若质量显著改变，首先视为需要定位的兼容问题。

## 7. 后续代码如何加载

```python
import torch
import hybrid_vlm  # 注册自定义 AutoClasses，必须执行
from transformers import AutoModelForImageTextToText, AutoProcessor

path = '/absolute/path/to/hf_phase1_sft_v1'
model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True).cuda().eval()
processor = AutoProcessor.from_pretrained(path, local_files_only=True)
prompt = processor.user_prompt('你好')
batch = processor(text=prompt, return_tensors='pt').to('cuda')
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    ids = model.generate(**batch, do_sample=False, max_new_tokens=32)
print(processor.tokenizer.decode(ids[0, batch['input_ids'].shape[1]:], skip_special_tokens=True))
```

图像可传 PIL 图片或单样本图片列表；多样本用嵌套列表。视频输入为已经解码的 PIL 帧列表，沿用 `encode_batch(..., videos_per_sample=..., fps_per_sample=...)`。文本需要已经渲染的原 ChatML prompt，或调用 `user_prompt()`；本阶段不替换聊天模板。训练时设置 tokenizer.padding_side='right' 并继续用原 collator，生成时用 left。

## 8. 问题定位与验收结果

| 现象 | 首先检查 |
|---|---|
| 找不到自定义 model_type | 是否安装包，是否先 import hybrid_vlm，transformers 是否 4.57.6 |
| 缺键/多键/形状错误 | stage 是否选对，checkpoint 是否完整，三个基础模型是否是训练使用版本 |
| 权重完全一致但 logits 不同 | 原服务器的 InternViT 自定义代码版本、apex 等可选 kernel、精度及 attention backend |
| 图像 token 数错误 | tokenizer special token、图片数与占位符、448 预处理及切块配置 |
| 离线加载仍访问原模型 | 是否误用了旧 build_model，是否指向完整导出目录 |
| CUDA OOM | 是否有其它训练占卡，先缩短生成并跑 basic；FP32 比 preserve 占显存更多 |
| checkpoint 加载成功但续训不一致 | HF 导出不包含优化器/调度器/RNG；继续从旧训练 checkpoint 恢复 |

InternViT 的源码已放入包内并保留许可证，来源与局部改动见 `hybrid_vlm/THIRD_PARTY.md`。它固定使用非 flash PyTorch 路径；若旧服务器额外安装 apex，必须以真实数值对照结果为准。

验收完成请提供：Git commit ID、转换日志、`sft_preserve.json`、`sft_fp32.json`、离线推理日志，以及新旧验证集 token_loss/records/supervised_tokens。不要上传模型权重。全部通过后，第二阶段再把 PT/SFT/DPO 的加载和保存入口切换到这个包，并单独验证训练步及恢复训练；优化 DPO 逻辑应另开变更，以便追踪收益或回归。
