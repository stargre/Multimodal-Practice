"""Unified functional smoke/regression test for Task 1.
Run from multimodal_training/ after copying the models/ directory here.

  python test_task1_all.py --qwen25 ./checkpoints/Qwen2.5-VL-3B-Instruct \
      --vit ./checkpoints/InternViT-300M-448px-V2_5 \
      --llm ./checkpoints/Qwen3-0.6B-Base --image ./assets/test.jpg

The default suite includes Beam Search and a checkpoint roundtrip.
It writes task1_test_report.json. No monkey-patching is used.
"""
import argparse
import gc
import json
import traceback
from pathlib import Path

import torch
from PIL import Image
import transformers
from transformers import AutoConfig, CLIPImageProcessor
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from models.qwen25_fork.build_model import build_model
from models.qwen25_fork.checkpoint import save_checkpoint, load_checkpoint
from models.qwen25_fork.processing_hybrid import HybridQwen25Processor
from models.qwen25_fork.qwen3_mrope import Qwen3MRoPE
from models.qwen25_fork.visual_intern import InternViTQwenVisual
from models.qwen25_fork.modeling_hybrid import HybridQwen25VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen25", default="./checkpoints/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--vit", default="./checkpoints/InternViT-300M-448px-V2_5")
    parser.add_argument("--llm", default="./checkpoints/Qwen3-0.6B-Base")
    parser.add_argument("--image", default="./assets/test.jpg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-beam", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--fp32",action="store_true")
    args = parser.parse_args()

    if transformers.__version__ != "4.57.6":
        raise RuntimeError(
            f"This source is version-pinned to transformers==4.57.6, found {transformers.__version__}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 or device.type == "cpu" else torch.bfloat16
    print(f"Environment: torch={torch.__version__}, transformers={transformers.__version__}, device={device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This test requires a GPU with BF16 support")
    torch.manual_seed(1234)

    report = {}
    def run(name, fn):
        print(f"\n========== {name} ==========")
        try:
            fn()
            report[name] = {"status": "PASS"}
            print("PASS:", name)
        except Exception as exc:
            report[name] = {"status": "FAIL", "error": repr(exc)}
            print("FAIL:", name)
            traceback.print_exc(limit=7)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    qcfg = AutoConfig.from_pretrained(args.llm)

    def check_rope():
        base = Qwen3RotaryEmbedding(qcfg).to(device)
        custom = Qwen3MRoPE(qcfg).to(device)
        dummy = torch.zeros((1, 8, qcfg.hidden_size), device=device, dtype=dtype)
        pos = torch.arange(8, device=device).unsqueeze(0)
        c0, s0 = base(dummy, pos)
        c1, s1 = custom(dummy, pos.unsqueeze(0).expand(3, -1, -1))
        torch.testing.assert_close(c0, c1)
        torch.testing.assert_close(s0, s1)
        coords = torch.stack([pos, pos // 2, pos % 2])
        c2, s2 = custom(dummy, coords)
        assert c2.shape == s2.shape == (1, 8, 128)
        assert torch.isfinite(c2).all()

    run("01 MRoPE numerical equivalence", check_rope)

    def check_patch_order():
        x = torch.arange(16).reshape(1, 4, 4, 1)
        x = x.reshape(1, 2, 2, 2, 2, 1).permute(0, 1, 3, 2, 4, 5).reshape(4, 4)
        expect = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        assert torch.equal(x, expect)

    run("02 2x2 patch spatial order", check_patch_order)

    try:
        model, tokenizer = build_model(
            device=str(device), qwen25_path=args.qwen25,
            vit_path=args.vit, llm_path=args.llm, dtype=dtype,
        )
        tokenizer.padding_side = "left"  # decoder-only batched inference
        image_processor = CLIPImageProcessor.from_pretrained(args.vit)
        processor = HybridQwen25Processor(tokenizer, image_processor, max_tiles_per_side=2)
        if Path(args.image).is_file():
            img = Image.open(args.image).convert("RGB")
        else:
            print("WARNING: test image missing, using a synthetic RGB image")
            img = Image.new("RGB", (448, 448), (80, 130, 180))
        # Normalize the *test* image to one tile; the 896x896 case is tested separately.
        img = img.resize((448, 448))
    except Exception:
        print("FATAL: failed to construct model or Processor")
        traceback.print_exc()
        raise

    def dev(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    prompt = processor.user_prompt("请描述这张图片。", num_images=1)
    one = dev(processor.encode_images([prompt], [[img]]))
    imid = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vidid = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def check_components():
        assert isinstance(model, HybridQwen25VLForConditionalGeneration)
        assert isinstance(model.model.visual, InternViTQwenVisual)
        assert model.model.language_model.config.hidden_size == 1024
        assert model.model.language_model.config.num_hidden_layers == 28
        assert model.lm_head.out_features == model.config.text_config.vocab_size
        assert int((one["input_ids"] == imid).sum()) == 256
        assert one["image_grid_thw"].tolist() == [[1, 32, 32]]
        print("Visual type:", type(model.model.visual.vit).__name__)
        print("Decoder type:", type(model.model.language_model).__name__)

    run("03 actual backbone replacement", check_components)

    def check_visual():
        with torch.inference_mode():
            feat = model.model.get_image_features(one["pixel_values"], one["image_grid_thw"])
        expected = one["image_grid_thw"].prod(-1).div(4, rounding_mode="floor").tolist()
        assert [len(x) for x in feat] == expected
        assert all(x.shape[1] == 1024 and torch.isfinite(x).all() for x in feat)
        print("Visual tokens per image:", expected)

    run("04 InternViT + Qwen2.5 patch merger", check_visual)

    def check_text():
        text_inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
        model.model.rope_deltas = None
        with torch.inference_mode():
            out = model(**text_inputs, use_cache=False)
        assert out.logits.shape == (1, text_inputs.input_ids.shape[1], model.lm_head.out_features)
        assert torch.isfinite(out.logits).all()

    run("05 plain-text forward", check_text)

    def check_single():
        model.model.rope_deltas = None
        with torch.inference_mode():
            out = model(**one, use_cache=False)
        assert out.logits.shape == (1, one["input_ids"].shape[1], model.lm_head.out_features)
        assert torch.isfinite(out.logits).all()
        pids, delta = model.model.get_rope_index(
            one["input_ids"], image_grid_thw=one["image_grid_thw"],
            attention_mask=one["attention_mask"],
        )
        assert pids.shape == (3, 1, one["input_ids"].shape[1])
        assert delta.shape == (1, 1)
        print("Single image logits:", tuple(out.logits.shape))

    run("06 single-image Qwen2.5-VL forward and MRoPE", check_single)

    large = img.resize((896, 896))
    multi_prompt = processor.user_prompt("比较两张图。", num_images=2)
    multi = dev(processor.encode_images([multi_prompt], [[img, large]]))
    def check_multi():
        expected = int((multi["image_grid_thw"].prod(-1) // 4).sum())
        assert int((multi["input_ids"] == imid).sum()) == expected
        assert multi["pixel_values"].shape[0] == 5
        model.model.rope_deltas = None
        with torch.inference_mode():
            out = model(**multi, use_cache=False)
        assert torch.isfinite(out.logits).all()
        print("Multi-image tokens:", expected, "tiles:", len(multi["pixel_values"]))

    run("07 multi-image and dynamic 2x2 tiling", check_multi)

    def check_training():
        answer_ids = tokenizer("这是一张图片。<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        train_ids = torch.cat([one["input_ids"], answer_ids], dim=1)
        labels = torch.full_like(train_ids, -100)
        labels[:, -answer_ids.shape[1]:] = answer_ids
        model.model.visual.vit.requires_grad_(False)
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
        model.model.visual.merger.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model.model.rope_deltas = None
        out = model(
            input_ids=train_ids, attention_mask=torch.ones_like(train_ids),
            pixel_values=one["pixel_values"], image_grid_thw=one["image_grid_thw"],
            labels=labels, use_cache=False,
        )
        assert torch.isfinite(out.loss)
        out.loss.backward()
        total = sum(
            p.grad.float().norm().item() for p in model.model.visual.merger.parameters()
            if p.grad is not None
        )
        assert total > 0 and math_isfinite(total)
        assert all(p.grad is None for p in model.model.language_model.parameters())
        print(
            "Loss:",
            out.loss.detach().item(),
            "Merger grad norm sum:",
            total,
        )
        model.zero_grad(set_to_none=True)
        model.eval()

    run("08 loss and backward to original Qwen patch merger", check_training)

    video_prompt = processor.user_prompt("请描述视频。", num_videos=1)
    video = dev(processor.encode_video(video_prompt, frames=[img, img.copy()], fps=2.0))

    def check_video():
        tokens = int((video["video_grid_thw"].prod(-1) // 4).sum())
        assert int((video["input_ids"] == vidid).sum()) == tokens
        with torch.inference_mode():
            feats = model.model.get_video_features(video["pixel_values_videos"], video["video_grid_thw"])
        assert sum(len(x) for x in feats) == tokens
        ids, _ = model.model.get_rope_index(
            video["input_ids"], video_grid_thw=video["video_grid_thw"],
            second_per_grid_ts=video["second_per_grid_ts"], attention_mask=video["attention_mask"],
        )
        video_positions = ids[0, 0, video["input_ids"][0] == vidid]
        half = tokens // 2
        assert video_positions[half] > video_positions[0]
        model.model.rope_deltas = None
        with torch.inference_mode():
            out = model(**video, use_cache=False)
        assert torch.isfinite(out.logits).all()
        print("Video grid:", video["video_grid_thw"].tolist(), "time steps:", video_positions[0].item(), video_positions[half].item())

    run("09 multi-frame video MRoPE and forward", check_video)

    def check_cache():
        L = one["input_ids"].shape[1]
        tok = tokenizer.encode("好", add_special_tokens=False)[0]
        next_ids = torch.tensor([[tok]], dtype=torch.long, device=device)
        model.model.rope_deltas = None
        with torch.inference_mode():
            prefill = model(**one, use_cache=True)
            assert prefill.past_key_values.get_seq_length() == L
            delta = prefill.rope_deltas
            next_pos = (delta[:, 0] + L).view(1, 1, 1).expand(3, 1, 1)
            cached = model(
                input_ids=next_ids,
                attention_mask=torch.ones((1, L+1), dtype=torch.long, device=device),
                past_key_values=prefill.past_key_values,
                position_ids=next_pos,
                cache_position=torch.tensor([L], device=device),
                use_cache=True,
            )
            model.model.rope_deltas = None
            full_ids = torch.cat([one["input_ids"], next_ids], dim=1)
            # Full Forward 使用的真实三维位置
            full_positions, full_delta = (
                model.model.get_rope_index(
                    input_ids=full_ids,
                    image_grid_thw=one["image_grid_thw"],
                    attention_mask=torch.ones_like(full_ids),
                )
            )

            # Prefill 使用的位置
            prefill_positions, _ = (
                model.model.get_rope_index(
                    input_ids=one["input_ids"],
                    image_grid_thw=one["image_grid_thw"],
                    attention_mask=one["attention_mask"],
                )
            )

            # 检查原 Prompt 的位置是否完全一致
            assert torch.equal(
                full_positions[:, :, :L],
                prefill_positions,
            )

            # 检查下一 Token 的位置是否完全一致
            assert torch.equal(
                full_positions[:, :, -1:],
                next_pos,
            ), (
                f"Full: {full_positions[:, :, -1:].tolist()}, "
                f"Cached: {next_pos.tolist()}"
            )

            print("CACHE POSITION MATCH PASS")
            full = model(
                input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                pixel_values=one["pixel_values"], image_grid_thw=one["image_grid_thw"],
                use_cache=False,
            )
        a, b = cached.logits[:, -1].float(), full.logits[:, -1].float()
        diff = (a - b).abs()
        print("Cache max diff:", diff.max().item())
        print("Cache mean diff:", diff.mean().item())
        print("Cache RMS diff:", diff.square().mean().sqrt().item())

        print("Cached top1:", a.argmax(dim=-1).tolist())
        print("Full top1:", b.argmax(dim=-1).tolist())

        print("Cached top10:", a.topk(10).indices.tolist())
        print("Full top10:", b.topk(10).indices.tolist())

        print("RoPE deltas:", delta.tolist())
        print("Next MRoPE pos:", next_pos.flatten().tolist())
        print("Cache max logit diff:", float((a - b).abs().max()))
        # 共同检查
        assert torch.isfinite(a).all()
        assert torch.isfinite(b).all()

        if args.fp32:
            torch.testing.assert_close(
                a,
                b,
                rtol=1e-4,
                atol=1e-4,
            )
            print("FP32 KV CACHE NUMERICAL PASS")

        else:
            assert torch.equal(
                a.argmax(dim=-1),
                b.argmax(dim=-1),
            )

            print("BF16 KV CACHE FUNCTIONAL PASS")
            print(
                "Note: strict BF16 numerical equivalence "
                "is not established."
            )
        

    run("10 cached decode versus full-sequence forward", check_cache)

    def check_generate(num_beams=1):
        call_count = [0]
        def hook(_module, _inputs, _output):
            call_count[0] += 1
        handle = model.model.visual.vit.register_forward_hook(hook)
        try:
            model.model.rope_deltas = None
            with torch.inference_mode():
                generated = model.generate(
                    **one, max_new_tokens=5, min_new_tokens=3,
                    num_beams=num_beams, do_sample=False,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
                )
        finally:
            handle.remove()
        assert call_count[0] == 1, f"ViT called {call_count[0]} times, expected 1"
        assert generated.shape[1] > one["input_ids"].shape[1]
        suffix = generated[0, one["input_ids"].shape[1]:]
        print("Generated:", tokenizer.decode(suffix, skip_special_tokens=True)[:100])

    run("11 generate + ViT prefill-only", lambda: check_generate(num_beams=1))
    if not args.skip_beam:
        run("12 two-beam generate + visual expansion", lambda: check_generate(num_beams=2))

    def check_batch_expand():
        p = processor.user_prompt("图片内容？", num_images=1)
        batch = dev(processor.encode_images([p, p], [[img], [large]]))
        ids, kw = model._expand_inputs_for_generation(
            expand_size=2, input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
        )
        assert ids.shape[0] == 4
        assert len(kw["pixel_values"]) == 2 * len(batch["pixel_values"])
        assert len(kw["image_grid_thw"]) == 4

    run("13 unequal-tile batch expansion", check_batch_expand)

    if not args.skip_checkpoint:
        def check_checkpoint():
            nonlocal model
            model.model.rope_deltas = None
            with torch.inference_mode():
                expected = model(**one, use_cache=False).logits.detach().cpu()
            dst = Path("./checkpoints/_hybrid_task1_test_roundtrip")
            save_checkpoint(model, tokenizer, image_processor, dst, args.qwen25, args.vit, args.llm)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            loaded, _ = load_checkpoint(dst, device=str(device), dtype=dtype)
            model = loaded
            model.model.rope_deltas = None
            with torch.inference_mode():
                actual = model(**one, use_cache=False).logits.detach().cpu()
            torch.testing.assert_close(expected.float(), actual.float(), rtol=1e-4, atol=1e-4)
            print("Checkpoint saved at:", dst)
        run("14 checkpoint save/reload equivalence", check_checkpoint)

    successes = sum(x["status"] == "PASS" for x in report.values())

    print("\n========= TASK 1 TEST SUMMARY =========")
    print("tokenizer.vocab_size:", tokenizer.vocab_size)
    print("len(tokenizer):", len(tokenizer))
    print("lm_head.out_features:", model.lm_head.out_features)
    for name, result in report.items():
        print(f"{result['status']:4} | {name}")
    print(f"PASSED {successes}/{len(report)} tests. Report: task1_test_report.json")
    if successes != len(report):
        raise SystemExit(1)


def math_isfinite(x):
    import math
    return math.isfinite(x)


if __name__ == "__main__":
    main()
