import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


MODEL_PATH = "./checkpoints/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "./assets/test.jpg"


def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_PATH,
                },
                {
                    "type": "text",
                    "text": "请详细描述这张图片。",
                },
            ],
        }
    ]

    # 构造 Qwen chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 读取图片
    image_inputs, video_inputs = process_vision_info(messages)

    # Processor:
    # text -> input_ids
    # image -> pixel_values
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    print("input_ids:", inputs["input_ids"].shape)
    print("pixel_values:", inputs["pixel_values"].shape)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # 去掉 prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n========== Result ==========")
    print(output_text[0])


if __name__ == "__main__":
    main()