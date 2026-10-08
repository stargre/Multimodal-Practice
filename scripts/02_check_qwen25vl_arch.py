from transformers import Qwen2_5_VLForConditionalGeneration


MODEL_PATH = "./checkpoints/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cpu",
)

print(model)

print("\n========== Vision ==========")
print(model.model.visual)

print("\n========== Language Model ==========")
print(model.model.language_model)

print("\n========== LM Head ==========")
print(model.lm_head)

print("\n========== Config ==========")
print("text hidden:",
      model.config.text_config.hidden_size)

print("vision hidden:",
      model.config.vision_config.hidden_size)