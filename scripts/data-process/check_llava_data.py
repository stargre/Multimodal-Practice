import json
from pathlib import Path

ROOT = Path("/mnt/data2/zhiyu/first-task/datasets/llava_stage2")

files = {
    "PT": ROOT / "pretrain" / "blip_laion_cc_sbu_558k.json",
    "SFT": ROOT / "sft" / "llava_instruct_150k.json",
}

for name, path in files.items():
    print(f"\n========== {name} ==========")

    assert path.is_file(), f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Expected a JSON list"
    assert len(data) > 0, "Dataset is empty"

    print("Samples:", len(data))

    sample = data[0]
    print("Keys:", list(sample.keys()))
    print("Image:", sample.get("image"))
    print("ID:", sample.get("id"))

    conversations = sample.get("conversations", [])
    print("Conversations:", len(conversations))

    for turn in conversations[:2]:
        role = turn.get("from")
        text = str(turn.get("value", ""))
        print(f"  {role}: {text[:200]}")

    assert "image" in sample
    assert "conversations" in sample

    print(f"{name} metadata check PASSED")

print("\nAll metadata checks PASSED")
