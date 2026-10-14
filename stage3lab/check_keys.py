from pathlib import Path
import copy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


from stage1_runtime import load_stage1_components


model, processor = load_stage1_components()


print("\n===== MODEL KEYS =====")

for i,k in enumerate(model.state_dict().keys()):

    print(k)

    if i > 30:
        break