"""Read LLaVA PT JSON and locate its images without changing the original data."""
import json
import random
from pathlib import Path
from torch.utils.data import Dataset


def find_image(root: Path, image_name: str) -> Path | None:
    rel = Path(image_name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe image path in annotation: {image_name}')
    for candidate in (root / rel, root / 'images' / rel):
        if candidate.is_file():
            return candidate
    return None


def parse_pair(row):
    conv = row.get('conversations') or []
    if len(conv) < 2 or conv[0].get('from') != 'human' or conv[1].get('from') != 'gpt':
        raise ValueError(f'Not a PT human->gpt pair: id={row.get("id")}')
    question = conv[0]['value'].replace('<image>', '').strip()
    answer = conv[1]['value'].strip()
    if not question or not answer:
        raise ValueError(f'Empty prompt or answer: id={row.get("id")}')
    return question, answer


class LLaVAPT(Dataset):
    def __init__(self, json_path, image_root, limit=None, seed=42):
        self.image_root = Path(image_root)
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, list) and data, 'Annotations must be a nonempty JSON list'
        if limit is not None:
            # Deterministic random subset; avoid testing only first samples.
            inds = random.Random(seed).sample(range(len(data)), k=min(limit, len(data)))
            data = [data[i] for i in inds]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        path = find_image(self.image_root, row['image'])
        if path is None:
            raise FileNotFoundError(f'Missing image {row["image"]!r} (root={self.image_root})')
        question, answer = parse_pair(row)
        return {'id': row['id'], 'image_path': str(path), 'question': question,
                'answer': answer}
