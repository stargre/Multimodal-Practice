"""Debug-only collator: BS=1 prevents mistakes in vision-grid concatenation.

Expand to variable-resolution BS>1 ONLY after verifying first-stage MRoPE
and image-grid packing semantics. Do not zero-pad pixel_values like text.
"""
from .adapter import encode_pt_example


class PTDebugCollator:
    def __init__(self, processor, max_tokens=3072):
        self.processor = processor
        self.max_tokens = max_tokens

    def __call__(self, examples):
        if len(examples) != 1:
            raise ValueError('PTDebugCollator intentionally supports batch_size=1')
        batch = encode_pt_example(self.processor, examples[0])
        seq_len = batch['input_ids'].shape[-1]
        if seq_len > self.max_tokens:
            raise ValueError(
                f'Expanded sequence {seq_len} > {self.max_tokens}; reduce image '
                'resolution / vision tokens in the original preprocessing; '
                'DO NOT truncate image tokens blindly.')
        return batch
