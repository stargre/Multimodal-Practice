# Third-party source provenance

`configuration_intern_vit.py` and `modeling_intern_vit.py` originate from OpenGVLab InternVL (MIT). See `LICENSE_InternVL`.

Source repository: https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5
Upstream revision observed during implementation: `8f86a5e87697180b439811ca69fabbfccd38d996`.
Original downloaded file SHA-256 values:

- `modeling_intern_vit.py`: `eb20457de0a1748d8fa7f45af99c35372ca25abf3429584daa1d95fd42ddab28`
- `configuration_intern_vit.py`: `5bf42e1fdd013b93e90ad355cb8f3abf9bfd35712f120d38d4a2fa26ac635675`

Local changes: remove optional flash-attn and apex autodetection; keep the original PyTorch normalization and non-flash attention path; import DropPath from timm.layers. No parameter names or visual tensor ordering are changed.

The project intentionally sets use_flash_attn=False. If the original server dynamically selected an apex normalization kernel, server numeric comparison is required before adopting this export.

Qwen components are imported from the separately installed transformers==4.57.6 distribution (Apache-2.0). qwen3_mrope.py, visual_intern.py, and multimodal position/generation methods preserve the repository qwen25_fork implementations from ab0fc63fb9019ccabd61017eada584b8f09c4cd9.
