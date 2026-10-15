"""Import once to register the local HF AutoClasses. No Hub publishing required."""
import transformers

if transformers.__version__ != "4.57.6":
    raise RuntimeError(f"hybrid_vlm phase 1 requires transformers==4.57.6, got {transformers.__version__}")

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoImageProcessor
from .configuration_hybrid import HybridVLMConfig
from .modeling_hybrid import HybridVLMForConditionalGeneration
from .image_processing_hybrid import HybridImageProcessor
from .processing_hybrid import HybridVLMProcessor

AutoConfig.register(HybridVLMConfig.model_type, HybridVLMConfig, exist_ok=True)
AutoModelForImageTextToText.register(HybridVLMConfig, HybridVLMForConditionalGeneration, exist_ok=True)
AutoImageProcessor.register(HybridVLMConfig, slow_image_processor_class=HybridImageProcessor, exist_ok=True)
AutoProcessor.register(HybridVLMConfig, HybridVLMProcessor, exist_ok=True)

__all__ = ["HybridVLMConfig", "HybridVLMForConditionalGeneration", "HybridVLMProcessor", "HybridImageProcessor"]
