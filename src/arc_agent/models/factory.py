"""Model factory for initializing Hugging Face Transformers models."""

from pathlib import Path
from typing import Optional
import os

from ..config import ModelConfig
from ..utils.locator import locate_hf_model_dir
from ..utils.suppression import silence_hf_warnings
from .gemma_transformers import GemmaTransformersChatModel, MockChatModel


class ModelFactory:
    """Factory for instantiating LangChain chat models backed by Hugging Face Transformers."""

    @staticmethod
    def create_model(
        config: ModelConfig,
        use_mock: bool = False,
    ) -> GemmaTransformersChatModel | MockChatModel:
        if use_mock or os.getenv("USE_MOCK_MODEL", "false").lower() == "true":
            print("🤖 [MODEL FACTORY] Instantiating MockChatModel for testing/dry-run.")
            return MockChatModel()

        import torch


        silence_hf_warnings()

        # Resolve model path / ID
        model_id = config.model_id
        if not Path(model_id).exists() and not model_id.startswith("http"):
            # Try offline Kaggle search
            candidates = ["gemma-4", "26b", "a4b", "nvfp4"]
            discovered_path = locate_hf_model_dir(candidates, search_root="/kaggle/input")
            if discovered_path:
                print(f"🔍 [MODEL FACTORY] Auto-located offline model weights at: {discovered_path}")
                model_id = discovered_path
            else:
                print(f"🌐 [MODEL FACTORY] Using model identifier: {model_id}")

        print(f"🚀 [MODEL FACTORY] Loading model '{model_id}' on {config.device} ({config.torch_dtype})...")


        # Compatibility hotfix for PIL / torchvision _Ink typing mismatch in Kaggle
        try:
            import PIL._typing
            if not hasattr(PIL._typing, "_Ink"):
                from typing import Union, Tuple, Sequence
                PIL._typing._Ink = Union[Tuple[int, ...], str, int, float, Sequence[int]]
        except Exception:
            pass

        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype.lower(), torch.bfloat16)

        # Attempt loading processor / tokenizer
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=config.trust_remote_code,
            )
        except Exception as e:
            print(f"⚠️ AutoProcessor failed ({e}), falling back to AutoTokenizer...")
            processor = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=config.trust_remote_code,
            )

        # Attempt model load with AutoModelForImageTextToText / AutoModelForCausalLM
        model = None
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device if torch.cuda.is_available() else "cpu",
        }

        if config.attn_implementation and config.attn_implementation != "default":
            load_kwargs["attn_implementation"] = config.attn_implementation

        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
            print("✅ Loaded as AutoModelForImageTextToText (multimodal vision-language model).")
        except Exception as e:
            print(f"ℹ️ AutoModelForImageTextToText fallback ({e}) -> trying AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            print("✅ Loaded as AutoModelForCausalLM.")

        model.eval()

        return GemmaTransformersChatModel(
            model=model,
            processor=processor,
            device=config.device,
            torch_dtype=config.torch_dtype,
            max_context_length=config.max_context_length,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repeat_penalty,
        )
