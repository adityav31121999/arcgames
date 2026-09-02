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

        import json
        import importlib.util
        import transformers
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            GemmaConfig,
            PretrainedConfig,
        )

        # Handle unrecognized model types like 'gemma4' or custom remote architectures
        model_dir = Path(model_id) if Path(model_id).is_dir() else None
        model_type = None
        if model_dir and (model_dir / "config.json").exists():
            try:
                with open(model_dir / "config.json", "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                    model_type = cfg_dict.get("model_type")
            except Exception:
                pass

        if model_type and model_type not in transformers.models.auto.configuration_auto.CONFIG_MAPPING:
            print(f"🔧 [MODEL FACTORY] Registering dynamic config handler for model_type='{model_type}'...")
            loaded_custom = False

            # 1. Check for custom Python modules in the model folder
            if model_dir:
                for py_file in sorted(model_dir.glob("*.py")):
                    try:
                        spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            for attr_name in dir(mod):
                                attr = getattr(mod, attr_name)
                                if (
                                    isinstance(attr, type)
                                    and issubclass(attr, PretrainedConfig)
                                    and attr is not PretrainedConfig
                                ):
                                    transformers.models.auto.configuration_auto.CONFIG_MAPPING.register(model_type, attr)
                                    loaded_custom = True
                                    print(f"✅ Auto-registered custom config '{attr_name}' for '{model_type}'")
                    except Exception as exc:
                        print(f"⚠️ Custom module load note ({py_file.name}): {exc}")

            # 2. Fallback dynamic registration
            if not loaded_custom:
                gemma_base_cfg = getattr(transformers, "Gemma2Config", GemmaConfig)
                gemma_base_model = getattr(transformers, "Gemma2ForCausalLM", getattr(transformers, "GemmaForCausalLM", None))

                class DynamicGemma4Config(gemma_base_cfg):  # type: ignore
                    model_type = model_type

                transformers.models.auto.configuration_auto.CONFIG_MAPPING.register(model_type, DynamicGemma4Config)
                if gemma_base_model is not None:
                    transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING.register(
                        DynamicGemma4Config, gemma_base_model
                    )
                print(f"✅ Dynamic fallback registration complete for '{model_type}'.")

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
            print("✅ Loaded AutoProcessor.")
        except Exception as e:
            print(f"ℹ️ AutoProcessor note ({e}) -> using AutoTokenizer...")
            try:
                processor = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=config.trust_remote_code,
                )
                print("✅ Loaded AutoTokenizer.")
            except Exception as e2:
                print(f"⚠️ AutoTokenizer fallback loading: {e2}")
                processor = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

        # Attempt model load with AutoModelForImageTextToText / AutoModelForCausalLM / AutoModel
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
            print(f"ℹ️ AutoModelForImageTextToText note ({e}) -> trying AutoModelForCausalLM...")
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                print("✅ Loaded as AutoModelForCausalLM.")
            except Exception as e2:
                print(f"ℹ️ AutoModelForCausalLM note ({e2}) -> trying AutoModel...")
                model = AutoModel.from_pretrained(model_id, **load_kwargs)
                print("✅ Loaded as AutoModel.")


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
