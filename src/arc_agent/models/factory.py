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

        # pyrefly: ignore [missing-import]
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

        # Base Gemma configuration & model classes
        gemma_base_cfg = getattr(transformers, "Gemma2Config", GemmaConfig)
        gemma_base_model = getattr(
            transformers, "Gemma2ForCausalLM", getattr(transformers, "GemmaForCausalLM", None)
        )

        # 1. Register 'gemma4' architecture directly with AutoConfig
        try:
            AutoConfig.register("gemma4", gemma_base_cfg)
            print("🔧 [MODEL FACTORY] Successfully mapped 'gemma4' model_type to Gemma configuration.")
        except Exception as e:
            print(f"ℹ️ AutoConfig registration note: {e}")

        # 2. Check for custom Python modules or other model_types in model directory
        model_dir = Path(model_id) if Path(model_id).is_dir() else None
        if model_dir and (model_dir / "config.json").exists():
            try:
                with open(model_dir / "config.json", "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                    detected_type = cfg_dict.get("model_type")
                    if detected_type and detected_type != "gemma4":
                        try:
                            AutoConfig.register(detected_type, gemma_base_cfg)
                            print(f"🔧 [MODEL FACTORY] Registered dynamic config for '{detected_type}'.")
                        except Exception:
                            pass
            except Exception:
                pass

            # Search and load any custom modeling code in the checkpoint directory
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
                                model_type_name = getattr(attr, "model_type", None)
                                if model_type_name:
                                    AutoConfig.register(model_type_name, attr)
                                    print(f"✅ Auto-registered custom config '{attr_name}' for '{model_type_name}'")
                except Exception as exc:
                    print(f"⚠️ Custom module load note ({py_file.name}): {exc}")

        # 3. Load and sanitize model configuration (resolving 'dict' object has no attribute 'to_dict')
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=config.trust_remote_code)
        except Exception as e:
            print(f"ℹ️ AutoConfig.from_pretrained note ({e}) -> trying gemma_base_cfg...")
            try:
                cfg = gemma_base_cfg.from_pretrained(model_id, trust_remote_code=config.trust_remote_code)
            except Exception:
                pass

        if cfg is not None:
            # Fix text_config if raw dict is returned
            if hasattr(cfg, "text_config") and isinstance(cfg.text_config, dict):
                try:
                    cfg.text_config = gemma_base_cfg(**cfg.text_config)
                except Exception:
                    pass
            # Ensure get_text_config returns a PretrainedConfig with .to_dict()
            if hasattr(cfg, "get_text_config"):
                orig_get_text = cfg.get_text_config

                def safe_get_text_config(*args, **kwargs):
                    res = orig_get_text(*args, **kwargs)
                    if isinstance(res, dict):
                        return gemma_base_cfg(**res)
                    return res

                cfg.get_text_config = safe_get_text_config

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype.lower(), torch.bfloat16)

        # 4. Attempt loading processor / tokenizer
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

        # 5. Multi-tier model loading with fallbacks
        model = None
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device if torch.cuda.is_available() else "cpu",
        }
        if cfg is not None:
            load_kwargs["config"] = cfg

        if config.attn_implementation and config.attn_implementation != "default":
            load_kwargs["attn_implementation"] = config.attn_implementation

        # Tier 1: AutoModelForImageTextToText
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
            print("✅ Loaded as AutoModelForImageTextToText (multimodal vision-language model).")
        except Exception as e:
            print(f"ℹ️ AutoModelForImageTextToText note ({e}) -> trying AutoModelForCausalLM...")

        # Tier 2: AutoModelForCausalLM
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                print("✅ Loaded as AutoModelForCausalLM.")
            except Exception as e:
                print(f"ℹ️ AutoModelForCausalLM note ({e}) -> trying AutoModel...")

        # Tier 3: AutoModel
        if model is None:
            try:
                model = AutoModel.from_pretrained(model_id, **load_kwargs)
                print("✅ Loaded as AutoModel.")
            except Exception as e:
                print(f"ℹ️ AutoModel note ({e}) -> trying direct Gemma2ForCausalLM loader...")

        # Tier 4: Direct Gemma Architecture Fallback
        if model is None and gemma_base_model is not None:
            try:
                model = gemma_base_model.from_pretrained(model_id, **load_kwargs)
                print("✅ Loaded directly via Gemma2ForCausalLM.")
            except Exception as e:
                raise RuntimeError(f"All model loading tiers failed for '{model_id}': {e}")




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
