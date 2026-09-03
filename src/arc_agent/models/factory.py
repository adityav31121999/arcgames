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

        # Compatibility hotfix for hf_api in transformers.utils
        try:
            import transformers.utils
            if not hasattr(transformers.utils, "hf_api"):
                from huggingface_hub import HfApi
                transformers.utils.hf_api = HfApi
        except Exception:
            pass

        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            GemmaConfig,
            PretrainedConfig,
        )

        # Base Gemma configuration & model classes — prefer native Gemma4 classes in Transformers >= 5.10
        gemma4_cfg_cls = getattr(transformers, "Gemma4Config", None)
        gemma4_model_cls = getattr(
            transformers,
            "Gemma4ForConditionalGeneration",
            getattr(transformers, "Gemma4ForCausalLM", None),
        )
        gemma_base_cfg = gemma4_cfg_cls if gemma4_cfg_cls is not None else getattr(transformers, "Gemma2Config", GemmaConfig)
        gemma_base_model = getattr(transformers, "Gemma2ForCausalLM", getattr(transformers, "GemmaForCausalLM", None))

        # Patch Gemma4Config to add missing pad_token_id / eos_token_id defaults expected by Gemma2 internals
        if gemma4_cfg_cls is not None and not hasattr(gemma4_cfg_cls, "_patched_pad_token"):
            try:
                _orig_gemma4_init = gemma4_cfg_cls.__init__
                def _gemma4_init_patch(self, *args, **kwargs):
                    _orig_gemma4_init(self, *args, **kwargs)
                    if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
                        self.pad_token_id = getattr(self, "eos_token_id", 1)
                    if not hasattr(self, "eos_token_id") or self.eos_token_id is None:
                        self.eos_token_id = 1
                gemma4_cfg_cls.__init__ = _gemma4_init_patch
                gemma4_cfg_cls._patched_pad_token = True
            except Exception:
                pass

        # 1. Register 'gemma4' with AutoConfig using the native Gemma4Config (if available) to avoid model_type mismatch warnings
        try:
            if gemma4_cfg_cls is not None:
                try:
                    AutoConfig.register("gemma4", gemma4_cfg_cls, exist_ok=True)
                except TypeError:
                    AutoConfig.register("gemma4", gemma4_cfg_cls)
            else:
                AutoConfig.register("gemma4", gemma_base_cfg)
            print("🔧 [MODEL FACTORY] Registered 'gemma4' in AutoConfig.")
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
        # Monkey-patch GenerationConfig.from_model_config globally to handle cases where model_config.get_text_config() returns a dict
        try:
            from transformers.generation.configuration_utils import GenerationConfig
            orig_from_model_config = GenerationConfig.from_model_config

            @classmethod
            def patched_from_model_config(cls, model_config):
                try:
                    if hasattr(model_config, "get_text_config"):
                        t_cfg = model_config.get_text_config(decoder=True)
                        if isinstance(t_cfg, dict):
                            # wrap raw dict into a PretrainedConfig
                            model_config.text_config = gemma_base_cfg(**t_cfg)
                except Exception:
                    pass
                try:
                    return orig_from_model_config(model_config)
                except AttributeError as ae:
                    if "'dict' object has no attribute 'to_dict'" in str(ae):
                        # Fallback: construct GenerationConfig from empty or base dictionary
                        return cls()
                    raise ae

            GenerationConfig.from_model_config = patched_from_model_config
        except Exception as patch_exc:
            print(f"ℹ️ GenerationConfig patch note: {patch_exc}")

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

        # Build load_kwargs without pre-passed config for Auto* loaders (let them resolve it natively)
        load_kwargs_noconfig = {k: v for k, v in load_kwargs.items() if k != "config"}

        # Tier 1: Gemma4ForConditionalGeneration (native, requires Transformers >= 5.10)
        if gemma4_model_cls is not None:
            try:
                model = gemma4_model_cls.from_pretrained(model_id, **load_kwargs_noconfig)
                print(f"✅ Loaded as {gemma4_model_cls.__name__} (native Gemma 4 class).")
            except Exception as e:
                print(f"ℹ️ {gemma4_model_cls.__name__} note ({e}) -> trying AutoModelForImageTextToText...")

        # Tier 2: AutoModelForImageTextToText
        if model is None:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs_noconfig)
                print("✅ Loaded as AutoModelForImageTextToText (multimodal vision-language model).")
            except Exception as e:
                print(f"ℹ️ AutoModelForImageTextToText note ({e}) -> trying AutoModelForCausalLM...")

        # Tier 3: AutoModelForCausalLM
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs_noconfig)
                print("✅ Loaded as AutoModelForCausalLM.")
            except Exception as e:
                print(f"ℹ️ AutoModelForCausalLM note ({e}) -> trying AutoModel...")

        # Tier 4: AutoModel
        if model is None:
            try:
                model = AutoModel.from_pretrained(model_id, **load_kwargs_noconfig)
                print("✅ Loaded as AutoModel.")
            except Exception as e:
                print(f"ℹ️ AutoModel note ({e}) -> trying direct Gemma2ForCausalLM loader...")

        # Tier 5: Direct Gemma2ForCausalLM fallback (last resort)
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
