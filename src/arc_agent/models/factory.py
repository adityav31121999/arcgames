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

        # Compatibility hotfix for safetensors.safe_open unexpected keyword argument 'backend'
        try:
            import safetensors
            _orig_safe_open = safetensors.safe_open

            def _compat_safe_open(*args, **kwargs):
                try:
                    return _orig_safe_open(*args, **kwargs)
                except TypeError as te:
                    if "backend" in str(te):
                        kwargs.pop("backend", None)
                        return _orig_safe_open(*args, **kwargs)
                    raise te

            safetensors.safe_open = _compat_safe_open
            if hasattr(safetensors, "torch"):
                safetensors.torch.safe_open = _compat_safe_open

            import sys
            for mod in list(sys.modules.values()):
                if hasattr(mod, "safe_open") and getattr(mod, "safe_open") is _orig_safe_open:
                    setattr(mod, "safe_open", _compat_safe_open)
        except Exception as e:
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

        # Patch Gemma4Config:
        # - Add pad_token_id / eos_token_id defaults so GenerationConfig doesn't crash
        # - Proxy vocab_size / hidden_size / etc. from text_config so Gemma2 internals don't crash
        if gemma4_cfg_cls is not None and not hasattr(gemma4_cfg_cls, "_patched_pad_token"):
            try:
                _orig_gemma4_init = gemma4_cfg_cls.__init__
                def _gemma4_init_patch(self, *args, **kwargs):
                    _orig_gemma4_init(self, *args, **kwargs)
                    if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
                        eos = getattr(self, "eos_token_id", 1)
                        self.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
                    if not hasattr(self, "eos_token_id") or self.eos_token_id is None:
                        self.eos_token_id = 1
                    # Ensure pad_token_id is always a plain int (never a list)
                    if isinstance(self.pad_token_id, (list, tuple)):
                        self.pad_token_id = self.pad_token_id[0] if self.pad_token_id else 1
                gemma4_cfg_cls.__init__ = _gemma4_init_patch
                gemma4_cfg_cls._patched_pad_token = True
            except Exception:
                pass

        # Proxy vocab_size and related attrs from text_config so Gemma2/dense model internals don't crash
        # when accidentally called with a Gemma4Config (multimodal) object.
        if gemma4_cfg_cls is not None and not hasattr(gemma4_cfg_cls, "_patched_vocab_proxy"):
            try:
                _TEXT_CFG_PROXIED = [
                    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "intermediate_size", "head_dim", "rms_norm_eps",
                ]
                def _make_proxy(attr):
                    def _proxy(self):
                        text_cfg = getattr(self, "text_config", None)
                        if text_cfg is not None and not isinstance(text_cfg, dict):
                            return getattr(text_cfg, attr, None)
                        if isinstance(text_cfg, dict):
                            return text_cfg.get(attr)
                        return None
                    _proxy.__name__ = attr
                    return property(_proxy)

                for _attr in _TEXT_CFG_PROXIED:
                    if not hasattr(gemma4_cfg_cls, _attr):
                        setattr(gemma4_cfg_cls, _attr, _make_proxy(_attr))

                gemma4_cfg_cls._patched_vocab_proxy = True
                print("🔧 [MODEL FACTORY] Patched Gemma4Config with text_config attribute proxies.")
            except Exception as _proxy_exc:
                print(f"ℹ️ Gemma4Config proxy patch note: {_proxy_exc}")

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

        # 3. Load and sanitize model configuration (resolving 'dict' object has no attribute 'to_dict' and list pad_token_id)
        # Monkey-patch GenerationConfig globally to handle list pad_token_id and raw text_config dicts
        try:
            from transformers.generation.configuration_utils import GenerationConfig
            orig_from_model_config = GenerationConfig.from_model_config
            orig_from_dict = GenerationConfig.from_dict

            def _sanitize_tokens_dict(d):
                if not isinstance(d, dict):
                    return d
                for k in ["pad_token_id", "eos_token_id", "bos_token_id"]:
                    val = d.get(k)
                    if isinstance(val, (list, tuple)):
                        d[k] = val[0] if len(val) > 0 and isinstance(val[0], int) else None
                return d

            @classmethod
            def patched_from_dict(cls, config_dict, **kwargs):
                config_dict = _sanitize_tokens_dict(config_dict)
                return orig_from_dict(config_dict, **kwargs)

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

                # Sanitize token IDs if they are lists (e.g. [1, 2] -> 1)
                for attr_name in ["pad_token_id", "eos_token_id", "bos_token_id"]:
                    val = getattr(model_config, attr_name, None)
                    if isinstance(val, (list, tuple)):
                        setattr(model_config, attr_name, val[0] if len(val) > 0 and isinstance(val[0], int) else None)

                try:
                    return orig_from_model_config(model_config)
                except Exception as ae:
                    if "'dict' object has no attribute 'to_dict'" in str(ae) or "'<' not supported" in str(ae):
                        # Fallback: construct default GenerationConfig safely
                        return cls()
                    raise ae

            GenerationConfig.from_model_config = patched_from_model_config
            GenerationConfig.from_dict = patched_from_dict
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
                print(f"⚠️ AutoTokenizer direct loading note: {e2}")
                try:
                    from transformers import GemmaTokenizerFast
                    processor = GemmaTokenizerFast.from_pretrained(model_id)
                    print("✅ Loaded GemmaTokenizerFast.")
                except Exception as e3:
                    raise RuntimeError(f"Could not load processor or tokenizer from local model '{model_id}': {e2} / {e3}")

        # 5. Multi-tier model loading with fallbacks
        model = None
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device if torch.cuda.is_available() else "cpu",
        }
        if cfg is not None:
            load_kwargs["config"] = cfg

        # Attempt to add attn_implementation — but test it first since sdpa may not be supported
        # for this model/driver combination (Gemma4 MoE on some Transformers versions)
        attn_impl = config.attn_implementation if config.attn_implementation and config.attn_implementation != "default" else None

        # Build load_kwargs without pre-passed config for Auto* loaders (let them resolve it natively)
        # Also exclude attn_implementation for initial tier attempts — add it only if needed
        load_kwargs_noconfig = {k: v for k, v in load_kwargs.items() if k not in ("config", "attn_implementation")}

        # Tier 1: Gemma4ForConditionalGeneration (native, requires Transformers >= 5.10)
        if gemma4_model_cls is not None:
            try:
                model = gemma4_model_cls.from_pretrained(model_id, **load_kwargs_noconfig)
                print(f"✅ Loaded as {gemma4_model_cls.__name__} (native Gemma 4 class).")
            except Exception as e:
                print(f"ℹ️ {gemma4_model_cls.__name__} note ({e}) -> trying AutoModelForMultimodalLM...")

        # Tier 2: AutoModelForMultimodalLM (new standard for omni/any-to-any multimodal models in Transformers 5+)
        if model is None:
            try:
                AutoModelForMultimodalLM = getattr(transformers, "AutoModelForMultimodalLM", None)
                if AutoModelForMultimodalLM is not None:
                    model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs_noconfig)
                    print("✅ Loaded as AutoModelForMultimodalLM.")
            except Exception as e:
                print(f"ℹ️ AutoModelForMultimodalLM note ({e}) -> trying AutoModelForImageTextToText...")

        # Tier 3: AutoModelForImageTextToText
        if model is None:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs_noconfig)
                print("✅ Loaded as AutoModelForImageTextToText (multimodal vision-language model).")
            except Exception as e:
                print(f"ℹ️ AutoModelForImageTextToText note ({e}) -> trying AutoModelForCausalLM...")

        # Tier 4: AutoModelForCausalLM
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs_noconfig)
                print("✅ Loaded as AutoModelForCausalLM.")
            except Exception as e:
                print(f"ℹ️ AutoModelForCausalLM note ({e}) -> trying AutoModel...")

        # Tier 5: AutoModel
        if model is None:
            try:
                model = AutoModel.from_pretrained(model_id, **load_kwargs_noconfig)
                print("✅ Loaded as AutoModel.")
            except Exception as e:
                print(f"ℹ️ AutoModel note ({e}) -> trying direct Gemma2ForCausalLM loader...")

        # Tier 6: Direct Gemma2ForCausalLM fallback (last resort) — skip if cfg is Gemma4Config
        # since Gemma2ForCausalLM needs vocab_size as a top-level attribute, which Gemma4Config doesn't have.
        cfg_is_gemma4 = gemma4_cfg_cls is not None and isinstance(cfg, gemma4_cfg_cls)
        if model is None and gemma_base_model is not None and not cfg_is_gemma4:
            try:
                tier6_kwargs = {k: v for k, v in load_kwargs.items() if k != "attn_implementation"}
                model = gemma_base_model.from_pretrained(model_id, **tier6_kwargs)
                print("✅ Loaded directly via Gemma2ForCausalLM.")
            except Exception as e:
                print(f"ℹ️ Gemma2ForCausalLM final fallback note: {e}")

        if model is None:
            raise RuntimeError(
                f"All model loading tiers failed for '{model_id}'. "
                f"Check that transformers>=5.10 is installed for Gemma4ForConditionalGeneration support. "
                f"Disk usage: {__import__('shutil').disk_usage('/tmp')}. "
                f"Last config type: {type(cfg).__name__ if cfg else 'None'}"
            )




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
