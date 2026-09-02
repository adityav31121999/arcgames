"""ARC-AGI-3 Agent Package

LangChain-powered inference agent for ARC-AGI-3 (ARC Prize 2026) using
Hugging Face Transformers (Gemma-4-26B-A4B-NVFP4).
"""

# Compatibility hotfix for PIL / torchvision _Ink typing mismatch in Python 3.12 / Kaggle
try:
    import PIL._typing
    if not hasattr(PIL._typing, "_Ink"):
        from typing import Union, Tuple, Sequence
        PIL._typing._Ink = Union[Tuple[int, ...], str, int, float, Sequence[int]]
except Exception:
    pass

__version__ = "0.1.0"
