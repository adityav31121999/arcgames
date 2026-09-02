"""Low-level stdout/stderr and logging suppressors."""

from contextlib import contextmanager
import logging
import os
import sys
import warnings


@contextmanager
def suppress_stdout_stderr():
    """Silences low-level C/C++ and runtime output streams by redirecting fd 1 and 2."""
    null_fd = os.open(os.devnull, os.O_WRONLY)
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        os.dup2(orig_stdout_fd, 1)
        os.dup2(orig_stderr_fd, 2)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)
        os.close(null_fd)


def silence_hf_warnings():
    """Configures transformers and PyTorch logging to minimize noise."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
