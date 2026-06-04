"""
Conftest: pre-install stub modules for heavy ML dependencies so that
pipeline.preprocessors can be imported without the real libraries present.
"""
import sys
from unittest.mock import MagicMock

# Stub out heavy ML libraries that are not installed in the test environment
_STUB_MODULES = [
    "controlnet_aux",
    "segment_anything",
    "huggingface_hub",
    "torch",
]

for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
