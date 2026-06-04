import pytest
from unittest.mock import patch, MagicMock

def test_load_pipeline_returns_object_for_each_mode():
    """Loader must return a non-None object for every valid mode."""
    with patch("pipeline.loader._load_base_pipeline") as mock_base, \
         patch("pipeline.loader._load_controlnet_pipeline") as mock_ctrl, \
         patch("pipeline.loader._load_inpaint_pipeline") as mock_inpaint:
        mock_base.return_value = MagicMock()
        mock_ctrl.return_value = MagicMock()
        mock_inpaint.return_value = MagicMock()

        from pipeline.loader import load_pipeline
        assert load_pipeline("color") is not None
        assert load_pipeline("pose") is not None
        assert load_pipeline("background") is not None

def test_load_pipeline_raises_on_unknown_mode():
    from pipeline.loader import load_pipeline
    with pytest.raises(ValueError, match="Unknown mode"):
        load_pipeline("invalid")
