import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
import numpy as np

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


def _make_dummy_image(path: Path, size=(512, 512)):
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(path)
    return img

def test_extract_pose_returns_pil_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    _make_dummy_image(img_path)
    with patch("pipeline.preprocessors.OpenposeDetector") as mock_cls:
        mock_detector = MagicMock()
        mock_cls.from_pretrained.return_value = mock_detector
        mock_detector.return_value = Image.new("RGB", (512, 512))
        from pipeline.preprocessors import extract_pose
        result = extract_pose(img_path)
    assert isinstance(result, Image.Image)

def test_segment_background_returns_image_and_mask(tmp_path):
    img_path = tmp_path / "test.jpg"
    _make_dummy_image(img_path)
    with patch("pipeline.preprocessors.sam_model_registry") as mock_reg, \
         patch("pipeline.preprocessors.SamPredictor") as mock_pred_cls:
        mock_sam = MagicMock()
        mock_reg.__getitem__.return_value = MagicMock(return_value=mock_sam)
        mock_predictor = MagicMock()
        mock_pred_cls.return_value = mock_predictor
        mock_predictor.predict.return_value = (
            np.array([np.ones((512, 512), dtype=bool)]),
            np.array([0.95]),
            np.array([0.0]),
        )
        from pipeline.preprocessors import segment_background
        image, mask = segment_background(img_path)
    assert isinstance(image, Image.Image)
    assert isinstance(mask, Image.Image)
