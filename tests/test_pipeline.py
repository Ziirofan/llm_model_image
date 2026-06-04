"""
Integration tests — require GPU + downloaded model weights.
Run manually: pytest tests/test_pipeline.py -v -s
"""
import pytest
from pathlib import Path

FIXTURE = Path("dataset/fixture.jpg")
OUTPUT_DIR = Path("outputs/integration_test")


@pytest.fixture(autouse=True)
def check_fixture():
    if not FIXTURE.exists():
        pytest.skip("dataset/fixture.jpg not found — add a photo to run integration tests")


@pytest.fixture(autouse=True)
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_color_variation_produces_output():
    from pipeline.loader import load_pipeline
    from pipeline.modes.color import generate_color_variation

    pipe = load_pipeline("color")
    output = OUTPUT_DIR / "color_result.png"
    generate_color_variation(pipe, FIXTURE, "red dress, vibrant color", output, resolution=768)

    assert output.exists()
    assert output.stat().st_size > 10_000  # at least 10KB — not an empty file


def test_background_variation_produces_output():
    from pipeline.loader import load_pipeline
    from pipeline.modes.background import generate_background_variation

    pipe = load_pipeline("background")
    output = OUTPUT_DIR / "background_result.png"
    generate_background_variation(pipe, FIXTURE, "in a bright sunny park", output, resolution=768)

    assert output.exists()
    assert output.stat().st_size > 10_000


def test_pose_variation_produces_output():
    """Uses the fixture itself as the pose reference — result should be similar pose."""
    from pipeline.loader import load_pipeline
    from pipeline.modes.pose import generate_pose_variation

    pipe = load_pipeline("pose")
    output = OUTPUT_DIR / "pose_result.png"
    generate_pose_variation(pipe, FIXTURE, FIXTURE, output, resolution=768)

    assert output.exists()
    assert output.stat().st_size > 10_000
