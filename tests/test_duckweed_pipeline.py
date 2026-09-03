"""Mock-mode duckweed tracker pipeline smoke test.

This test imports the real `run_duckweed_tracker` script and executes the Sacred
experiment in mock mode with observers disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sj_duckweed.scripts import run_duckweed_tracker as tracker

logger = logging.getLogger(__name__)


@pytest.mark.vision
def test_duckweed_pipeline_mock_run():
    # Disable Sacred observers (e.g., Mongo) for test/offline execution.
    tracker.ex.observers.clear()

    repo_root = Path(__file__).resolve().parents[1]
    mock_image_path = (
        repo_root
        / "data"
        / "filtered_images"
        / "test_duckweed_image.png"
    )
    config_updates = {
        "hardware": False,
        "use_ai": False,
        "session_env_mock": ".env.mock",
        "mock_image_path": str(mock_image_path),
        # A destination slot can contain multiple wells; the tracker visits each one.
        "source_slot": "0",
        "dest_slot": "1",
        "supplementary_offset_xyz": [-10.0, 10.0, 0.0],
        "image_settle": 0.0,
    }

    print("\nRunning duckweed tracker smoke test in mock mode")
    print(f"  Mock image: {mock_image_path}")
    print(
        "  Deck slots: "
        f"source={config_updates['source_slot']}, "
        f"destination={config_updates['dest_slot']}"
    )
    print(f"  Image exists: {mock_image_path.is_file()}")

    result = tracker.ex.run(
        config_updates=config_updates
    )

    print(f"  Sacred run status: {result.status}")
    assert result.status == "COMPLETED"
    logger.info("Mock duckweed tracker run completed successfully.")
