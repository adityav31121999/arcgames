import pytest
from arc_agent.core.diff import clear_hud_pixels


@pytest.fixture(autouse=True)
def reset_hud_pixels_fixture():
    """Ensure global HUD pixel registry is cleared before and after each test."""
    clear_hud_pixels()
    yield
    clear_hud_pixels()
