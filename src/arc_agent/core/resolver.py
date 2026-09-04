"""Game state resolution helper across various ARC engine versions."""

from typing import Any


class GameStateResolver:
    """Safely checks terminal states and win/loss status."""

    def __init__(self, game_state_enum: Any = None):
        self._enum = game_state_enum
        self.WIN = getattr(game_state_enum, "WIN", None)
        self.GAME_OVER = getattr(game_state_enum, "GAME_OVER", None)
        self.LEVEL_UP = getattr(game_state_enum, "LEVEL_UP", None)
        self.NOT_FINISHED = getattr(
            game_state_enum, "NOT_FINISHED", getattr(game_state_enum, "PLAYING", None)
        )
        if self.WIN is None and self.GAME_OVER is None and self.LEVEL_UP is None:
            # Fallback for integer/string based status
            pass

    def is_win(self, state: Any) -> bool:
        if self.WIN is not None and state == self.WIN:
            return True
        return str(state).upper() in ("WIN", "GAMESTATE.WIN", "SUCCESS")

    def is_game_over(self, state: Any) -> bool:
        if self.GAME_OVER is not None and state == self.GAME_OVER:
            return True
        return str(state).upper() in ("GAME_OVER", "GAMESTATE.GAME_OVER", "LOSS", "FAILED")

    def is_level_up(self, state: Any) -> bool:
        if self.LEVEL_UP is not None and state == self.LEVEL_UP:
            return True
        return str(state).upper() in ("LEVEL_UP", "GAMESTATE.LEVEL_UP", "NEXT_LEVEL", "LEVEL_COMPLETE", "GAMESTATE.LEVEL_COMPLETE")

    def is_terminal(self, state: Any) -> bool:
        return self.is_win(state) or self.is_game_over(state) or self.is_level_up(state)

    def name(self, state: Any) -> str:
        return getattr(state, "name", str(state))
