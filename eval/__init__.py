"""UniChessTransformer evaluation and benchmarking tools."""
from .arena import EngineSpec, elo_with_error, play_game, play_match
from .puzzle_bench import BUILTIN_PUZZLES, evaluate_puzzles, load_puzzles_from_file

__all__ = [
    "EngineSpec",
    "elo_with_error",
    "play_game",
    "play_match",
    "BUILTIN_PUZZLES",
    "evaluate_puzzles",
    "load_puzzles_from_file",
]
