"""C++ accelerated MCTS engine module."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Callable

# Ensure conda bin is in PATH for ninja and compilers
conda_bin = "/home/jeefy/miniconda3/envs/unichess/bin"
if conda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"

import torch
from torch.utils.cpp_extension import load

_CPP_DIR = Path(__file__).resolve().parent

_CACHED_EXT = None

def load_cpp_extension():
    """Load or JIT-compile the C++ MCTS extension module."""
    global _CACHED_EXT
    if _CACHED_EXT is not None:
        return _CACHED_EXT

    sources = [
        str(_CPP_DIR / "mcts_pybind.cpp"),
    ]
    extra_cflags = ["-O3", "-std=c++17", f"-I{_CPP_DIR}"]

    ext = load(
        name="unichess_mcts_cpp",
        sources=sources,
        extra_cflags=extra_cflags,
        verbose=False,
    )
    _CACHED_EXT = ext
    return _CACHED_EXT

def load_cpp_mcts():
    """Load or JIT-compile the C++ MCTSCpp class."""
    ext = load_cpp_extension()
    return ext.MCTSCpp

def perft(fen: str, depth: int) -> int:
    return load_cpp_extension().perft(fen, depth)

def get_legal_moves(fen: str) -> list[str]:
    return load_cpp_extension().get_legal_moves(fen)

def encode_planes(fen: str) -> torch.Tensor:
    return load_cpp_extension().encode_planes(fen)

def piece_count(fen: str) -> int:
    return load_cpp_extension().piece_count(fen)

def make_syzygy_probe(tablebase: Any) -> Callable[[str], tuple[bool, float]] | None:
    """Create a Syzygy probe function (fen: str) -> tuple[bool, float] from a chess.syzygy.Tablebase or path."""
    if tablebase is None:
        return None
    import chess
    import chess.syzygy

    if isinstance(tablebase, (str, Path)):
        tablebase = chess.syzygy.open_tablebase(str(tablebase))

    def probe(fen: str) -> tuple[bool, float]:
        board = chess.Board(fen)
        if len(board.piece_map()) > 5:
            return (False, 0.0)
        try:
            wdl = tablebase.probe_wdl(board)
        except Exception:
            return (False, 0.0)
        if abs(wdl) == 2 and board.halfmove_clock:
            try:
                dtz = abs(tablebase.probe_dtz(board))
                if board.halfmove_clock + dtz >= 100:
                    return (False, 0.0)
            except Exception:
                return (False, 0.0)
        val = float((wdl == 2) - (wdl == -2))
        return (True, val)

    return probe

__all__ = [
    "load_cpp_mcts",
    "load_cpp_extension",
    "perft",
    "get_legal_moves",
    "encode_planes",
    "piece_count",
    "make_syzygy_probe",
]
