"""Batched PUCT Monte Carlo Tree Search with virtual loss, Dirichlet root noise, and Syzygy support."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import chess
import numpy as np

from core.encoding import orient_move
from core.moves import move_to_index, move_to_promo_index


@dataclass
class MCTSConfig:
    simulations: int = 800
    batch_size: int = 128          # Leaf batch size for GPU evaluation
    c_puct: float = 1.8            # Base exploration constant
    c_puct_base: float = 19652.0   # Scale factor for visit-dependent c_puct (AlphaZero formula)
    c_puct_init: float = 1.8
    dirichlet_alpha: float = 0.3   # Alpha for chess Dirichlet noise
    dirichlet_eps: float = 0.25    # Noise weight at root
    fpu_reduction: float = 0.2     # First-Play Urgency reduction
    temperature: float = 0.0       # 0 = greedy argmax(N)
    temp_moves: int = 0            # Initial plies with temperature sampling
    virtual_loss: float = 1.0      # Virtual loss penalty during leaf accumulation
    tablebase_pieces: int = 5      # Maximum pieces for Syzygy probe
    claim_draw: bool = False       # Treat 3-fold repetition / 50-move rule as immediate draw
    max_collision: int = 8         # Max retry attempts for batch collection
    root_min_visits: int = 1       # Ensure every legal root move receives minimum visits


class Node:
    """MCTS Node storing child statistics in numpy arrays for fast vectorized PUCT selection."""

    __slots__ = (
        "moves",
        "P",
        "N",
        "W",
        "VL",
        "children",
        "expanded",
        "terminal_value",
        "sum_N",
    )

    def __init__(self):
        self.moves: list[chess.Move] = []
        self.P: np.ndarray = np.zeros(0, dtype=np.float32)   # Move priors
        self.N: np.ndarray = np.zeros(0, dtype=np.int32)     # Visit counts
        self.W: np.ndarray = np.zeros(0, dtype=np.float32)   # Accumulated value
        self.VL: np.ndarray = np.zeros(0, dtype=np.float32)  # Virtual loss
        self.children: list[Node | None] = []
        self.expanded: bool = False
        self.terminal_value: float | None = None
        self.sum_N: int = 0

    def expand(self, moves: list[chess.Move], priors: np.ndarray) -> None:
        n = len(moves)
        self.moves = moves
        self.P = priors.astype(np.float32)
        self.N = np.zeros(n, dtype=np.int32)
        self.W = np.zeros(n, dtype=np.float32)
        self.VL = np.zeros(n, dtype=np.float32)
        self.children = [None] * n
        self.expanded = True

    def q(self) -> np.ndarray:
        denom = self.N + self.VL
        out = np.zeros_like(self.W)
        nz = denom > 0
        out[nz] = self.W[nz] / denom[nz]
        return out

    def best_child(self, cfg: MCTSConfig) -> int:
        """PUCT selection: argmax( Q + c * P * sqrt(sum_N) / (1 + N + VL) )."""
        total = max(self.sum_N, 1)
        c = math.log((1 + total + cfg.c_puct_base) / cfg.c_puct_base) + cfg.c_puct_init

        denom = self.N + self.VL
        q = np.zeros_like(self.W)
        visited = denom > 0
        q[visited] = (self.W[visited] - cfg.virtual_loss * self.VL[visited]) / denom[visited]

        # First Play Urgency (FPU)
        if visited.any():
            parent_q = float((self.W[visited] - cfg.virtual_loss * self.VL[visited]).sum() / denom[visited].sum())
        else:
            parent_q = 0.0
        q[~visited] = parent_q - cfg.fpu_reduction

        u = c * self.P * math.sqrt(total) / (1.0 + denom)
        return int(np.argmax(q + u))


def priors_from_policy(
    board: chess.Board, policy: np.ndarray, promo: np.ndarray
) -> tuple[list[chess.Move], np.ndarray]:
    """Map 4096 policy logits/probabilities and 4 promo probabilities to legal moves."""
    moves = list(board.legal_moves)
    if not moves:
        return [], np.zeros(0, dtype=np.float32)

    scores = np.empty(len(moves), dtype=np.float32)
    turn = board.turn
    for i, mv in enumerate(moves):
        om = orient_move(mv, turn)
        s = float(policy[move_to_index(om)])
        pi = move_to_promo_index(om)
        if pi is not None:
            s *= float(promo[pi])
        scores[i] = s

    total = scores.sum()
    if total <= 0:
        scores[:] = 1.0 / len(moves)
    else:
        scores /= total
    return moves, scores


class MCTS:
    """Batched MCTS search engine with virtual loss for neural net batching."""

    def __init__(
        self,
        evaluator: Callable[[list[chess.Board]], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cfg: MCTSConfig | None = None,
        tablebase=None,
        rng: np.random.Generator | None = None,
    ):
        self.evaluator = evaluator
        self.cfg = cfg or MCTSConfig()
        self.tablebase = tablebase
        self.rng = rng or np.random.default_rng()
        self.last_metrics: dict[str, int | float | bool] = {}

    def _exact_value(self, board: chess.Board) -> float | None:
        """Return exact terminal or tablebase value from current player's perspective, or None."""
        if board.is_checkmate():
            return -1.0
        if self.cfg.claim_draw and (board.is_repetition(3) or board.is_fifty_moves()):
            return 0.0
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0.0

        if (
            self.tablebase is not None
            and chess.popcount(board.occupied) <= self.cfg.tablebase_pieces
            and board.is_valid()
        ):
            try:
                wdl = self.tablebase.probe_wdl(board)
            except Exception:
                return None
            if abs(wdl) == 2 and board.halfmove_clock:
                try:
                    dtz = abs(self.tablebase.probe_dtz(board))
                except Exception:
                    return None
                if board.halfmove_clock + dtz >= 100:
                    return None
            return float((wdl == 2) - (wdl == -2))

        return None

    def _collect(
        self, root: Node, root_board: chess.Board, want: int
    ) -> tuple[list[tuple[Node, chess.Board, list[tuple[Node, int]]]], int]:
        leaves: list[tuple[Node, chess.Board, list[tuple[Node, int]]]] = []
        terminal_sims = 0
        pending: set[int] = set()
        spins = 0
        cfg_root_min = self.cfg.root_min_visits
        budget = want

        while (len(leaves) + terminal_sims) < budget and spins < self.cfg.max_collision * want:
            node = root
            board = root_board.copy(stack=True)
            path: list[tuple[Node, int]] = []

            first = True
            while node.expanded and node.terminal_value is None:
                if not node.moves:
                    break
                if first and cfg_root_min > 0:
                    unvisited = np.flatnonzero(node.N + node.VL < cfg_root_min)
                    i = int(unvisited[0]) if unvisited.size else node.best_child(self.cfg)
                else:
                    i = node.best_child(self.cfg)
                first = False

                node.VL[i] += self.cfg.virtual_loss
                path.append((node, i))
                board.push(node.moves[i])

                child = node.children[i]
                if child is None:
                    child = Node()
                    node.children[i] = child
                node = child

            if node.terminal_value is not None:
                self._backup(path, node.terminal_value)
                terminal_sims += 1
                spins += 1
                continue

            if node.expanded:
                self._backup(path, 0.0)
                terminal_sims += 1
                spins += 1
                continue

            exact = self._exact_value(board)
            if exact is not None:
                node.expanded = True
                node.terminal_value = exact
                self._backup(path, exact)
                terminal_sims += 1
                spins += 1
                continue

            if id(node) in pending:
                self.last_metrics["collisions"] = int(self.last_metrics.get("collisions", 0)) + 1
                for parent, edge in path:
                    parent.VL[edge] -= self.cfg.virtual_loss
                spins += 1
                continue

            pending.add(id(node))
            leaves.append((node, board, path))
            spins += 1

        return leaves, terminal_sims

    def _backup(self, path: list[tuple[Node, int]], value: float) -> None:
        v = value
        for node, i in reversed(path):
            v = -v
            node.N[i] += 1
            node.W[i] += v
            node.VL[i] -= self.cfg.virtual_loss
            node.sum_N += 1

    def _evaluate_and_expand(
        self, leaves: list[tuple[Node, chess.Board, list[tuple[Node, int]]]]
    ) -> None:
        if not leaves:
            return
        self.last_metrics["network_positions"] = int(self.last_metrics.get("network_positions", 0)) + len(leaves)
        self.last_metrics["network_batches"] = int(self.last_metrics.get("network_batches", 0)) + 1
        current_max_depth = int(self.last_metrics.get("max_depth", 0))
        self.last_metrics["max_depth"] = max(current_max_depth, max(len(p) for _, _, p in leaves))

        boards = [b for _, b, _ in leaves]
        policy, promo, wdl = self.evaluator(boards)
        for k, (node, board, path) in enumerate(leaves):
            moves, priors = priors_from_policy(board, policy[k], promo[k])
            if not moves:
                node.expanded = True
                node.terminal_value = 0.0
                self._backup(path, 0.0)
                continue
            node.expand(moves, priors)
            v = float(wdl[k][0] - wdl[k][2])  # Q = P(win) - P(loss)
            self._backup(path, v)

    def search(
        self,
        board: chess.Board,
        simulations: int | None = None,
        add_noise: bool = False,
        root: Node | None = None,
    ) -> Node:
        root_reused = root is not None and root.expanded
        self.last_metrics = {
            "network_positions": 0,
            "network_batches": 0,
            "max_depth": 0,
            "collisions": 0,
            "reused_root": root_reused,
        }
        cfg = self.cfg
        sims = simulations or cfg.simulations
        root = root or Node()

        if not root_reused:
            exact = self._exact_value(board)
            if exact is not None:
                root.expanded = True
                root.terminal_value = exact
                return root

            policy, promo, wdl = self.evaluator([board])
            self.last_metrics["network_positions"] = 1
            self.last_metrics["network_batches"] = 1
            moves, priors = priors_from_policy(board, policy[0], promo[0])
            if not moves:
                root.expanded = True
                root.terminal_value = 0.0
                return root
            root.expand(moves, priors)

        if add_noise and len(root.moves) > 1:
            noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(root.moves))
            root.P = ((1 - cfg.dirichlet_eps) * root.P + cfg.dirichlet_eps * noise).astype(np.float32)

        done = 0
        while done < sims:
            want = min(cfg.batch_size, sims - done)
            leaves, terminal_sims = self._collect(root, board, want)
            if not leaves and terminal_sims == 0:
                break
            self._evaluate_and_expand(leaves)
            done += len(leaves) + terminal_sims

        return root

    @staticmethod
    def advance_root(root: Node | None, move: chess.Move) -> Node | None:
        if root is None or not root.expanded:
            return None
        for index, candidate in enumerate(root.moves):
            if candidate == move:
                child = root.children[index]
                if child is not None:
                    child.VL.fill(0.0)
                return child
        return None

    def best_move(
        self,
        board: chess.Board,
        simulations: int | None = None,
        temperature: float | None = None,
        add_noise: bool = False,
        root: Node | None = None,
    ) -> tuple[chess.Move, Node]:
        root = self.search(board, simulations, add_noise=add_noise, root=root)
        if not root.moves:
            legal = list(board.legal_moves)
            if not legal:
                raise ValueError("No legal moves available")
            return legal[0], root

        t = self.cfg.temperature if temperature is None else temperature
        if t <= 0:
            i = int(np.argmax(root.N))
        else:
            counts = root.N.astype(np.float64) ** (1.0 / t)
            s = counts.sum()
            if s <= 0:
                i = int(np.argmax(root.N))
            else:
                i = int(self.rng.choice(len(counts), p=counts / s))
        return root.moves[i], root

    @staticmethod
    def visit_policy(root: Node) -> list[tuple[chess.Move, float]]:
        total = int(root.N.sum())
        if total <= 0:
            return []
        return [(mv, int(n) / total) for mv, n in zip(root.moves, root.N)]

    @staticmethod
    def root_value(root: Node) -> float:
        denom = int(root.N.sum())
        if denom <= 0:
            return 0.0 if root.terminal_value is None else root.terminal_value
        return float(root.W.sum() / denom)
