"""Multi-process Parallel MCTS with Central GPU Evaluator."""
from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Callable

import chess
import numpy as np
import torch

from core.encoding import encode, orient_move
from core.moves import move_to_index, move_to_promo_index
from search.mcts import MCTS, MCTSConfig, Node, priors_from_policy


def _gpu_evaluator_process(
    req_queue: mp.Queue,
    res_pipes: list[mp.connection.Connection],
    stop_event: mp.Event,
    model_preset: str = "transformer_tiny",
    ckpt_path: str | None = None,
    batch_size: int = 128,
    device_str: str = "cuda",
    timeout_ms: float = 2.0,
):
    """Central GPU evaluator loop collecting and batching evaluation requests across all CPU workers."""
    import sys
    # Ensure torch operates cleanly in separate process
    torch.set_num_threads(2)
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")

    from model.transformer import ChessTransformer, TransformerConfig, create_transformer

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = TransformerConfig.from_dict(ckpt.get("cfg", {}))
        model = ChessTransformer(cfg).to(device).eval()
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state)
    else:
        model = create_transformer(model_preset).to(device).eval()

    # Compile or warm up if CUDA
    use_fp16 = device.type == "cuda"

    timeout_sec = timeout_ms / 1000.0

    while not stop_event.is_set():
        batch_items = []
        try:
            item = req_queue.get(timeout=0.01)
            batch_items.append(item)
            deadline = time.perf_counter() + timeout_sec
            while len(batch_items) < batch_size and time.perf_counter() < deadline:
                try:
                    item = req_queue.get_nowait()
                    batch_items.append(item)
                except Exception:
                    # Brief pause if queue momentarily empty
                    time.sleep(0.0001)
                    if stop_event.is_set():
                        break
        except Exception:
            continue

        if not batch_items:
            continue

        # Each item: (worker_id, req_id, np_array_boards)
        worker_ids = [it[0] for it in batch_items]
        req_ids = [it[1] for it in batch_items]
        arrays = [it[2] for it in batch_items]
        sizes = [arr.shape[0] for arr in arrays]

        full_array = np.concatenate(arrays, axis=0)
        tensor = torch.from_numpy(full_array).to(device, non_blocking=True)

        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    p_logits, pr_logits, w_logits = model(tensor)
            else:
                p_logits, pr_logits, w_logits = model(tensor)

            p_probs = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            pr_probs = torch.softmax(pr_logits.float(), dim=-1).cpu().numpy()
            w_probs = torch.softmax(w_logits.float(), dim=-1).cpu().numpy()

        offset = 0
        for wid, rid, sz in zip(worker_ids, req_ids, sizes):
            p_slice = p_probs[offset : offset + sz]
            pr_slice = pr_probs[offset : offset + sz]
            w_slice = w_probs[offset : offset + sz]
            res_pipes[wid].send((rid, p_slice, pr_slice, w_slice))
            offset += sz


def _cpu_worker_search(
    worker_id: int,
    res_pipe: mp.connection.Connection,
    req_queue: mp.Queue,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
):
    """CPU worker process executing tree simulations and communicating with central GPU evaluator."""
    req_counter = 0

    def remote_evaluator(boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal req_counter
        req_counter += 1
        req_id = req_counter
        planes = np.stack([encode(b) for b in boards], axis=0)
        req_queue.put((worker_id, req_id, planes))

        # Block until evaluator returns corresponding result
        while True:
            ret_id, p, pr, w = res_pipe.recv()
            if ret_id == req_id:
                return p, pr, w

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.02)
        except Exception:
            continue

        if task is None or stop_event.is_set():
            break

        task_id, fen, sims, mcts_cfg_dict, seed = task
        board = chess.Board(fen)
        cfg = MCTSConfig(**mcts_cfg_dict)
        rng = np.random.default_rng(seed)

        mcts = MCTS(remote_evaluator, cfg=cfg, rng=rng)
        root = mcts.search(board, simulations=sims, add_noise=True)

        moves_uci = [m.uci() for m in root.moves]
        visits = root.N.tolist()
        values = root.W.tolist()
        sum_n = int(root.sum_N)

        result_queue.put((task_id, worker_id, moves_uci, visits, values, sum_n))


class ParallelMCTS:
    """Multi-process MCTS coordinator managing CPU worker pool and central GPU evaluator."""

    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 128,
        model_preset: str = "transformer_tiny",
        ckpt_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cfg: MCTSConfig | None = None,
        eval_timeout_ms: float = 2.0,
    ):
        self.num_workers = max(1, min(num_workers, 32))
        self.batch_size = batch_size
        self.model_preset = model_preset
        self.ckpt_path = ckpt_path
        self.device = device
        self.cfg = cfg or MCTSConfig()
        self.eval_timeout_ms = eval_timeout_ms

        self.ctx = mp.get_context("spawn")
        self.req_queue: mp.Queue = self.ctx.Queue()
        self.task_queue: mp.Queue = self.ctx.Queue()
        self.result_queue: mp.Queue = self.ctx.Queue()
        self.stop_event: mp.Event = self.ctx.Event()

        # Pipes: pipes[i][0] is worker read end, pipes[i][1] is evaluator write end
        self.pipes = [self.ctx.Pipe(duplex=False) for _ in range(self.num_workers)]

        # Start GPU evaluator
        eval_writers = [p[1] for p in self.pipes]
        self.eval_proc = self.ctx.Process(
            target=_gpu_evaluator_process,
            args=(
                self.req_queue,
                eval_writers,
                self.stop_event,
                self.model_preset,
                self.ckpt_path,
                self.batch_size,
                self.device,
                self.eval_timeout_ms,
            ),
        )
        self.eval_proc.start()

        # Start CPU workers
        self.workers: list[mp.Process] = []
        for wid in range(self.num_workers):
            p = self.ctx.Process(
                target=_cpu_worker_search,
                args=(
                    wid,
                    self.pipes[wid][0],
                    self.req_queue,
                    self.task_queue,
                    self.result_queue,
                    self.stop_event,
                ),
            )
            p.start()
            self.workers.append(p)

    def close(self):
        """Cleanly terminate all processes and queues."""
        self.stop_event.set()
        for _ in range(self.num_workers):
            try:
                self.task_queue.put_nowait(None)
            except Exception:
                pass

        for w in self.workers:
            w.join(timeout=1.0)
            if w.is_alive():
                w.terminate()

        self.eval_proc.join(timeout=1.0)
        if self.eval_proc.is_alive():
            self.eval_proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def search(
        self,
        board: chess.Board,
        simulations: int = 400,
        cfg: MCTSConfig | None = None,
        base_seed: int = 42,
    ) -> tuple[chess.Move, dict[str, int]]:
        """Run parallel MCTS search distributed across all CPU workers, aggregating root visit counts."""
        search_cfg = cfg or self.cfg
        cfg_dict = {
            "c_puct": search_cfg.c_puct,
            "c_puct_base": search_cfg.c_puct_base,
            "c_puct_init": search_cfg.c_puct_init,
            "dirichlet_alpha": search_cfg.dirichlet_alpha,
            "dirichlet_eps": search_cfg.dirichlet_eps,
            "fpu_reduction": search_cfg.fpu_reduction,
            "virtual_loss": search_cfg.virtual_loss,
            "batch_size": search_cfg.batch_size,
        }

        task_id = f"search_{int(time.time()*1000)}"
        sims_per_worker = max(1, simulations // self.num_workers)

        for wid in range(self.num_workers):
            seed = base_seed + wid * 10007
            self.task_queue.put((task_id, board.fen(), sims_per_worker, cfg_dict, seed))

        # Collect results from all workers
        all_moves: set[str] = set()
        worker_results = []
        for _ in range(self.num_workers):
            res = self.result_queue.get()
            worker_results.append(res)
            for m in res[2]:
                all_moves.add(m)

        # Aggregate visits and values
        aggregated_visits: dict[str, int] = {m: 0 for m in all_moves}
        for _, _, moves_uci, visits, _, _ in worker_results:
            for m, n in zip(moves_uci, visits):
                aggregated_visits[m] += n

        if not aggregated_visits:
            legal = list(board.legal_moves)
            return legal[0], {}

        best_uci = max(aggregated_visits.items(), key=lambda kv: kv[1])[0]
        return chess.Move.from_uci(best_uci), aggregated_visits


def benchmark_parallel_mcts(
    worker_counts: list[int] = (1, 4, 8, 16, 32),
    batch_sizes: list[int] = (64, 128, 256),
    simulations_per_test: int = 400,
    model_preset: str = "transformer_tiny",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fen: str = chess.STARTING_FEN,
) -> dict[str, dict[str, float]]:
    """Benchmark simulations/sec throughput across different worker counts and batch sizes."""
    board = chess.Board(fen)
    results = {}

    print(f"\n========================================================")
    print(f"Parallel MCTS Benchmark (Model: {model_preset}, Device: {device})")
    print(f"========================================================")
    print(f"{'Workers':<10}{'Batch Size':<12}{'Simulations':<14}{'Time (s)':<12}{'Sims/sec':<12}")
    print("-" * 60)

    for num_w in worker_counts:
        results[num_w] = {}
        for b_sz in batch_sizes:
            pmcts = ParallelMCTS(
                num_workers=num_w,
                batch_size=b_sz,
                model_preset=model_preset,
                device=device,
            )
            try:
                # Warmup run
                pmcts.search(board, simulations=min(50, simulations_per_test))

                t0 = time.perf_counter()
                best_move, visits = pmcts.search(board, simulations=simulations_per_test)
                dt = time.perf_counter() - t0

                total_visits = sum(visits.values())
                actual_sims = max(total_visits, simulations_per_test)
                sims_per_sec = actual_sims / dt if dt > 0 else 0.0

                results[num_w][b_sz] = sims_per_sec
                print(f"{num_w:<10}{b_sz:<12}{actual_sims:<14}{dt:<12.3f}{sims_per_sec:<12.1f}")
            finally:
                pmcts.close()

    print("========================================================\n")
    return results


if __name__ == "__main__":
    benchmark_parallel_mcts(worker_counts=[1, 4, 8], batch_sizes=[64, 128], simulations_per_test=200)
