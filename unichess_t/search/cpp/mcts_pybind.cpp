#include <torch/extension.h>
#include "chess_board.hpp"
#include "mcts.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <chrono>
#include <random>

namespace py = pybind11;

static uint64_t perft_recursive(chess::Board& board, int depth) {
    if (depth <= 0) return 1;
    std::vector<chess::Move> moves;
    board.generate_legal_moves(moves);
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const auto& m : moves) {
        chess::Board next_b = board;
        next_b.make_move(m);
        nodes += perft_recursive(next_b, depth - 1);
    }
    return nodes;
}

uint64_t perft(const std::string& fen, int depth) {
    chess::Board board(fen);
    return perft_recursive(board, depth);
}

std::vector<std::string> get_legal_moves(const std::string& fen) {
    chess::Board board(fen);
    std::vector<chess::Move> moves;
    board.generate_legal_moves(moves);
    std::vector<std::string> res;
    res.reserve(moves.size());
    for (const auto& m : moves) {
        res.push_back(m.to_uci());
    }
    return res;
}

torch::Tensor encode_planes(const std::string& fen) {
    chess::Board board(fen);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::zeros({19, 8, 8}, options);
    board.encode_planes(tensor.data_ptr<float>());
    return tensor;
}

class MCTSCpp {
public:
    chess::MCTSConfig cfg;
    std::mt19937 rng;
    py::object syzygy_obj = py::none();

    std::unique_ptr<chess::Node> stored_root;
    chess::Board stored_root_board;
    std::string stored_root_fen;

    MCTSCpp(
        float c_puct_init = 1.8f,
        float c_puct_base = 19652.0f,
        float virtual_loss = 1.0f,
        py::object syzygy = py::none()
    ) {
        cfg.c_puct_init = c_puct_init;
        cfg.c_puct_base = c_puct_base;
        cfg.virtual_loss = virtual_loss;
        set_syzygy_fn(syzygy);
        std::random_device rd;
        rng.seed(rd());
    }

    void set_seed(uint64_t seed) {
        rng.seed(seed);
    }

    void set_syzygy_fn(py::object syzygy) {
        syzygy_obj = syzygy;
    }

    void reset() {
        stored_root.reset();
        stored_root_fen.clear();
        stored_root_board.clear();
    }

    bool reuse_root(const std::string& move_uci) {
        if (!stored_root || !stored_root->expanded) return false;

        chess::Move m = chess::Move::from_uci(move_uci);
        for (size_t i = 0; i < stored_root->moves.size(); ++i) {
            if (stored_root->moves[i] == m) {
                if (!stored_root->children[i]) return false;
                std::fill(stored_root->children[i]->VL.begin(), stored_root->children[i]->VL.end(), 0.0f);
                stored_root_board.make_move(m);
                stored_root_fen = stored_root_board.to_fen();
                stored_root = std::move(stored_root->children[i]);
                return true;
            }
        }
        return false;
    }

    static chess::TablebaseProbeFn resolve_probe_fn(py::object obj) {
        if (obj.is_none()) return nullptr;

        py::object callable_fn = obj;
        if (py::hasattr(obj, "probe_wdl") || py::isinstance<py::str>(obj)) {
            py::object search_cpp = py::module_::import("unichess_t.search.cpp");
            callable_fn = search_cpp.attr("make_syzygy_probe")(obj);
        }

        return [callable_fn](const std::string& fen) -> std::pair<bool, float> {
            try {
                py::tuple res = callable_fn(fen);
                return {res[0].cast<bool>(), res[1].cast<float>()};
            } catch (const std::exception&) {
                return {false, 0.0f};
            }
        };
    }

    struct LeafPath {
        chess::Node* node;
        chess::Board board;
        std::vector<std::pair<chess::Node*, int>> path;
    };

    void backup(const std::vector<std::pair<chess::Node*, int>>& path, float value, float contempt, int root_N) {
        float v = value;
        float draw_bias = 0.0f;
        if (contempt != 0.0f && root_N > 0 && std::abs(v) < 1e-6f) {
            draw_bias = contempt * (1.0f / (1.0f + (float)root_N));
        }

        for (size_t i = 0; i < path.size(); ++i) {
            v = -v;
            if (i == path.size() - 1 && draw_bias != 0.0f) {
                v += draw_bias;
            }
            chess::Node* n = path[path.size() - 1 - i].first;
            int edge = path[path.size() - 1 - i].second;
            n->N[edge] += 1;
            n->W[edge] += v;
            n->VL[edge] -= cfg.virtual_loss;
            n->sum_N += 1;
        }
    }

    py::tuple search(
        const std::string& fen,
        py::function evaluator,
        int simulations = 800,
        int batch_size = 64,
        bool add_noise = false,
        float temperature = 0.0f,
        py::object syzygy = py::none(),
        py::object syzygy_path = py::none(),
        bool reuse = false,
        const std::string& previous_root_fen = "",
        float contempt = 0.0f,
        float c_fpu = 0.5f
    ) {
        cfg.simulations = simulations;
        cfg.batch_size = batch_size;
        cfg.temperature = temperature;
        cfg.contempt = contempt;
        if (c_fpu > 0.0f) cfg.c_fpu = c_fpu;

        chess::TablebaseProbeFn tb_probe = nullptr;
        if (!syzygy_path.is_none()) {
            tb_probe = resolve_probe_fn(syzygy_path);
        } else if (!syzygy.is_none()) {
            tb_probe = resolve_probe_fn(syzygy);
        } else if (!syzygy_obj.is_none()) {
            tb_probe = resolve_probe_fn(syzygy_obj);
        } else if (cfg.tablebase_probe_fn) {
            tb_probe = cfg.tablebase_probe_fn;
        }

        std::unique_ptr<chess::Node> root_uptr;
        chess::Node* root = nullptr;
        chess::Board root_board;
        bool reused = false;

        if (reuse) {
            if (!previous_root_fen.empty()) {
                chess::Board prev(previous_root_fen);
                std::vector<chess::Move> legal_moves;
                prev.generate_legal_moves(legal_moves);
                std::string played_move;
                for (const auto& m : legal_moves) {
                    chess::Board next = prev;
                    next.make_move(m);
                    if (next.to_fen() == fen) {
                        played_move = m.to_uci();
                        break;
                    }
                }
                if (!played_move.empty() && reuse_root(played_move)) {
                    root = stored_root.get();
                    root_board = stored_root_board;
                    reused = true;
                }
            }
            if (!reused && stored_root_fen == fen && stored_root && stored_root->expanded) {
                root = stored_root.get();
                root_board = stored_root_board;
                reused = true;
            }
        }

        if (!root) {
            root_uptr = std::make_unique<chess::Node>();
            root = root_uptr.get();
            root_board = chess::Board(fen);
        }

        int root_term = root_board.check_terminal(cfg.claim_draw);
        if (root_term != 2) {
            root->expanded = true;
            root->is_terminal = true;
            root->terminal_value = (float)root_term;
            py::dict metrics;
            metrics["visits"] = 0;
            metrics["root_value"] = root->terminal_value;
            metrics["tablebase_hits"] = 0;
            metrics["reused_root"] = reused;
            if (!reused && root_uptr) {
                stored_root = std::move(root_uptr);
                stored_root_fen = fen;
            }
            return py::make_tuple("", metrics);
        }

        if (!reused) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor root_plane_tensor = torch::zeros({1, 19, 8, 8}, options);
            root_board.encode_planes(root_plane_tensor.data_ptr<float>());

            py::tuple eval_res = evaluator(root_plane_tensor);
            torch::Tensor p_t = eval_res[0].cast<torch::Tensor>().contiguous().cpu();
            torch::Tensor pr_t = eval_res[1].cast<torch::Tensor>().contiguous().cpu();

            std::vector<chess::Move> legal_moves;
            root_board.generate_legal_moves(legal_moves);

            if (legal_moves.empty()) {
                root->expanded = true;
                root->is_terminal = true;
                root->terminal_value = 0.0f;
                py::dict metrics;
                metrics["visits"] = 0;
                metrics["root_value"] = 0.0f;
                metrics["tablebase_hits"] = 0;
                metrics["reused_root"] = false;
                stored_root = std::move(root_uptr);
                stored_root_fen = fen;
                return py::make_tuple("", metrics);
            }

            const float* p_ptr = p_t.data_ptr<float>();
            const float* pr_ptr = pr_t.data_ptr<float>();
            std::vector<float> priors;
            chess::priors_from_policy(root_board, p_ptr, pr_ptr, legal_moves, priors);

            if (add_noise && cfg.dirichlet_alpha > 0.0f && !legal_moves.empty() && !reused) {
                std::gamma_distribution<float> gamma(cfg.dirichlet_alpha, 1.0f);
                std::vector<float> noise(legal_moves.size());
                float sum_noise = 0.0f;
                for (size_t i = 0; i < noise.size(); ++i) {
                    noise[i] = gamma(rng);
                    sum_noise += noise[i];
                }
                if (sum_noise > 1e-7f) {
                    float eps = cfg.dirichlet_eps;
                    for (size_t i = 0; i < noise.size(); ++i) {
                        priors[i] = (1.0f - eps) * priors[i] + eps * (noise[i] / sum_noise);
                    }
                }
            }

            root->expand(legal_moves, priors);
        }

        int network_batches = reused ? 0 : 1;
        int network_positions = reused ? 0 : 1;
        int tablebase_hits = 0;

        std::vector<LeafPath> leaves;
        leaves.reserve(batch_size);

        while (root->sum_N < simulations) {
            leaves.clear();

            for (int b = 0; b < batch_size && root->sum_N + (int)leaves.size() < simulations; ++b) {
                chess::Node* curr = root;
                chess::Board curr_board = root_board;
                std::vector<std::pair<chess::Node*, int>> path;

                bool reached_terminal = false;

                while (curr->expanded && !curr->is_terminal) {
                    int best_action = -1;
                    float best_score = -1e9f;

                    int N_parent = curr->sum_N;
                    float c = std::log((1.0f + N_parent + cfg.c_puct_base) / cfg.c_puct_base) + cfg.c_puct_init;
                    float sqrt_N = std::sqrt((float)std::max(1, N_parent));

                    float fpu_q = chess::compute_fpu_q(*curr, cfg);

                    for (size_t a = 0; a < curr->moves.size(); ++a) {
                        int n_child = curr->N[a];
                        float vl = curr->VL[a];
                        float total_n = n_child + vl;

                        float q = (total_n > 0.0f) ? (curr->W[a] - vl * cfg.virtual_loss) / total_n : fpu_q;
                        float u = c * curr->P[a] * sqrt_N / (1.0f + total_n);
                        float score = q + u;

                        if (score > best_score) {
                            best_score = score;
                            best_action = (int)a;
                        }
                    }

                    if (best_action < 0) {
                        break;
                    }

                    path.emplace_back(curr, best_action);
                    curr->VL[best_action] += cfg.virtual_loss;
                    curr_board.make_move(curr->moves[best_action]);

                    if (!curr->children[best_action]) {
                        curr->children[best_action] = std::make_unique<chess::Node>();
                    }
                    curr = curr->children[best_action].get();

                    int term = curr_board.check_terminal(cfg.claim_draw);
                    if (term != 2) {
                        curr->is_terminal = true;
                        curr->terminal_value = (float)term;
                        reached_terminal = true;
                        break;
                    }
                }

                if (reached_terminal || curr->is_terminal) {
                    backup(path, curr->terminal_value, cfg.contempt, root->sum_N);
                } else if (!curr->expanded) {
                    bool tb_hit = false;
                    if (tb_probe && curr_board.piece_count() <= cfg.tablebase_pieces) {
                        auto [is_tb, exact_val] = tb_probe(curr_board.to_fen());
                        if (is_tb) {
                            curr->expanded = true;
                            curr->is_terminal = true;
                            curr->terminal_value = exact_val;
                            backup(path, exact_val, cfg.contempt, root->sum_N);
                            tablebase_hits++;
                            tb_hit = true;
                        }
                    }
                    if (!tb_hit) {
                        leaves.push_back({curr, curr_board, path});
                    }
                } else {
                    for (auto& p : path) {
                        p.first->VL[p.second] -= cfg.virtual_loss;
                    }
                }
            }

            if (leaves.empty()) {
                if (root->sum_N >= simulations || root->is_terminal) break;
                continue;
            }

            int num_leaves = (int)leaves.size();
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor batch_tensor = torch::zeros({num_leaves, 19, 8, 8}, options);
            float* plane_ptr = batch_tensor.data_ptr<float>();

            for (int idx = 0; idx < num_leaves; ++idx) {
                leaves[idx].board.encode_planes(plane_ptr + idx * 19 * 64);
            }

            py::tuple eval_res = evaluator(batch_tensor);
            torch::Tensor policy_batch = eval_res[0].cast<torch::Tensor>().contiguous().cpu();
            torch::Tensor promo_batch = eval_res[1].cast<torch::Tensor>().contiguous().cpu();
            torch::Tensor wdl_batch = eval_res[2].cast<torch::Tensor>().contiguous().cpu();

            const float* p_data = policy_batch.data_ptr<float>();
            const float* pr_data = promo_batch.data_ptr<float>();
            const float* wdl_data = wdl_batch.data_ptr<float>();

            network_batches += 1;
            network_positions += num_leaves;

            for (int idx = 0; idx < num_leaves; ++idx) {
                LeafPath& leaf = leaves[idx];
                float p_win = wdl_data[idx * 3 + 0];
                float p_loss = wdl_data[idx * 3 + 2];
                float v = p_win - p_loss;

                std::vector<chess::Move> legal_moves;
                leaf.board.generate_legal_moves(legal_moves);

                if (legal_moves.empty()) {
                    leaf.node->expanded = true;
                    leaf.node->is_terminal = true;
                    int term = leaf.board.check_terminal(cfg.claim_draw);
                    float term_v = (term != 2) ? (float)term : 0.0f;
                    leaf.node->terminal_value = term_v;
                    backup(leaf.path, term_v, cfg.contempt, root->sum_N);
                } else {
                    const float* p_sub = p_data + idx * 4096;
                    const float* pr_sub = pr_data + idx * 4;
                    std::vector<float> priors;
                    chess::priors_from_policy(leaf.board, p_sub, pr_sub, legal_moves, priors);
                    leaf.node->expand(legal_moves, priors);
                    backup(leaf.path, v, cfg.contempt, root->sum_N);
                }
            }
        }

        if (!reused) {
            stored_root = std::move(root_uptr);
            stored_root_fen = fen;
        }

        if (root->moves.empty()) {
            py::dict metrics;
            metrics["visits"] = 0;
            metrics["root_value"] = root->terminal_value;
            metrics["tablebase_hits"] = tablebase_hits;
            metrics["reused_root"] = reused;
            return py::make_tuple("", metrics);
        }

        int best_idx = 0;
        if (temperature <= 0.01f) {
            int max_n = -1;
            for (size_t i = 0; i < root->moves.size(); ++i) {
                if (root->N[i] > max_n) {
                    max_n = root->N[i];
                    best_idx = (int)i;
                }
            }
        } else {
            std::vector<double> probs(root->moves.size());
            double sum_p = 0.0;
            for (size_t i = 0; i < root->moves.size(); ++i) {
                probs[i] = std::pow((double)root->N[i], 1.0 / cfg.temperature);
                sum_p += probs[i];
            }
            if (sum_p <= 0.0) {
                best_idx = 0;
            } else {
                std::uniform_real_distribution<double> dist(0.0, sum_p);
                double r = dist(rng);
                double acc = 0.0;
                for (size_t i = 0; i < probs.size(); ++i) {
                    acc += probs[i];
                    if (r <= acc) {
                        best_idx = i;
                        break;
                    }
                }
            }
        }

        std::string best_move_uci = root->moves[best_idx].to_uci();

        py::dict metrics;
        metrics["visits"] = root->sum_N;
        metrics["network_batches"] = network_batches;
        metrics["network_positions"] = network_positions;
        metrics["tablebase_hits"] = tablebase_hits;
        metrics["reused_root"] = reused;

        py::dict visits_dict;
        for (size_t i = 0; i < root->moves.size(); ++i) {
            visits_dict[py::str(root->moves[i].to_uci())] = root->N[i];
        }
        metrics["policy"] = visits_dict;

        float root_v = 0.0f;
        if (root->sum_N > 0) {
            float sum_w = 0.0f;
            for (float w : root->W) sum_w += w;
            root_v = sum_w / root->sum_N;
        }
        metrics["root_value"] = root_v;

        return py::make_tuple(best_move_uci, metrics);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("perft", &perft, py::arg("fen"), py::arg("depth"), "Compute perft node count");
    m.def("get_legal_moves", &get_legal_moves, py::arg("fen"), "Get legal moves in UCI format");
    m.def("encode_planes", &encode_planes, py::arg("fen"), "Encode board into 19x8x8 tensor");
    m.def("piece_count", [](const std::string& fen) {
        return chess::Board(fen).piece_count();
    }, py::arg("fen"), "Get piece count on board");

    py::class_<MCTSCpp>(m, "MCTSCpp")
        .def(py::init<float, float, float, py::object>(),
             py::arg("c_puct_init") = 1.8f,
             py::arg("c_puct_base") = 19652.0f,
             py::arg("virtual_loss") = 1.0f,
             py::arg("syzygy") = py::none())
        .def("set_seed", &MCTSCpp::set_seed, py::arg("seed"))
        .def("set_syzygy_fn", &MCTSCpp::set_syzygy_fn, py::arg("syzygy"))
        .def("reset", &MCTSCpp::reset, "Reset stored tree for fresh search")
        .def("reuse_root", &MCTSCpp::reuse_root, py::arg("move_uci"),
             "Promote child node matching move_uci to new root for tree reuse")
        .def("search", &MCTSCpp::search,
             py::arg("fen"),
             py::arg("evaluator"),
             py::arg("simulations") = 800,
             py::arg("batch_size") = 64,
             py::arg("add_noise") = false,
             py::arg("temperature") = 0.0f,
             py::arg("syzygy") = py::none(),
             py::arg("syzygy_path") = py::none(),
             py::arg("reuse") = false,
             py::arg("previous_root_fen") = "",
             py::arg("contempt") = 0.0f,
             py::arg("c_fpu") = 0.5f);
}
