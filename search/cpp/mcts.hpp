#pragma once

#include "chess_board.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <unordered_set>
#include <functional>

namespace chess {

// Policy index mapping: from_sq * 64 + to_sq
inline int move_to_index(const Move& m) {
    return (int)m.from_sq * 64 + (int)m.to_sq;
}

using TablebaseProbeFn = std::function<std::pair<bool, float>(const std::string&)>;

struct MCTSConfig {
    int simulations = 800;
    int batch_size = 64;
    float c_puct = 1.8f;
    float c_puct_base = 19652.0f;
    float c_puct_init = 1.8f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_eps = 0.25f;
    float fpu_reduction = 0.2f;
    float temperature = 0.0f;
    float virtual_loss = 1.0f;
    bool claim_draw = false;
    int max_collision = 8;
    int root_min_visits = 1;
    int tablebase_pieces = 5;
    TablebaseProbeFn tablebase_probe_fn = nullptr;
};

struct Node {
    std::vector<Move> moves;
    std::vector<float> P;   // Priors
    std::vector<int> N;     // Visit counts
    std::vector<float> W;   // Value sums
    std::vector<float> VL;  // Virtual losses
    std::vector<std::unique_ptr<Node>> children;

    bool expanded = false;
    bool is_terminal = false;
    float terminal_value = 0.0f;
    int sum_N = 0;

    Node() = default;

    void expand(const std::vector<Move>& legal_moves, const std::vector<float>& priors) {
        moves = legal_moves;
        P = priors;
        int n = moves.size();
        N.assign(n, 0);
        W.assign(n, 0.0f);
        VL.assign(n, 0.0f);
        children.resize(n);
        expanded = true;
    }

    int best_child(const MCTSConfig& cfg) const {
        int total = std::max(sum_N, 1);
        float c = std::log((1.0f + total + cfg.c_puct_base) / cfg.c_puct_base) + cfg.c_puct_init;

        int best_idx = 0;
        float best_score = -1e9f;

        // FPU calculation
        float sum_visited_w = 0.0f;
        float sum_visited_denom = 0.0f;
        int visited_count = 0;

        for (size_t i = 0; i < moves.size(); ++i) {
            float denom = N[i] + VL[i];
            if (denom > 0.0f) {
                sum_visited_w += (W[i] - cfg.virtual_loss * VL[i]);
                sum_visited_denom += denom;
                visited_count++;
            }
        }

        float parent_q = (visited_count > 0 && sum_visited_denom > 0.0f)
                         ? (sum_visited_w / sum_visited_denom)
                         : 0.0f;
        float fpu_q = parent_q - cfg.fpu_reduction;

        for (size_t i = 0; i < moves.size(); ++i) {
            float denom = N[i] + VL[i];
            float q = (denom > 0.0f) ? ((W[i] - cfg.virtual_loss * VL[i]) / denom) : fpu_q;
            float u = c * P[i] * std::sqrt((float)total) / (1.0f + denom);
            float score = q + u;
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        return best_idx;
    }
};

inline void priors_from_policy(
    const Board& board,
    const float* policy,   // 4096 floats
    const float* promo,    // 4 floats (Q, R, B, N)
    const std::vector<Move>& moves,
    std::vector<float>& scores
) {
    scores.resize(moves.size());
    if (moves.empty()) return;

    float sum = 0.0f;
    for (size_t i = 0; i < moves.size(); ++i) {
        Move om = board.orient_move(moves[i]);
        int idx = move_to_index(om);
        float s = policy[idx];
        if (om.promo != PROMO_NONE) {
            uint8_t pi = promo_to_index(om.promo);
            if (pi < 4) {
                s *= promo[pi];
            }
        }
        scores[i] = s;
        sum += s;
    }

    if (sum <= 0.0f) {
        float uniform = 1.0f / moves.size();
        for (size_t i = 0; i < moves.size(); ++i) scores[i] = uniform;
    } else {
        float inv = 1.0f / sum;
        for (size_t i = 0; i < moves.size(); ++i) scores[i] *= inv;
    }
}

} // namespace chess
