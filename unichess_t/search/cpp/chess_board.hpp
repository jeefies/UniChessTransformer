#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace chess {

// Square indices 0..63: 0 = a1, 7 = h1, 56 = a8, 63 = h8
// row = sq / 8 (rank 0..7), col = sq % 8 (file 0..7)

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
    NO_COLOR = 2
};

enum PieceType : uint8_t {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    NONE = 6
};

// Promotion pieces in policy index order: Queen=0, Rook=1, Bishop=2, Knight=3
enum PromoType : uint8_t {
    PROMO_NONE = 0,
    PROMO_QUEEN = 1,
    PROMO_ROOK = 2,
    PROMO_BISHOP = 3,
    PROMO_KNIGHT = 4
};

inline uint8_t promo_to_index(PromoType pt) {
    switch (pt) {
        case PROMO_QUEEN: return 0;
        case PROMO_ROOK: return 1;
        case PROMO_BISHOP: return 2;
        case PROMO_KNIGHT: return 3;
        default: return 255;
    }
}

inline PromoType index_to_promo(uint8_t idx) {
    switch (idx) {
        case 0: return PROMO_QUEEN;
        case 1: return PROMO_ROOK;
        case 2: return PROMO_BISHOP;
        case 3: return PROMO_KNIGHT;
        default: return PROMO_NONE;
    }
}

struct Move {
    uint8_t from_sq = 0;
    uint8_t to_sq = 0;
    PromoType promo = PROMO_NONE;

    Move() = default;
    Move(uint8_t f, uint8_t t, PromoType p = PROMO_NONE) : from_sq(f), to_sq(t), promo(p) {}

    bool operator==(const Move& o) const {
        return from_sq == o.from_sq && to_sq == o.to_sq && promo == o.promo;
    }
    bool operator!=(const Move& o) const {
        return !(*this == o);
    }

    std::string to_uci() const {
        std::string s;
        s += (char)('a' + (from_sq % 8));
        s += (char)('1' + (from_sq / 8));
        s += (char)('a' + (to_sq % 8));
        s += (char)('1' + (to_sq / 8));
        if (promo == PROMO_QUEEN) s += 'q';
        else if (promo == PROMO_ROOK) s += 'r';
        else if (promo == PROMO_BISHOP) s += 'b';
        else if (promo == PROMO_KNIGHT) s += 'n';
        return s;
    }

    static Move from_uci(const std::string& uci) {
        if (uci.size() < 4) return Move();
        uint8_t f = (uci[0] - 'a') + (uci[1] - '1') * 8;
        uint8_t t = (uci[2] - 'a') + (uci[3] - '1') * 8;
        PromoType p = PROMO_NONE;
        if (uci.size() >= 5) {
            char pr = uci[4];
            if (pr == 'q' || pr == 'Q') p = PROMO_QUEEN;
            else if (pr == 'r' || pr == 'R') p = PROMO_ROOK;
            else if (pr == 'b' || pr == 'B') p = PROMO_BISHOP;
            else if (pr == 'n' || pr == 'N') p = PROMO_KNIGHT;
        }
        return Move(f, t, p);
    }
};

// Castling rights bitmask
constexpr uint8_t CASTLE_WK = 1;
constexpr uint8_t CASTLE_WQ = 2;
constexpr uint8_t CASTLE_BK = 4;
constexpr uint8_t CASTLE_BQ = 8;

// Square mirroring: row -> 7 - row
inline uint8_t square_mirror(uint8_t sq) {
    return sq ^ 56;
}

// Bitboard attacks tables & rays
struct AttackTables {
    std::array<uint64_t, 64> knight_attacks{};
    std::array<uint64_t, 64> king_attacks{};
    std::array<std::array<uint64_t, 64>, 2> pawn_attacks{}; // [color][square]

    AttackTables() {
        for (int sq = 0; sq < 64; ++sq) {
            int r = sq / 8;
            int c = sq % 8;

            // Knight
            uint64_t n_mask = 0;
            static const int knight_dr[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
            static const int knight_dc[8] = {-1, 1, -2, 2, -2, 2, -1, 1};
            for (int d = 0; d < 8; ++d) {
                int nr = r + knight_dr[d];
                int nc = c + knight_dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                    n_mask |= (1ULL << (nr * 8 + nc));
                }
            }
            knight_attacks[sq] = n_mask;

            // King
            uint64_t k_mask = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr;
                    int nc = c + dc;
                    if (nr >= 0 && nr < 8 && nc >= 0 && nc < 8) {
                        k_mask |= (1ULL << (nr * 8 + nc));
                    }
                }
            }
            king_attacks[sq] = k_mask;

            // White pawn attacks
            uint64_t wp_mask = 0;
            if (r < 7) {
                if (c > 0) wp_mask |= (1ULL << ((r + 1) * 8 + c - 1));
                if (c < 7) wp_mask |= (1ULL << ((r + 1) * 8 + c + 1));
            }
            pawn_attacks[WHITE][sq] = wp_mask;

            // Black pawn attacks
            uint64_t bp_mask = 0;
            if (r > 0) {
                if (c > 0) bp_mask |= (1ULL << ((r - 1) * 8 + c - 1));
                if (c < 7) bp_mask |= (1ULL << ((r - 1) * 8 + c + 1));
            }
            pawn_attacks[BLACK][sq] = bp_mask;
        }
    }
};

inline const AttackTables& get_attack_tables() {
    static AttackTables tables;
    return tables;
}

inline uint64_t get_rook_attacks(int sq, uint64_t occupied) {
    uint64_t attacks = 0;
    int r = sq / 8;
    int c = sq % 8;

    // Up
    for (int nr = r + 1; nr < 8; ++nr) {
        int s = nr * 8 + c;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    // Down
    for (int nr = r - 1; nr >= 0; --nr) {
        int s = nr * 8 + c;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    // Right
    for (int nc = c + 1; nc < 8; ++nc) {
        int s = r * 8 + nc;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    // Left
    for (int nc = c - 1; nc >= 0; --nc) {
        int s = r * 8 + nc;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    return attacks;
}

inline uint64_t get_bishop_attacks(int sq, uint64_t occupied) {
    uint64_t attacks = 0;
    int r = sq / 8;
    int c = sq % 8;

    // Up-Right
    for (int nr = r + 1, nc = c + 1; nr < 8 && nc < 8; ++nr, ++nc) {
        int s = nr * 8 + nc;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    // Up-Left
    for (int nr = r + 1, nc = c - 1; nr < 8 && nc >= 0; ++nr, --nc) {
        int s = nr * 8 + nc;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    // Down-Right
    for (int nr = r - 1, nc = c + 1; nr >= 0 && nc < 8; --nr, ++nc) {
        int s = nr * 8 + nc;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    // Down-Left
    for (int nr = r - 1, nc = c - 1; nr >= 0 && nc >= 0; --nr, --nc) {
        int s = nr * 8 + nc;
        attacks |= (1ULL << s);
        if (occupied & (1ULL << s)) break;
    }
    return attacks;
}

inline uint64_t get_queen_attacks(int sq, uint64_t occupied) {
    return get_rook_attacks(sq, occupied) | get_bishop_attacks(sq, occupied);
}

// Zobrist Hashing for repetition tracking
struct Zobrist {
    uint64_t piece[2][6][64]{};
    uint64_t castling[16]{};
    uint64_t ep_file[8]{};
    uint64_t side_to_move{};

    Zobrist() {
        uint64_t seed = 0x98f107b53c059a1fULL;
        auto next_rand = [&seed]() -> uint64_t {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            return seed * 0x2545F4914F6CDD1DULL;
        };

        for (int c = 0; c < 2; ++c) {
            for (int p = 0; p < 6; ++p) {
                for (int s = 0; s < 64; ++s) {
                    piece[c][p][s] = next_rand();
                }
            }
        }
        for (int i = 0; i < 16; ++i) {
            castling[i] = next_rand();
        }
        for (int i = 0; i < 8; ++i) {
            ep_file[i] = next_rand();
        }
        side_to_move = next_rand();
    }
};

inline const Zobrist& get_zobrist() {
    static Zobrist z;
    return z;
}

class Board {
public:
    std::array<uint64_t, 2> colors{}; // 0 = White, 1 = Black
    std::array<uint64_t, 6> pieces{}; // PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    Color turn = WHITE;
    uint8_t castling_rights = 0; // CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ
    int8_t ep_square = -1;       // 0..63 or -1
    uint16_t halfmove_clock = 0;
    uint16_t fullmove_number = 1;
    uint64_t zobrist_hash = 0;
    std::vector<uint64_t> history{}; // Zobrist history for repetition checking

    Board() {
        from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    explicit Board(const std::string& fen) {
        from_fen(fen);
    }

    uint64_t occupied() const {
        return colors[WHITE] | colors[BLACK];
    }

    int piece_count() const {
        return __builtin_popcountll(colors[WHITE] | colors[BLACK]);
    }

    PieceType piece_at(uint8_t sq, Color& color) const {
        uint64_t mask = 1ULL << sq;
        if (colors[WHITE] & mask) {
            color = WHITE;
        } else if (colors[BLACK] & mask) {
            color = BLACK;
        } else {
            color = NO_COLOR;
            return NONE;
        }

        for (int p = 0; p < 6; ++p) {
            if (pieces[p] & mask) {
                return static_cast<PieceType>(p);
            }
        }
        return NONE;
    }

    void clear() {
        colors.fill(0);
        pieces.fill(0);
        turn = WHITE;
        castling_rights = 0;
        ep_square = -1;
        halfmove_clock = 0;
        fullmove_number = 1;
        zobrist_hash = 0;
        history.clear();
    }

    void compute_zobrist() {
        const auto& z = get_zobrist();
        uint64_t h = 0;
        for (int c = 0; c < 2; ++c) {
            for (int p = 0; p < 6; ++p) {
                uint64_t b = colors[c] & pieces[p];
                while (b) {
                    int sq = __builtin_ctzll(b);
                    h ^= z.piece[c][p][sq];
                    b &= b - 1;
                }
            }
        }
        h ^= z.castling[castling_rights];
        if (ep_square >= 0) {
            h ^= z.ep_file[ep_square % 8];
        }
        if (turn == BLACK) {
            h ^= z.side_to_move;
        }
        zobrist_hash = h;
    }

    bool from_fen(const std::string& fen) {
        clear();
        std::istringstream ss(fen);
        std::string placement, active, castling, ep;
        int halfmove = 0, fullmove = 1;

        if (!(ss >> placement >> active >> castling >> ep)) {
            return false;
        }
        ss >> halfmove >> fullmove;

        int row = 7, col = 0;
        for (char c : placement) {
            if (c == '/') {
                row--;
                col = 0;
            } else if (c >= '1' && c <= '8') {
                col += (c - '0');
            } else {
                int sq = row * 8 + col;
                Color colr = (c >= 'a' && c <= 'z') ? BLACK : WHITE;
                char uc = std::toupper(c);
                PieceType pt = NONE;
                if (uc == 'P') pt = PAWN;
                else if (uc == 'N') pt = KNIGHT;
                else if (uc == 'B') pt = BISHOP;
                else if (uc == 'R') pt = ROOK;
                else if (uc == 'Q') pt = QUEEN;
                else if (uc == 'K') pt = KING;

                if (pt != NONE && sq >= 0 && sq < 64) {
                    colors[colr] |= (1ULL << sq);
                    pieces[pt] |= (1ULL << sq);
                }
                col++;
            }
        }

        turn = (active == "b" || active == "B") ? BLACK : WHITE;

        castling_rights = 0;
        for (char c : castling) {
            if (c == 'K') castling_rights |= CASTLE_WK;
            else if (c == 'Q') castling_rights |= CASTLE_WQ;
            else if (c == 'k') castling_rights |= CASTLE_BK;
            else if (c == 'q') castling_rights |= CASTLE_BQ;
        }

        if (ep != "-") {
            int ec = ep[0] - 'a';
            int er = ep[1] - '1';
            ep_square = er * 8 + ec;
        } else {
            ep_square = -1;
        }

        halfmove_clock = halfmove;
        fullmove_number = fullmove;
        compute_zobrist();
        history.push_back(zobrist_hash);
        return true;
    }

    std::string to_fen() const {
        std::ostringstream ss;
        for (int r = 7; r >= 0; --r) {
            int empty = 0;
            for (int c = 0; c < 8; ++c) {
                int sq = r * 8 + c;
                Color col;
                PieceType pt = piece_at(sq, col);
                if (pt == NONE) {
                    empty++;
                } else {
                    if (empty > 0) {
                        ss << empty;
                        empty = 0;
                    }
                    static const char pchars[2][6] = {
                        {'P', 'N', 'B', 'R', 'Q', 'K'},
                        {'p', 'n', 'b', 'r', 'q', 'k'}
                    };
                    ss << pchars[col][pt];
                }
            }
            if (empty > 0) ss << empty;
            if (r > 0) ss << '/';
        }

        ss << (turn == WHITE ? " w " : " b ");

        std::string cstr;
        if (castling_rights & CASTLE_WK) cstr += 'K';
        if (castling_rights & CASTLE_WQ) cstr += 'Q';
        if (castling_rights & CASTLE_BK) cstr += 'k';
        if (castling_rights & CASTLE_BQ) cstr += 'q';
        if (cstr.empty()) cstr = "-";
        ss << cstr << ' ';

        if (ep_square >= 0) {
            ss << (char)('a' + (ep_square % 8));
            ss << (char)('1' + (ep_square / 8));
        } else {
            ss << '-';
        }

        ss << ' ' << halfmove_clock << ' ' << fullmove_number;
        return ss.str();
    }

    bool is_attacked(int sq, Color attacker) const {
        const auto& tables = get_attack_tables();
        uint64_t occ = occupied();

        Color defender = (attacker == WHITE) ? BLACK : WHITE;
        if (tables.pawn_attacks[defender][sq] & (colors[attacker] & pieces[PAWN])) return true;

        if (tables.knight_attacks[sq] & (colors[attacker] & pieces[KNIGHT])) return true;

        if (tables.king_attacks[sq] & (colors[attacker] & pieces[KING])) return true;

        uint64_t bq = colors[attacker] & (pieces[BISHOP] | pieces[QUEEN]);
        if (bq && (get_bishop_attacks(sq, occ) & bq)) return true;

        uint64_t rq = colors[attacker] & (pieces[ROOK] | pieces[QUEEN]);
        if (rq && (get_rook_attacks(sq, occ) & rq)) return true;

        return false;
    }

    bool is_in_check() const {
        Color opp = (turn == WHITE) ? BLACK : WHITE;
        uint64_t k = colors[turn] & pieces[KING];
        if (!k) return false;
        int ksq = __builtin_ctzll(k);
        return is_attacked(ksq, opp);
    }

    void generate_legal_moves(std::vector<Move>& moves) const {
        moves.clear();
        moves.reserve(64);

        const auto& tables = get_attack_tables();
        Color us = turn;
        Color them = (us == WHITE) ? BLACK : WHITE;
        uint64_t occ = occupied();
        uint64_t our_pieces = colors[us];
        uint64_t their_pieces = colors[them];

        auto add_pawn_move = [&](uint8_t f, uint8_t t) {
            int to_r = t / 8;
            if ((us == WHITE && to_r == 7) || (us == BLACK && to_r == 0)) {
                moves.emplace_back(f, t, PROMO_QUEEN);
                moves.emplace_back(f, t, PROMO_ROOK);
                moves.emplace_back(f, t, PROMO_BISHOP);
                moves.emplace_back(f, t, PROMO_KNIGHT);
            } else {
                moves.emplace_back(f, t, PROMO_NONE);
            }
        };

        // 1. Pawns
        uint64_t pawns = our_pieces & pieces[PAWN];
        while (pawns) {
            int f = __builtin_ctzll(pawns);
            pawns &= pawns - 1;
            int r = f / 8;

            if (us == WHITE) {
                int t1 = f + 8;
                if (t1 < 64 && !(occ & (1ULL << t1))) {
                    add_pawn_move(f, t1);
                    int t2 = f + 16;
                    if (r == 1 && !(occ & (1ULL << t2))) {
                        moves.emplace_back(f, t2, PROMO_NONE);
                    }
                }
                uint64_t att = tables.pawn_attacks[WHITE][f] & their_pieces;
                while (att) {
                    int t = __builtin_ctzll(att);
                    att &= att - 1;
                    add_pawn_move(f, t);
                }
                if (ep_square >= 0 && (tables.pawn_attacks[WHITE][f] & (1ULL << ep_square))) {
                    moves.emplace_back(f, ep_square, PROMO_NONE);
                }
            } else {
                int t1 = f - 8;
                if (t1 >= 0 && !(occ & (1ULL << t1))) {
                    add_pawn_move(f, t1);
                    int t2 = f - 16;
                    if (r == 6 && !(occ & (1ULL << t2))) {
                        moves.emplace_back(f, t2, PROMO_NONE);
                    }
                }
                uint64_t att = tables.pawn_attacks[BLACK][f] & their_pieces;
                while (att) {
                    int t = __builtin_ctzll(att);
                    att &= att - 1;
                    add_pawn_move(f, t);
                }
                if (ep_square >= 0 && (tables.pawn_attacks[BLACK][f] & (1ULL << ep_square))) {
                    moves.emplace_back(f, ep_square, PROMO_NONE);
                }
            }
        }

        // 2. Knights
        uint64_t knights = our_pieces & pieces[KNIGHT];
        while (knights) {
            int f = __builtin_ctzll(knights);
            knights &= knights - 1;
            uint64_t att = tables.knight_attacks[f] & ~our_pieces;
            while (att) {
                int t = __builtin_ctzll(att);
                att &= att - 1;
                moves.emplace_back(f, t, PROMO_NONE);
            }
        }

        // 3. Bishops
        uint64_t bishops = our_pieces & pieces[BISHOP];
        while (bishops) {
            int f = __builtin_ctzll(bishops);
            bishops &= bishops - 1;
            uint64_t att = get_bishop_attacks(f, occ) & ~our_pieces;
            while (att) {
                int t = __builtin_ctzll(att);
                att &= att - 1;
                moves.emplace_back(f, t, PROMO_NONE);
            }
        }

        // 4. Rooks
        uint64_t rooks = our_pieces & pieces[ROOK];
        while (rooks) {
            int f = __builtin_ctzll(rooks);
            rooks &= rooks - 1;
            uint64_t att = get_rook_attacks(f, occ) & ~our_pieces;
            while (att) {
                int t = __builtin_ctzll(att);
                att &= att - 1;
                moves.emplace_back(f, t, PROMO_NONE);
            }
        }

        // 5. Queens
        uint64_t queens = our_pieces & pieces[QUEEN];
        while (queens) {
            int f = __builtin_ctzll(queens);
            queens &= queens - 1;
            uint64_t att = get_queen_attacks(f, occ) & ~our_pieces;
            while (att) {
                int t = __builtin_ctzll(att);
                att &= att - 1;
                moves.emplace_back(f, t, PROMO_NONE);
            }
        }

        // 6. King
        uint64_t king = our_pieces & pieces[KING];
        if (king) {
            int f = __builtin_ctzll(king);
            uint64_t att = tables.king_attacks[f] & ~our_pieces;
            while (att) {
                int t = __builtin_ctzll(att);
                att &= att - 1;
                moves.emplace_back(f, t, PROMO_NONE);
            }

            // Castling
            if (us == WHITE && f == 4 && !is_attacked(4, BLACK)) {
                // White kingside: e1 (4) -> g1 (6), f1 (5) and g1 (6) empty and safe
                // Also verify rook at h1
                if ((castling_rights & CASTLE_WK) && (colors[WHITE] & pieces[ROOK] & (1ULL << 7)) &&
                    !(occ & ((1ULL << 5) | (1ULL << 6))) &&
                    !is_attacked(5, BLACK) && !is_attacked(6, BLACK)) {
                    moves.emplace_back(4, 6, PROMO_NONE);
                }
                // White queenside: e1 (4) -> c1 (2), b1 (1), c1 (2), d1 (3) empty, d1 and c1 safe
                // Also verify rook at a1
                if ((castling_rights & CASTLE_WQ) && (colors[WHITE] & pieces[ROOK] & (1ULL << 0)) &&
                    !(occ & ((1ULL << 1) | (1ULL << 2) | (1ULL << 3))) &&
                    !is_attacked(3, BLACK) && !is_attacked(2, BLACK)) {
                    moves.emplace_back(4, 2, PROMO_NONE);
                }
            } else if (us == BLACK && f == 60 && !is_attacked(60, WHITE)) {
                // Black kingside: e8 (60) -> g8 (62), f8 (61) and g8 (62) empty and safe
                // Also verify rook at h8
                if ((castling_rights & CASTLE_BK) && (colors[BLACK] & pieces[ROOK] & (1ULL << 63)) &&
                    !(occ & ((1ULL << 61) | (1ULL << 62))) &&
                    !is_attacked(61, WHITE) && !is_attacked(62, WHITE)) {
                    moves.emplace_back(60, 62, PROMO_NONE);
                }
                // Black queenside: e8 (60) -> c8 (58), b8 (57), c8 (58), d8 (59) empty, d8 and c8 safe
                // Also verify rook at a8
                if ((castling_rights & CASTLE_BQ) && (colors[BLACK] & pieces[ROOK] & (1ULL << 56)) &&
                    !(occ & ((1ULL << 57) | (1ULL << 58) | (1ULL << 59))) &&
                    !is_attacked(59, WHITE) && !is_attacked(58, WHITE)) {
                    moves.emplace_back(60, 58, PROMO_NONE);
                }
            }
        }

        // Filter for king safety (legality)
        size_t write_idx = 0;
        for (size_t i = 0; i < moves.size(); ++i) {
            Board copy = *this;
            copy.make_move(moves[i]);
            // King must not be attacked by opposing color in copy
            uint64_t k = copy.colors[us] & copy.pieces[KING];
            if (k) {
                int ksq = __builtin_ctzll(k);
                if (!copy.is_attacked(ksq, them)) {
                    moves[write_idx++] = moves[i];
                }
            }
        }
        moves.resize(write_idx);
    }

    void make_move(const Move& m) {
        Color us = turn;
        Color them = (us == WHITE) ? BLACK : WHITE;

        Color dummy_col;
        PieceType moving_piece = piece_at(m.from_sq, dummy_col);
        PieceType captured_piece = piece_at(m.to_sq, dummy_col);

        // Remove moving piece from source
        colors[us] ^= (1ULL << m.from_sq);
        pieces[moving_piece] ^= (1ULL << m.from_sq);

        bool is_capture = (captured_piece != NONE);
        if (is_capture) {
            colors[them] ^= (1ULL << m.to_sq);
            pieces[captured_piece] ^= (1ULL << m.to_sq);
        }

        // En passant capture
        if (moving_piece == PAWN && m.to_sq == ep_square) {
            int cap_sq = (us == WHITE) ? (ep_square - 8) : (ep_square + 8);
            colors[them] ^= (1ULL << cap_sq);
            pieces[PAWN] ^= (1ULL << cap_sq);
            is_capture = true;
        }

        // Castling move: move the corresponding rook
        if (moving_piece == KING) {
            if (m.from_sq == 4 && m.to_sq == 6) { // WK
                colors[WHITE] ^= (1ULL << 7) | (1ULL << 5);
                pieces[ROOK] ^= (1ULL << 7) | (1ULL << 5);
            } else if (m.from_sq == 4 && m.to_sq == 2) { // WQ
                colors[WHITE] ^= (1ULL << 0) | (1ULL << 3);
                pieces[ROOK] ^= (1ULL << 0) | (1ULL << 3);
            } else if (m.from_sq == 60 && m.to_sq == 62) { // BK
                colors[BLACK] ^= (1ULL << 63) | (1ULL << 61);
                pieces[ROOK] ^= (1ULL << 63) | (1ULL << 61);
            } else if (m.from_sq == 60 && m.to_sq == 58) { // BQ
                colors[BLACK] ^= (1ULL << 56) | (1ULL << 59);
                pieces[ROOK] ^= (1ULL << 56) | (1ULL << 59);
            }
        }

        // Place piece on destination
        PieceType placed_piece = moving_piece;
        if (m.promo != PROMO_NONE) {
            if (m.promo == PROMO_QUEEN) placed_piece = QUEEN;
            else if (m.promo == PROMO_ROOK) placed_piece = ROOK;
            else if (m.promo == PROMO_BISHOP) placed_piece = BISHOP;
            else if (m.promo == PROMO_KNIGHT) placed_piece = KNIGHT;
        }
        colors[us] |= (1ULL << m.to_sq);
        pieces[placed_piece] |= (1ULL << m.to_sq);

        // Update castling rights if King or Rook moved or Rook captured
        if (moving_piece == KING) {
            if (us == WHITE) castling_rights &= ~(CASTLE_WK | CASTLE_WQ);
            else castling_rights &= ~(CASTLE_BK | CASTLE_BQ);
        } else if (moving_piece == ROOK) {
            if (m.from_sq == 0) castling_rights &= ~CASTLE_WQ;
            else if (m.from_sq == 7) castling_rights &= ~CASTLE_WK;
            else if (m.from_sq == 56) castling_rights &= ~CASTLE_BQ;
            else if (m.from_sq == 63) castling_rights &= ~CASTLE_BK;
        }
        if (captured_piece == ROOK) {
            if (m.to_sq == 0) castling_rights &= ~CASTLE_WQ;
            else if (m.to_sq == 7) castling_rights &= ~CASTLE_WK;
            else if (m.to_sq == 56) castling_rights &= ~CASTLE_BQ;
            else if (m.to_sq == 63) castling_rights &= ~CASTLE_BK;
        }

        // Update en passant square
        if (moving_piece == PAWN && std::abs((int)m.to_sq - (int)m.from_sq) == 16) {
            ep_square = (m.from_sq + m.to_sq) / 2;
        } else {
            ep_square = -1;
        }

        // Update halfmove clock
        if (moving_piece == PAWN || is_capture) {
            halfmove_clock = 0;
        } else {
            halfmove_clock++;
        }

        if (us == BLACK) {
            fullmove_number++;
        }

        turn = them;
        compute_zobrist();
        history.push_back(zobrist_hash);
    }

    int repetition_count() const {
        int count = 0;
        for (auto h : history) {
            if (h == zobrist_hash) count++;
        }
        return count > 0 ? (count - 1) : 0;
    }

    bool is_repetition(int n = 3) const {
        int count = 0;
        for (auto h : history) {
            if (h == zobrist_hash) {
                count++;
                if (count >= n) return true;
            }
        }
        return false;
    }

    bool is_fifty_moves() const {
        return halfmove_clock >= 100;
    }

    // Terminal check:
    // Returns: -1 if side to move checkmated, 0 for draw, 2 for not terminal
    int check_terminal(bool claim_draw = false) const {
        if (claim_draw && (is_repetition(3) || is_fifty_moves())) {
            return 0; // Draw
        }
        if (halfmove_clock >= 150) { // 75-move rule
            return 0;
        }
        if (is_repetition(5)) { // 5-fold repetition
            return 0;
        }

        // Check if insufficient material: K vs K, K+B vs K, K+N vs K
        uint64_t non_kings = occupied() & ~pieces[KING];
        if (!non_kings) return 0; // K vs K
        if (__builtin_popcountll(non_kings) == 1) {
            if (pieces[BISHOP] || pieces[KNIGHT]) return 0;
        }

        std::vector<Move> legals;
        generate_legal_moves(legals);
        if (legals.empty()) {
            if (is_in_check()) {
                return -1; // Current player checkmated (loss)
            } else {
                return 0; // Stalemate (draw)
            }
        }

        return 2; // Not terminal
    }

    // 19-plane tensor encoder matching core/encoding.py:
    // Shape: (19, 8, 8) in row-major float32 buffer [19 * 64]
    // Always oriented from side to move!
    // If Black to move: mirror row (row -> 7 - row), swap colors, mirror castling, mirror ep!
    void encode_planes(float* out_planes) const {
        std::memset(out_planes, 0, 19 * 64 * sizeof(float));

        Color us = turn;
        Color them = (us == WHITE) ? BLACK : WHITE;

        // Plane 0-5: Friendly pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        for (int p = 0; p < 6; ++p) {
            uint64_t b = colors[us] & pieces[p];
            float* plane = out_planes + p * 64;
            while (b) {
                int sq = __builtin_ctzll(b);
                b &= b - 1;
                int r = sq / 8;
                int c = sq % 8;
                if (us == BLACK) r = 7 - r;
                plane[r * 8 + c] = 1.0f;
            }
        }

        // Plane 6-11: Opponent pieces
        for (int p = 0; p < 6; ++p) {
            uint64_t b = colors[them] & pieces[p];
            float* plane = out_planes + (6 + p) * 64;
            while (b) {
                int sq = __builtin_ctzll(b);
                b &= b - 1;
                int r = sq / 8;
                int c = sq % 8;
                if (us == BLACK) r = 7 - r;
                plane[r * 8 + c] = 1.0f;
            }
        }

        // Plane 12: Own kingside castling rights
        // Plane 13: Own queenside castling rights
        // Plane 14: Opponent kingside castling rights
        // Plane 15: Opponent queenside castling rights
        bool own_k = (us == WHITE) ? (castling_rights & CASTLE_WK) : (castling_rights & CASTLE_BK);
        bool own_q = (us == WHITE) ? (castling_rights & CASTLE_WQ) : (castling_rights & CASTLE_BQ);
        bool opp_k = (us == WHITE) ? (castling_rights & CASTLE_BK) : (castling_rights & CASTLE_WK);
        bool opp_q = (us == WHITE) ? (castling_rights & CASTLE_BQ) : (castling_rights & CASTLE_WQ);

        if (own_k) {
            float* p12 = out_planes + 12 * 64;
            for (int i = 0; i < 64; ++i) p12[i] = 1.0f;
        }
        if (own_q) {
            float* p13 = out_planes + 13 * 64;
            for (int i = 0; i < 64; ++i) p13[i] = 1.0f;
        }
        if (opp_k) {
            float* p14 = out_planes + 14 * 64;
            for (int i = 0; i < 64; ++i) p14[i] = 1.0f;
        }
        if (opp_q) {
            float* p15 = out_planes + 15 * 64;
            for (int i = 0; i < 64; ++i) p15[i] = 1.0f;
        }

        // Plane 16: En passant target square (single 1.0)
        if (ep_square >= 0) {
            int r = ep_square / 8;
            int c = ep_square % 8;
            if (us == BLACK) r = 7 - r;
            out_planes[16 * 64 + r * 8 + c] = 1.0f;
        }

        // Plane 17: Halfmove clock / 100
        float hmc = std::min((float)halfmove_clock, 100.0f) / 100.0f;
        float* p17 = out_planes + 17 * 64;
        for (int i = 0; i < 64; ++i) p17[i] = hmc;

        // Plane 18: Repetition count / 2
        int rep = repetition_count();
        float rep_val = std::min((float)rep, 2.0f) / 2.0f;
        float* p18 = out_planes + 18 * 64;
        for (int i = 0; i < 64; ++i) p18[i] = rep_val;
    }

    // Orient a move to perspective of side to move
    Move orient_move(const Move& m) const {
        if (turn == WHITE) return m;
        return Move(square_mirror(m.from_sq), square_mirror(m.to_sq), m.promo);
    }
};

} // namespace chess
