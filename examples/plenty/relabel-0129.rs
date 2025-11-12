/// This is much slower than it could be for the sake of simplicity.
/// To improve speed you would need to asynchronously read and write data,
/// and asynchronously copy next batch to GPU whilst current batch is computing.
use std::{
    fs::{self, File as FsFile},
    io::BufWriter,
    path::Path,
    time::Instant,
};

use acyclib::{graph::like::GraphLike, trainer::dataloader::PreparedBatchDevice};
use bullet_lib::{
    game::{outputs::MaterialCount},
    nn::optimiser,
    value::{ValueTrainerBuilder, loader::ViriBinpackLoader},
};
use viriformat::{chess::piece::Colour, dataformat::{Filter, Game}};

use bullet_lib::game::inputs;
use bullet_lib::game::inputs::Chess768;
use bullet_lib::game::formats::bulletformat::ChessBoard;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

macro_rules! init_add_assign {
    (|$sq:ident, $init:expr, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size + 1];
        let mut val = $init;
        while $sq < $size {
            res[$sq] = val;
            val += {$($rest)+};
            $sq += 1;
        }

        res[$size] = val;

        res
    }};
}

pub struct Side;
impl Side {
    pub const WHITE: usize = 0;
    pub const BLACK: usize = 1;
}

pub struct Piece;
impl Piece {
    pub const EMPTY: usize = 0;
    pub const PAWN: usize = 2;
    pub const KNIGHT: usize = 3;
    pub const BISHOP: usize = 4;
    pub const ROOK: usize = 5;
    pub const QUEEN: usize = 6;
    pub const KING: usize = 7;
}

pub mod offsets {
    use super::indices;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = PAWN + 6 * indices::PAWN; // pawn attacks: 6 different pieces: own pawn, own knight, own rook, opp pawn, opp knight, opp rook
    pub const BISHOP: usize = KNIGHT + 12 * indices::KNIGHT[64]; // knight attacks: all 12 pieces possible
    pub const ROOK: usize = BISHOP + 10 * indices::BISHOP[64]; // bishop attacks: queens are skipped
    pub const QUEEN: usize = ROOK + 10 * indices::ROOK[64]; // rook attacks: queens are skipped
    pub const KING: usize = QUEEN + 12 * indices::QUEEN[64]; // queen attacks: all 12 pieces possible
    pub const END: usize = KING + 8 * indices::KING[64]; // king attacks: queen and king skipped
}

pub mod indices {
    use super::attacks;

    pub const PAWN: usize = 84;
    pub const KNIGHT: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KNIGHT[sq].count_ones() as usize); // for every square (including index 64, aka total) stores the start index of threats of that square
    pub const BISHOP: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::BISHOP[sq].count_ones() as usize);
    pub const ROOK: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::ROOK[sq].count_ones() as usize);
    pub const QUEEN: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::QUEEN[sq].count_ones() as usize);
    pub const KING: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KING[sq].count_ones() as usize);
}

pub mod attacks {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = A << 7;

    const DIAGS: [u64; 15] = [
        0x0100_0000_0000_0000,
        0x0201_0000_0000_0000,
        0x0402_0100_0000_0000,
        0x0804_0201_0000_0000,
        0x1008_0402_0100_0000,
        0x2010_0804_0201_0000,
        0x4020_1008_0402_0100,
        0x8040_2010_0804_0201,
        0x0080_4020_1008_0402,
        0x0000_8040_2010_0804,
        0x0000_0080_4020_1008,
        0x0000_0000_8040_2010,
        0x0000_0000_0080_4020,
        0x0000_0000_0000_8040,
        0x0000_0000_0000_0080,
    ];

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);

    pub const KING: [u64; 64] = init!(|sq, 64| {
        let mut k = 1 << sq;
        k |= (k << 8) | (k >> 8);
        k |= ((k & !A) >> 1) | ((k & !H) << 1);
        k ^ (1 << sq)
    });
}

pub struct Attacks;
impl Attacks {
    pub fn of_piece<const PC: usize>(from: usize, occ: u64) -> u64 {
        match PC {
            Piece::KNIGHT => Attacks::knight(from),
            Piece::BISHOP => Attacks::bishop(from, occ),
            Piece::ROOK => Attacks::rook(from, occ),
            Piece::QUEEN => Attacks::queen(from, occ),
            Piece::KING => Attacks::king(from),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn pawn(sq: usize, side: usize) -> u64 {
        LOOKUP.pawn[side][sq]
    }

    #[inline]
    pub fn knight(sq: usize) -> u64 {
        LOOKUP.knight[sq]
    }

    #[inline]
    pub fn king(sq: usize) -> u64 {
        LOOKUP.king[sq]
    }

    // hyperbola quintessence
    // this gets automatically vectorised when targeting avx or better
    #[inline]
    pub fn bishop(sq: usize, occ: u64) -> u64 {
        let mask = LOOKUP.bishop[sq];

        let mut diag = occ & mask.diag;
        let mut rev1 = diag.swap_bytes();
        diag = diag.wrapping_sub(mask.bit);
        rev1 = rev1.wrapping_sub(mask.swap);
        diag ^= rev1.swap_bytes();
        diag &= mask.diag;

        let mut anti = occ & mask.anti;
        let mut rev2 = anti.swap_bytes();
        anti = anti.wrapping_sub(mask.bit);
        rev2 = rev2.wrapping_sub(mask.swap);
        anti ^= rev2.swap_bytes();
        anti &= mask.anti;

        diag | anti
    }

    // shifted lookup
    // files and ranks are mapped to 1st rank and looked up by occupancy
    #[inline]
    pub fn rook(sq: usize, occ: u64) -> u64 {
        let flip = ((occ >> (sq & 7)) & File::A).wrapping_mul(DIAG);
        let file_sq = (flip >> 57) & 0x3F;
        let files = LOOKUP.file[sq][file_sq as usize];

        let rank_sq = (occ >> RANK_SHIFT[sq]) & 0x3F;
        let ranks = LOOKUP.rank[sq][rank_sq as usize];

        ranks | files
    }

    #[inline]
    pub fn queen(sq: usize, occ: u64) -> u64 {
        Self::bishop(sq, occ) | Self::rook(sq, occ)
    }

    #[inline]
    pub fn xray_rook(sq: usize, occ: u64, blockers: u64) -> u64 {
        let attacks = Self::rook(sq, occ);
        attacks ^ Self::rook(sq, occ ^ (attacks & blockers))
    }

    #[inline]
    pub fn xray_bishop(sq: usize, occ: u64, blockers: u64) -> u64 {
        let attacks = Self::bishop(sq, occ);
        attacks ^ Self::bishop(sq, occ ^ (attacks & blockers))
    }

    pub const fn white_pawn_setwise(pawns: u64) -> u64 {
        ((pawns & !File::A) << 7) | ((pawns & !File::H) << 9)
    }

    pub const fn black_pawn_setwise(pawns: u64) -> u64 {
        ((pawns & !File::A) >> 9) | ((pawns & !File::H) >> 7)
    }
}

struct File;
impl File {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = Self::A << 7;
}

const EAST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) ^ (1 << sq) ^ WEST[sq]);
const WEST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) & ((1 << sq) - 1));
const DIAG: u64 = DIAGS[7];
const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

// masks for hyperbola quintessence bishop attacks
#[derive(Clone, Copy)]
struct Mask {
    bit: u64,
    diag: u64,
    anti: u64,
    swap: u64,
}

struct Lookup {
    pawn: [[u64; 64]; 2],
    knight: [u64; 64],
    king: [u64; 64],
    bishop: [Mask; 64],
    rank: [[u64; 64]; 64],
    file: [[u64; 64]; 64],
}

static LOOKUP: Lookup = Lookup { pawn: PAWN, knight: KNIGHT, king: KING, bishop: BISHOP, rank: RANK, file: FILE };

const PAWN: [[u64; 64]; 2] = [
    init!(|sq, 64| (((1 << sq) & !File::A) << 7) | (((1 << sq) & !File::H) << 9)),
    init!(|sq, 64| (((1 << sq) & !File::A) >> 9) | (((1 << sq) & !File::H) >> 7)),
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !File::A) >> 1) | ((k & !File::H) << 1);
    k ^ (1 << sq)
});

const BISHOP: [Mask; 64] = init!(|sq, 64|
    let bit = 1 << sq;
    let file = sq & 7;
    let rank = sq / 8;
    Mask {
        bit,
        diag: bit ^ DIAGS[7 + file - rank],
        anti: bit ^ DIAGS[    file + rank].swap_bytes(),
        swap: bit.swap_bytes()
    }
);

const RANK_SHIFT: [usize; 64] = init!(|sq, 64| sq - (sq & 7) + 1);

const RANK: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| {
    let file = sq & 7;
    let mask = (occ << 1) as u64;
    let east = ((EAST[file] & mask) | (1 << 63)).trailing_zeros() as usize;
    let west = ((WEST[file] & mask) | 1).leading_zeros() as usize ^ 63;
    (EAST[file] ^ EAST[east] | WEST[file] ^ WEST[west]) << (sq - file)
}));

const FILE: [[u64; 64]; 64] =
    init!(|sq, 64| init!(|occ, 64| (RANK[7 - sq / 8][occ].wrapping_mul(DIAG) & File::H) >> (7 - (sq & 7))));

pub const fn line_through(i: usize, j: usize) -> u64 {
    let sq = 1 << j;

    let rank = i / 8;
    let file = i % 8;

    let files = File::A << file;
    if files & sq > 0 {
        return files;
    }

    let ranks = 0xFF << (8 * rank);
    if ranks & sq > 0 {
        return ranks;
    }

    let diags = DIAGS[7 + file - rank];
    if diags & sq > 0 {
        return diags;
    }

    let antis = DIAGS[file + rank].swap_bytes();
    if antis & sq > 0 {
        return antis;
    }

    0
}

const TOTAL_THREATS: usize = 2 * offsets::END;
const TOTAL: usize = TOTAL_THREATS + 768;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static SQRED: AtomicUsize = AtomicUsize::new(0);
static EVALS: AtomicUsize = AtomicUsize::new(0);
static MAX: AtomicUsize = AtomicUsize::new(0);
const TRACK: bool = false;

pub fn print_feature_stats() {
    let count = COUNT.load(Ordering::Relaxed);
    let sqred = SQRED.load(Ordering::Relaxed);
    let evals = EVALS.load(Ordering::Relaxed);
    let max = MAX.load(Ordering::Relaxed);

    let mean = count as f64 / evals as f64;
    let var = sqred as f64 / evals as f64 - mean.powi(2);
    let pct = 1.96 * var.sqrt();

    println!("Total Evals: {evals}");
    println!("Maximum Active Features: {max}");
    println!("Active Features: {mean:.3} +- {pct:.3} (95%)");
}

pub fn map_piece_threat(
    // maps a threat to a feature index (still need to add half width in case of nstm)
    piece: usize,  // piece type
    src: usize,    // square of piece type
    dest: usize,   // square of interaction
    target: usize, // piece (including color) on square of interaction
    enemy: bool,   // indicates whether "target" is an enemy piece (threat or protection)
) -> Option<usize> {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest, target, enemy),
        Piece::KNIGHT => map_knight_threat(src, dest, target),
        Piece::BISHOP => map_bishop_threat(src, dest, target),
        Piece::ROOK => map_rook_threat(src, dest, target),
        Piece::QUEEN => map_queen_threat(src, dest, target),
        Piece::KING => map_king_threat(src, dest, target),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

const fn offset_mapping<const N: usize>(a: [usize; N]) -> [usize; 12] {
    let mut res = [usize::MAX; 12];

    let mut i = 0;
    while i < N {
        res[a[i] - 2] = i; // PieceType - 2 gives the STM index (0..<6)
        res[a[i] + 4] = i + N; // PieceType + 4 gives the OPP index (6..<12)
        i += 1;
    }

    res
}

fn target_is(target: usize, piece: usize) -> bool {
    target % 6 == piece - 2
}

fn map_pawn_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    // target is still a colored piece
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::ROOK]);
    // this MAP call results in the following array:
    // [0, 1, MAX, 2, MAX, MAX, 3, 4, MAX, 5, MAX, MAX]

    // pawn <-> bishop threats are covered by the bishop attacking the pawn, same for queen
    // for pawn <-> pawn threats, we don't cover cases where dest > src to avoid duplicates
    // for pawn <-> pawn threats of the same color, there are never duplicates
    if MAP[target] == usize::MAX || (enemy && dest > src && target_is(target, Piece::PAWN)) {
        None
    } else {
        let up = usize::from(dest > src);
        let diff = dest.abs_diff(src);
        let id = if diff == [9, 7][up] { 0 } else { 1 };
        let attack = 2 * (src % 8) + id - 1;
        let threat = offsets::PAWN + MAP[target] * indices::PAWN + (src / 8 - 1) * 14 + attack;

        assert!(threat < offsets::KNIGHT, "{threat}");

        Some(threat)
    }
}

fn map_knight_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    // don't duplicate knight threats, only allow where dest < src
    if dest > src && target_is(target, Piece::KNIGHT) {
        None
    } else {
        let idx = indices::KNIGHT[src] + below(src, dest, &attacks::KNIGHT);
        let threat = offsets::KNIGHT + target * indices::KNIGHT[64] + idx;

        assert!(threat >= offsets::KNIGHT, "{threat}");
        assert!(threat < offsets::BISHOP, "{threat}");

        Some(threat)
    }
}

fn map_bishop_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::KING]);
    // queen attacks bishop => bishop attacks queen, so ont used here
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::BISHOP) {
        None
    } else {
        let idx = indices::BISHOP[src] + below(src, dest, &attacks::BISHOP);
        let threat = offsets::BISHOP + MAP[target] * indices::BISHOP[64] + idx;

        assert!(threat >= offsets::BISHOP, "{threat}");
        assert!(threat < offsets::ROOK, "{threat}");

        Some(threat)
    }
}

fn map_rook_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::KING]);
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::ROOK) {
        None
    } else {
        let idx = indices::ROOK[src] + below(src, dest, &attacks::ROOK);
        let threat = offsets::ROOK + MAP[target] * indices::ROOK[64] + idx;

        assert!(threat >= offsets::ROOK, "{threat}");
        assert!(threat < offsets::QUEEN, "{threat}");

        Some(threat)
    }
}

fn map_queen_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    if dest > src && target_is(target, Piece::QUEEN) {
        None
    } else {
        let idx = indices::QUEEN[src] + below(src, dest, &attacks::QUEEN);
        let threat = offsets::QUEEN + target * indices::QUEEN[64] + idx;

        assert!(threat >= offsets::QUEEN, "{threat}");
        assert!(threat < offsets::KING, "{threat}");

        Some(threat)
    }
}

fn map_king_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);
    if MAP[target] == usize::MAX {
        None
    } else {
        let idx = indices::KING[src] + below(src, dest, &attacks::KING);
        // piece offset + attacking piece offset + local index (made up of square offset + square-local index)
        let threat = offsets::KING + MAP[target] * indices::KING[64] + idx;

        assert!(threat >= offsets::KING, "{threat}");
        assert!(threat < offsets::END, "{threat}");

        Some(threat)
    }
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

fn map_features<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let mut count = 0;

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                f(TOTAL_THREATS + [0, 384][side] + 64 * (piece - 2) + sq);
                count += 1;
                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(piece, sq, dest, pieces[dest], enemy) {
                        f(side_offset + idx);
                        count += 1;
                    }
                });
            });
        }
    }

    if TRACK {
        COUNT.fetch_add(count, Ordering::Relaxed);
        SQRED.fetch_add(count * count, Ordering::Relaxed);
        let evals = EVALS.fetch_add(1, Ordering::Relaxed);
        MAX.fetch_max(count, Ordering::Relaxed);

        if (evals + 1) % (16384 * 6104) == 0 {
            print_feature_stats();
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TOTAL
    }

    fn max_active(&self) -> usize {
        128 + 32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL}")
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}

fn map_features_bucketed<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let mut count = 0;

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                count += 1;
                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(piece, sq, dest, pieces[dest], enemy) {
                        f(side_offset + idx);
                        count += 1;
                    }
                });
            });
        }
    }

    if TRACK {
        COUNT.fetch_add(count, Ordering::Relaxed);
        SQRED.fetch_add(count * count, Ordering::Relaxed);
        let evals = EVALS.fetch_add(1, Ordering::Relaxed);
        MAX.fetch_max(count, Ordering::Relaxed);

        if (evals + 1) % (16384 * 6104) == 0 {
            print_feature_stats();
        }
    }
}

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val)
    }
    max + 1
}

#[derive(Clone, Copy, Debug)]
pub struct ThreatInputsBucketsMirrored {
    buckets: [usize; 64],
    num_buckets: usize,
}
impl Default for ThreatInputsBucketsMirrored {
    fn default() -> Self {
        Self { buckets: [0; 64], num_buckets: 1 }
    }
}
impl ThreatInputsBucketsMirrored {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);

        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }

        Self { buckets: expanded, num_buckets }
    }
}
impl inputs::SparseInputType for ThreatInputsBucketsMirrored {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 + TOTAL_THREATS + 768 * self.num_buckets
    }

    fn max_active(&self) -> usize {
        128 + 32 // integrated factoriser
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.buckets[usize::from(ksq)]);
        let (stm_flip, stm_bucket) = get(pos.our_ksq());
        let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());
        Chess768.map_features(pos, |stm, ntm| {
            let bucketed_offset = 768 + TOTAL_THREATS;
            f(bucketed_offset + stm_bucket + (stm ^ stm_flip), bucketed_offset + ntm_bucket + (ntm ^ ntm_flip)); // bucketed feature
            f(stm ^ stm_flip, ntm ^ ntm_flip) // factorised feature
        });

        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features_bucketed(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features_bucketed(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(768 + stm, 768 + ntm); // factoriser offset
        }
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL_THREATS}+{}x{}", Chess768.shorthand(), self.num_buckets)
    }

    fn description(&self) -> String {
        "Threat inputs bucketed mirrored factorised".to_string()
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------

fn main() {
    let threads = 4;

    #[rustfmt::skip]
    let inputs = ThreatInputsBucketsMirrored::new([
            00, 01, 02, 03,
            04, 05, 06, 07,
            08, 08, 09, 09,
            10, 10, 10, 10,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
        ]);
    const KING_BUCKETS: usize = 12;
    const OUTPUT_BUCKETS: usize = 8;
    const L1_SIZE: usize = 640;
    const L2_SIZE: usize = 16;
    const L3_SIZE: usize = 32;

    const EVAL_SCALE: f32 = 287.0;

    #[rustfmt::skip]
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .datapoint_weight_function(|_datapoint| {
            let mut rng = rand::rng();
            if rng.random_range(0.0..1.0) < 0.05 {
                0.0
            } else {
                1.0
            }
        })
        .save_format(&[])
        .build(|builder, stm, ntm, buckets| {
            // Build layers
            let l0 = builder.new_affine("l0", 768 + TOTAL_THREATS + 768 * KING_BUCKETS, L1_SIZE);
            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE + 2 * L2_SIZE, OUTPUT_BUCKETS);

            // Crelu + Pairwise
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let pairwise_out = stm_subnet.concat(ntm_subnet);
            // Dual activation
            let l1_out = l1.forward(pairwise_out).select(buckets);
            let l1_out = l1_out.concat(l1_out.abs_pow(2.0));
            let l1_out = l1_out.crelu();
            // L2 + L3 forward
            let l2_out = l2.forward(l1_out).select(buckets).screlu();
            let l3_out = l3.forward(l2_out.concat(l1_out)).select(buckets);

            l3_out
        });
    trainer.load_from_checkpoint("/mnt/d/Chess Data/Selfgen/Training/0129r/net-0129r-400");

    // let correct = (EVAL_SCALE * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) as i16;
    // println!("{correct}");
    // return;

    const VIRIFORMAT_PATH: &str = "/mnt/e/Chess/Data/Viriformat";

    for path in fs::read_dir(VIRIFORMAT_PATH).unwrap() {
        let path = path.unwrap().path();
        let path_str = path.as_os_str().to_str().unwrap();
        let relabel_path_str = path_str.to_owned() + ".rlbd.2";

        if Path::new(relabel_path_str.as_str()).exists() || !path_str.ends_with(".rlbd") {
            println!("Skipping {path_str}");
            continue;
        }

        println!("Relabeling {path_str}");

        let loader = ViriBinpackLoader::new(path_str, 1024, threads, Filter::UNRESTRICTED);

        let state = trainer.state.clone();

        let mut file = BufWriter::new(FsFile::create(relabel_path_str).unwrap());

        let t = Instant::now();

        let mut relabelled = 0usize;
        let mut relabelled_games = 0usize;

        let mut b = 0;
        loader.map_batches_relabel(0, 16384 * 32, |batch, games| {
            assert_eq!(batch.len(), games.iter().map(|g| g.len()).sum());
            if batch.len() == 0 {
                return false;
            }

            // let prepared = state.prepare(batch, threads, 1.0, 1.0);
            // let mut device_data = PreparedBatchDevice::new(trainer.optimiser.graph.devices(), &prepared).unwrap();
            // device_data.load_into_graph(&mut trainer.optimiser.graph).unwrap();
            // trainer.optimiser.graph.execute_fn("forward").unwrap();
            // trainer.optimiser.graph.synchronise().unwrap();
            // let new_evals = trainer.get_output_values();

            // assert_eq!(batch.len(), new_evals.len());

            let mut i = 0;
            for game in games {
                let evals: Vec<f32> = game.moves.iter().map(|(_, score)| score.get() as f32).collect();

                let mut relabeled_game = Game { initial_position: game.initial_position, moves: Vec::new() };
                let mut game_i = 0;
                let initial_white = relabeled_game.initial_position().turn() == Colour::White;
                for m in game.moves() {
                    let stm_factor = if game_i % 2 == 0 { if initial_white { 1 } else { -1 } } else { if initial_white { -1 } else { 1 } };
                    // relabeled_game.add_move(m, (EVAL_SCALE * new_evals[i] * stm_factor as f32) as i16);
                    relabeled_game.add_move(m, (evals[game_i] * stm_factor as f32) as i16);
                    i += 1;
                    game_i += 1;
                }
                relabeled_game.serialise_into(&mut file).unwrap();

                // relabeled_game.visit_positions(|viriboard, score| {
                //     let bf = viriboard.to_bulletformat(0, 0).unwrap();

                //     let prepared = state.prepare(&[bf], 1, 1.0, 1.0);
                //     let mut device_data = PreparedBatchDevice::new(trainer.optimiser.graph.devices(), &prepared).unwrap();
                //     device_data.load_into_graph(&mut trainer.optimiser.graph).unwrap();
                //     trainer.optimiser.graph.execute_fn("forward").unwrap();
                //     trainer.optimiser.graph.synchronise().unwrap();
                //     let correct = (EVAL_SCALE * trainer.get_output_values()[0]) as i16;
                //     let abs_diff = correct.abs_diff(score as i16);

                //     assert!(abs_diff < 10);
                // });
            }
            assert_eq!(i, batch.len());

            relabelled += batch.len();
            relabelled_games += games.len();

            if b % 16 == 0 {
                let pos_per_sec = relabelled as f64 / t.elapsed().as_secs_f64();
                println!("Relabelled {relabelled} pos (~{pos_per_sec:.0} pos/sec)");
            }
            b += 1;

            false
        });

        println!("\nEnd of relabeling. Total positions: {relabelled}, Total games: {relabelled_games}");
    }
}
