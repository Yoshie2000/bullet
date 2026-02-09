use bullet_core::optimiser::adam::AdamW;
use bullet_lib::{
    default::{
        inputs::{self, SparseInputType},
        loader::DirectSequentialDataLoader,
        outputs::MaterialCount,
    },
    lr::{self, LrScheduler},
    nn::{optimiser, InitSettings},
    trainer::{save::SavedFormat},
    value::{ValueTrainer, ValueTrainerBuilder},
    wdl::{self, WdlScheduler},
    game::inputs::{Chess768, Factorises},
    ExecutionContext, LocalSettings, Shape, TrainingSchedule, TrainingSteps,
};
use bulletformat::ChessBoard;
use rand::Rng;

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub const fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;

    while i < N {
        if arr[i] > max {
            max = arr[i];
        }

        i += 1;
    }
    max + 1
}

#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsThreatenedMirrored {
    buckets: [usize; 64],
    num_buckets: usize,
}

impl Default for ChessBucketsThreatenedMirrored {
    fn default() -> Self {
        Self { buckets: [0; 64], num_buckets: 1 }
    }
}

impl ChessBucketsThreatenedMirrored {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);

        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }

        Self { buckets: expanded, num_buckets }
    }
}

impl SparseInputType for ChessBucketsThreatenedMirrored {
    type RequiredDataType = ChessBoard;

    /// The total number of inputs
    fn num_inputs(&self) -> usize {
        (768 + 128) * self.num_buckets
    }

    /// The maximum number of active inputs
    fn max_active(&self) -> usize {
        32 + 64
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // Standard 768 features
        let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, (768 + 128) * self.buckets[usize::from(ksq)]);
        let (stm_mirrored, stm_bucket) = get(pos.our_ksq());
        let (ntm_mirrored, ntm_bucket) = get(pos.opp_ksq());

        Chess768.map_features(pos, |stm, ntm| f(stm_bucket + (stm ^ stm_mirrored), ntm_bucket + (ntm ^ ntm_mirrored)));

        // 64 one-hot threat features (threats on stm pieces)
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut threat_features = [0 as u64; 2];
        
        let occ = bbs[0] | bbs[1];
        for side in [Side::WHITE, Side::BLACK] {
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
                    } & opps;

                    threat_features[side] |= threats;
                });
            }
        }

        // White == STM
        map_bb(threat_features[Side::WHITE], |sq| {
            f(stm_bucket + 768 + (sq ^ stm_mirrored), ntm_bucket + 768 + 64 + (sq ^ ntm_mirrored));
        });
        // Black == NTM
        map_bb(threat_features[Side::BLACK], |sq| {
            f(stm_bucket + 768 + 64 + (sq ^ stm_mirrored), ntm_bucket + 768 + (sq ^ ntm_mirrored));
        });
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String {
        format!("(768+128)x{}hm", self.num_buckets)
    }

    /// Description of the input type
    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed, threat psqt chess inputs".to_string()
    }
}

impl Factorises<ChessBucketsThreatenedMirrored> for Chess768 {
    fn derive_feature(&self, _: &ChessBucketsThreatenedMirrored, feat: usize) -> Option<usize> {
        Some(feat % (768 + 128))
    }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

const TRAINING_DIR: &str = "/mnt/d/Chess Data/Selfgen/Training";

fn make_trainer() -> ValueTrainer<AdamW<ExecutionContext>, ChessBucketsThreatenedMirrored, MaterialCount<8>> {
    #[rustfmt::skip]
    let inputs = ChessBucketsThreatenedMirrored::new([
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
    const L1_SIZE: usize = 1792;
    const L2_SIZE: usize = 16;
    const L3_SIZE: usize = 32;

    #[rustfmt::skip]
    return ValueTrainerBuilder::default()
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
        .save_format(&[
            SavedFormat::id("l0f"),
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build(|builder, stm, ntm, buckets| {
            // Build fast factoriser
            let mut l0 = builder.new_affine("l0", (768 + 128) * KING_BUCKETS, L1_SIZE);

            let l0f = builder.new_weights("l0f", Shape::new((768 + 128) * L1_SIZE, 1), InitSettings::Zeroed);
            let ones = builder.new_constant(Shape::new(1, KING_BUCKETS), &[1.0; KING_BUCKETS]);
            let expanded = l0f.matmul(ones).reshape(Shape::new(L1_SIZE, inputs.num_inputs()));
            l0.weights = l0.weights + expanded;

            // Build layers
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
}

fn make_settings(experiment_name: &str) -> LocalSettings {
    return LocalSettings {
        threads: 4,
        output_directory: format!("{}/{}", TRAINING_DIR, experiment_name),
        test_set: None,
        batch_queue_size: 512,
    };
}

fn train<WDL: WdlScheduler, LR: LrScheduler>(
    data_path: &str,
    wdl_scheduler: WDL,
    lr_scheduler: LR,
    net: NetConfig,
    load_from: Option<NetConfig>,
) {
    let mut trainer = make_trainer();

    if let Some(load_from_net) = load_from.clone() {
        if load_from_net.name != net.name {
            let _ = trainer
                .optimiser
                .load_weights_from_file(
                    format!(
                        "/mnt/d/Chess Data/Selfgen/Training/{}/net-{}-{}/optimiser_state/weights.bin",
                        load_from_net.name, load_from_net.name, load_from_net.superbatch
                    )
                    .as_str(),
                )
                .unwrap();
        } else {
            let _ = trainer.load_from_checkpoint(
                format!(
                    "/mnt/d/Chess Data/Selfgen/Training/{}/net-{}-{}",
                    load_from_net.name, load_from_net.name, load_from_net.superbatch
                )
                .as_str(),
            );
        }
    }

    let start_superbatch = if load_from.clone().map(|load_from| load_from.name == net.name).unwrap_or(false) {
        load_from.unwrap().superbatch + 1
    } else {
        1
    };

    let schedule = TrainingSchedule {
        net_id: format!("net-{}", net.name).to_string(),
        eval_scale: 450.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: start_superbatch,
            end_superbatch: net.superbatch,
        },
        wdl_scheduler: wdl_scheduler,
        lr_scheduler: lr_scheduler,
        save_rate: 1,
    };

    trainer.optimiser.set_params(optimiser::AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    });

    let settings = make_settings(net.name);
    let data_loader = DirectSequentialDataLoader::new(&[data_path]);
    trainer.run(&schedule, &settings, &data_loader);
}

fn main() {
    // Step 1
    train(
        "/mnt/d/Chess Data/Selfgen/20ksn-plentyChonker.data",
        wdl::ConstantWDL { value: 0.15 },
        lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3 * 0.3 * 0.3, final_superbatch: 300 },
        NetConfig { name: "0115", superbatch: 300 },
        None,
    );

    // Step 2
    train(
        "/mnt/d/Chess Data/Selfgen/5ksn.data",
        wdl::ConstantWDL { value: 0.3 },
        lr::CosineDecayLR { initial_lr: 0.00025, final_lr: 0.00025 * 0.3 * 0.3 * 0.3, final_superbatch: 300 },
        NetConfig { name: "0115r", superbatch: 300 },
        Some(NetConfig { name: "0115", superbatch: 300 }),
    );

    // Step 3
    train(
        "/mnt/d/Chess Data/Selfgen/20ksn.data",
        wdl::ConstantWDL { value: 0.6 },
        lr::CosineDecayLR { initial_lr: 0.00025, final_lr: 0.00025 * 0.3 * 0.3 * 0.3, final_superbatch: 400 },
        NetConfig { name: "0115rr", superbatch: 400 },
        Some(NetConfig { name: "0115r", superbatch: 300 }),
    );

    // Step 4
    train(
        "/mnt/d/Chess Data/Selfgen/20ksn-adversarial-plentychonker.data",
        wdl::ConstantWDL { value: 1.0 },
        lr::CosineDecayLR { initial_lr: 0.000025, final_lr: 0.000025 * 0.3 * 0.3 * 0.3, final_superbatch: 300 },
        NetConfig { name: "0115rrr", superbatch: 300 },
        Some(NetConfig { name: "0115rr", superbatch: 400 }),
    );
}
