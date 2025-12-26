use acyclib::trainer::optimiser::adam::AdamW;
use bullet_cuda_backend::CudaDevice;
use bullet_lib::LocalSettings;
use bullet_lib::TrainingSchedule;
use bullet_lib::TrainingSteps;
use bullet_lib::game::formats::bulletformat::ChessBoard;
use bullet_lib::game::inputs::ChessBucketsMirrored;
use bullet_lib::game::outputs::MaterialCount;
use bullet_lib::lr;
use bullet_lib::lr::LrScheduler;
use bullet_lib::nn::InitSettings;
use bullet_lib::nn::Shape;
use bullet_lib::nn::optimiser;
use bullet_lib::trainer::save::SavedFormat;
use bullet_lib::value::ValueTrainer;
use bullet_lib::value::ValueTrainerBuilder;
use bullet_lib::value::loader::ViriBinpackLoader;
use bullet_lib::value::loader::viribinpack::ViriFilter;
use bullet_lib::wdl;
use bullet_lib::wdl::WdlScheduler;
use rand::{rng};
use viriformat::chess::board::Board;
use viriformat::chess::chessmove::Move;
use viriformat::dataformat::Filter;
use viriformat::dataformat::WDL;

#[derive(Clone)]
struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

const TRAINING_DIR: &str = "/mnt/d/Chess Data/Selfgen/Training";

fn make_trainer() -> ValueTrainer<AdamW<CudaDevice>, ChessBucketsMirrored, MaterialCount<8>> {
    #[rustfmt::skip]
    let inputs = ChessBucketsMirrored::new([
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
    const L1_SIZE: usize = 128;
    const L2_SIZE: usize = 16;
    const L3_SIZE: usize = 32;

    #[rustfmt::skip]
    return ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
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
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(L1_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(KING_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * KING_BUCKETS, L1_SIZE);
            l0.weights = l0.weights + expanded_factoriser;
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
            l3.forward(l2_out.concat(l1_out)).select(buckets)
        });
}

fn make_settings(experiment_name: &str) -> LocalSettings<'_> {
    LocalSettings {
        threads: 4,
        output_directory: format!("{}/{}", TRAINING_DIR, experiment_name),
        test_set: None,
        batch_queue_size: 512,
    }
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let default_viri_filter = Filter {
        min_ply: 8,
        min_pieces: 0,
        max_eval: 32000,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.05,
        wdl_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.5,
    };
    let mut rng = rng();
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    fn should_filter_custom(board: &ChessBoard) -> bool {
        const PIECE_VALUES: [i32; 7] = [208, 781, 825, 1276, 2538, 0, 0];
        let mut side_evaluations = [0, 0];

        let mut bbs = [0; 8];
        for (pc, sq) in board.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
            side_evaluations[c] += PIECE_VALUES[pt - 2];
        }
        let material_diff = side_evaluations[0].abs_diff(side_evaluations[1]);
        material_diff <= 931 // filter out positions with low material imbalance
    }

    !default_viri_filter.should_filter(mv, eval as i32, board, wdl, &mut rng) && !should_filter_custom(&board.to_bulletformat(wdl as u8, eval).unwrap())
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
            trainer
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
            trainer.load_from_checkpoint(
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
            start_superbatch,
            end_superbatch: net.superbatch,
        },
        wdl_scheduler,
        lr_scheduler,
        save_rate: 100,
    };

    trainer.optimiser.set_params(optimiser::AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    });

    let settings = make_settings(net.name);
    let data_loader = ViriBinpackLoader::new(
        data_path,
        8192,
        20,
        ViriFilter::Custom(filter),
    );
    trainer.run(&schedule, &settings, &data_loader);
}

fn main() {
    // Step 1
    train(
        "/mnt/e/Chess/Data/combined.vf",
        wdl::LinearWDL { start: 0.15, end: 0.3 },
        lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3 * 0.3 * 0.3, final_superbatch: 200 },
        NetConfig { name: "0152", superbatch: 200 },
        None,
    );
}
