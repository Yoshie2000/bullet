use bullet_core::optimiser::adam::AdamW;
use bullet_lib::{
    default::{
        inputs,
        loader::{DataLoader, DirectSequentialDataLoader},
        outputs::{self, MaterialCount},
        Loss, Trainer, TrainerBuilder,
    },
    game::inputs::ChessBucketsMirroredFactorised,
    lr::{self, LrScheduler},
    nn::optimiser,
    trainer::NetworkTrainer,
    value::{loader::ViriBinpackLoader, ValueTrainerBuilder},
    wdl::{self, WdlScheduler},
    Activation, ExecutionContext, LocalSettings, TrainingSchedule, TrainingSteps,
};
use bulletformat::ChessBoard;
use viriformat::dataformat::Filter;

struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

fn make_trainer() -> Trainer<AdamW<ExecutionContext>, ChessBucketsMirroredFactorised, NoOutputBuckets> {
    #[rustfmt::skip]
    return ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .inputs(inputs::ChessBucketsMirroredFactorised::new([
            00, 01, 02, 03,
            04, 05, 06, 07,
            08, 08, 09, 09,
            10, 10, 10, 10,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
        ]))
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
        .build(|builder, stm, ntm| {
            const KING_BUCKETS: usize = 12;
            const OUTPUT_BUCKETS: usize = 1;
            const L1_SIZE: usize = 1792;
            const L2_SIZE: usize = 16;
            const L3_SIZE: usize = 32;

            let mut l0 = builder.new_affine("l0", 768 * KING_BUCKETS, L1_SIZE);

            let l0f = builder.new_weights("l0f", Shape::new(768 * L1_SIZE, 1), InitSettings::Zeroed);
            let ones = builder.new_constant(Shape::new(1, KING_BUCKETS), &[1.0; KING_BUCKETS]);
            let expanded = l0f.matmul(ones).reshape(Shape::new(L1_SIZE, inputs.num_inputs()));
            l0.weights = l0.weights + expanded;

            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, OUTPUT_BUCKETS * L1_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS);

            // Crelu + Pairwise
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let out = stm_subnet.concat(ntm_subnet);
            // Dual activation
            let out = l1.forward(out); // TODO output buckets
            let out = out.concat(out.abs_pow(2.0));
            let out = out.crelu();
            // L2 + L3 forward
            let out = l2.forward(out).screlu(); // TODO output buckets
            let out = l3.forward(out); // TODO output buckets

            out
        });
    );
}

fn make_settings(experiment_name: &str) -> LocalSettings {
    return LocalSettings {
        threads: 4,
        output_directory: format!("/mnt/d/Chess Data/Selfgen/Training/{}", experiment_name),
        test_set: None,
        batch_queue_size: 512,
    };
}

fn train<WDL: WdlScheduler, LR: LrScheduler, DL: DataLoader<ChessBoard>>(
    data_loader: DL,
    wdl_scheduler: WDL,
    lr_scheduler: LR,
    net: NetConfig,
    load_from: Option<NetConfig>,
) {
    let mut trainer = make_trainer();

    if let Some(load_from_net) = load_from {
        let _ = trainer.optimiser_mut().load_weights_from_file(
            format!(
                "/mnt/d/Chess Data/Selfgen/Training/{}/net-{}-{}/optimiser_state/weights.bin",
                load_from_net.name, load_from_net.name, load_from_net.superbatch
            )
            .as_str(),
        );
    }

    let schedule = TrainingSchedule {
        net_id: format!("net-{}", net.name).to_string(),
        eval_scale: 450.0f,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: net.superbatch,
        },
        wdl_scheduler: wdl_scheduler,
        lr_scheduler: lr_scheduler,
        save_rate: 1,
    };

    trainer.set_optimiser_params(optimiser::AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    });

    let settings = make_settings(net.name);
    trainer.run(&schedule, &settings, &data_loader);
}

fn main() {
    // Step 4
    train(
        ViriBinpackLoader::new(
            "/mnt/d/Chess Data/Selfgen/20ksn.viri",
            4096,
            4,
            Filter {
                min_ply: 24,
                min_pieces: 3,
                max_eval: 32000,
                filter_tactical: true,
                filter_check: true,
                filter_castling: false,
                max_eval_incorrectness: u32::MAX,
            },
        ),
        wdl::ConstantWDL { value: 0.6 },
        lr::CosineDecayLR {
            initial_lr: 0.00025 * 0.3,
            final_lr: 0.00025 * 0.3 * 0.3 * 0.3 * 0.3,
            final_superbatch: 200,
        },
        NetConfig { name: "0102rr4", superbatch: 200 },
        Some(NetConfig { name: "0102rr", superbatch: 400 }),
    );
}
