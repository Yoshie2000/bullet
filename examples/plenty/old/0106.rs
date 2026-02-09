use bullet_core::optimiser::adam::AdamW;
use bullet_lib::{
    default::{
        inputs::{self, SparseInputType},
        loader::{DataLoader, DirectSequentialDataLoader},
        outputs::{self, MaterialCount},
        Loss, Trainer, TrainerBuilder,
    },
    game::inputs::ChessBucketsMirrored,
    lr::{self, LrScheduler},
    nn::{optimiser, InitSettings},
    trainer::{save::SavedFormat, NetworkTrainer},
    value::{loader::ViriBinpackLoader, NoOutputBuckets, ValueTrainerBuilder},
    wdl::{self, WdlScheduler},
    Activation, ExecutionContext, LocalSettings, Shape, TrainingSchedule, TrainingSteps,
};
use bulletformat::ChessBoard;
use viriformat::dataformat::Filter;

struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

const TRAINING_DIR: &str = "/mnt/d/Chess Data/Selfgen/Training";

fn make_trainer() -> Trainer<AdamW<ExecutionContext>, ChessBucketsMirrored, MaterialCount<8>> {
    #[rustfmt::skip]
    let inputs = inputs::ChessBucketsMirrored::new([
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        8, 8, 8, 8,
        8, 8, 8, 8,
    ]);
    const KING_BUCKETS: usize = 9;
    const OUTPUT_BUCKETS: usize = 8;
    const L1_SIZE: usize = 4096;
    const L2_SIZE: usize = 96;
    const L3_SIZE: usize = 192;
    const L4_SIZE: usize = 192;

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
            SavedFormat::id("l4w"),
            SavedFormat::id("l4b"),
        ])
        .build(|builder, stm, ntm, buckets| {
            // Build fast factoriser
            let mut l0 = builder.new_affine("l0", 768 * KING_BUCKETS, L1_SIZE);

            let l0f = builder.new_weights("l0f", Shape::new(768 * L1_SIZE, 1), InitSettings::Zeroed);
            let ones = builder.new_constant(Shape::new(1, KING_BUCKETS), &[1.0; KING_BUCKETS]);
            let expanded = l0f.matmul(ones).reshape(Shape::new(L1_SIZE, inputs.num_inputs()));
            l0.weights = l0.weights + expanded;

            // Build layers
            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS * L4_SIZE);
            let l4 = builder.new_affine("l4", L4_SIZE, OUTPUT_BUCKETS);

            // Crelu + Pairwise
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let out = stm_subnet.concat(ntm_subnet);
            // Dual activation
            let out = l1.forward(out).select(buckets);
            let out = out.concat(out.abs_pow(2.0));
            let out = out.crelu();
            // L2 + L3 + L4 forward
            let out = l2.forward(out).select(buckets).screlu();
            let out = l3.forward(out).select(buckets).screlu();
            let out = l4.forward(out).select(buckets);

            out
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
                "{}/{}/net-{}-{}/optimiser_state/weights.bin",
                TRAINING_DIR, load_from_net.name, load_from_net.name, load_from_net.superbatch
            )
            .as_str(),
        );
    }

    let schedule = TrainingSchedule {
        net_id: format!("net-{}", net.name).to_string(),
        eval_scale: 450.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: net.superbatch,
        },
        wdl_scheduler: wdl_scheduler,
        lr_scheduler: lr_scheduler,
        save_rate: 50,
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
    // Step 1
    train(
        DirectSequentialDataLoader::new(&["/mnt/d/Chess Data/Selfgen/5ksn.data"]),
        wdl::ConstantWDL { value: 0.3 },
        lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3 * 0.3 * 0.3, final_superbatch: 500 },
        NetConfig { name: "0106", superbatch: 500 },
        None,
    );

    // Step 2
    train(
        DirectSequentialDataLoader::new(&["/mnt/d/Chess Data/Selfgen/20ksn.data"]),
        wdl::ConstantWDL { value: 0.6 },
        lr::CosineDecayLR { initial_lr: 0.00025, final_lr: 0.00025 * 0.3 * 0.3 * 0.3, final_superbatch: 500 },
        NetConfig { name: "0106r", superbatch: 500 },
        Some(NetConfig { name: "0106", superbatch: 500 }),
    );
}
