use bullet_core::optimiser::adam::AdamW;
use bullet_lib::{
    default::{
        inputs::{self, SparseInputType},
        loader::DirectSequentialDataLoader,
        outputs::MaterialCount,
    },
    game::inputs::ChessBucketsMirrored,
    lr::{self, LrScheduler},
    nn::{optimiser, InitSettings},
    trainer::{save::SavedFormat},
    value::{ValueTrainer, ValueTrainerBuilder},
    wdl::{self, WdlScheduler},
    ExecutionContext, LocalSettings, Shape, TrainingSchedule, TrainingSteps,
};
use rand::Rng;

#[derive(Clone)]
struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

const TRAINING_DIR: &str = "/mnt/d/Chess Data/Selfgen/Training";

fn make_trainer() -> ValueTrainer<AdamW<ExecutionContext>, ChessBucketsMirrored, MaterialCount<8>> {
    #[rustfmt::skip]
    let inputs = inputs::ChessBucketsMirrored::new([
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
            let mut l0 = builder.new_affine("l0", 768 * KING_BUCKETS, L1_SIZE);

            let l0f = builder.new_weights("l0f", Shape::new(768 * L1_SIZE, 1), InitSettings::Zeroed);
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
    // Step 4
    train(
        "/mnt/d/Chess Data/Selfgen/20ksn-only-adversarial-plentychonker.data",
        wdl::ConstantWDL { value: 1.0 },
        lr::CosineDecayLR { initial_lr: 0.000025, final_lr: 0.000025 * 0.3 * 0.3 * 0.3, final_superbatch: 300 },
        NetConfig { name: "0109rrr4", superbatch: 300 },
        Some(NetConfig { name: "0109rr", superbatch: 400 }),
    );
}
