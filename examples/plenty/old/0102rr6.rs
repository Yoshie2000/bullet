use std::fs;

use bullet_core::optimiser::adam::AdamW;
use bullet_lib::{
    default::{
        inputs,
        loader::DirectSequentialDataLoader,
        outputs::{self, MaterialCount},
        Loss, Trainer, TrainerBuilder,
    },
    game::inputs::ChessBucketsMirroredFactorised,
    lr::{self, LrScheduler},
    nn::optimiser,
    trainer::NetworkTrainer,
    wdl::{self, WdlScheduler},
    Activation, ExecutionContext, LocalSettings, TrainingSchedule, TrainingSteps,
};
use rand::Rng;

const HIDDEN_SIZE: usize = 1792;
const SCALE: f32 = 450.0;

#[derive(Clone)]
struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

fn make_trainer() -> Trainer<AdamW<ExecutionContext>, ChessBucketsMirroredFactorised, MaterialCount<8>> {
    #[rustfmt::skip]
    return TrainerBuilder::default()
        .optimiser(optimiser::AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .datapoint_weight_function(|datapoint| {
            let mut rng = rand::rng();
            if rng.random_range(0.0..1.0) < 0.05 {
                0.0
            } else {
                1.0
            }
        })
        .input(inputs::ChessBucketsMirroredFactorised::new([
            00, 01, 02, 03,
            04, 05, 06, 07,
            08, 08, 09, 09,
            10, 10, 10, 10,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
        ]))
        .output_buckets(outputs::MaterialCount::<8>)
        .feature_transformer(HIDDEN_SIZE)
        .activate(Activation::CReLU)
        .add_pairwise_mul()
        .add_layer(16)
        .add_dual_activation()
        .add_layer(32)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build();
}

fn make_settings(experiment_name: &str) -> LocalSettings {
    return LocalSettings {
        threads: 4,
        output_directory: format!("/mnt/d/Chess Data/Selfgen/Training/{}", experiment_name),
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
            let _ = trainer.optimiser_mut().load_weights_from_file(
                format!(
                    "/mnt/d/Chess Data/Selfgen/Training/{}/net-{}-{}/optimiser_state/weights.bin",
                    load_from_net.name, load_from_net.name, load_from_net.superbatch
                )
                .as_str(),
            ).unwrap();
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

    let start_superbatch = if load_from.clone().map(|load_from| load_from.name == net.name).unwrap_or(false) { load_from.unwrap().superbatch + 1 } else { 1 };

    let schedule = TrainingSchedule {
        net_id: format!("net-{}", net.name).to_string(),
        eval_scale: SCALE,
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

    trainer.set_optimiser_params(optimiser::AdamWParams {
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
        "/mnt/d/Chess Data/Selfgen/20ksn.data",
        wdl::ConstantWDL { value: 0.6 },
        lr::CosineDecayLR { initial_lr: 0.00025, final_lr: 0.00025 * 0.3 * 0.3 * 0.3, final_superbatch: 400 },
        NetConfig { name: "0102rr6", superbatch: 400 },
        Some(NetConfig { name: "0102r", superbatch: 300 }),
    );
}
