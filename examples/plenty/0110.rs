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

pub mod argmax {
    use bullet_core::{
        device::OperationError,
        graph::{
            builder::Shape,
            instruction::{GraphInstruction, LinearCombination, MaybeUpdateBatchSize},
            ir::{
                node::AnnotatedNode,
                operation::{util, GraphIROperation, GraphIROperationCompilable},
                BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
            },
            Graph, GraphFunction, NodeId, NodeIdTy,
        },
    };
    use bullet_cuda_backend::{cudarc::driver::{LaunchConfig, PushKernelArg}, CudaDevice, CudaError, CudaMarker};

    #[derive(Debug)]
    pub struct ArgMax {
        pub input: AnnotatedNode,
    }

    impl<B: BackendMarker> GraphIROperation<B> for ArgMax {
        fn nodes(&self) -> Vec<AnnotatedNode> {
            vec![self.input]
        }

        fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
            util::check_dense_eq(ir, &self.input, true)?;

            Ok(self.input.shape)
        }
    }

    impl GraphIROperationCompilable<CudaMarker> for ArgMax {
        fn forward_pass(&self, _: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
            let input = NodeId::new(self.input.idx, NodeIdTy::Values);
            let output = NodeId::new(output_node, NodeIdTy::Values);

            let mut func = GraphFunction::default();

            func.push(MaybeUpdateBatchSize { input, output });
            func.push(ArgMaxFwd { input, output });

            func
        }

        fn backward_pass(&self, _: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
            let input = NodeId::new(output_node, NodeIdTy::Gradients);
            let output = NodeId::new(self.input.idx, NodeIdTy::Gradients);

            let mut func = GraphFunction::default();

            func.push(MaybeUpdateBatchSize { input, output });
            func.push(LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });

            func
        }
    }

    #[derive(Debug)]
    struct ArgMaxFwd {
        input: NodeId,
        output: NodeId,
    }

    impl GraphInstruction<CudaDevice> for ArgMaxFwd {
        fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
            let input = graph.get(self.input)?;
            let input = input.dense()?;

            let mut output = graph.get_mut(self.output)?;
            let output = output.dense_mut()?;

            if input.batch_size() != output.batch_size() {
                return Err(OperationError::MismatchedBatchSizes);
            }

            if input.single_size() != output.single_size() {
                return Err(OperationError::InvalidTensorFormat);
            }

            let device = input.buf.device.clone();

            unsafe {
                let func = device.get_custom_func_or_rtc("ArgMaxKernel", || include_str!("argmax.cu").to_string())?;

                let batch_size = input.batch_size().unwrap_or(1);
                let single_size = input.single_size();
                let blocks = batch_size.div_ceil(512) as u32;
                let grid_dim = (blocks, 1, 1);

                device
                    .stream()
                    .launch_builder(&func)
                    .arg(&(single_size as i32))
                    .arg(&(batch_size as i32))
                    .arg(&input.buf.buf)
                    .arg(&mut output.buf.buf)
                    .launch(LaunchConfig { grid_dim, block_dim: (512, 1, 1), shared_mem_bytes: 0 })
                    .map_err(CudaError::Driver)?;
            }

            Ok(())
        }
    }
}

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
            let router = builder.new_affine("router", 2 * L2_SIZE, OUTPUT_BUCKETS);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE + 2 * L2_SIZE, OUTPUT_BUCKETS);

            // FT w/ Crelu + Pairwise
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let pairwise_out = stm_subnet.concat(ntm_subnet);

            // L1 w/ Dual activation
            let l1_out = l1.forward(pairwise_out).select(buckets);
            let l1_out = l1_out.concat(l1_out.abs_pow(2.0));
            let l1_out = l1_out.crelu();

            // Output bucket router
            let router_bucket = builder.apply(argmax::ArgMax { input: router.forward(l1_out).annotated_node() });

            // L2 + L3 forward
            let l2_out = l2.forward(l1_out).reshape(Shape::new(L3_SIZE, OUTPUT_BUCKETS)).matmul(router_bucket).screlu();
            let l3_out = l3.forward(l2_out.concat(l1_out)).reshape(Shape::new(1, OUTPUT_BUCKETS)).matmul(router_bucket);

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
        NetConfig { name: "0110", superbatch: 300 },
        None,
    );

    // Step 2
    train(
        "/mnt/d/Chess Data/Selfgen/5ksn.data",
        wdl::ConstantWDL { value: 0.3 },
        lr::CosineDecayLR { initial_lr: 0.00025, final_lr: 0.00025 * 0.3 * 0.3 * 0.3, final_superbatch: 300 },
        NetConfig { name: "0110r", superbatch: 300 },
        Some(NetConfig { name: "0110", superbatch: 300 }),
    );

    // Step 3
    train(
        "/mnt/d/Chess Data/Selfgen/20ksn.data", 
        wdl::ConstantWDL { value: 0.6 },
        lr::CosineDecayLR { initial_lr: 0.00025, final_lr: 0.00025 * 0.3 * 0.3 * 0.3, final_superbatch: 400 },
        NetConfig { name: "0110rr", superbatch: 400 },
        Some(NetConfig { name: "0110r", superbatch: 300 }),
    );

    // Step 4
    train(
        "/mnt/d/Chess Data/Selfgen/20ksn-adversarial-plentychonker.data",
        wdl::ConstantWDL { value: 1.0 },
        lr::CosineDecayLR { initial_lr: 0.000025, final_lr: 0.000025 * 0.3 * 0.3 * 0.3, final_superbatch: 300 },
        NetConfig { name: "0110rrr", superbatch: 300 },
        Some(NetConfig { name: "0110rr", superbatch: 400 }),
    );
}
