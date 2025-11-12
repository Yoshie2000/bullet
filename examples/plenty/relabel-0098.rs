/// This is much slower than it could be for the sake of simplicity.
/// To improve speed you would need to asynchronously read and write data,
/// and asynchronously copy next batch to GPU whilst current batch is computing.
use std::{
    fs::{self, File},
    io::BufWriter,
    path::Path,
    time::Instant,
};

use acyclib::{graph::like::GraphLike, trainer::dataloader::PreparedBatchDevice};
use bullet_lib::{
    game::{inputs::ChessBucketsMirroredFactorised, outputs::MaterialCount},
    nn::optimiser,
    value::{ValueTrainerBuilder, loader::ViriBinpackLoader},
};
use viriformat::dataformat::{Filter, Game};

fn main() {
    let threads = 4;

    #[rustfmt::skip]
    let inputs = ChessBucketsMirroredFactorised::new([
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
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[])
        .build(|builder, stm, ntm, buckets| {
            // Build fast factoriser
            let l0 = builder.new_affine("l0", 768 + 768 * KING_BUCKETS, L1_SIZE);

            // let l0f = builder.new_weights("l0f", Shape::new(768 * L1_SIZE, 1), InitSettings::Zeroed);
            // let ones = builder.new_constant(Shape::new(1, KING_BUCKETS), &[1.0; KING_BUCKETS]);
            // let expanded = l0f.matmul(ones).reshape(Shape::new(L1_SIZE, inputs.num_inputs()));
            // l0.weights = l0.weights + expanded;

            // Build layers
            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS * L4_SIZE);
            let l4 = builder.new_affine("l4", L4_SIZE, OUTPUT_BUCKETS);

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
            let l3_out = l3.forward(l2_out).select(buckets).screlu();
            let l4_out = l4.forward(l3_out).select(buckets).screlu();

            l4_out
        });
    trainer.load_from_checkpoint("/mnt/d/Chess Data/Selfgen/Training/0098/net-0098-1000");

    // let correct = (400.0 * trainer.eval("r2qkb1r/1p3p1p/2ppbp2/p3p2Q/3nP3/P2B3N/1PPP1PPP/R1B1K2R b KQkq - 1 1")) as i16;
    // println!("{correct}");
    // return;

    const VIRIFORMAT_PATH: &str = "/mnt/e/Chess/Data/Viriformat";

    for path in fs::read_dir(VIRIFORMAT_PATH).unwrap() {
        let path = path.unwrap().path();
        let path_str = path.as_os_str().to_str().unwrap();
        let relabel_path_str = path_str.to_owned() + ".rlbd";

        if Path::new(relabel_path_str.as_str()).exists() || path_str.ends_with(".rlbd") {
            println!("Skipping {path_str}");
            continue;
        }

        println!("Relabeling {path_str}");

        let loader = ViriBinpackLoader::new(path_str, 1024, threads, Filter::UNRESTRICTED);

        let state = trainer.state.clone();

        let mut file = BufWriter::new(File::create(relabel_path_str).unwrap());

        let t = Instant::now();

        let mut relabelled = 0usize;
        let mut relabelled_games = 0usize;

        let mut b = 0;
        loader.map_batches_relabel(0, 16384 * 4, |batch, games| {
            assert_eq!(batch.len(), games.iter().map(|g| g.len()).sum());
            if batch.len() == 0 {
                return false;
            }

            let prepared = state.prepare(batch, threads, 1.0, 1.0);
            let mut device_data = PreparedBatchDevice::new(trainer.optimiser.graph.devices(), &prepared).unwrap();
            device_data.load_into_graph(&mut trainer.optimiser.graph).unwrap();
            trainer.optimiser.graph.execute_fn("forward").unwrap();
            trainer.optimiser.graph.synchronise().unwrap();
            let new_evals = trainer.get_output_values();

            assert_eq!(batch.len(), new_evals.len());

            let mut i = 0;
            for game in games {
                let mut relabeled_game = Game { initial_position: game.initial_position, moves: Vec::new() };
                for m in game.moves() {
                    relabeled_game.add_move(m, (400.0 * new_evals[i]) as i16);
                    i += 1;
                }
                relabeled_game.serialise_into(&mut file).unwrap();

                // relabeled_game.visit_positions(|viriboard, score| {
                //     let bf = viriboard.to_bulletformat(0, 0).unwrap();

                //     let prepared = state.prepare(&[bf], 1, 1.0, 1.0);
                //     let mut device_data = PreparedBatchDevice::new(trainer.optimiser.graph.devices(), &prepared).unwrap();
                //     device_data.load_into_graph(&mut trainer.optimiser.graph).unwrap();
                //     trainer.optimiser.graph.execute_fn("forward").unwrap();
                //     trainer.optimiser.graph.synchronise().unwrap();
                //     let correct = (400.0 * trainer.get_output_values()[0]) as i16;
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
