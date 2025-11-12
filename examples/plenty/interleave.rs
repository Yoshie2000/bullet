use std::{fs::{self, File}, io::{BufReader, BufWriter, Write}, path::{Path, PathBuf}};

use viriformat::dataformat::Game;

struct Rand(u64);

impl Default for Rand {
    fn default() -> Self {
        Self(
            (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("valid").as_nanos()
                & 0xFFFF_FFFF_FFFF_FFFF) as u64,
        )
    }
}

impl Rand {
    fn rand(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

pub fn run(inputs: Vec<PathBuf>, output: PathBuf) -> anyhow::Result<()> {
    println!("Writing to {:#?}", output);
    println!("Reading from:\n{:#?}", inputs);
    let mut streams = Vec::new();
    let mut total = 0;

    let target = File::create(&output)?;
    let mut writer = BufWriter::new(target);

    let mut total_input_file_size = 0;
    for path in &inputs {
        let file = File::open(path)?;

        let count = file.metadata()?.len();

        total_input_file_size += count;

        if count > 0 {
            let fname = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "<unknown>".into());
            streams.push((count, BufReader::new(file), fname));
            total += count;
        }
    }

    let mut remaining = total;
    let mut rng = Rand::default();

    const INTERVAL: u64 = 1024 * 1024 * 256;
    let mut prev = remaining / INTERVAL;

    let mut buffer = Vec::new();
    let mut games = 0usize;

    while remaining > 0 {
        let mut spot = rng.rand() % remaining;
        let mut idx = 0;
        while streams[idx].0 < spot {
            spot -= streams[idx].0;
            idx += 1;
        }

        let (count, reader, _) = &mut streams[idx];

        buffer.clear();
        Game::deserialise_fast_into_buffer(reader, &mut buffer)?;
        writer.write_all(&buffer)?;
        games += 1;

        let size = buffer.len() as u64;

        remaining -= size;
        *count -= size;
        if *count == 0 {
            println!("Finished reading {}", streams[idx].2);
            streams.swap_remove(idx);
        }

        if remaining / INTERVAL < prev {
            prev = remaining / INTERVAL;
            let written = total - remaining;
            print!("Written {written}/{total} Bytes ({:.2}%)\r", written as f64 / total as f64 * 100.0);
            let _ = std::io::stdout().flush();
        }
    }

    writer.flush()?;

    println!();
    println!("Written {games} games to {:#?}", output);

    let output_file = File::open(&output)?;
    let output_file_size = output_file.metadata()?.len();
    if output_file_size != total_input_file_size {
        anyhow::bail!("Output file size {output_file_size} does not match input file size {total_input_file_size}");
    }

    Ok(())
}

fn main() {
    const VIRIFORMAT_PATH: &str = "/mnt/e/Chess/Data/Viriformat";

    let inputs: Vec<PathBuf> = fs::read_dir(VIRIFORMAT_PATH).unwrap().map(|file| file.unwrap().path()).filter(|file| file.as_os_str().to_str().unwrap().ends_with(".rlbd") && !file.as_os_str().to_str().unwrap().ends_with("12859.vf.rlbd")).collect();
    let output = Path::new("/mnt/e/Chess/Data/combined.vf").to_path_buf();
    run(inputs, output).unwrap();
}
