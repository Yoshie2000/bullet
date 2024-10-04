use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use super::{to_slice_with_lifetime, BulletFormat, DataLoader};

#[derive(Clone)]
pub struct DirectSequentialDataLoader {
    file_paths: Vec<String>,
}

impl DirectSequentialDataLoader {
    pub fn new(file_paths: &[&str]) -> Self {
        let file_paths = file_paths.iter().map(|path| path.to_string()).collect::<Vec<_>>();

        for path in &file_paths {
            let path_buf: PathBuf = path.parse().unwrap();
            assert!(path_buf.exists(), "File not found: {path}");
        }

        Self { file_paths }
    }
}

impl<T: BulletFormat + 'static> DataLoader<T> for DirectSequentialDataLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        let data_size = std::mem::size_of::<T>() as u64;

        let mut file_size = 0;

        for file in self.file_paths.iter() {
            let this_size = std::fs::metadata(file).unwrap().len();

            if this_size % data_size != 0 {
                panic!("File [{file}] does not have a multiple of {data_size} size!");
            }

            file_size += this_size;
        }

        Some(file_size / data_size)
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, batch_size: usize, mut f: F) {
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size: usize = std::mem::size_of::<T>();
        let batches_per_load = buffer_size / data_size / batch_size;
        let cap = data_size * batch_size * batches_per_load;

        'dataloading: loop {
            let mut loader_files = vec![];
            for file in self.file_paths.iter() {
                loader_files.push(File::open(file).unwrap());
            }

            for loader_file in loader_files.iter() {
                let mut file = BufReader::with_capacity(cap, loader_file);
                while let Ok(buf) = file.fill_buf() {
                    if buf.is_empty() {
                        break;
                    }

                    let data: &[T] = unsafe { to_slice_with_lifetime(buf) };

                    for batch in data.chunks(batch_size) {
                        let should_break = f(batch);

                        if should_break {
                            break 'dataloading;
                        }
                    }

                    let consumed = buf.len();
                    file.consume(consumed);
                }
            }
        }
    }
}