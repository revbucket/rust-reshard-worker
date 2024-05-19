use core::sync::atomic::{AtomicUsize, Ordering};
use std::io::Read;
use std::time::Instant;
use std::hash::{Hash, Hasher, DefaultHasher};
use anyhow::{anyhow, bail, Result, Error};
use clap::Parser;
use std::path::{PathBuf};
use std::io::{BufReader, Cursor, Write, BufRead};
use std::fs::{File};
use std::thread::available_parallelism;
use std::sync::{Arc, Mutex};
use threadpool::ThreadPool;
use crate::s3::{is_s3, expand_s3_dir, get_reader_from_s3, write_cursor_to_s3};

use glob::glob;
use indicatif::{ProgressBar,ProgressStyle};
use zstd::stream::read::Decoder as ZstdDecoder;

use rand::prelude::*;
use rand::SeedableRng;

use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use zstd::stream::write::Encoder as ZstdEncoder;
pub mod s3;


/*=========================================
=                    ARGS                 =
=========================================*/

#[derive(Parser, Debug)]
struct Args {
    /// (List of) directories/files (on s3 or local) that are jsonl.gz or jsonl.zstd files
    #[arg(required=true, long)]
    input: Vec<PathBuf>,


    /// Output location (may be an s3 uri)
    #[arg(required=true, long)]
    output: PathBuf,

    /// Shard prefix: Files will be named {shard_prefix}_{shard_num}.jsonl.gz
    #[arg(long, default_value_t=String::from("reshard_file"))]
    prefix: String,

    /// Reshard batch size: Glues x many files together 
    #[arg(long, required=true)]
    batch_size: usize,

    /// Subsample rate: Subsamples docs independently at this rate 
    #[arg(long, default_value_t=1.0)]
    subsample_rate: f64,

    /// Random seed: 
    #[arg(long, default_value_t=1234)]
    seed: usize,

    /// Number of threads to use 
    #[arg(long, default_value_t=0)]
    threads: usize,

    /// Extension to save output to 
    #[arg(long, default_value_t=String::from("gz"))] // zstd is the ONLY other valid choice!
    ext: String,

}


/*==============================================
=                   Utilities                  =
==============================================*/

pub(crate) fn expand_dirs(paths: Vec<PathBuf>, ext: Option<&str>) -> Result<Vec<PathBuf>> {
    // For local directories -> does a glob over each directory to get all files with given extension
    // For s3 directories -> does an aws s3 ls to search for files

    let mut files: Vec<PathBuf> = Vec::new();
    let runtime = tokio::runtime::Runtime::new().unwrap();


    for path in paths {
        if is_s3(path.clone()) {
            // Use async_std to block until we scour the s3 directory for files
            runtime.block_on(async {
                let s3_paths = expand_s3_dir(&path, ext).await.unwrap();
                files.extend(s3_paths);                
            });                
        }
        else if path.is_dir() {
            let ext = ext.unwrap_or(".jsonl.gz"); // Defaults to jsonl.gz, json.gz
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
            let mut num_hits = 0;
            //for entry in glob(&format!("{}/**/*.json*.gz", path_str))? {
            for entry in glob(&format!("{}/**/*{}", path_str, ext))? {

                files.push(entry?.to_path_buf());
                num_hits += 1;
            }
            if num_hits == 0 {
                bail!("No JSON Gz files found in '{}'", path_str);
            }
        } else {
            files.push(path.clone());
        }
    }

    Ok(files)
}


fn read_pathbuf_to_mem(input_file: &PathBuf) -> Result<BufReader<Cursor<Vec<u8>>>, Error> {
    // Generic method to read local or s3 file into memory
    let reader = if is_s3(input_file) {
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        match rt.block_on(get_reader_from_s3(input_file, Some(5))) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Error! {:?}", err);
                return Err(err.into());
            }
        }
    } else {
        let contents = read_local_file_into_memory(input_file).expect("Failed to read contents into memory");
        BufReader::new(contents)
    };
    Ok(reader)
} 


fn write_mem_to_pathbuf(contents: &[u8], output_file: &PathBuf) -> Result<(), Error> {
    if is_s3(output_file) {
        let cursor = Cursor::new(contents.to_vec());
        let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();   
        match rt.block_on(write_cursor_to_s3(&output_file, cursor)) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Error! {:?}", err);
                return Err(err.into());
            }
        };
    } else {
        let mut file = File::create(output_file).expect(format!("Unable to create output file {:?}", output_file).as_str());
        file.write_all(contents).expect(format!("Unable to write to {:?}", output_file).as_str());

    }
    Ok(())
}


fn read_local_file_into_memory(input_file: &PathBuf) ->Result<Cursor<Vec<u8>>, Error>{
    // Takes a local file (must be local!) and reads it into a Cursor of bytes
    let mut file = File::open(input_file).expect("Failed to open file");

    let mut contents = Vec::new();
    let ext = input_file.extension().unwrap().to_string_lossy().to_lowercase();
    if ext == "gz" {
        // Gzip case        
        let mut decoder = MultiGzDecoder::new(file);
        decoder.read_to_end(&mut contents).expect("Failed to read loca gzip file");
    } else if ext == "zstd" || ext == "zst" {
        // Zstd case
        let mut decoder = ZstdDecoder::new(file).unwrap();
        decoder.read_to_end(&mut contents).expect("Failed to read local zstd file");
    } else {
        file.read_to_end(&mut contents).expect("Failed to read local file");

        // No compression case 
    }
    Ok(Cursor::new(contents))
}

fn bernoulli(rng: &mut StdRng , p: f64) -> bool {
    // Samples bernoulli RV with given rng and seed 
    let val = rng.next_u32();
    ((val as f64 / u32::MAX as f64) as f64) < p // TODO ACTUALLY GET MAX?
}

fn get_output_name(output_dir: &PathBuf, prefix: &String, shard_id: usize) -> PathBuf {
    output_dir.join(format!("{}_{:08}.jsonl.gz", prefix, shard_id))
}


/*================================================
=                    Meat                        =
================================================*/


fn reshard_batch(batch: &Vec<PathBuf>, output_dir: &PathBuf, prefix: &String, ext: &String,
                       reshard_counter: Arc<AtomicUsize>, subsample_rate: f64, seed: usize,
                       total_docs: Arc<AtomicUsize>, surviving_docs: Arc<AtomicUsize>) -> Result<(), Error> {
    let shard_id = reshard_counter.fetch_add(1, Ordering::SeqCst) as usize;
    let output_pathbuf = get_output_name(output_dir, prefix, shard_id);
    let mut output_lines: Vec<String> = Vec::new();


    // Gather all lines into memory (subsampling as we go)
    let mut local_total = 0;
    let mut local_surviving = 0;
    for file in batch {
        let reader = read_pathbuf_to_mem(file).unwrap();
        let mut hasher = DefaultHasher::new();
        (file, seed).hash(&mut hasher);
        let rng_seed = hasher.finish();
        let mut rng = StdRng::seed_from_u64(rng_seed);
        for line in reader.lines() {
            local_total += 1;
            let line = line?;
            if subsample_rate == 1.0 || bernoulli(&mut rng, subsample_rate) {
                local_surviving += 1;
                output_lines.push(line);
            }
        }
    }
    let output_bytes = output_lines.join("\n");
    let output_bytes = output_bytes.as_bytes();

    // Join output lines together and get the thing we want to write
    let bytes_to_write = match ext.as_str() {
        "gz" => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&output_bytes).unwrap();            
            encoder.finish().unwrap()            
        },
        "zstd" | "zst" => {
            let mut encoder = ZstdEncoder::new(Vec::new(), 0).unwrap();
            encoder.write_all(&output_bytes).unwrap();
            encoder.finish().unwrap()
        },
        _ => output_bytes.to_vec()
    };

    let _ = total_docs.fetch_add(local_total, Ordering::SeqCst);
    let _ = surviving_docs.fetch_add(local_surviving, Ordering::SeqCst);
    write_mem_to_pathbuf(&bytes_to_write, &output_pathbuf).unwrap();
    Ok(())
}


/*=====================================
=                     Main            =
=====================================*/


fn main() -> Result<()> {
    let start_time = Instant::now();    
    let args = Args::parse();
    let threads = if args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        args.threads
    };    

    let ext: Option<&str> = if args.ext.len() == 0 {
        None
    } else {
        Some(&args.ext)
    };
    let input_files: Vec<PathBuf> =  expand_dirs(args.input, ext).unwrap();
    let num_inputs = input_files.len();
    let batches: Vec<&[PathBuf]> = input_files.chunks(args.batch_size).collect();
    let num_outputs = batches.len();

    let pbar = ProgressBar::new(num_outputs as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );


    let pbar = Arc::new(Mutex::new(pbar));
    pbar.lock().unwrap().inc(0); // setup pbar 

    let threadpool = ThreadPool::new(threads);
    let reshard_counter = Arc::new(AtomicUsize::new(0));
    let total_docs = Arc::new(AtomicUsize::new(0));
    let surviving_docs = Arc::new(AtomicUsize::new(0));
    for batch in batches { 
        let batch = batch.to_vec();
        let output = args.output.clone();
        let prefix = args.prefix.clone();
        let ext = args.ext.clone();
        let reshard_counter = reshard_counter.clone();        
        let subsample_rate = args.subsample_rate.clone();
        let seed = args.seed.clone();
        let total_docs = total_docs.clone();
        let surviving_docs = surviving_docs.clone();
        let pbar = pbar.clone();
        threadpool.execute(move || {        
            reshard_batch(&batch, &output, &prefix, &ext, reshard_counter, 
                          subsample_rate, seed, total_docs, surviving_docs).unwrap();
            pbar.lock().unwrap().inc(1);
        });
    }
    threadpool.join();

    println!("Finishing exact dedup run!");
    println!("-------------------------------");
    println!("Ran in {:?} (s)", start_time.elapsed().as_secs());
    println!("Resharded {:?} input files into {:?} output files", 
              num_inputs, num_outputs);
    println!("Saw {:?} input docs | Wrote {:?} output docs",
             total_docs.fetch_add(0, Ordering::SeqCst) as usize,
             surviving_docs.fetch_add(0, Ordering::SeqCst) as usize);


    Ok(())
}
