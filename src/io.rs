// #![feature(test)]



use std::io;
use std::f64;
use std::usize;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::cmp::{PartialOrd,Ordering,Eq};
use std::fmt::Debug;

use num_cpus;

use ndarray::Array2;

use crate::SampleValue;

//::/ Author: Boris Brenerman
//::/ Created: 2017 Academic Year, Johns Hopkins University, Department of Biology, Taylor Lab

//::/ This is a forest-based regression/classification software package designed with single-cell RNAseq data in mind.
//::/
//::/ Currently implemented features are to generate Decision Trees that segment large 2-dimensional matrices, and prediction of samples based on these decision trees
//::/
//::/ Features to be implemented include interaction analysis, python-based node clustering and trajectory analysis using minimum spanning trees of clusters, feature correlation analysis, and finally subsampling-based gradient boosting for sequential tree generation.

//::/ The general structure of the program is as follows:
//::/
//::/ The outer-most class is the Random Forest
//::/


//::/ Random Forests:
//::/
//::/ Random Forest contains:
//::/     - The matrix to be analyzed
//::/     - Decision Trees
//::/
//::/     - Important methods:
//::/         - Method that generates decision trees and calls on them to grow branches
//::/         - Method that generates predicted values for a matrix of samples
//::/
//::/



//::/ Trees:
//::/
//::/ Trees contain:
//::/     - Root Node
//::/     - Feature Thread Pool Sender Channel
//::/     - Drop Mode
//::/
//::/ Each tree contains a subsampling of both rows and columns of the original matrix. The subsampled rows and columns are contained in a root node, which is the only node the tree has direct access to.
//::/

//::/ Feature Thread Pool:
//::/
//::/ Feature Thread Pool contains:
//::/     - Worker Threads
//::/     - Reciever Channel for jobs
//::/
//::/     - Important methods:
//::/         - A wrapper method to compute a set of medians and MADs for each job passed to the pool. Core method logic is in Rank Vector
//::/
//::/ Feature Thread Pools are containers of Worker threads. Each pool contains a multiple in, single out channel locked with a Mutex. Each Worker contained in the pool continuously requests jobs from the channel. If the Mutex is unlocked and has a job, a Worker thread receives it.
//::/
//::/     Jobs:
//::/         Jobs in the pool channel consist of a channel to pass back the solution to the underlying problem and a freshly spawned Rank Vector (see below). The job consists of calling a method on the RV that consumes it and produces the medians and Median Absolute Deviations (MAD) from the Median of the vector if a set of samples is removed from it in a given order. This allows us to determine what the Median Absolute Deviation from the Median would be given the split of that feature by some draw order. The draw orders given to each job are usually denoting that the underlying matrix was sorted by another feature.
//::/
//::/ Worker threads are simple anonymous threads kept in a vector in the pool, requesting jobs on loop from the channel.

    //
    // let mut rnd_forest = Forest::initialize(&input_array,&output_array, arc_params.clone(), report_address);
    //
    // rnd_forest.generate(arc_params.clone(),false).unwrap();
    //





#[derive(Clone,Debug)]
pub struct ParameterBook<V>
where
    V: SampleValue
{
    pub auto: bool,
    pub input_count_array_file: String,
    pub input_array: Option<Array2<V>>,
    pub output_count_array_file: String,
    pub output_array: Option<Array2<V>>,
    pub input_feature_header_file: Option<String>,
    pub input_feature_names: Vec<String>,
    pub output_feature_header_file: Option<String>,
    pub output_feature_names: Vec<String>,
    pub sample_header_file: Option<String>,
    pub sample_names: Vec<String>,
    pub report_address: String,

    pub processor_limit: usize,
    pub tree_limit: usize,
    pub leaf_size_cutoff: usize,
    pub depth_cutoff: usize,
    pub max_splits: usize,

    pub feature_subsample: usize,
    pub sample_subsample: usize,
    pub input_feature_subsample: usize,
    pub output_feature_subsample: usize,

    pub braid_thickness: usize,
    pub scaling:bool,
    pub reduce_input:bool,
    pub reduce_output:bool,

    pub norm_mode: NormMode,
    pub dispersion_mode: DispersionMode,
    pub split_fraction_regularization: f64,
    pub big_mem: bool,

    pub backups: Option<String>,
    pub backup_vec: Option<Vec<String>>,

}

pub fn read<V:SampleValue,T: Iterator<Item = String>>(args: &mut T) -> ParameterBook<V> {

    let mut arg_struct = ParameterBook::blank();

    let mut supress_warnings = false;
    let mut continuation_flag = false;
    let mut continuation_argument: String = "".to_string();

    let _location = args.next();

    while let Some((i,arg)) = args.enumerate().next() {

        // println!("READING:{:?}", arg);

        if arg.chars().next().unwrap_or('_') == '-' {
            continuation_flag = false;

        }
        match &arg[..] {
            "-sw" | "-suppress_warnings" => {
                if i!=1 {
                    println!("If the supress warnings flag is not given first it may not function correctly.");
                }
                supress_warnings = true;
            },
            "-auto" | "-a"=> {
                // arg_struct.auto = true;
                // arg_struct.auto()
            },
            // "-stdin" => {
            //     let single_array = Some(read_standard_in());
            //     arg_struct.input_array = single_array.clone();
            //     arg_struct.output_array = single_array;
            // }
            "-c" | "-counts" => {
                let single_count_array_file = args.next().expect("Error parsing count location!");
                let single_array = Some({read_matrix(&single_count_array_file)});
                arg_struct.input_count_array_file = single_count_array_file.clone();
                arg_struct.input_array = single_array.clone();
                arg_struct.output_count_array_file = single_count_array_file;
                arg_struct.output_array = single_array;
            },
            "-scale" | "-scaling" => {
                arg_struct.scaling = true;
            },
            "-reduce_input" | "-ri" => {
                arg_struct.reduce_input = args.next().unwrap().parse::<bool>().expect("Error parsing reduction argument");
            },
            "-reduce_output" | "-ro" => {
                arg_struct.reduce_output = args.next().unwrap().parse::<bool>().expect("Error parsing reduction argument");
            },

            "-ic" | "-input_counts" | "-input" => {
                arg_struct.input_array = Some(read_matrix(&args.next().expect("Error parsing input count location!")));
            }
            "-oc" | "-output_counts" | "-output" => {
                arg_struct.output_array = Some(read_matrix(&args.next().expect("Error parsing output count location!")));
            }
            "-backups" | "-bk" => {
                arg_struct.backups = Some(args.next().expect("Error parsing tree locations"));
            },
            "-dm" | "-dispersion_mode" => {
                arg_struct.dispersion_mode = DispersionMode::read(&args.next().expect("Failed to read split mode"));
            },
            "-split_fraction_regularization" | "-sfr" => {
                arg_struct.split_fraction_regularization = args.next().expect("Error processing SFR").parse::<f64>().expect("Error parsing SFR");
            }
            "-n" | "-norm" | "-norm_mode" => {
                arg_struct.norm_mode = NormMode::read(&args.next().expect("Failed to read norm mode"));
            },
            "-t" | "-trees" => {
                arg_struct.tree_limit = args.next().expect("Error processing tree count").parse::<usize>().expect("Error parsing tree count");
            },
            "-tg" | "-tree_glob" => {
                continuation_flag = true;
                continuation_argument = arg.clone();
            },
            "-p" | "-processors" | "-threads" => {
                arg_struct.processor_limit = args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit");
                rayon::ThreadPoolBuilder::new().num_threads(arg_struct.processor_limit).build_global().unwrap();
                std::env::set_var("OMP_NUM_THREADS",format!("{}",arg_struct.processor_limit));
            },
            "-o" | "-output_location" => {
                arg_struct.report_address = args.next().expect("Error processing output destination")
            },
            "-ifh" | "-ih" | "-input_feature_header" => {
                arg_struct.input_feature_header_file = Some(args.next().expect("Error processing feature file"));
                arg_struct.input_feature_names = read_header(arg_struct.input_feature_header_file.as_ref().unwrap());
            },
            "-ofh" | "-oh" | "-output_feature_header" => {
                arg_struct.output_feature_header_file = Some(args.next().expect("Error processing feature file"));
                arg_struct.output_feature_names = read_header(arg_struct.output_feature_header_file.as_ref().unwrap());
            },
            "-h" | "-header" => {
                let header_file = args.next().expect("Error processing feature file");
                let header = read_header(&header_file);

                arg_struct.input_feature_header_file = Some(header_file.clone());
                arg_struct.output_feature_header_file = Some(header_file);

                arg_struct.input_feature_names = header.clone();
                arg_struct.output_feature_names = header;
            },
            "-s" | "-samples" => {
                arg_struct.sample_header_file = Some(args.next().expect("Error processing feature file"));
                arg_struct.sample_names = read_header(arg_struct.sample_header_file.as_ref().unwrap());
            }
            "-l" | "-leaves" => {
                arg_struct.leaf_size_cutoff = args.next().expect("Error processing leaf limit").parse::<usize>().expect("Error parsing leaf limit");
            },
            "-depth" => {
                arg_struct.depth_cutoff = args.next().expect("Error processing depth").parse::<usize>().expect("Error parsing depth");
            }
            "-max_splits" => {
                arg_struct.max_splits = args.next().expect("Error processing max_splits").parse::<usize>().expect("Error parsing max_splits");
            }
            "-if" | "-ifs" | "-in_features" | "-in_feature_subsample" | "-input_feature_subsample" => {
                arg_struct.input_feature_subsample = args.next().expect("Error processing in feature arg").parse::<usize>().expect("Error in feature  arg");
            },
            "-of" | "-ofs" | "-out_features" | "-out_feature_subsample" | "-output_feature_subsample" => {
                arg_struct.output_feature_subsample = args.next().expect("Error processing out feature arg").parse::<usize>().expect("Error out feature arg");
            },
            "-fs" | "-feature_sub" | "-feature_subsample" | "-feature_subsamples" => {
                arg_struct.feature_subsample = args.next().expect("Error processing feature subsample arg").parse::<usize>().expect("Error feature subsample arg");
            },
            "-ss" | "-sample_sub" | "-sample_subsample" | "-sample_subsamples" => {
                arg_struct.sample_subsample = args.next().expect("Error processing sample subsample arg").parse::<usize>().expect("Error sample subsample arg");
            },
            "-braid" | "-braids" | "-braid_thickness" => {
                arg_struct.braid_thickness = args.next().expect("Error reading braid thickness").parse::<usize>().expect("-braid not a number");
            },
            &_ => {
                if continuation_flag {
                    match &continuation_argument[..] {
                        "-tg" | "-tree_glob" => {
                            arg_struct.backup_vec.get_or_insert(vec![]).push(arg);
                        }
                        &_ => {
                            panic!("Continuation flag set but invalid continuation argument, debug prediction arg parse!");
                        }
                    }
                }
                else if !supress_warnings {
                    // eprintln!("Warning, detected unexpected argument:{}. Ignoring, press enter to continue, or CTRL-C to stop. Were you trying to input multiple arguments? Only some options take multiple arguments. Watch out for globs(*, also known as wild cards), these count as multiple arguments!",arg);
                    // stdin().read_line(&mut String::new());
                    panic!(format!("Unexpected argument:{}",arg));
                }
            }

        }
    }

    assert!(arg_struct.input_array.as_ref().expect("Please specify input").dim() == arg_struct.output_array.as_ref().expect("Please specify output").dim(), "Unequal dimensions in input and output!");

    if arg_struct.input_feature_header_file.is_none() {
        let cols = arg_struct.input_array.as_ref().unwrap().dim().1;
        arg_struct.input_feature_names = (0..cols).map(|x| x.to_string()).collect()
    }
    if arg_struct.output_feature_header_file.is_none() {
        let cols = arg_struct.output_array.as_ref().unwrap().dim().1;
        arg_struct.output_feature_names = (0..cols).map(|x| x.to_string()).collect()
    }
    if arg_struct.sample_header_file.is_none() {
        let rows = arg_struct.input_array.as_ref().unwrap().dim().0;
        arg_struct.sample_names = (0..rows).map(|i| format!("{:?}",i).to_string()).collect()
    }

    eprintln!("INPUT ARRAY DIMENSION:{:?}", arg_struct.input_array.as_ref().unwrap().dim());
    eprintln!("OUTPUT ARRAY DIMENSION:{:?}", arg_struct.output_array.as_ref().unwrap().dim());
    eprintln!("SAMPLE HEADER:{}", arg_struct.sample_names.len());

    arg_struct

}

impl<V: SampleValue> ParameterBook<V> {
    fn blank() -> ParameterBook<V> {
        ParameterBook
        {
            auto: false,
            input_count_array_file: "".to_string(),
            input_array: None,
            output_count_array_file: "".to_string(),
            output_array: None,
            input_feature_header_file: None,
            input_feature_names: vec![],
            output_feature_header_file: None,
            output_feature_names: vec![],
            sample_header_file: None,
            sample_names: vec![],
            report_address: "./".to_string(),

            processor_limit: 1,
            tree_limit: 1,
            leaf_size_cutoff: usize::MAX,
            depth_cutoff: 1,
            max_splits: 1,

            feature_subsample: 1,
            sample_subsample: 1,
            input_feature_subsample: 1,
            output_feature_subsample: 1,

            braid_thickness: 3,
            scaling:false,
            reduce_input:false,
            reduce_output:false,

            norm_mode: NormMode::L2,
            dispersion_mode: DispersionMode::SSME,
            split_fraction_regularization: 0.5,
            big_mem: false,

            backups: None,
            backup_vec: None,


        }
    }

    fn auto(&mut self) {

        let input_counts = self.input_array.as_ref().expect("Please specify counts file(s) before the \"-auto\" argument.");
        let output_counts = self.output_array.as_ref().expect("Please specify counts file(s) before the \"-auto\" argument.");

        let input_features = input_counts.dim().1;
        let output_features = output_counts.dim().1;
        let samples = input_counts.dim().0;

        let output_feature_subsample = ((output_features as f64 / (output_features as f64).log10()) as usize).min(output_features);

        let input_feature_subsample = ((input_features as f64 / (input_features as f64).log10()) as usize).min(input_features);

        let feature_subsample = output_features;

        let sample_subsample: usize;

        if samples < 10 {
            eprintln!("Warning, you seem to be using suspiciously few samples, are you sure you specified the right file? If so, trees may not be the right solution to your problem.");
            sample_subsample = samples;
        }
        else if samples < 1000 {
            sample_subsample = (samples/3)*2;
        }
        else if samples < 5000 {
            sample_subsample = samples/2;
        }
        else {
            sample_subsample = samples/4;
        }

        let leaf_size_cutoff = (sample_subsample as f64).sqrt() as usize;

        let depth_cutoff = ((samples as f64).log(5.) as usize).max(2);

        let trees = 100;

        let processors = num_cpus::get();

        let max_splits = 100;

        // let dropout: DropMode;
        //
        // if input_counts.iter().flat_map(|x| x).any(|x| x.is_nan()) || output_counts.iter().flat_map(|x| x).any(|x| x.is_nan()) {
        //     dropout = DropMode::NaNs;
        // }
        // else if input_counts.iter().flat_map(|x| x.iter().map(|y| if *y == 0. {1.} else {0.})).sum::<f64>() > ((samples * input_features) as f64 / 4.) ||
        //         output_counts.iter().flat_map(|x| x.iter().map(|y| if *y == 0. {1.} else {0.})).sum::<f64>() > ((samples * output_features) as f64 / 4.)
        // {
        //     dropout = DropMode::Zeros;
        // }
        // else {
        //     dropout = DropMode::No;
        // }


        println!("Automatic parameters:");
        // println!("fs:{:?}",feature_subsample);
        println!("ss:{:?}",sample_subsample);
        println!("if:{:?}",input_features);
        println!("of:{:?}",output_features);
        println!("p:{:?}",processors);
        println!("t:{:?}",trees,);
        println!("l:{:?}",leaf_size_cutoff);

        self.auto = true;

        self.feature_subsample = feature_subsample;
        self.sample_subsample = sample_subsample;
        self.input_feature_subsample = input_feature_subsample;
        self.output_feature_subsample = output_feature_subsample;


        self.processor_limit = processors;
        self.tree_limit = trees;
        self.leaf_size_cutoff = leaf_size_cutoff;
        self.depth_cutoff = depth_cutoff;
        self.max_splits = max_splits;


    }


    pub fn test(array:Array2<f64>) -> ParameterBook<f64> {
        ParameterBook
        {
            auto: false,
            input_count_array_file: "".to_string(),
            input_array: Some(array.clone()),
            output_count_array_file: "".to_string(),
            output_array: Some(array.clone()),
            input_feature_header_file: None,
            input_feature_names: (0..array.dim().1).map(|x| format!("{:}",x)).collect(),
            output_feature_header_file: None,
            output_feature_names: (0..array.dim().1).map(|x| format!("{:}",x)).collect(),
            sample_header_file: None,
            sample_names: vec![],
            report_address: "./".to_string(),

            processor_limit: 10,
            tree_limit: 100,
            leaf_size_cutoff: 5,
            depth_cutoff: 5,
            max_splits: 100,

            feature_subsample: 2,
            sample_subsample: 20,
            input_feature_subsample: 2,
            output_feature_subsample: 2,

            braid_thickness: 1,
            scaling:false,
            reduce_input:true,
            reduce_output:false,

            norm_mode: NormMode::L1,
            dispersion_mode: DispersionMode::SSME,
            split_fraction_regularization: 0.5,
            big_mem: false,

            backups: None,
            backup_vec: None,


        }
    }


}
//
// // Various modes that are included in Parameters, serving as control elements for program internals. Each mode can parse strings that represent alternative options for that mode. Enums were chosen because they compile down to extremely small memory footprint.
//

#[derive(Clone,Copy,Debug)]
pub enum BoostMode {
    Additive,
    Subsampling,
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum DispersionMode {
    MAD,
    Variance,
    SME,
    SSME,
    Entropy,
    Mixed,
}

impl DispersionMode {
    pub fn read(input: &str) -> DispersionMode {
        match input {
            "var" | "variance" => DispersionMode::Variance,
            "mad"  => DispersionMode::MAD,
            "mix" | "mixed" => DispersionMode::Mixed,
            "ssme" => DispersionMode::SSME,
            "sme" => DispersionMode::SME,
            "entropy" => DispersionMode::Entropy,
            _ => panic!("Not a valid dispersion mode, choose var, mad, or mixed")

        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum NormMode {
    L1,
    L2,
}

impl NormMode {
    pub fn read(input: &str) -> NormMode {
        match input {
            "1" | "L1" | "l1" => NormMode::L1,
            "2" | "L2" | "l2" => NormMode::L2,
            _ => panic!("Not a valid norm, choose l1 or l2")
        }
    }

    pub fn int(self) -> i32 {
        match self {
            NormMode::L1 => 1,
            NormMode::L2 => 2,
        }
    }
}


fn read_matrix<V: SampleValue>(location:&str) -> Array2<V> {

    let mut count_array_file = File::open(location).expect("Count file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let (mut rows, mut columns) = (0,0);
    let mut padding = 0;
    if let Some(Ok(line)) = count_array_lines.next() {
        columns = line.split_whitespace().count();
        padding += 1;
    }
    rows = count_array_lines.by_ref().count()+padding;

    // let mut array = self.blank_array((rows,columns));
    let mut array = Array2::zeros((rows,columns));

    std::mem::drop(count_array_lines);

    count_array_file.seek(io::SeekFrom::Start(0)).expect("Count file error, seeking");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();


    for (i,line_res) in count_array_lines.by_ref().enumerate() {
        let line = line_res.expect("Readline error");
        // if i%200==0{print!("{}:",i);}
        for (j,string_value) in line.split_whitespace().enumerate() {
            // if i%200==0 && j%200 == 0 {print!(",{:?}", string_value.parse::<V>().unwrap_or(V::zero()) );}
            match string_value.parse::<V>() {
                Ok(numeric_value) => {
                    array[[i,j]] = numeric_value;
                },
                Err(_) => {
                        println!("Couldn't parse a cell in the text file");
                        println!("Cell content: {:?}", string_value);
                }
            }
        }
        // if i%200==0{print!("\r");}
    };
    println!("Read matrix:{:?}",array.dim());
    // println!("Returning array: {:?}",array);
    array
}

fn read_header(location: &str) -> Vec<String> {

    println!("Reading header: {}", location);

    let mut header_map = HashMap::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = io::BufReader::new(&header_file).lines();

    for (i,line) in header_file_iterator.by_ref().enumerate() {
        let str_representation  = line.unwrap_or("error".to_string());
        let mut renamed = str_representation.clone();
        let mut j = 1;
        while header_map.contains_key(&renamed) {
            renamed = [str_representation.clone(),j.to_string()].join("");
            eprintln!("WARNING: Two individual features were named the same thing: {}",str_representation);
            j += 1;
        }
        header_map.insert(renamed,i);
    };

    let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x|
        {
            if let Ok(feature) = x.0.parse::<String>() { (feature,*x.1) }
            else {panic!("{:?}",x.0)}
        }).collect();
    header_inter.sort_unstable_by_key(|x| x.1);
    let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

    println!("Read {} lines", header_vector.len());

    header_vector
}

//
// pub trait NumberMode
//
// #[derive(Debug,Clone,Copy,Serialize,Deserialize,PartialEq,Eq)]
// pub enum NumberMode {
//     I32,
//     F32,
//     F64,
// }
//
// impl NumberMode {
//     fn blank_array<V:SampleValue,D:ndarray::Dimension>(&self,dim:D) -> Array2<V> {
//         match self {
//             NumberMode::I32 => Array2::<i>::zeros(dim),
//         }
//     }
// }

//
// pub fn read_standard_in() -> Array2<> {
//
//     let stdin = io::stdin();
//     let count_array_pipe_guard = stdin.lock();
//
//     let mut count_array: Vec<Vec<f64>> = Vec::new();
//     // let mut samples = 0;
//
//     for (_i,line) in count_array_pipe_guard.lines().enumerate() {
//
//         // samples += 1;
//         let mut gene_vector = Vec::new();
//
//         for (_j,gene) in line.as_ref().expect("readline error").split_whitespace().enumerate() {
//
//             match gene.parse::<f64>() {
//                 Ok(exp_val) => {
//
//                     gene_vector.push(exp_val);
//
//                 },
//                 Err(msg) => {
//
//                     if gene != "nan" && gene != "NAN" {
//                         println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
//                         println!("Cell content: {:?}", gene);
//                     }
//                     gene_vector.push(f64::NAN);
//                 }
//             }
//
//         }
//
//         count_array.push(gene_vector);
//
//     };
//
//     // eprintln!("Counts read:");
//     // eprintln!("{:?}", counts);
//
//     matrix_flip(&count_array)
// }

fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
    out
}

// fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
//     let mut out = input.iter().cloned().enumerate().collect::<Vec<(usize,f64)>>();
//     out.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//     out
// }

fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {

    input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")

}

fn median(input: &Vec<f64>) -> (usize,f64) {
    let index;
    let value;

    let mut sorted_input = input.clone();
    sorted_input.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    if sorted_input.len() % 2 == 0 {
        index = sorted_input.len()/2;
        value = (sorted_input[index-1] + sorted_input[index]) / 2.
    }
    else {
        if sorted_input.len() % 2 == 1 {
            index = (sorted_input.len()-1)/2;
            value = sorted_input[index]
        }
        else {
            panic!("Median failed!");
        }
    }
    (index,value)
}

//
// mod manual_testing {
//
//     use super::*;
//
//     pub fn test_command_predict_full() {
//         let mut args = vec!["predict", "-m","branching","-b","tree.txt","-tg","tree.0","tree.1","tree.2","-c","counts.txt","-p","3","-o","./elsewhere/","-f","header_backup.txt"].into_iter().map(|x| x.to_string());
//
//         let command = Command::parse(&args.next().unwrap());
//
//         println!("{:?}",command);
//
//         // panic!();
//
//     }
//
// }

#[cfg(test)]
pub mod primary_testing {

    use super::*;
    //
    // #[test]
    // fn test_command_trivial() {
    //
    //     match Command::parse("construct") {
    //         Command::Construct => {},
    //         _ => panic!("Failed prediction parse")
    //     };
    //
    //     match Command::parse("predict") {
    //         Command::Predict => {},
    //         _ => panic!("Failed prediction parse")
    //     };
    //
    //     match Command::parse("combined") {
    //         Command::Combined => {},
    //         _ => panic!("Failed prediction parse")
    //     };
    //
    // }
    //
    // #[test]
    // #[should_panic]
    // fn test_command_wrong() {
    //     Command::parse("abc");
    // }
    //
    // #[test]
    // fn test_matrix_flip() {
    //     let mtx1 = vec![
    //         vec![0,1,2],
    //         vec![3,4,5],
    //         vec![6,7,8]
    //     ];
    //
    //     let mtx2 = vec![
    //         vec![0,3,6],
    //         vec![1,4,7],
    //         vec![2,5,8]
    //     ];
    //
    //     assert_eq!(matrix_flip(&mtx1),mtx2);
    //
    // }
    //
    // #[test]
    // fn test_pearsonr() {
    //     let vec1 = vec![1.,2.,3.,4.,5.];
    //     let vec2 = vec![2.,3.,4.,5.,6.];
    //
    //     println!("{:?}",pearsonr(&vec1,&vec2));
    //
    //     if (pearsonr(&vec1,&vec2)-1.) > 0.00001 {
    //         panic!("Correlation error")
    //     }
    // }

    // #[test]
    // fn test_parameters_args() {
    //     let mut args_iter = vec!["predict", "-m","branching","-b","tree.txt","-tg","tree.0","tree.1","tree.2","-c","testing/iris.drop","-p","3","-o","./elsewhere/","-f","header_backup.txt"].into_iter().map(|x| x.to_string());
    //
    //     let args = Parameters::read(&mut args_iter);
    //
    //     match args.prediction_mode.unwrap() {
    //         PredictionMode::Branch => {},
    //         _ => panic!("Branch mode not read correctly")
    //     }
    //
    //     assert_eq!(args.backup_vec.unwrap(), vec!["tree.0".to_string(),"tree.1".to_string(),"tree.2".to_string()]);
    //     assert_eq!(args.backups.unwrap(), "tree.txt".to_string());
    //
    //     assert_eq!(args.count_array_file, "counts.txt".to_string());
    //     assert_eq!(args.feature_header_file.unwrap(), "header_backup.txt".to_string());
    //     assert_eq!(args.sample_header_file, None);
    //     assert_eq!(args.report_address, "./elsewhere/".to_string());
    //
    //     assert_eq!(args.processor_limit.unwrap(), 3);
    //
    // }

    //
    // #[test]
    // fn test_read_counts_trivial() {
    //     assert_eq!(read_matrix("../testing/trivial.txt"),Vec::<Vec<f64>>::with_capacity(0))
    // }
    //
    // #[test]
    // fn test_read_counts_simple() {
    //     assert_eq!(read_matrix("../testing/simple.txt"), vec![vec![10.,5.,-1.,0.,-2.,10.,-3.,20.]])
    // }
    //
    // #[test]
    // fn test_read_header_trivial() {
    //     assert_eq!(read_header("../testing/trivial.txt"),Vec::<String>::with_capacity(0))
    // }
    //
    // #[test]
    // fn test_read_header_simple() {
    //     assert_eq!(read_header("../testing/iris.features"),vec!["petal_length","petal_width","sepal_length","sepal_width"])
    // }
    //



}
