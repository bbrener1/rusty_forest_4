// #![feature(test)]

// extern crate test;
// use test::Bencher;
//
#![allow(dead_code)]
#![allow(unused_assignments)]

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate ndarray;

extern crate serde;
extern crate serde_json;
extern crate num_cpus;
extern crate rand;
extern crate time;
extern crate smallvec;
extern crate rayon;

extern crate num_traits;

mod rank_vector;
mod rank_matrix;
mod io;
mod node;
mod nipals;
mod rv_dict;

use std::hash::Hash;
use std::cmp::{Eq,PartialOrd};
use std::fmt::{Debug};
use num_traits::{Zero,One,Num,NumCast,Pow,Bounded,Signed};
use std::str::FromStr;
use std::ops::{SubAssign,AddAssign};
use std::iter::Sum;
use std::sync::Arc;
use std::convert::Into;
use std::collections::BTreeMap;

use ndarray::prelude::*;
use ndarray::{LinalgScalar};

use crate::io::{ParameterBook};
use crate::node::{Node,ComputeNode,FastNode};
use std::env;

use rayon::prelude::*;

fn main() {
    let mut arg_iter = env::args();

    let parameters: ParameterBook<f64> = crate::io::read(&mut arg_iter);

    println!("Read parameters");

    let forest = ForestUF::from_parameters(parameters);
    let report_address = forest.parameters().report_address.clone();
    let tree_limit = forest.parameters().tree_limit;

    println!("Starting loop");

    (0..tree_limit)
        // .into_iter()
        .into_par_iter()
        .flat_map(|i| {
            let mut root = FastNode::from_forest(&forest);
            println!("Computing tree {:?}",i);
            // println!("#######################");

            let mut leaf_splits: Vec<(&mut FastNode<f64>,(SampleFilter<InputFeatureUF<f64>>,SampleFilter<InputFeatureUF<f64>>),f64)> = vec![];
            if let Some(root_split) = root.best_reduced_split(false,true) {
                leaf_splits.push(root_split)
            }
            else { return None }

            // println!("Initial candidates computed");
            //
            // for j in 0..forest.parameters().max_splits {
            //     if let Some((best_index,_)) = leaf_splits.iter().map(|(_,_,d)| d).argmax_v() {
            //         let (node,(left,right),_) = leaf_splits.remove(best_index);
            //         if let Some(stem) = node.split(left,right) {
            //             let (left_slice,right_slice) = stem.mut_children().split_at_mut(1);
            //             let left_split = left_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
            //             let right_split = right_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
            //             leaf_splits.extend(left_split);
            //             leaf_splits.extend(right_split);
            //         };
            //     };
            // }

            while leaf_splits.len() > 0 {
                let (node,(left,right),_) = leaf_splits.pop().unwrap();
                if let Some(stem) = node.split(left,right) {
                    // println!("depth:{:?}",stem.depth);
                    let (left_slice,right_slice) = stem.mut_children().split_at_mut(1);
                    if left_slice[0].depth < forest.parameters().depth_cutoff {
                        let left_split = left_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
                        let right_split = right_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
                        leaf_splits.extend(left_split);
                        leaf_splits.extend(right_split);
                    }
                };
            };

            // while leaf_splits.len() > 0 {
            //     let (node,(left,right),_) = leaf_splits.pop().unwrap();
            //     if let Some(stem) = node.split(left,right) {
            //         // println!("depth:{:?}",stem.depth);
            //         let (left_slice,right_slice) = stem.mut_children().split_at_mut(1);
            //         if left_slice[0].depth < forest.parameters().depth_cutoff {
            //             let left_split = left_slice[0].best_split();
            //             let right_split = right_slice[0].best_split();
            //             leaf_splits.extend(left_split);
            //             leaf_splits.extend(right_split);
            //         }
            //     };
            // };

            root.to_sidxn().dump(format!("{}.{}.compact",report_address,i).as_str());
            Some(())
        })
        .for_each(drop);
        // .collect::<Vec<()>>();


    // println!("CHILDREN:{:?}",children);

}

//
// KEY VALUE TRAITS
//

pub trait SampleKey: Hash + Eq + Copy + Clone + Debug {}

impl SampleKey for usize {}
impl SampleKey for &str {}

pub trait FeatureKey: Hash + Eq + Clone + Debug {}

impl FeatureKey for &str {}
impl FeatureKey for usize {}

// pub trait SampleValue: Num + Zero + FromStr + Clone + Copy + Into<f64> + LinalgScalar + Debug + PartialOrd + Send + Sync + SubAssign + AddAssign + Signed + FromPrimitive + Sum + Bounded + ToPrimitive + NumCast + Pow<u8,Output=Self> {}
pub trait SampleValue: Num + Zero + FromStr + Clone + Copy + Into<f64> + LinalgScalar + Debug + PartialOrd + Send + Sync + SubAssign + AddAssign + Signed + Sum + Bounded + NumCast + Pow<u8,Output=Self> {}

//
// SAMPLE/FEATURE TRAITS
//


impl SampleValue for f64 {}
impl SampleValue for f32 {}
impl SampleValue for i32 {}

pub trait DrawOrder<K:SampleKey>: Iterator<Item=K> {}

pub trait Sample: Clone + Debug + Send + Sync
{
    type Prototype: Prototype<Value=Self::Value>;
    type Key: SampleKey;
    type Value: SampleValue;

    fn from_index(index:usize,prototype:Arc<Self::Prototype>) -> Self;
    fn index(&self) -> usize;
    fn prototype(&self) -> &Arc<Self::Prototype>;

    fn output_slice(&self) -> ArrayView1<Self::Value>{
        self.prototype().output_array().slice(s![self.index(),..])
    }

    fn input_slice(&self) -> ArrayView1<Self::Value> {
        self.prototype().input_array().slice(s![self.index(),..])
    }

    fn output_feature<F:OutputFeature>(&self,feature:&F) -> Self::Value {
        self.output_slice()[feature.index()]
    }

    fn input_feature<F:InputFeature>(&self,feature:&F) -> Self::Value {
        self.input_slice()[feature.index()]
    }


}

pub trait Feature : Clone + Debug + Send + Sync
{
    type Prototype: Prototype<Value=Self::Value>;
    type Sample: Sample<Value=Self::Value>;
    type Key: FeatureKey;
    type Value: SampleValue;


    fn from_index(index:usize,prototype: Arc<Self::Prototype>) -> Self;
    fn index(&self) -> usize;
    fn prototype(&self) -> &Self::Prototype;
    fn slice(&self) -> ArrayView1<Self::Value>;

    fn sample(&self,sample:&Self::Sample) -> Self::Value {
        self.slice()[sample.index()]
    }

}


pub trait InputFeature : Feature {

    fn sorted_indices(&self) -> ArrayView1<usize> {
        self.prototype().sorted_index_array().slice(s![self.index(),..])
    }

    fn slice(&self) -> ArrayView1<Self::Value> {
        self.prototype().input_array().slice(s![..,self.index()])
    }

}

pub trait OutputFeature : Feature {
    fn slice(&self) -> ArrayView1<Self::Value> {
        self.prototype().output_array().slice(s![..,self.index()])
    }
}

#[derive(Clone,Debug)]
pub struct SampleFilter<IF:InputFeature> {
    reduction: Reduction<IF>,
    split: IF::Value,
    orientation: bool,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SerializedFilter {
    reduction: SerializedReduction,
    split: f64,
    orientation:bool,
}

impl<IF:InputFeature> SampleFilter<IF> {

    // NOTE AN EMPTY REDUCTION TRANSFORMS A SAMPLE TO A SCORE OF 0. BECAUSE OF THIS A BLANK FILTER
    // IS LEFT-ORIENTED (eg, all samples are <= 0)

    fn blank() -> SampleFilter<IF> {
        SampleFilter {
            reduction: Reduction::<IF>::blank(),
            split: IF::Value::zero(),
            orientation: false,
        }
    }

    fn from_reduction(reduction:Reduction<IF>,split:IF::Value) -> (SampleFilter<IF>,SampleFilter<IF>) {
        let left_filter = SampleFilter {
            reduction: reduction.clone(),
            split:split.clone(),
            orientation:false,
        };
        let right_filter = SampleFilter {
            reduction: reduction.clone(),
            split:split.clone(),
            orientation:true,
        };
        (left_filter,right_filter)
    }

    fn from_feature_sample(feature:&IF,sample:&IF::Sample) -> (SampleFilter<IF>,SampleFilter<IF>) {
        let reduction = Reduction::from_feature_sample(feature, sample);
        let split = reduction.transform_sample(sample);
        Self::from_reduction(reduction, split)
    }

    fn filter_samples(&self,samples:&[IF::Sample]) -> Vec<IF::Sample> {
        let mut new = Vec::with_capacity(samples.len());
        if self.orientation {
            for sample in samples {
                let compound_score = self.reduction.transform_sample(sample);
                if compound_score > self.split {
                    new.push(sample.clone())
                }
            }
        }
        else {
            for sample in samples {
                let compound_score = self.reduction.transform_sample(sample);
                if compound_score <= self.split {
                    new.push(sample.clone())
                }
            }
        }
        new
    }


    fn filter_samples_scaled(&self,samples:&[IF::Sample]) -> Vec<IF::Sample> {
        let mut new = Vec::with_capacity(samples.len());
        if self.orientation {
            for sample in samples {
                let compound_score = self.reduction.transform_sample_scaled(sample);
                if compound_score > self.split {
                    new.push(sample.clone())
                }
            }
        }
        else {
            for sample in samples {
                let compound_score = self.reduction.transform_sample_scaled(sample);
                if compound_score <= self.split {
                    new.push(sample.clone())
                }
            }
        }
        new
    }

    fn serialize(&self) -> SerializedFilter {
        SerializedFilter {
            reduction: self.reduction.serialize(),
            split: IF::Value::into(self.split),
            orientation: self.orientation,
        }
    }

}

#[derive(Clone,Debug)]
pub struct Reduction<F:Feature> {
    features: Vec<F>,
    scores:Vec<F::Value>,
    means:Vec<F::Value>,
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SerializedReduction {
    features: Vec<usize>,
    scores: Vec<f64>,
    means: Vec<f64>,
}

impl<F:Feature> Reduction<F> {

    fn blank() -> Reduction<F> {
        Reduction {
            features:vec![],
            scores:vec![],
            means:vec![],
        }
    }


    fn from_feature_sample(feature:&F,sample:&F::Sample) -> Self {
        Reduction {
            features:vec![feature.clone()],
            scores:vec![F::Value::one()],
            means:vec![F::Value::zero()],
        }
    }


    fn from(features:Vec<F>,scores:Vec<F::Value>,means:Vec<F::Value>) -> Self {
        Reduction {
            features:features,
            scores:scores,
            means:means,
        }
    }

    fn transform_sample(&self,sample:&F::Sample) -> F::Value {
        let mut compound_score = F::Value::zero();
        for i in 0..self.features.len() {
            compound_score += (self.features[i].sample(sample) - self.means[i]) * self.scores[i];
        }
        compound_score
    }

    fn transform_sample_scaled(&self,sample:&F::Sample) -> F::Value {
        let mut compound_score = F::Value::zero();
        let mut feature_sum = F::Value::zero();
        for i in 0..self.features.len() {
            let v = self.features[i].sample(sample);
            feature_sum += v;
            compound_score += (v - self.means[i]) * self.scores[i];
        }
        compound_score / feature_sum
    }

    fn serialize(&self) -> SerializedReduction {
        SerializedReduction {
            features: self.features.iter().map(|f| f.index()).collect(),
            scores: self.scores.iter().map(|s| F::Value::into(*s)).collect(),
            means: self.means.iter().map(|s| F::Value::into(*s)).collect(),
        }
    }

}

//
// PROTOTYPE TRAIT(S)
//

pub trait Prototype : Clone + Debug + Send + Sync
{
    type InputFeature: InputFeature<Prototype=Self,Value=Self::Value,Sample=Self::Sample>;
    type OutputFeature: OutputFeature<Prototype=Self,Value=Self::Value,Sample=Self::Sample>;
    type Sample: Sample<Prototype=Self,Value=Self::Value>;
    type Value: SampleValue;

    fn input_array(&self) -> &Array2<Self::Value>;
    fn output_array(&self) -> &Array2<Self::Value>;
    fn parameters(&self) -> &ParameterBook<Self::Value>;
    fn sorted_index_array(&self) -> &Array2<usize>;

    fn double_select_input(&self,samples:&[Self::Sample],features:&[Self::InputFeature]) -> Array2<Self::Value> {
        let mut selection = Array2::zeros((samples.len(),features.len()));
        for (i,si) in samples.iter().map(|s| s.index()).enumerate() {
            for (j,fj) in features.iter().map(|f| f.index()).enumerate() {
                selection[[i,j]] = self.input_array()[[si,fj]]
            }
        }
        selection
    }

    fn double_select_output(&self,samples:&[Self::Sample],features:&[Self::OutputFeature]) -> Array2<Self::Value> {
        let mut selection = Array2::zeros((samples.len(),features.len()));
        for (i,si) in samples.iter().map(|s| s.index()).enumerate() {
            for (j,fj) in features.iter().map(|f| f.index()).enumerate() {
                selection[[i,j]] = self.output_array()[[si,fj]]
            }
        }
        selection
    }
}

#[derive(Debug,Clone)]
pub struct ForestUF<V:SampleValue> {
    prototype:Arc<PrototypeUF<V>>,
    input_features:Vec<InputFeatureUF<V>>,
    output_features:Vec<OutputFeatureUF<V>>,
    samples:Vec<SampleUF<V>>,
}

impl<'a,V:SampleValue> Forest for ForestUF<V> {
    type Value = V;
    type Sample = SampleUF<V>;
    type InputFeature = InputFeatureUF<V>;
    type OutputFeature = OutputFeatureUF<V>;
    type Prototype = PrototypeUF<V>;

    fn from_parameters(parameters:ParameterBook<V>) -> Self {
        let prototype = Arc::new(PrototypeUF::from_parameters(parameters));
        let input_features = (0..prototype.parameters.input_feature_names.len()).map(|i| Self::InputFeature::from_index(i,prototype.clone())).collect::<Vec<Self::InputFeature>>();
        let output_features = (0..prototype.parameters.output_feature_names.len()).map(|i| Self::OutputFeature::from_index(i,prototype.clone())).collect();
        let samples = (0..prototype.parameters.sample_names.len()).map(|i| Self::Sample::from_index(i,prototype.clone())).collect();
        let forest =  ForestUF {
            prototype,
            input_features,
            output_features,
            samples,
        };

        forest
    }

    fn parameters(&self) -> &ParameterBook<Self::Value> {
        &self.prototype.parameters
    }
    fn prototype(&self) -> &Self::Prototype {
        &self.prototype
    }
    fn input_features(&self) -> &[Self::InputFeature] {
        &self.input_features
    }
    fn output_featues(&self) -> &[Self::OutputFeature] {
        &self.output_features
    }
    fn samples(&self) -> &[Self::Sample] {
        &self.samples
    }
}

pub trait Forest: Send + Sync {
    type Value: SampleValue;
    type Sample: Sample<Prototype=Self::Prototype,Value=Self::Value>;
    type InputFeature: InputFeature<Prototype=Self::Prototype,Value=Self::Value,Sample=Self::Sample>;
    type OutputFeature: OutputFeature<Prototype=Self::Prototype,Value=Self::Value,Sample=Self::Sample>;
    type Prototype: Prototype<Value=Self::Value,Sample=Self::Sample,InputFeature=Self::InputFeature,OutputFeature=Self::OutputFeature>;

    fn from_parameters(ParameterBook<Self::Value>) -> Self;
    fn parameters(&self) -> &ParameterBook<Self::Value>;
    fn prototype(&self) -> &Self::Prototype;
    fn input_features(&self) -> &[Self::InputFeature];
    fn output_featues(&self) -> &[Self::OutputFeature];
    fn samples(&self) -> &[Self::Sample];

    fn subsample_input_features(&self) -> Vec<Self::InputFeature> {
        let draws = self.parameters().input_feature_subsample;
        let mut new_collection: Vec<Self::InputFeature> = Vec::with_capacity(draws);
        let mut sampler = bounded_sampler(self.input_features().len());
        for _ in 0..draws {
            new_collection.push(self.input_features()[sampler()].clone());
        }
        new_collection
    }


    fn subsample_output_features(&self) -> Vec<Self::OutputFeature> {
        let draws = self.parameters().output_feature_subsample;
        let mut new_collection: Vec<Self::OutputFeature> = Vec::with_capacity(draws);
        let mut sampler = bounded_sampler(self.output_featues().len());
        for _ in 0..draws {
            new_collection.push(self.output_featues()[sampler()].clone());
        }
        new_collection
    }

    fn subsample_samples(&self) -> Vec<&Self::Sample> {
        let draws = self.parameters().sample_subsample;
        let mut new_collection: Vec<&Self::Sample> = Vec::with_capacity(draws);
        let mut sampler = bounded_sampler(self.samples().len());
        for _ in 0..draws {
            new_collection.push(&self.samples()[sampler()]);
        }
        new_collection
    }



}

#[derive(Clone,Debug)]
pub struct PrototypeUF<V:SampleValue> {
    inputs: Array2<V>,
    outputs: Array2<V>,
    sorted_input_indices: Array2<usize>,
    parameters: ParameterBook<V>,
}

impl<V:SampleValue> PrototypeUF<V> {
    fn from_parameters(mut parameters: ParameterBook<V>) -> PrototypeUF<V> {
        let inputs = parameters.input_array.take().unwrap();
        let outputs = parameters.output_array.take().unwrap();
        let mut sorted_input_indices: Array2<usize> = Array2::zeros((inputs.dim().1,inputs.dim().0));
        for (i,feature) in inputs.axis_iter(Axis(1)).enumerate() {
            let valsorted: Vec<(usize,&V)> = valsort(feature.iter());
            for (j,(si,_)) in valsorted.into_iter().enumerate() {
                sorted_input_indices[[i,j]]=si;
            }
            // sorted_input_indices.row_mut(i).assign(&valsorted.iter().map(|x| x.0).collect());
        }

        PrototypeUF {
            inputs,
            outputs,
            sorted_input_indices,
            parameters: parameters,
        }
    }
}

impl<'a,V:SampleValue> Prototype for PrototypeUF<V> {
    type InputFeature = InputFeatureUF<V>;
    type OutputFeature = OutputFeatureUF<V>;
    type Sample = SampleUF<V>;
    type Value = V;

    fn input_array(&self) -> &Array2<Self::Value> {
        &self.inputs
    }
    fn output_array(&self) -> &Array2<Self::Value> {
        &self.outputs
    }

    fn parameters(&self) -> &ParameterBook<Self::Value> {
        &self.parameters
    }

    fn sorted_index_array(&self) -> &Array2<usize> {
        &self.sorted_input_indices
    }

}

#[derive(Clone)]
pub struct SampleUF<V:SampleValue> {
    index: usize,
    prototype: Arc<PrototypeUF<V>>
}

impl<V:SampleValue> Sample for SampleUF<V> {

    type Prototype = PrototypeUF<Self::Value>;
    type Key = usize;
    type Value = V;

    fn from_index(index:usize,prototype:Arc<PrototypeUF<Self::Value>>) -> SampleUF<V> {
        SampleUF {
            index:index,
            prototype:prototype,
        }
    }

    fn index(&self) -> usize {
        self.index
    }

    fn prototype(&self) -> &Arc<Self::Prototype> {
        &self.prototype
    }

}



#[derive(Clone)]
pub struct InputFeatureUF<V:SampleValue> {
    index: usize,
    prototype: Arc<PrototypeUF<V>>
}

impl<'a,V:SampleValue> Feature for InputFeatureUF<V> {
    type Prototype = PrototypeUF<V>;
    type Sample = SampleUF<V>;
    type Key = usize;
    type Value = V;

    fn from_index(index:usize,prototype:Arc<PrototypeUF<V>>) -> InputFeatureUF<V> {
        InputFeatureUF {
            index: index,
            prototype: prototype,
        }
    }

    fn index(&self) -> usize {
        self.index
    }
    fn prototype(&self) -> &Self::Prototype {
        &self.prototype
    }

    fn slice(&self) -> ArrayView1<Self::Value> {
        InputFeature::slice(self)
    }
}

impl<'a,V:SampleValue> InputFeature for InputFeatureUF<V> {}

#[derive(Clone)]
pub struct OutputFeatureUF<V: SampleValue> {
    index: usize,
    prototype: Arc<PrototypeUF<V>>
}
//
// impl<'a,V:SampleValue> OutputFeatureUF<V> {
//     fn from(index:usize,prototype: &'a PrototypeUF<V>) -> OutputFeatureUF<V> {
//         OutputFeatureUF {
//             index: index,
//             prototype: prototype,
//         }
//     }
// }

impl<'a,V:SampleValue> Feature for OutputFeatureUF<V> {
    type Prototype = PrototypeUF<V>;
    type Sample = SampleUF<V>;
    type Key = usize;
    type Value = V;

    fn from_index(index:usize,prototype: Arc<PrototypeUF<V>>) -> Self {
        OutputFeatureUF {
            index: index,
            prototype: prototype,
        }
    }
    fn index(&self) -> usize {
        self.index
    }
    fn prototype(&self) -> &Self::Prototype {
        &self.prototype
    }

    fn slice(&self) -> ArrayView1<Self::Value> {
        OutputFeature::slice(self)
    }
}

impl<'a,V:SampleValue> OutputFeature for OutputFeatureUF<V> {}



impl<V:SampleValue> Debug for SampleUF<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SampleUF {{ index: {:?} }}",self.index)
    }
}

impl<V:SampleValue> Debug for InputFeatureUF<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InputFeatureUF {{ index: {:?} }}",self.index)
    }
}

impl<V:SampleValue> Debug for OutputFeatureUF<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OutputFeatureUF {{ index: {:?} }}",self.index)
    }
}

fn valsort<I: Iterator<Item=T>,T: PartialOrd + Clone>(s:I) -> Vec<(usize,T)>{
    let mut paired: Vec<(usize,T)> = s.map(|t| t.clone()).enumerate().collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    paired
}

fn bounded_sampler(bound:usize) -> Box<dyn FnMut() -> usize> {
    use rand::distributions::{Distribution, Uniform};
    let between = Uniform::from(0..bound);
    let mut rng = rand::thread_rng();
    let sample = move || between.sample(&mut rng);
    Box::new(sample)
}

fn subsample<T:Clone>(collection:&[T],draws:usize) -> Vec<T> {
    let mut new_collection: Vec<T> = Vec::with_capacity(draws);
    let mut sampler = bounded_sampler(collection.len());
    for _ in 0..draws {
        new_collection.push(collection[sampler()].clone());
    }
    new_collection
}

fn logistic<T:NumCast>(input:T) -> f64 {
    let f_cast: f64 = NumCast::from(input).unwrap();
    1./ (1. + (std::f64::consts::E).powf(-1. * f_cast))
}

use std::cmp::Ordering;

pub fn argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = maximum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Less => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Greater => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        maximum = check;

    };
    maximum.map(|(i,_)| i)
}


pub fn argmax_v<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<(usize,U)> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = maximum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Less => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Greater => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        maximum = check;

    };
    maximum
}


pub fn argmin<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut minimum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = minimum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Greater => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Less => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        minimum = check;

    };
    minimum.map(|(i,_)| i)
}


pub fn argmin_v<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<(usize,U)> {
    let mut minimum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = minimum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Greater => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Less => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        minimum = check;

    };
    minimum
}

pub trait ArgMinMax<I:PartialOrd+PartialEq> : Iterator<Item=I> + Sized {

    fn argmax(self) -> Option<usize> {
        argmax(self)
    }

    fn argmax_v(self) -> Option<(usize,I)> {
        argmax_v(self)
    }

    fn argmin(self) -> Option<usize> {
        argmin(self)
    }

    fn argmin_v(self) -> Option<(usize,I)> {
        argmin_v(self)
    }
}

impl<I:Iterator<Item=IT>,IT:PartialOrd+PartialEq> ArgMinMax<IT> for I {}

#[cfg(test)]
mod main_tests {

    use crate::iris_array;
    use super::*;

    #[test]
    fn main_test() {
        let iris = iris_array().t().to_owned();
        let book = ParameterBook::<f64>::test(iris);
        let forest = ForestUF::from_parameters(book);


        (0..10)
            // .into_iter()
            .into_par_iter()
            .flat_map(|i| {
                let mut root = FastNode::from_forest(&forest);
                println!("Computing tree {:?}",i);
                // println!("#######################");

                let mut leaf_splits: Vec<(&mut FastNode<f64>,(SampleFilter<InputFeatureUF<f64>>,SampleFilter<InputFeatureUF<f64>>),f64)> = vec![];
                if let Some(root_split) = root.best_reduced_split(false,true) {
                    leaf_splits.push(root_split)
                }
                else { return None }

                // println!("Initial candidates computed");
                //
                // for j in 0..forest.parameters().max_splits {
                //     if let Some((best_index,_)) = leaf_splits.iter().map(|(_,_,d)| d).argmax_v() {
                //         let (node,(left,right),_) = leaf_splits.remove(best_index);
                //         if let Some(stem) = node.split(left,right) {
                //             let (left_slice,right_slice) = stem.mut_children().split_at_mut(1);
                //             let left_split = left_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
                //             let right_split = right_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
                //             leaf_splits.extend(left_split);
                //             leaf_splits.extend(right_split);
                //         };
                //     };
                // }

                while leaf_splits.len() > 0 {
                    let (node,(left,right),_) = leaf_splits.pop().unwrap();
                    if let Some(stem) = node.split(left,right) {
                        // println!("depth:{:?}",stem.depth);
                        let (left_slice,right_slice) = stem.mut_children().split_at_mut(1);
                        if left_slice[0].depth < forest.parameters().depth_cutoff {
                            let left_split = left_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
                            let right_split = right_slice[0].best_reduced_split(forest.parameters().reduce_input,forest.parameters().reduce_output);
                            leaf_splits.extend(left_split);
                            leaf_splits.extend(right_split);
                        }
                    };
                };

                // while leaf_splits.len() > 0 {
                //     let (node,(left,right),_) = leaf_splits.pop().unwrap();
                //     if let Some(stem) = node.split(left,right) {
                //         // println!("depth:{:?}",stem.depth);
                //         let (left_slice,right_slice) = stem.mut_children().split_at_mut(1);
                //         if left_slice[0].depth < forest.parameters().depth_cutoff {
                //             let left_split = left_slice[0].best_split();
                //             let right_split = right_slice[0].best_split();
                //             leaf_splits.extend(left_split);
                //             leaf_splits.extend(right_split);
                //         }
                //     };
                // };

                Some(())
            })
            .for_each(drop);
            // .collect::<Vec<()>>();


        // println!("CHILDREN:{:?}",children);
    }

}


pub fn iris_array() -> Array2<f64> {
    array![[5.1,3.5,1.4,0.2],
    [4.9,3.0,1.4,0.2],
    [4.7,3.2,1.3,0.2],
    [4.6,3.1,1.5,0.2],
    [5.0,3.6,1.4,0.2],
    [5.4,3.9,1.7,0.4],
    [4.6,3.4,1.4,0.3],
    [5.0,3.4,1.5,0.2],
    [4.4,2.9,1.4,0.2],
    [4.9,3.1,1.5,0.1],
    [5.4,3.7,1.5,0.2],
    [4.8,3.4,1.6,0.2],
    [4.8,3.0,1.4,0.1],
    [4.3,3.0,1.1,0.1],
    [5.8,4.0,1.2,0.2],
    [5.7,4.4,1.5,0.4],
    [5.4,3.9,1.3,0.4],
    [5.1,3.5,1.4,0.3],
    [5.7,3.8,1.7,0.3],
    [5.1,3.8,1.5,0.3],
    [5.4,3.4,1.7,0.2],
    [5.1,3.7,1.5,0.4],
    [4.6,3.6,1.0,0.2],
    [5.1,3.3,1.7,0.5],
    [4.8,3.4,1.9,0.2],
    [5.0,3.0,1.6,0.2],
    [5.0,3.4,1.6,0.4],
    [5.2,3.5,1.5,0.2],
    [5.2,3.4,1.4,0.2],
    [4.7,3.2,1.6,0.2],
    [4.8,3.1,1.6,0.2],
    [5.4,3.4,1.5,0.4],
    [5.2,4.1,1.5,0.1],
    [5.5,4.2,1.4,0.2],
    [4.9,3.1,1.5,0.1],
    [5.0,3.2,1.2,0.2],
    [5.5,3.5,1.3,0.2],
    [4.9,3.1,1.5,0.1],
    [4.4,3.0,1.3,0.2],
    [5.1,3.4,1.5,0.2],
    [5.0,3.5,1.3,0.3],
    [4.5,2.3,1.3,0.3],
    [4.4,3.2,1.3,0.2],
    [5.0,3.5,1.6,0.6],
    [5.1,3.8,1.9,0.4],
    [4.8,3.0,1.4,0.3],
    [5.1,3.8,1.6,0.2],
    [4.6,3.2,1.4,0.2],
    [5.3,3.7,1.5,0.2],
    [5.0,3.3,1.4,0.2],
    [7.0,3.2,4.7,1.4],
    [6.4,3.2,4.5,1.5],
    [6.9,3.1,4.9,1.5],
    [5.5,2.3,4.0,1.3],
    [6.5,2.8,4.6,1.5],
    [5.7,2.8,4.5,1.3],
    [6.3,3.3,4.7,1.6],
    [4.9,2.4,3.3,1.0],
    [6.6,2.9,4.6,1.3],
    [5.2,2.7,3.9,1.4],
    [5.0,2.0,3.5,1.0],
    [5.9,3.0,4.2,1.5],
    [6.0,2.2,4.0,1.0],
    [6.1,2.9,4.7,1.4],
    [5.6,2.9,3.6,1.3],
    [6.7,3.1,4.4,1.4],
    [5.6,3.0,4.5,1.5],
    [5.8,2.7,4.1,1.0],
    [6.2,2.2,4.5,1.5],
    [5.6,2.5,3.9,1.1],
    [5.9,3.2,4.8,1.8],
    [6.1,2.8,4.0,1.3],
    [6.3,2.5,4.9,1.5],
    [6.1,2.8,4.7,1.2],
    [6.4,2.9,4.3,1.3],
    [6.6,3.0,4.4,1.4],
    [6.8,2.8,4.8,1.4],
    [6.7,3.0,5.0,1.7],
    [6.0,2.9,4.5,1.5],
    [5.7,2.6,3.5,1.0],
    [5.5,2.4,3.8,1.1],
    [5.5,2.4,3.7,1.0],
    [5.8,2.7,3.9,1.2],
    [6.0,2.7,5.1,1.6],
    [5.4,3.0,4.5,1.5],
    [6.0,3.4,4.5,1.6],
    [6.7,3.1,4.7,1.5],
    [6.3,2.3,4.4,1.3],
    [5.6,3.0,4.1,1.3],
    [5.5,2.5,4.0,1.3],
    [5.5,2.6,4.4,1.2],
    [6.1,3.0,4.6,1.4],
    [5.8,2.6,4.0,1.2],
    [5.0,2.3,3.3,1.0],
    [5.6,2.7,4.2,1.3],
    [5.7,3.0,4.2,1.2],
    [5.7,2.9,4.2,1.3],
    [6.2,2.9,4.3,1.3],
    [5.1,2.5,3.0,1.1],
    [5.7,2.8,4.1,1.3],
    [6.3,3.3,6.0,2.5],
    [5.8,2.7,5.1,1.9],
    [7.1,3.0,5.9,2.1],
    [6.3,2.9,5.6,1.8],
    [6.5,3.0,5.8,2.2],
    [7.6,3.0,6.6,2.1],
    [4.9,2.5,4.5,1.7],
    [7.3,2.9,6.3,1.8],
    [6.7,2.5,5.8,1.8],
    [7.2,3.6,6.1,2.5],
    [6.5,3.2,5.1,2.0],
    [6.4,2.7,5.3,1.9],
    [6.8,3.0,5.5,2.1],
    [5.7,2.5,5.0,2.0],
    [5.8,2.8,5.1,2.4],
    [6.4,3.2,5.3,2.3],
    [6.5,3.0,5.5,1.8],
    [7.7,3.8,6.7,2.2],
    [7.7,2.6,6.9,2.3],
    [6.0,2.2,5.0,1.5],
    [6.9,3.2,5.7,2.3],
    [5.6,2.8,4.9,2.0],
    [7.7,2.8,6.7,2.0],
    [6.3,2.7,4.9,1.8],
    [6.7,3.3,5.7,2.1],
    [7.2,3.2,6.0,1.8],
    [6.2,2.8,4.8,1.8],
    [6.1,3.0,4.9,1.8],
    [6.4,2.8,5.6,2.1],
    [7.2,3.0,5.8,1.6],
    [7.4,2.8,6.1,1.9],
    [7.9,3.8,6.4,2.0],
    [6.4,2.8,5.6,2.2],
    [6.3,2.8,5.1,1.5],
    [6.1,2.6,5.6,1.4],
    [7.7,3.0,6.1,2.3],
    [6.3,3.4,5.6,2.4],
    [6.4,3.1,5.5,1.8],
    [6.0,3.0,4.8,1.8],
    [6.9,3.1,5.4,2.1],
    [6.7,3.1,5.6,2.4],
    [6.9,3.1,5.1,2.3],
    [5.8,2.7,5.1,1.9],
    [6.8,3.2,5.9,2.3],
    [6.7,3.3,5.7,2.5],
    [6.7,3.0,5.2,2.3],
    [6.3,2.5,5.0,1.9],
    [6.5,3.0,5.2,2.0],
    [6.2,3.4,5.4,2.3],
    [5.9,3.0,5.1,1.8]]
}
