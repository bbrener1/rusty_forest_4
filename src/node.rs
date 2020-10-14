
use std::cmp::PartialOrd;
use std::f64;
use std::collections::{HashMap};
use serde_json;

extern crate rand;

use rayon::prelude::*;

use crate::{InputFeature,OutputFeature,Forest,Reduction,Feature};
use crate::Sample;
use crate::SampleValue;
use crate::Prototype;
use crate::SampleFilter;
use crate::subsample;
// use crate::io::DispersionMode;
use crate::io::ParameterBook;
use crate::rank_vector::{SegmentedVector,FeatureVector};
use crate::valsort;
use crate::rank_vector::MedianArray;
use crate::ArgMinMax;
use nipals::Projector;
use ndarray::{Array1,Array2};
use rank_matrix::split as mtx_split;

use crate::{PrototypeUF,InputFeatureUF,OutputFeatureUF,SampleUF,ForestUF,SerializedFilter};

pub trait Node<'a> : Clone
{
    type Forest: Forest<Prototype=Self::Prototype,Value=Self::Value,Sample=Self::Sample,InputFeature=Self::InputFeature,OutputFeature=Self::OutputFeature>;
    type Prototype: Prototype<Value=Self::Value,Sample=Self::Sample,InputFeature=Self::InputFeature,OutputFeature=Self::OutputFeature>;
    type Value: SampleValue;
    type Sample: Sample<Prototype=Self::Prototype,Value=Self::Value>;
    type InputFeature: InputFeature<Prototype=Self::Prototype,Sample=Self::Sample,Value=Self::Value>;
    type OutputFeature: OutputFeature<Prototype=Self::Prototype,Value=Self::Value,Sample=Self::Sample>;

    fn from_forest(forest:&'a Self::Forest) -> Self;
    fn filter(&self) -> &SampleFilter<Self::InputFeature>;
    fn forest(&self) -> &Self::Forest;
    fn samples(&self) -> & [Self::Sample];
    fn samples_mut(&mut self) -> & mut [Self::Sample];
    fn prototype(&self) -> &Self::Prototype;
    fn children(&self) -> & Vec<Self>;
    fn mut_children(&mut self) -> &mut Vec<Self>;
    fn parameters(&self) -> &'a ParameterBook<Self::Value>;

    fn to_sidxn(&self) -> SampleIndexNode {
        let samples = self.samples().iter().map(|s| s.index()).collect();
        let children = self.children().iter().map(|c| c.to_sidxn()).collect();
        SampleIndexNode{
            samples,
            filter: self.filter().serialize().clone(),
            children
        }
    }

}

pub trait ComputeNode<'a>: Node<'a>
{

    fn derive(&self,SampleFilter<Self::InputFeature>) -> Option<Self>;
    fn derive_scaled(&self,SampleFilter<Self::InputFeature>) -> Option<Self>;

    fn subsample(&self) -> Option<(Vec<Self::InputFeature>,Vec<Self::OutputFeature>,Vec<Self::Sample>,Array2<Self::Value>,Array2<Self::Value>)> {
        let input_feature_subsample: Vec<Self::InputFeature> = self.forest().subsample_input_features();
        let output_feature_subsample: Vec<Self::OutputFeature> = self.forest().subsample_output_features();
        let (in_bag,out_bag) = self.sample_bags();
        let sample_subsample = subsample(&in_bag, self.forest().parameters().sample_subsample);

        if in_bag.len() < 3 {
            return None
        }

        let input_intermediate = self.prototype().double_select_input(&sample_subsample,&input_feature_subsample);
        let output_intermediate = self.prototype().double_select_output(&sample_subsample,&output_feature_subsample);

        Some((
            input_feature_subsample,
            output_feature_subsample,
            sample_subsample,
            input_intermediate,
            output_intermediate
        ))
    }

    fn best_split(&mut self) -> Option<(&mut Self, (SampleFilter<Self::InputFeature>,SampleFilter<Self::InputFeature>),f64)> {

        let (
                input_feature_subsample,
                output_feature_subsample,
                sample_subsample,
                input_intermediate,
                output_intermediate
            ) = self.subsample()?;

        let (best_feature_index,best_sample_index,best_dispersion) =
            mtx_split(
                &input_intermediate,
                &output_intermediate,
                self.parameters().split_fraction_regularization,
                self.parameters().norm_mode.int()
            )?;

        let (best_feature,best_sample) = (
            input_feature_subsample[best_feature_index].clone(),
            sample_subsample[best_sample_index].clone()
        );


        if best_sample_index < 2 || best_sample_index > (sample_subsample.len() - 2) {
            return None
        }

        let (left_filter,right_filter) = SampleFilter::from_feature_sample(&best_feature, &best_sample);


        Some((self,(left_filter,right_filter),best_dispersion))
    }

    fn best_reduced_split(&mut self,reduce_input:bool,reduce_output:bool) -> Option<(&mut Self, (SampleFilter<Self::InputFeature>,SampleFilter<Self::InputFeature>),f64)> {
        use num_traits::{NumCast};

        // println!("Reducing");

        let mut input_reduction: Option<Reduction<Self::InputFeature>> = None;

        let (
                input_feature_subsample,
                output_feature_subsample,
                sample_subsample,
                input_intermediate,
                output_intermediate
            ) = self.subsample()?;


        let reduced_input =
            if reduce_input {

                let (
                    mut reduced_input,
                    input_scores,
                    input_means,
                    input_scales
                ) = Projector::from(input_intermediate).calculate_n_projections(1)?;

                if self.parameters().scaling {
                    reduced_input /= &input_scales;
                };
                input_reduction =
                    Some(Reduction::from(
                       input_feature_subsample.clone(),
                       input_scores.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect(),
                       input_means.into_iter().map(|s| NumCast::from(*s).expect("Cast failure")).collect()
                   ));
                reduced_input.t().to_owned()
            }
            else {
                input_intermediate.mapv(|v| NumCast::from(v).expect("Cast failure"))
            };

        let reduced_output =
            if reduce_output {
                let (mut r_o,_,_,o_s) =
                    Projector::from(output_intermediate)
                        .calculate_n_projections(self.parameters().braid_thickness)?;
                if self.parameters().scaling {
                    r_o /= &o_s;
                };
                r_o.t().to_owned()
            }
            else {
                output_intermediate.mapv(|v| NumCast::from(v).expect("Cast failure"))
            };

        // println!("Features reduced");

        let (best_feature_index,best_sample_index,best_dispersion) =
            mtx_split(
                &reduced_input,
                &reduced_output,
                self.parameters().split_fraction_regularization,
                self.parameters().norm_mode.int()
            )?;

        if best_sample_index < 2 || best_sample_index > (sample_subsample.len() - 2) {
            return None
        }

        let (best_feature,best_sample) = (
            input_feature_subsample[best_feature_index].clone(),
            sample_subsample[best_sample_index].clone()
        );

        // println!("Computing filters");

        let (left_filter,right_filter) =
            if reduce_input {
                let ir_u = input_reduction?;
                let split = ir_u.transform_sample_scaled(&best_sample);
                SampleFilter::from_reduction(ir_u, split)
            }
            else {
                SampleFilter::from_feature_sample(&best_feature, &best_sample)
            };

        // println!("Filters done, returning");

        Some((self,(left_filter,right_filter),best_dispersion))
    }

    fn split(&mut self,left_filter:SampleFilter<Self::InputFeature>,right_filter:SampleFilter<Self::InputFeature>) -> Option<&mut Self> {

        // println!("Deriving");

        if let (Some(left_child),Some(right_child)) = (self.derive(left_filter),self.derive(right_filter)) {

            self.mut_children().push(left_child);
            self.mut_children().push(right_child);

            Some(self)

        }
        else {
            // println!("Failed to derive");
            None }
    }


    fn check_predictions(filter:&SampleFilter<Self::InputFeature>,output_reduction:&Reduction<Self::OutputFeature>,in_bag:&[Self::Sample],out_bag:&[Self::Sample])
    {
        let filtered_inbag = filter.filter_samples(in_bag);
        let filtered_outbag = filter.filter_samples(out_bag);

        let reduced_inbag: Array1<Self::Value> = filtered_inbag.iter().map(|s| output_reduction.transform_sample(s)).collect();
        let sorted_reduced_inbag:Vec<(usize,Self::Value)> = valsort(reduced_inbag.to_vec().into_iter());
        let mv = MedianArray::link(&sorted_reduced_inbag);
        let prediction = mv.central_tendency();

        let reduced_outbag: Array1<Self::Value> = filtered_outbag.iter().map(|s| output_reduction.transform_sample(s)).collect();

    }

    fn sample_bags(&self) -> (Vec<Self::Sample>,Vec<Self::Sample>) {
        use rand::seq::SliceRandom;
        use rand::prelude::thread_rng;

        let mut in_bag: Vec<Self::Sample> = self.samples().to_vec();
        let bag_size = (in_bag.len() as f64 * 0.66) as usize;
        &mut in_bag.partial_shuffle(&mut thread_rng(),bag_size);
        let out_bag = in_bag.split_off(bag_size);
        (in_bag,out_bag)
    }

}

#[derive(Clone,Debug,Serialize)]
pub struct SampleIndexNode {
    samples: Vec<usize>,
    filter: SerializedFilter,
    children: Vec<SampleIndexNode>,
}

impl SampleIndexNode
{
    pub fn dump(&mut self,filename:&str) -> Option<()> {
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut handle = OpenOptions::new().write(true).truncate(true).create(true).open(filename).ok()?;
        handle.write(serde_json::to_string(self).ok()?.as_bytes()).ok()?;
        Some(())
    }
}

#[derive(Clone,Debug)]
pub struct FastNode<'a,V:SampleValue> {
    forest: &'a ForestUF<V>,
    parameters: &'a ParameterBook<V>,
    samples: Vec<SampleUF<V>>,
    filter: SampleFilter<InputFeatureUF<V>>,
    children: Vec<FastNode<'a,V>>,
    pub depth: usize,
}

impl<'a,V:SampleValue> Node<'a> for FastNode<'a,V> {
    type Forest = ForestUF<V>;
    type Prototype = PrototypeUF<V>;
    type Value = V;
    type Sample = SampleUF<V>;
    type InputFeature = InputFeatureUF<V>;
    type OutputFeature = OutputFeatureUF<V>;

    fn from_forest(forest:&'a ForestUF<V>) -> FastNode<V> {
        FastNode {
            samples: forest.samples().to_vec(),
            forest: forest,
            parameters: forest.parameters(),
            filter: SampleFilter::<InputFeatureUF<V>>::blank(),
            children: vec![],
            depth: 0,
        }
    }

    fn filter(&self) -> &SampleFilter<Self::InputFeature> {
        &self.filter
    }

    fn samples(&self) -> & [SampleUF<V>] {
        &self.samples
    }

    fn samples_mut(&mut self) -> &mut [SampleUF<V>] {
        &mut self.samples
    }

    fn children(&self) -> &Vec<Self> {
        &self.children
    }

    fn mut_children(&mut self) -> &mut Vec<Self> {
        &mut self.children
    }

    fn forest(&self) -> &ForestUF<V> {
        &self.forest
    }

    fn parameters(&self) -> &'a ParameterBook<V> {
        &self.parameters
    }

    fn prototype(&self) -> &PrototypeUF<V> {
        self.forest.prototype()
    }
}

impl<'a,V:SampleValue> ComputeNode<'a> for FastNode<'a,V> {
    fn derive(&self,filter:SampleFilter<InputFeatureUF<V>>) -> Option<FastNode<'a,V>> {
        if self.depth >= self.forest.parameters().depth_cutoff { return None };
        let new_samples = filter.filter_samples(&self.samples);
        if new_samples.len() > 2 {
            Some(FastNode {
                samples: new_samples,
                forest: self.forest,
                parameters: self.parameters,
                filter: filter,
                children: vec![],
                depth: self.depth + 1,
            })
        }
        else { None }
    }

    fn derive_scaled(&self,filter:SampleFilter<InputFeatureUF<V>>) -> Option<FastNode<'a,V>> {
        if self.depth >= self.forest.parameters().depth_cutoff { return None };
        let new_samples = filter.filter_samples_scaled(&self.samples);
        if new_samples.len() > 2 {
            Some(FastNode {
                samples: new_samples,
                forest: self.forest,
                parameters: self.parameters,
                filter: filter,
                children: vec![],
                depth: self.depth + 1,
            })
        }
        else { None }
    }
}

#[derive(Clone,Debug)]
struct CachedSorter {
    subsample: Vec<usize>,
    stencil: HashMap<usize,(usize,usize)>,
    draw_order: Vec<usize>,
}

impl CachedSorter {
    fn from(subsample:Vec<usize>) -> Self {

        // First we count how many times each key occurrs in the subsample;

        let mut stencil_map: HashMap<usize,(usize,usize)> = HashMap::with_capacity(subsample.len());
        for i in subsample.iter() {
            stencil_map.entry(*i).or_insert((0,0)).1 += 1;
        }
        CachedSorter {
            draw_order: vec![subsample.len();subsample.len()],
            subsample:subsample,
            stencil: stencil_map,
         }
    }

    fn sort<'a,I:ExactSizeIterator<Item=&'a usize>>(&mut self,sorted_indices:I) {

        // For a given sort order, we must now determine the minimum rank of each present index.
        // We can do this by iterating through the sorted indices and keeping track of the quantity
        // seen so far.

        let mut current_subsample_rank = 0;
        for key in sorted_indices {
            if let Some((minimum_rank,quantity)) = self.stencil.get_mut(&(key)) {
                *minimum_rank = current_subsample_rank;
                current_subsample_rank += *quantity;
            }
        }

        // For each sample key present in the subsample we now know the minimum rank and quantity.
        // We proceed through the subsample and insert the subsample index at a the minimum rank of
        // that subsample


        for (ss_index,key) in self.subsample.iter().enumerate() {
            if let Some((minimum_rank,_)) = self.stencil.get_mut(key) {
                self.draw_order[*minimum_rank] = ss_index;
                *minimum_rank += 1;
            }
        }
    }

    fn draw_order(&self) -> &[usize] {
        &self.draw_order
    }
}
