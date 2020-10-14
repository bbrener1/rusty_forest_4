use ndarray::prelude::*;
use crate::rank_vector::{FeatureVector,SegmentedVector};
use crate::{SampleKey,SampleValue,DrawOrder};
use crate::rank_vector::{MedianArray,MedianVector};
use crate::valsort;
use crate::ArgMinMax;
use num_traits::NumCast;
use rayon::prelude::*;

pub fn split<V1:SampleValue,V2:SampleValue>(input:&Array2<V1>,output:&Array2<V2>,sfr:f64,l:i32) -> Option<(usize,usize,f64)> {

    // println!("Sorting outputs");

    let mut output_vectors: Vec<MedianArray<V2>> =
        output.axis_iter(Axis(1))
        .into_par_iter()
        .map(|column| {
            let valsorted = valsort(column.iter().cloned());
            MedianArray::<V2>::link(&valsorted)
            // println!("O:{:?}",output_vectors.last().unwrap());
        })
        .collect();

    // println!("Sorting inputs");

    let mut draw_orders: Vec<Vec<usize>> =
        input.axis_iter(Axis(1))
        .into_par_iter()
        .map(|column|
        {
            valsort(column.iter()).into_iter().map(|(i,_)| i).collect()
        })
        .collect();

    let minima: Vec<Option<(usize,usize,f64)>> = draw_orders
        // .into_iter()
        .into_par_iter()
        .enumerate()
        .map(|(i,draw_order)| {
            let mut dispersions: Vec<f64> = vec![0.;draw_order.len()];
            let ss_len = draw_order.len();
            for (k,rv) in output_vectors.iter().enumerate() {
                let mut rv_f = rv.clone();
                for (j,index) in draw_order.iter().enumerate() {
                    rv_f.pop(*index);
                    let regularization = ((ss_len - j) as f64 / ss_len as f64).powf(sfr);
                    dispersions[j] +=  rv_f.dispersion().to_f64().expect("cast error").powi(l) * regularization;
                }
                let mut rv_r = rv.clone();
                for (j,index) in draw_order.iter().rev().enumerate() {
                    rv_r.pop(*index);
                    let regularization = ((ss_len - j) as f64 / ss_len as f64).powf(sfr);
                    dispersions[ss_len - j - 1] += rv_r.dispersion().to_f64().expect("cast error").powi(l) * regularization;
                }
            }
            dispersions.into_iter().argmin_v().map(|(local_index,dispersion)| (i,draw_order[local_index],dispersion))
        })
        .collect();

    let (feature,sample,dispersion) = minima.iter().flat_map(|m| m).min_by(|&a,&b| (a.2).partial_cmp(&b.2).unwrap())?;
    // Some((*feature,*sample,*dispersion))

    // let feature_index = minima.iter().map(|m| m.map(|(f,s,d)| d).unwrap_or(std::f64::MAX)).argmin()?;
    // let (feature,sample,dispersion) = minima[feature_index]?;
    let mut initial_dispersion: f64 = 0.;

    for rv in output_vectors.iter() {
        let dispersion: f64 = NumCast::from(rv.dispersion()).unwrap();
        initial_dispersion += dispersion;
    }

    // println!("Split successful");
    // println!("{}",output_vectors.len());
    let delta_dispersion = initial_dispersion - dispersion;
    Some((*feature,*sample,delta_dispersion))


}


#[cfg(test)]
mod matrix_tests {

    use crate::iris_array;
    use super::*;

    #[test]
    fn split_test() {
        let input = iris_array().t().to_owned();
        let output = iris_array().t().to_owned();
        println!("{:?}",split(&input,&output,0.,1));
        panic!();
    }

}
