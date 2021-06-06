use std::collections::HashMap;
use std::hash::Hash;
use std::cmp::Eq;
use std::ops::{Add,AddAssign,Sub,SubAssign};
use num_traits::Float;

pub trait Key: Eq + Hash {}
pub trait Value: Float + SubAssign {}

pub struct VNode<K: Key,V: Value> {
    key: K,
    next: K,
    previous: K,
    value:V
}

pub struct Segment<K: Key,V: Value> {
    map: HashMap<K,VNode<K,V>>,
    head: K,
    tail: K,
    sum: V,
    squared_sum: V,
}

impl<K: Key,V: Value> Segment<K,V> {
    pub fn pop(&mut self, key:K) -> Option<VNode<K,V>> {
        let vn = self.map.remove(&key)?;
        self.sum -= vn.value;
        self.squared_sum -= vn.value.powi(2);

        Some(vn)
    }
}
