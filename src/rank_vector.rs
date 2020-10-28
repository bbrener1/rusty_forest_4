use std::fmt;
use num_traits::{Pow};
use std::ops::{Index,IndexMut};
use std::fmt::Debug;
use std::clone::Clone;
use std::iter::{Take,Skip};
use crate::{SampleKey,SampleValue};
use std::cmp::Ordering;


#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
pub struct Node<K,V>
where
    K: SampleKey,
    V: SampleValue,
{
    value: V,
    squared_value:V,
    key: K,
    previous: K,
    next: K,
    segment: usize,
}

impl<K,V> Node<K,V>
where
    K: SampleKey,
    V: SampleValue,
{

    fn blank(key:K) -> Node<K,V> {
        Node {
            value:V::zero(),
            squared_value:V::zero(),
            key: key,
            previous:key,
            next:key,
            segment:0,
        }
    }

}


pub trait LinkedVector: Sized
{
    type K: SampleKey;
    type V: SampleValue;
    type Arena: NodeArena<Self::K,Self::V>;


    fn arena(&self) -> &Self::Arena;
    fn arena_mut(&mut self) -> &mut Self::Arena;

    fn unlink_node(&mut self,target_key:Self::K) -> Node<Self::K,Self::V> {
        let target = self.arena()[target_key];

        // OPTIONAL INTEGRITY CODE, CHECK SPEED IMPACT
        let previous = self.arena()[target.previous];
        let next = self.arena()[target.next];
        if !(previous.next == target.key && next.previous == target.key) {
            println!("Double pop!");
            println!("{:?}",previous);
            println!("{:?}",target);
            println!("{:?}",next);
            panic!();
        }

        self.link_nodes(target.previous, target.next);

        // OPTIONAL INTEGRITY CODE, CHECK SPEED IMPACT? (THIS WOULD NEED TO BE FOLLOWED UP IN SEGMENT POP ALSO TO MAINTAIN SUM INTEGRITY)
        // self.link_nodes(target.previous, target.next);

        target
    }

    fn link_nodes(&mut self, left:Self::K,right:Self::K) {
        {self.arena_mut()[left].next = right}
        {self.arena_mut()[right].previous = left}
        let seg = self.arena()[left].segment;
        {self.arena_mut()[right].segment = seg}

    }

    fn insert_node(&mut self, previous_key:Self::K, mut target: Node<Self::K,Self::V>, next_key:Self::K) {
        target.previous = previous_key;
        target.next = next_key;
        target.segment = self.arena()[previous_key].segment;
        let target_key = target.key;
        {self.arena_mut()[previous_key].next = target_key}
        {self.arena_mut()[next_key].previous = target_key}
        {self.arena_mut()[target_key] = target}
    }

    fn left_crawler(&self,start:Self::K) -> LeftCrawler<Self> {
        LeftCrawler{vector:&self,key:start}
    }
    // fn left_crawler(&self,start:Self::K) -> Self::Crawler {
    //     LeftCrawler{vector:&self,key:start}
    // }

    fn right_crawler(&self,start:Self::K) -> RightCrawler<Self> {
        RightCrawler{vector:&self,key:start}
    }


}

pub trait NodeArena<K:SampleKey,V:SampleValue>: Index<K,Output=Node<K,V>> + IndexMut<K,Output=Node<K,V>> + Debug + Clone {

    fn with_capacity(capacity:usize) -> Self;
    fn len(&self) -> usize;
}

pub trait SegmentedVector: LinkedVector
{
    type Segment: Segment<Self::K,Self::V>;

    fn len(&self) -> usize;
    fn balance(&mut self);
    fn segments(&self) -> &[Self::Segment];
    fn segments_mut(&mut self) -> &mut [Self::Segment];
    fn endcaps(&self,segment:usize) -> (Node<Self::K,Self::V>,Node<Self::K,Self::V>);


    fn crawl_segment_reverse(&self,segment:usize) -> Take<Skip<LeftCrawler<Self>>> {
        self.left_crawler(self.segments()[segment].right()).skip(1).take(self.segments()[segment].len())
    }

    fn crawl_segment_ordered(&self,segment:usize) -> Take<Skip<RightCrawler<Self>>> {
        self.right_crawler(self.segments()[segment].left()).skip(1).take(self.segments()[segment].len())
    }

    fn ordered_values(&self) -> Vec<Self::V>{
        (0..self.segments().len()).flat_map(|s| self.crawl_segment_ordered(s)).map(|n| n.value).collect()
    }

    fn ordered_indices(&self) -> Vec<Self::K>{
        (0..self.segments().len()).flat_map(|s| self.crawl_segment_ordered(s)).map(|n| n.key).collect()
    }

    fn pop(&mut self,key:Self::K) -> Self::V {
        let target = LinkedVector::unlink_node(self,key);
        let segment = &mut self.segments_mut()[target.segment];
        segment.pop(&target);
        self.balance();
        target.value
    }

    fn push_segment_left(&mut self,segment:usize,node:Node<Self::K,Self::V>) {
        let left_cap = {self.segments()[segment].left()};
        let left_edge = {self.arena()[left_cap].next};
        self.segments_mut()[segment].push(&node);
        self.insert_node(left_cap, node, left_edge)
    }

    fn pop_segment_left(&mut self,segment:usize) -> Node<Self::K,Self::V> {
        let left_cap = {self.segments()[segment].left()};
        let left_edge = {self.arena()[left_cap].next};
        let node = self.unlink_node(left_edge);
        self.segments_mut()[segment].pop(&node);
        node
    }

    fn push_segment_right(&mut self,segment:usize, node:Node<Self::K,Self::V>) {
        let right_cap = {self.segments()[segment].right()};
        let right_edge = {self.arena()[right_cap].previous};
        self.segments_mut()[segment].push(&node);
        self.insert_node(right_edge, node, right_cap)
    }

    fn pop_segment_right(&mut self,segment:usize) -> Node<Self::K,Self::V> {
        let right_cap = {self.segments()[segment].right()};
        let right_edge = {self.arena()[right_cap].previous};
        let node = self.unlink_node(right_edge);
        self.segments_mut()[segment].pop(&node);
        node
    }

    fn shift_boundary_left(&mut self,segment_left:usize,segment_right:usize) {
        let target = self.pop_segment_right(segment_left);
        self.push_segment_left(segment_right, target);
        // println!("shifted bound left {:?}",(segment_left,segment_right));
        // println!("{:?}",self.segments());
    }

    fn shift_boundary_right(&mut self,segment_left:usize,segment_right:usize) {
        let target = self.pop_segment_left(segment_right);
        self.push_segment_right(segment_left, target);
        // println!("shifted bound left {:?}",(segment_left,segment_right));
        // println!("{:?}",self.segments());
    }

    fn initialize(&mut self) {
        let seg_len = {self.segments().len()};
        for i in 0..seg_len {
            let (e1,e2) = self.endcaps(i);
            let seg = Segment::from_endcaps(&e1,&e2);
            {self.segments_mut()[i] = seg;}
            {self.arena_mut()[e1.key] = e1;}
            {self.arena_mut()[e2.key] = e2;}
        }
    }


    fn link(&mut self, sorted_input:&[(Self::K,Self::V)]) -> &mut Self {
        let mut previous_key = self.segments()[0].left();
        for ((key,value),(next_key,_)) in sorted_input.iter().zip(sorted_input.iter().skip(1)) {
            let node = Node {
                next:*next_key,
                previous: previous_key,
                value: *value,
                squared_value: Pow::<u8>::pow(*value,2),
                key: *key,
                segment: 0,
            };
            {self.segments_mut()[0].push(&node)}
            {self.arena_mut()[previous_key].next = *key}
            {self.arena_mut()[*key] = node};
            previous_key = *key;
            // println!("finished key {:?}",previous_key);
        }
        if let Some((final_key,final_value)) = sorted_input.last() {
            let cap_key = self.segments()[0].right();
            let node = Node {
                next:cap_key,
                previous: previous_key,
                value: *final_value,
                squared_value: Pow::<u8>::pow(*final_value,2),
                key: *final_key,
                segment: 0,
            };
            {self.segments_mut()[0].push(&node)}
            {self.arena_mut()[*final_key] = node};
            {self.arena_mut()[cap_key].previous = *final_key}
        }
        self.balance();
        // println!("Linking almost done");
        // println!("{:?}",self.arena());
        // println!("{:?}",self.segments());
        self
    }


    fn link_iterator<T:Iterator<Item=(Self::K,Self::V)>>(&mut self, sorted_input:T) -> &mut Self {
        let mut previous_key = self.segments()[0].left();
        for (key,value) in sorted_input {
            let node = Node {
                next:key,
                previous: previous_key,
                value: value,
                squared_value: Pow::<u8>::pow(value,2),
                key: key,
                segment: 0,
            };
            {self.segments_mut()[0].push(&node)}
            {self.arena_mut()[previous_key].next = key}
            {self.arena_mut()[key] = node};
            previous_key = key;
            // println!("finished key {:?}",previous_key);
        }
        let cap_key = self.segments()[0].right();
        {self.arena_mut()[previous_key].next = cap_key};
        {self.arena_mut()[cap_key].previous = previous_key}
        self.balance();
        // println!("Linking almost done");
        // println!("{:?}",self.arena());
        // println!("{:?}",self.segments());
        self
    }

}


pub struct RightCrawler<'v,LV>
where
    LV:LinkedVector,

{
    vector: &'v LV,
    key: LV::K,
}

impl<'v,LV> Iterator for RightCrawler<'v,LV>
where
    LV: LinkedVector,
{
    type Item = &'v Node<LV::K,LV::V>;

    fn next(&mut self) -> Option<&'v Node<LV::K,LV::V>> {
        // println!("Crawling right:{:?}",self.key);
        let node = &self.vector.arena()[self.key];
        self.key = node.next;
        return Some(node)
    }
}


pub struct LeftCrawler<'v,LV>
where
    LV:LinkedVector
 {
    vector: &'v LV,
    key: LV::K,
}

impl<'v,LV> Iterator for LeftCrawler<'v,LV>
where
    LV:LinkedVector,
{
    type Item = &'v Node<LV::K,LV::V>;

    fn next(&mut self) -> Option<&'v Node<LV::K,LV::V>> {

        let node = &self.vector.arena()[self.key];
        self.key = node.previous;
        return Some(node)
    }
}


pub trait Segment<K:SampleKey,V:SampleValue>
{

    fn left(&self) -> K;
    fn right(&self) -> K;
    fn len(&self) -> usize;
    fn sum(&self) -> V;
    fn squared_sum(&self) -> V;

    fn len_mut(&mut self) -> &mut usize;
    fn sum_mut(&mut self) -> &mut V;
    fn squared_sum_mut(&mut self) -> &mut V;

    fn from_endcaps(e1:&Node<K,V>,e2:&Node<K,V>) -> Self;

    fn pop(&mut self,node:&Node<K,V>) {
        // println!("Popping {:?}", node);
        *self.sum_mut() -= node.value;
        *self.squared_sum_mut() -= node.squared_value;
        *self.len_mut() -= 1;
        // println!("Done {:?}",(self.sum(),self.len()));
    }
    fn push(&mut self, node:&Node<K,V>) {
        // println!("Pushing {:?}", node);
        *self.sum_mut() += node.value;
        *self.squared_sum_mut() += node.squared_value;
        *self.len_mut() += 1;
        // println!("Done {:?}",(self.sum(),self.len()));
    }
}

pub trait FeatureVector: SegmentedVector
{
    fn central_tendency(&self) -> Self::V;
    fn dispersion(&self) -> Self::V;

}

#[derive(Clone,Copy,Debug,Serialize,Deserialize)]
pub struct IndexSegment<V:SampleValue> {
    left:usize,
    right:usize,
    sum:V,
    squared_sum:V,
    len:usize,
}

impl<V:SampleValue> Segment<usize,V> for IndexSegment<V> {

    fn left(&self) -> usize {
        self.left
    }
    fn right(&self) -> usize {
        self.right
    }
    fn len(&self) -> usize {
        self.len
    }
    fn sum(&self) -> V {
        self.sum
    }
    fn squared_sum(&self) -> V {
        self.squared_sum
    }

    fn len_mut(&mut self) -> &mut usize {
        &mut self.len
    }
    fn sum_mut(&mut self) -> &mut V {
        &mut self.sum
    }
    fn squared_sum_mut(&mut self) -> &mut V {
        &mut self.squared_sum
    }

    fn from_endcaps(e1:&Node<usize,V>,e2:&Node<usize,V>) -> Self {
        IndexSegment {
            left: e1.key,
            right: e2.key,
            sum: V::zero(),
            squared_sum: V::zero(),
            len: 0,
        }
    }

}
//
// trait IndexedSegmentedVector: SegmentedVector<K=usize,Segment=IndexSegment<Self>> {}

impl<V:SampleValue> IndexSegment<V> {

    fn blank() -> IndexSegment<V> {
        IndexSegment {
            left:0,
            right:0,
            sum:V::zero(),
            squared_sum:V::zero(),
            len:0,
        }
    }

}

impl<V> NodeArena<usize,V> for Vec<Node<usize,V>>
where
    V: SampleValue,
{
    fn with_capacity(capacity:usize) -> Vec<Node<usize,V>> {
        vec![Node::<usize,V>::blank(0);capacity]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<V> NodeArena<usize,V> for NodeArray<V>
where
    V: SampleValue,
{
    fn with_capacity(capacity:usize) -> NodeArray<V> {
        if capacity <= 1018 { NodeArray([Node::blank(0);1024])}
        else {panic!("Exceeded capacity of array: {:?}",capacity)}
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone)]
pub struct NodeArray<V:SampleValue>([Node<usize,V>;1024]);

impl<V:SampleValue> Index<usize> for NodeArray<V>
where
    V: SampleValue,
{
    type Output = Node<usize,V>;
    fn index(&self,index:usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<V> IndexMut<usize> for NodeArray<V>
where
    V: SampleValue,
{
    fn index_mut(&mut self,index:usize) -> &mut Node<usize,V> {
        &mut self.0[index]
    }
}

impl<V:SampleValue> Debug for NodeArray<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let end = self.0.iter().enumerate().find(|(_,n)| n.previous == 0 && n.next == 0).map(|(i,_)| i).unwrap_or(0);
        write!(f, "{:?}",self.0[..end].to_vec())
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct MedianArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    segments: [IndexSegment<V>;3],
    arena: A,
}

pub type MedianArray<V> = MedianArena<V,NodeArray<V>>;
pub type MedianVector<V> = MedianArena<V,Vec<Node<usize,V>>>;
pub type MeanArray<V> = MeanArena<V,NodeArray<V>>;
pub type MADArray<V> = MADArena<V,NodeArray<V>>;

impl<A,V> LinkedVector for MedianArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    type K = usize;
    type V = V;
    type Arena = A;

    fn arena(&self) -> &Self::Arena {
        &self.arena
    }

    fn arena_mut(&mut self) -> &mut Self::Arena {
        &mut self.arena
    }
}


impl<A,V> SegmentedVector for MedianArena<V,A>
where
    V: SampleValue,
    A: NodeArena<usize,V>,
{

    type Segment = IndexSegment<Self::V>;

    fn len(&self) -> usize {
        self.segments().iter().map(|s| s.len()).sum()
    }

    fn segments(&self) -> &[IndexSegment<Self::V>] {
        &self.segments
    }
    fn segments_mut(&mut self) -> &mut [IndexSegment<Self::V>] {
        &mut self.segments
    }

    fn endcaps(&self,segment:usize) -> (Node<usize,V>,Node<usize,V>) {
        let offset = (self.arena().len() - 6) + (segment*2);
        let i1 = offset;
        let i2 = offset + 1;
        let e1 = Node {
            key: i1,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment: segment,
        };
        let e2 = Node {
            key: i2,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment:segment,
        };
        (e1,e2)
    }

    fn balance(&mut self) {
        self.size_median();
        // println!("Balancing");
        while {self.segments()[0].len()} < {self.segments()[2].len()} {
            // println!("Shifting median left");
            self.shift_median_right();
        }
        while {self.segments()[0].len()} > {self.segments()[2].len()} {
            // println!("Shifting median right");
            self.shift_median_left();
        }
    }

}

impl<V:SampleValue,A> FeatureVector for MedianArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    fn central_tendency(&self) -> V {
        self.median()
    }
    fn dispersion(&self) -> Self::V {
        // self.ssme()
        self.sme()
    }

}

impl<'a,A,V> MedianArena<V,A>
where
    V: SampleValue + 'a,
    A: NodeArena<usize,V> + 'a,
{

        pub fn link(sorted_input:&[(usize,V)]) -> Self {
            let mut mv = Self::with_capacity(sorted_input.len());
            SegmentedVector::link(&mut mv,sorted_input);
            mv.balance();
            mv
        }

        pub fn link_iterator<T:Iterator<Item=&'a (usize,V)>>(sorted_input: T,length:usize) -> Self {
            let mut mv = Self::with_capacity(length);
            SegmentedVector::link_iterator(&mut mv,sorted_input.cloned());
            mv.balance();
            mv
        }

        fn with_capacity(capacity:usize) -> MedianArena<V,A> {

            let mut mv = MedianArena{
                segments: [IndexSegment::blank();3],
                arena: NodeArena::<usize,V>::with_capacity(capacity+6),
            };
            mv.initialize();
            mv
        }

        fn size_median(&mut self) {
            // println!("Initializing median");

            if self.segments[1].len() < 1 {
                // println!("Own length less than 1");
                // println!("{:?}",self);
                if self.segments[0].len() > 0 {
                    // println!("Shifting boundary left");
                    self.shift_boundary_left(0, 1)
                }
                else if self.segments[2].len() > 0 {
                    // println!("Shifting boundary left");
                    self.shift_boundary_right(1,2)
                }
                // else {println!("No shift"); panic!()}
            }
            // println!("Initialized to at least 1");
            while self.segments[1].len > 2 {
                self.shift_boundary_left(1,2)
            }

            // println!("{:?}",self.segments());
        }


        fn shift_median_left(&mut self) {
            match self.segments[1].len() {
                1 => {self.shift_boundary_left(0,1)},
                2 => {self.shift_boundary_left(1,2)},
                _ => {panic!(format!("Median de-synchronized:{:?}",self))}
            }
        }

        fn shift_median_right(&mut self) {
            match self.segments[1].len() {
                1 => {self.shift_boundary_right(1,2)},
                2 => {self.shift_boundary_right(0,1)},
                _ => {panic!(format!("Median de-synchronized:{:?}",self))}
            }
        }

        fn median(&self) -> V {
            if self.segments[1].len() > 0 {
                self.segments[1].sum / (V::from(self.segments[1].len()).expect("Cast failure"))
            }
            else {V::zero()}
        }

        pub fn ssme(&self) -> V {
            let median = self.median();
            let squared_sum = self.segments()[0].squared_sum + self.segments()[1].squared_sum + self.segments()[2].squared_sum;
            let sum = self.segments()[0].sum + self.segments()[1].sum + self.segments()[2].sum;
            squared_sum - ((V::from(2).expect("Cast failure"))*median*sum) + ((V::from(self.len()).expect("Cast failure")) * (median.pow(2)))
        }

        pub fn sme(&self) -> V {
            let median = self.median();
            let mut left_sum = (V::from(self.segments()[0].len()).expect("Cast failure") * median) - self.segments()[0].sum();
            let mut right_sum = self.segments()[2].sum() - (V::from(self.segments()[2].len()).expect("Cast failure") * median);
            if self.segments[1].len() == 2 {
                let left_median_key = self.arena()[self.segments[1].left].next;
                let left_median_value = self.arena()[left_median_key].value;
                let right_median_key = self.arena()[self.segments[1].right].previous;
                let right_median_value = self.arena()[right_median_key].value;
                left_sum += median - left_median_value;
                right_sum += right_median_value - median;
            }
            left_sum + right_sum
        }

        pub fn check_integrity(&self) {
            use NumCast;

            let ordered_values = self.ordered_values();
            let own_ssme:f64 = NumCast::from(self.ssme()).unwrap();
            let correct_ssme = slow_ssme(&ordered_values.iter().map(|v| NumCast::from(*v).unwrap()).collect());
            assert!((own_ssme-correct_ssme).abs() < 0.0001);
        }

}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct MeanArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    segments: [IndexSegment<V>;1],
    arena: A,
}

impl<V,A> LinkedVector for MeanArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    type K = usize;
    type V = V;
    type Arena = A;


    fn arena(&self) -> &Self::Arena {
        &self.arena
    }
    fn arena_mut(&mut self) -> &mut Self::Arena{
        &mut self.arena
    }

}

impl<V,A> SegmentedVector for MeanArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{

    type Segment = IndexSegment<Self::V>;

    fn len(&self) -> usize {
        self.segments[0].len()
    }

    fn balance(&mut self) {}

    fn segments(&self) -> &[Self::Segment] {
        &self.segments
    }
    fn segments_mut(&mut self) -> &mut [Self::Segment] {
        &mut self.segments
    }

    fn endcaps(&self,segment:usize) -> (Node<usize,V>,Node<usize,V>) {
        let offset = self.arena().len() - 2;
        let i1 = offset;
        let i2 = offset + 1;
        let e1 = Node {
            key: i1,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment: segment,
        };
        let e2 = Node {
            key: i2,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment:segment,
        };
        (e1,e2)
    }

}

impl<V,A> MeanArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    fn mean(&self) -> V {
        self.segments[0].sum / V::from(self.segments.len()).expect("Cast failure")
    }

    fn variance(&self) -> V {
        (self.segments[0].squared_sum / V::from(self.segments.len()).expect("Cast failure")) - self.mean().pow(2)
    }


    pub fn link(sorted_input:&[(usize,V)]) -> Self {
        let mut mv = Self::with_capacity(sorted_input.len());
        SegmentedVector::link(&mut mv,sorted_input);
        mv.balance();
        mv
    }

    fn with_capacity(capacity:usize) -> MeanArena<V,A> {

        let mut mv = MeanArena{
            segments: [IndexSegment::blank();1],
            arena: NodeArena::<usize,V>::with_capacity(capacity+2),
        };
        mv.initialize();
        mv
    }
}

impl<V,A> FeatureVector for MeanArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    fn central_tendency(&self) -> V {
        self.mean()
    }
    fn dispersion(&self) -> Self::V {
        self.variance()
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct MADArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    segments: [IndexSegment<V>;5],
    arena: A,
}



impl<V,A> LinkedVector for MADArena<V,A>
where
    V:SampleValue,
    A:NodeArena<usize,V>,
{
    type K = usize;
    type V = V;
    type Arena = A;


    fn arena(&self) -> &Self::Arena {
        &self.arena
    }
    fn arena_mut(&mut self) -> &mut Self::Arena{
        &mut self.arena
    }

}

impl<A,V> SegmentedVector for MADArena<V,A>
where
    V: SampleValue,
    A: NodeArena<usize,V>,
{

    type Segment = IndexSegment<Self::V>;

    fn len(&self) -> usize {
        self.segments().iter().map(|s| s.len()).sum()
    }

    fn segments(&self) -> &[IndexSegment<Self::V>] {
        &self.segments
    }
    fn segments_mut(&mut self) -> &mut [IndexSegment<Self::V>] {
        &mut self.segments
    }

    fn endcaps(&self,segment:usize) -> (Node<usize,V>,Node<usize,V>) {
        let offset = (self.arena().len() - 10) + (segment*2);
        let i1 = offset;
        let i2 = offset + 1;
        let e1 = Node {
            key: i1,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment: segment,
        };
        let e2 = Node {
            key: i2,
            value: V::zero(),
            squared_value: V::zero(),
            previous: i1,
            next: i2,
            segment:segment,
        };
        (e1,e2)
    }

    fn balance(&mut self) {



        let outer_sum = self.segments[0].sum() + self.segments[4].sum();
        let inner_sum = self.segments[1].sum() + self.segments[2].sum() + self.segments[3].sum();
        if outer_sum > inner_sum {
            self.expand_1();
        }
    }

}

impl<'a,V,A> MADArena<V,A>
where
    V: SampleValue,
    A: NodeArena<usize,V>,
{
    pub fn link(sorted_input:&[(usize,V)]) -> Self {
        let mut mv = Self::with_capacity(sorted_input.len());
        SegmentedVector::link(&mut mv,sorted_input);
        mv.balance();
        mv
    }

    pub fn link_iterator<T:Iterator<Item=&'a (usize,V)>>(sorted_input: T,length:usize) -> Self {
        let mut mv = Self::with_capacity(length);
        SegmentedVector::link_iterator(&mut mv,sorted_input.cloned());
        mv.balance();
        mv
    }

    fn with_capacity(capacity:usize) -> Self {

        let mut mv = MADArena{
            segments: [IndexSegment::blank();5],
            arena: NodeArena::<usize,V>::with_capacity(capacity+10),
        };
        mv.initialize();
        mv
    }

    fn outer_left(&self) -> Option<&Node<usize,V>> {
        if self.segments()[0].len() > 0 {
            let ol_key = self.arena()[self.segments[0].right].previous;
            Some(&self.arena()[ol_key])
        }
        else {None}
    }

    fn outer_right(&self) -> Option<&Node<usize,V>> {
        if self.segments()[4].len() > 0 {
            let or_key = self.arena()[self.segments[4].left].next;
            Some(&self.arena()[or_key])
        }
        else {None}
    }

    fn inner_left(&self) -> Option<&Node<usize,V>> {
        let cl_key;
        if self.segments()[1].len() > 0 {
            cl_key = self.arena()[self.segments[1].left].next;
        }
        else {
            cl_key = self.arena()[self.segments[2].left].next;
        }
        Some(&self.arena()[cl_key])
    }

    fn inner_right(&self) -> Option<&Node<usize,V>> {
        let cr_key;
        if self.segments()[3].len() > 0 {
            cr_key = self.arena()[self.segments[3].right].previous;
        }
        else {
            cr_key = self.arena()[self.segments[2].right].previous;
        }
        Some(&self.arena()[cr_key])
    }

    fn outer_population(&self) -> usize {
        self.segments[0].len() + self.segments[4].len()
    }

    fn inner_population(&self) -> usize {
        self.segments[1].len() + self.segments[2].len() + self.segments[3].len()
    }

    fn shift_left(&mut self) {
        self.shift_boundary_left(0,1);
        self.shift_boundary_left(1,2);
        self.shift_boundary_left(2,3);
    }

    fn shift_right(&mut self) {
        self.shift_boundary_right(0,1);
        self.shift_boundary_right(1,2);
        self.shift_boundary_right(2,3);
    }

    fn expand_1(&mut self) {
        let left = self.outer_left();
        let right = self.outer_right();

        if let (Some(Node{value:left_value,..}),Some(Node{value:right_value,..})) = (left,right) {
            let median = self.median();
            if (median - *left_value) > (*right_value - median) {
                self.shift_boundary_right(3,4);
            }
            else {
                self.shift_boundary_left(0,1)
            }
        }
        else if left.is_some() {
            self.shift_boundary_left(0,1);
        }
        else if right.is_some() {
            self.shift_boundary_right(3,4);
        }
        else {panic!("Boundary violation: {:?}",self)}
    }

    fn contract_1(&mut self) {
        let left = self.inner_left();
        let right = self.inner_right();

        if let (Some(Node{value:left_value,..}),Some(Node{value:right_value,..})) = (left,right) {
            let median = self.median();
            if (median - *left_value) > (*right_value - median) {
                self.shift_boundary_right(1,2);
            }
            else {
                self.shift_boundary_left(2,3)
            }
        }
        else if left.is_some() {
            self.shift_boundary_right(1,2);
        }
        else if right.is_some() {
            self.shift_boundary_left(3,4);
        }
        else {panic!("Boundary violation: {:?}",self)}
    }

    fn size_center(&mut self) {
        while (self.outer_population() > self.inner_population()) {
            self.expand_1();
        }
    }

    fn size_median(&mut self) {
        // println!("Initializing median");

        if self.segments[2].len() < 1 {
            // println!("Own length less than 1");
            // println!("{:?}",self);
            if self.segments[1].len() > 0 {
                // println!("Shifting boundary left");
                self.shift_boundary_left(1, 2)
            }
            else if self.segments[3].len() > 0 {
                // println!("Shifting boundary left");
                self.shift_boundary_right(2,3)
            }
            // else {println!("No shift"); panic!()}
        }
        // println!("Initialized to at least 1");
        while self.segments[2].len > 2 {
            self.shift_boundary_left(2,3)
        }

        // println!("{:?}",self.segments());
    }


    fn shift_median_left(&mut self) {
        match self.segments[2].len() {
            1 => {self.shift_boundary_left(1,2)},
            2 => {self.shift_boundary_left(2,3)},
            _ => {panic!(format!("Median de-synchronized:{:?}",self))}
        }
    }

    fn shift_median_right(&mut self) {
        match self.segments[1].len() {
            1 => {self.shift_boundary_right(2,3)},
            2 => {self.shift_boundary_right(1,2)},
            _ => {panic!(format!("Median de-synchronized:{:?}",self))}
        }
    }

    fn median(&self) -> V {
        if self.segments[2].len() > 0 {
            self.segments[2].sum / (V::from(self.segments[2].len()).expect("Cast failure"))
        }
        else {V::zero()}
    }

}

enum DispersionArray<V:SampleValue> {
    SME {vector:MedianArray<V>},
    SSME {vector:MedianArray<V>},
    Variance {vector:MeanArray<V>},
    MAD {vector:MADArray<V>},
}

use crate::io::DispersionMode;
impl<V:SampleValue> DispersionArray<V>{

    fn link(sorted_input:&[(usize,V)],dispersion:DispersionMode) -> Self {

        match dispersion {
            DispersionMode::SME => {DispersionArray::SME{vector:MedianArray::link(sorted_input)}},
            DispersionMode::SSME => {DispersionArray::SSME{vector:MedianArray::link(sorted_input)}},
            DispersionMode::Variance => {DispersionArray::Variance{vector:MeanArray::link(sorted_input)}},
            DispersionMode::MAD => {DispersionArray::MAD{vector:MADArray::link(sorted_input)}},
            _ => {unimplemented!()},
        }
    }

    fn dispersion(&self) -> V {
        match self {
            DispersionArray::SME{vector} => {vector.sme()},
            DispersionArray::SSME{vector} => {vector.ssme()},
            DispersionArray::Variance{vector} => {vector.variance()},
            DispersionArray::MAD{vector} => {unimplemented!()},
        }
    }
}


fn slow_median(values: &Vec<f64>) -> f64 {
    let median: f64;
    if values.len() < 1 {
        return 0.
    }

    if values.len()%2==0 {
        median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
    }
    else {
        median = values[(values.len()-1)/2];
    }

    median

}

fn slow_mad(values: &Vec<f64>) -> f64 {
    let median: f64;
    if values.len() < 1 {
        return 0.
    }
    if values.len()%2==0 {
        median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
    }
    else {
        median = values[(values.len()-1)/2];
    }

    let mut abs_deviations: Vec<f64> = values.iter().map(|x| (x-median).abs()).collect();

    abs_deviations.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    let mad: f64;
    if abs_deviations.len()%2==0 {
        mad = (abs_deviations[abs_deviations.len()/2] + abs_deviations[abs_deviations.len()/2 - 1]) as f64 / 2.;
    }
    else {
        mad = abs_deviations[(abs_deviations.len()-1)/2];
    }

    mad

}

pub fn slow_ssme(values: &Vec<f64>) -> f64 {
    let median = slow_median(values);
    values.iter().map(|x| (x - median).powi(2)).sum()
}

fn slow_sme(values: &Vec<f64>) -> f64 {
    let median = slow_median(values);
    values.iter().map(|x| (x - median).abs()).sum()
}

#[cfg(test)]
mod random_forest_tests {

    use rand::prelude::*;
    use super::*;


    fn simple_values() -> Vec<f64> {
        vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]
    }

    fn random_draw_order() -> Vec<usize> {
        let mut d = (0..1000).collect::<Vec<usize>>();
        &mut d[..].shuffle(&mut thread_rng());
        d
    }

    fn random_floats() -> Vec<f64> {
        (0..1000).map(|_| thread_rng().gen::<f64>()).collect()
    }


    fn argsorted() -> Vec<(usize,f64)> {
        let s = simple_values();
        let mut paired: Vec<(usize,f64)> = s.into_iter().enumerate().collect();
        paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        return paired
    }

    #[test]
    fn ordered_value_test() {
        let argsorted = argsorted();
        let ordered_values = vec![-3.,-2.,-1.,0.,5.,10.,15.,20.];
        println!("{:?}",ordered_values);
        let mut mv = MedianVector::<f64>::with_capacity(8);
        mv.link(&argsorted);
        println!("{:?}",ordered_values);
        assert_eq!(ordered_values,mv.ordered_values());
    }

    #[test]
    fn median_test() {
        let mut mv = MedianVector::<f64>::with_capacity(8);
        mv.link(&argsorted());
        println!("{:?}",mv);
        println!("{:?}",mv.median());
        assert_eq!(2.5,mv.median());
        mv.pop(0);
        println!("{:?}",mv);
        assert_eq!(0.,mv.median());

    }

    #[test]
    fn random_median_test() {
        use crate::valsort;

        let floats = random_floats();
        let draw_order = random_draw_order();
        let argsorted = valsort(floats.into_iter());
        let mut mv = MedianArray::link(&argsorted);
        for i in draw_order {
            mv.pop(i);
            let ordered_values = mv.ordered_values();
            if (slow_median(&ordered_values)-mv.median()).abs() > 0.000001 {
                println!("{:?}",(slow_median(&ordered_values),mv.median()));
                println!("{:?}",(slow_median(&ordered_values)-mv.median()).abs());
                panic!();
            }
            if (slow_ssme(&ordered_values)-mv.ssme()).abs() > 0.000001 {
                println!("{:?}",(slow_ssme(&ordered_values),mv.ssme()));
                println!("{:?}",(slow_ssme(&ordered_values)-mv.ssme()).abs());
                panic!();
            }

        }
    }



    #[test]
    fn random_median_iter_test() {

        let floats = random_floats();
        let draw_order = random_draw_order();
        let mut argsorted: Vec<(usize,f64)> = floats.into_iter().enumerate().collect();
        argsorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut mv = MedianVector::link_iterator(argsorted.iter(),1000);
        for i in draw_order {
            mv.pop(i);
            let ordered_values = mv.ordered_values();
            if (slow_median(&ordered_values)-mv.median()).abs() > 0.000001 {
                println!("{:?}",(slow_median(&ordered_values),mv.median()));
                println!("{:?}",(slow_median(&ordered_values)-mv.median()).abs());
                panic!();
            }
            if (slow_ssme(&ordered_values)-mv.ssme()).abs() > 0.000001 {
                println!("{:?}",(slow_ssme(&ordered_values),mv.ssme()));
                println!("{:?}",(slow_ssme(&ordered_values)-mv.ssme()).abs());
                panic!();
            }

        }
    }

}



// #[derive(Clone,Debug,Serialize,Deserialize)]
// pub struct ArrayArena<V>
// where
//     V: SampleValue,
// {
//     array:[Node<usize,V>;1024],
// }
//
// impl<V> Index<usize> for ArrayArena<V>
// where
//     V: SampleValue,
// {
//     type Output = Node<usize,V>;
//     fn index(&self,index:usize) -> Node<usize,V> {
//         self.array[index]
//     }
// }
//
// impl<V> NodeArena<usize,V> for ArrayArena<V>
// where
//     V: SampleValue,
// {
//     fn with_capacity(template:Node<>)
// }
//
// #[derive(Clone,Debug,Serialize,Deserialize)]
// pub struct MedianVector<A,V>
// where
//     V: SampleValue,
//     A: NodeArena<usize,V>,
// {
//     segments: [IndexSegment<V>;3],
//     arena: A,
// }
//
//
