use Asterix::*;



fn main() {
  let layout = vec![2,4,3,2];
  let input: Vec<f32> = vec![1.0,1.0];
  let m1 = Network::new(layout);
  let k1 = m1.cluster;
  for i in 0..k1.len() {
    println!("Data {:?}", k1[i]);
  }
}
