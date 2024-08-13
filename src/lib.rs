pub mod matrix;
use matrix::*;

#[derive(Clone, Debug)]
pub struct Network {
  pub network_layout: Vec<usize>,
  pub cluster: Vec<Matrix>,
}
// Gell
impl Network {
  pub fn new(
    network_layout: Vec<usize>
  ) -> Self {
      let length = network_layout.len() - 1;
      let mut network: Vec<Matrix> = Vec::new();
      network.push(Matrix::zeros(network_layout[0], 1)); // Input neurons
      network.push(Matrix::random(network_layout[1], network_layout[0])); // Input -> First hidden weights
      network.push(Matrix::random(network_layout[1], 1)); // Bias vector, first hidden layer
      
      for i in 1..network_layout.len() - 2 {
          network.push(Matrix::random(network_layout[i + 1], network_layout[i]));
          network.push(Matrix::random(network_layout[i + 1], 1));
      }
      
      network.push(Matrix::random(network_layout[length], network_layout[length - 1])); // Last hidden -> Output weights
      network.push(Matrix::random(network_layout[length], 1)); // Bias vector, output layer
      network.push(Matrix::zeros(network_layout[length], 1)); // Output layer
      
      Network {
          network_layout,
          cluster: network,
      }
  }
  pub fn to_data(
    &self,
  ) -> Vec<Vec<f32>> {
    let cluster = &self.cluster;
    let mut result: Vec<Vec<f32>> = Vec::new();
    for i in 0..cluster.len(){
      result.push(cluster[i].data());
    }
    result
  }
  pub fn from_data(
    &mut self,
    data: Vec<Vec<f32>>,
  ) {
    let cluster = &mut self.cluster;
    for (i, matrix_data) in data.into_iter().enumerate() {
      let rows = cluster[i].rows;
      let cols = cluster[i].cols; 
      if matrix_data.len() != rows * cols {
        panic!("Data soze does not match the matrix dimensions");
      }
      cluster[i].fill(matrix_data);
    }
  }
  pub fn forward(
    &self, inputs: Vec<f32>
  ) -> Vec<f32> {
    let cluster = &self.cluster;

    // Convert the input Vec<f32> to a Matrix
    let input_matrix = Matrix::new(cluster[0].rows, 1, inputs);

    let mut input = input_matrix; // Start with the input matrix

    // Iterate through each hidden layer
    for i in (1..cluster.len() - 3).step_by(2) {
        let Z = (cluster[i].mul(&input)).add(&cluster[i + 1]);
        input = Z.relu();
    }

    // Last hidden layer to output layer
    let Z_last = (cluster[cluster.len() - 3].mul(&input)).add(&cluster[cluster.len() - 2]);

    let A_last = Z_last; // For regression, leave it as is

    A_last.data() // Return the final output
  }
  pub fn display(
    &self,
  )
  {
    let cluster = &self.cluster;
    for i in 0..cluster.len(){
      if i == cluster.len() -1{
        println!("{} Output Neurons:  :: {:?}", i, cluster[i]);
      }
      else if i == 0 {
        println!("{} Input Neurons:   :: {:?}", i, cluster[i]);
      } 
      else if i % 2 == 0 {
        println!("{} Neuron Bias:     :: {:?}", i, cluster[i]);
      } 
      else {
        println!("{} Weights:         :: {:?}", i, cluster[i]);
      }
    }
  }
  pub fn mutate(&mut self, factor: f32) {
      use rand::Rng;
      let mut rng = rand::thread_rng();
  
      for matrix in &mut self.cluster {
          for i in 0..matrix.data.len() {
              let mutation = rng.gen_range(-factor..factor);
              matrix.data[i] += mutation;
          }
      }
  }
}