use nalgebra::*;
use rand::prelude::*;
use termion::*;
use ::core::f32;
use std::io::{stdout, Write};
use std::process::Command;
#[derive(Copy, Clone)]
pub enum ActivationFunction {
    Sigmoid,
    Relu,
    Tanh,
    Softmax,
}

impl ActivationFunction {
    fn apply(&self, x: &DVector<f32>) -> DVector<f32> {
        match self {
            ActivationFunction::Sigmoid => x.map(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Relu => x.map(|v| v.max(0.0)),
            ActivationFunction::Tanh => x.map(|v| v.tanh()),
            ActivationFunction::Softmax => {
                let max_val = x.max();
                let exp_vals = x.map(|v| (v - max_val).exp());
                let sum_exp = exp_vals.sum();
                exp_vals / sum_exp
            }
        }
    }
}
#[derive(Clone)]
pub struct Genome {
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<DVector<f32>>,
}

impl Genome {
    pub fn mutate(&mut self, mutation_rate: f32) {
        let mut rng = rand::thread_rng();
        let mutation_dist = rand::distributions::Uniform::new(-mutation_rate, mutation_rate);

        self.weights.iter_mut().for_each(|weight_matrix| {
            weight_matrix.iter_mut().for_each(|val| *val += rng.sample(mutation_dist));
        });

        self.biases.iter_mut().for_each(|bias_vector| {
            bias_vector.iter_mut().for_each(|val| *val += rng.sample(mutation_dist));
        });
    }
}
#[derive(Clone)]
pub struct Network {
    pub genome: Genome,
    pub layout: Vec<usize>,
    pub hidden_activation: ActivationFunction,
    pub output_activation: ActivationFunction,
    pub fitness: f32,
}

impl Network {
    pub fn new(layout: Vec<usize>, hidden_activation: ActivationFunction, output_activation: ActivationFunction) -> Self {
        let mut rng = rand::thread_rng();

        let (weights, biases): (Vec<_>, Vec<_>) = layout.windows(2).map(|w| {
            (
                DMatrix::from_fn(w[1], w[0], |_, _| rng.gen_range(-1.0..1.0)),
                DVector::from_fn(w[1], |_, _| rng.gen_range(-1.0..1.0))
            )
        }).unzip();

        Network {
            genome: Genome { weights, biases },
            layout,
            hidden_activation,
            output_activation,
            fitness: 0.0,
        }
    }

    pub fn forward(&self, input: DVector<f32>) -> Vec<DVector<f32>> {
        let mut activations = vec![input];

        for (i, (weights, biases)) in self.genome.weights.iter().zip(&self.genome.biases).enumerate() {
            let z = weights * activations.last().unwrap() + biases;
            let activation = if i == self.genome.weights.len() - 1 {
                self.output_activation.apply(&z)
            } else {
                self.hidden_activation.apply(&z)
            };
            activations.push(activation);
        }

        activations
    }
    pub fn display_activations(&self, activations: Vec<DVector<f32>>, fitness: f32) {
        let mut stdout = stdout();

        // Clear the screen and move the cursor to the top-left corner
        write!(stdout, "{}{}", clear::All, cursor::Goto(1, 1)).unwrap();

        // Display the input as a 10x10 grid
        let input_activations = &activations[0];
        for (i, &activation) in input_activations.iter().enumerate() {
            if i % 28 == 0 && i != 0 {
                println!(); // New line after every 10 characters
            }
            // Map the activation value to a yellow intensity (0-255)
            let color_value = ((activation.max(0.0).min(1.0) * 255.0) as u8).min(255);
            let color_block = color::Fg(color::Rgb(0,0,color_value)); // Yellow color with intensity mapping
            write!(stdout, "{}█{}", color_block, style::Reset).unwrap();
        }
        println!();

        // Display hidden and output layers
        for (layer_index, layer_activations) in activations.iter().enumerate().skip(1) {
            for &activation in layer_activations.iter() {
                // Map the activation value to a yellow intensity (0-255)
                let color_value = (activation.sin() * 255.0) as u8;
                //let color_value = ((activation.max(0.0).min(1.0) * 255.0) as u8).min(255);
                let color_block = color::Fg(color::Rgb(0,color_value,color_value)); // Yellow color with intensity mapping

                // Represent each neuron as a colored block (using "█" character)
                write!(stdout, "{}█{}", color_block, style::Reset).unwrap();
            }
            println!(); // Newline for each layer
        }
        let last_layer = activations.last().unwrap();
                let (max_index, _) = last_layer
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
        
                println!("Classification: {}", max_index);
        println!("Fitness {}", fitness);
    }
}

