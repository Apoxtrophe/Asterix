use nalgebra::*;
use rand::prelude::*;
use rayon::prelude::*;

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

    pub fn crossover(&self, other: &Genome) -> Genome {
        let mut rng = rand::thread_rng();
        let crossover_dist = rand::distributions::Uniform::new(0.0, 1.0);

        let weights = self.weights.iter().zip(&other.weights).map(|(w1, w2)| {
            DMatrix::from_fn(w1.nrows(), w1.ncols(), |r, c| {
                if rng.sample(crossover_dist) < 0.5 {
                    w1[(r, c)]
                } else {
                    w2[(r, c)]
                }
            })
        }).collect();

        let biases = self.biases.iter().zip(&other.biases).map(|(b1, b2)| {
            DVector::from_fn(b1.len(), |i, _| {
                if rng.sample(crossover_dist) < 0.5 {
                    b1[i]
                } else {
                    b2[i]
                }
            })
        }).collect();

        Genome { weights, biases }
    }
}

#[derive(Clone)]
pub struct Network {
    pub genome: Genome,
    pub layout: Vec<usize>,
    pub hidden_activation: ActivationFunction,
    pub output_activation: ActivationFunction,
    pub fitness: f32,
    pub outputs: Vec<f32>,
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
            outputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut activations = DVector::from_column_slice(input); // Use a slice to avoid reallocation
        for (i, (weights, biases)) in self.genome.weights.iter().zip(&self.genome.biases).enumerate() {
            // Use references to avoid cloning
            let z = weights * &activations + biases;
            activations = if i == self.genome.weights.len() - 1 {
                self.output_activation.apply(&z)
            } else {
                self.hidden_activation.apply(&z)
            };
        }
        self.outputs = activations.data.as_vec().clone(); // Only clone once
        self.outputs.clone()
    }
}

pub struct Population {
    pub networks: Vec<Network>,
    pub mutation_rate: f32,
    pub elite_fraction: f32,
}

impl Population {
    pub fn new(
        size: usize,
        layout: Vec<usize>,
        hidden_activation: ActivationFunction,
        output_activation: ActivationFunction,
        elite_fraction: f32,
    ) -> Self {
        let networks = (0..size)
            .map(|_| Network::new(layout.clone(), hidden_activation, output_activation))
            .collect();

        Population {
            networks,
            mutation_rate: 1.0,
            elite_fraction,
        }
    }

    fn adjust_mutation_rate(&mut self) {
        // Calculate the fitness difference between the top and bottom performers
        let max_fitness = self.networks.first().map_or(0.0, |n| n.fitness);
        let min_fitness = self.networks.last().map_or(0.0, |n| n.fitness);
        let fitness_range = max_fitness - min_fitness;
        let fitness_ratio = if max_fitness != 0.0 {
            fitness_range / max_fitness
        } else {
            0.0
        };

        if fitness_ratio > 0.05 {
            self.mutation_rate *= 0.9;
        } else {
            self.mutation_rate *= 1.1;
        }
    }

    pub fn evolve(&mut self) {
        self.networks.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.adjust_mutation_rate();
        println!("LOWEST: {} HIGHEST: {} MUT_RATE: {}", self.networks.last().unwrap().fitness / 10000.0 * 100.0,self.networks[0].fitness / 10000.0 * 100.0,self.mutation_rate);
        let elite_count = (self.networks.len() as f32 * self.elite_fraction).ceil() as usize;
        let elites = self.networks[..elite_count].to_vec();
        
        for i in elite_count..self.networks.len() {
            let parent1 = &elites[rand::thread_rng().gen_range(0..elite_count)];
            let parent2 = &elites[rand::thread_rng().gen_range(0..elite_count)];
            let mut child = Network {
                genome: parent1.genome.crossover(&parent2.genome),
                layout: parent1.layout.clone(),
                hidden_activation: parent1.hidden_activation,
                output_activation: parent1.output_activation,
                fitness: 0.0,
                outputs: Vec::new(),
            };
            child.genome.mutate(self.mutation_rate);
            self.networks[i] = child;
        }
    }

    pub fn forward(&mut self, inputs: &[Vec<f32>]) -> Vec<Vec<Vec<f32>>> {
            self.networks
                .par_iter_mut() // Parallelize over networks
                .map(|network| {
                    inputs
                        .iter()
                        .map(|input| network.forward(input)) // Use existing forward method without unnecessary cloning
                        .collect()
                })
                .collect()
        }
    `
}
