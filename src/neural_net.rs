
use rand::{thread_rng, Rng};

#[derive(Debug, Clone)]
pub struct NeuralNet {
    layers_widths: Vec<usize>,
    weights_widths: Vec<usize>,
    layers: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
}

impl NeuralNet {
    pub fn new(layers_widths: Vec<usize>) -> NeuralNet {
        let layers: Vec<Vec<f64>> = layers_widths
            .iter()
            .map(|width| vec![0.0; *width])
            .collect();
        let mut weights_widths: Vec<usize> = vec![];
        for i in 0..layers_widths.len() - 1 {
            weights_widths.push(layers_widths[i] * layers_widths[i + 1])
        }
        let weights: Vec<Vec<f64>> = weights_widths
            .iter()
            .map(|width| vec![(); *width].iter().map(|_| thread_rng().gen_range(0.0..1.0)).collect())
            .collect();
        NeuralNet {
            layers_widths,
            weights_widths,
            layers,
            weights,
        }
    }
    pub fn run(&mut self, layer_inputs: Vec<f64>) -> Result<&Vec<f64>, ()> {
        match layer_inputs {
            inputs if inputs.len() != self.layers_widths[0] => Err(()),
            inputs => {
                self.layers[0] = inputs; // set the input layer
                for i in 1..self.layers.len() {
                    // for every layer after the input layer, we want to set all the nodes in the next node layer
                    // based on the weights and the nodes from the previous layer
                    for j in 0..self.layers_widths[i] {
                        for k in 0..self.layers_widths[i - 1] {
                            //add the last (node * weight)
                            let prev_node = self.layers[i - 1][k];
                            let weight = self.weights[i][j * self.layers_widths[i - 1] + k];
                            self.layers[i][j] += prev_node * weight;
                        }
                        //do activation function
                        self.layers[i][j] = relu(self.layers[i][j])
                    }
                }
                Ok(self.layers.last().unwrap())
            }
        }
    }
}

#[inline(always)]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}



// fn reward_function() {
//     todo!()
// }


// fn train_networks(racers_per_map: usize, racers_per_generation: usize, num_generations: usize) {
//     assert!(racers_per_generation % racers_per_map == 0 && racers_per_map < racers_per_generation, "The racers_per_generation should be divisible by racers_per_map");
//     let num_maps_per_generation = racers_per_generation / racers_per_map;
//     for generation in 0..num_generations {
//         let racers = vec![Racer::new()];
//         for map_idx in 0..num_maps_per_generation {
//             let curr_racers = &racers[map_idx*racers_per_map..(map_idx + 1)*racers_per_map];
//             let curr_map = Map::new(100, 100);
//         }
        
//     }
// }

//& reference 