
use std::collections::HashMap;

use image::{self, GenericImageView};
use rs_graph::{self, Buildable, Builder};

trait ImageEnergy {
    fn dimensions(&self) -> (usize, usize);
    fn single_energy(&self, x : usize, y : usize, label : u32) -> u32;
    fn dual_energy(&self, x1 : usize, y1: usize, label1 : u32, x2 : usize, y2: usize, label2 : u32) -> u32;
}

pub fn img_to_graph(current_labeling : Vec<Vec<u32>>, energy : &dyn ImageEnergy, alpha_label : u32) {
    let g = rs_graph::VecGraph::<u32>::new_with(|b| {
        let current_label_node = b.add_node();
        let alpha_node = b.add_node();
        let (w, h) = energy.dimensions();
        let pixel_id = |x : usize, y : usize| -> usize { (x * w) + y};
        let pixel_nodes = b.add_nodes(w * h);
        let mut edge_flows: HashMap<rs_graph::vecgraph::Edge, u32>  = HashMap::new();
        for x in 0..w {
            for y in 0..h {
                let pid = pixel_id(x, y);
                let node = *pixel_nodes.get(pid).unwrap();
                let current_label = *((current_labeling.get(x).unwrap().get(y)).unwrap());
                
                // First, add the single energy edges. 
                        
                if current_label != alpha_label {
                    let e1 = b.add_edge(current_label_node, node);
                    edge_flows.insert(e1, energy.single_energy(x, y, current_label));
                }

                let e2 = b.add_edge(alpha_node, node);
                edge_flows.insert(e2, energy.single_energy(x, y, alpha_label));


                // Then, add the dual/smooth-energy edges and additional nodes. 
                
                if x != 0 {
                    
                }
            }
        }
    });
    return g;    
}

fn main() {
    let img: image::DynamicImage = image::open("./resources/v1/0001.png").unwrap();
    img.dimensions();
    img_to_graph(img);
    let mut n = 0;
    const K : u32 = 100;
    for i in 0..K {
        n += i;
    }
    let n2 = (K + 1) * (K / 2);
    println!("n2: {}, n1: {}", n2, n);
    println!("Hello, world!");
}
