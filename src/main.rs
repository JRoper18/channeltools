
use std::collections::HashMap;

use image::{self, GenericImageView};
use rs_graph::{self, Buildable, Builder, IndexGraph};

trait ImageEnergy {
    fn dimensions(&self) -> (usize, usize);
    fn single_energy(&self, x : usize, y : usize, label : u32) -> u32;
    fn dual_energy(&self, x1 : usize, y1: usize, label1 : u32, x2 : usize, y2: usize, label2 : u32) -> u32;
}

pub fn img_alpha_expansion(current_labeling : Vec<Vec<u32>>, energy : &dyn ImageEnergy, alpha_label : u32) -> Vec<(usize, usize)> {
    let mut edge_flows: HashMap<rs_graph::vecgraph::Edge, u32>  = HashMap::new();
    let mut b: rs_graph::vecgraph::VecGraphBuilder<u32> = rs_graph::VecGraph::new_builder();
    let current_label_node = b.add_node();
    let alpha_node = b.add_node();
    let (w, h) = energy.dimensions();
    let pixel_id = |x : usize, y : usize| -> usize { (x * w) + y};
    let pixel_nodes = b.add_nodes(w * h);
    let mut node_id_to_pixel : HashMap<usize, (usize, usize)> = HashMap::new();
    for x in 0..w {
        for y in 0..h {
            let pid = pixel_id(x, y);
            let node = pixel_nodes[pid];
            node_id_to_pixel.insert(b.node2id(node), (x, y)); 
            let current_label = current_labeling[x][y];
            
            // First, add the single energy edges. 
                    
            if current_label != alpha_label {
                let e1 = b.add_edge(current_label_node, node);
                edge_flows.insert(e1, energy.single_energy(x, y, current_label));
            }

            let e2 = b.add_edge(alpha_node, node);
            edge_flows.insert(e2, energy.single_energy(x, y, alpha_label));


            // Then, add the dual/smooth-energy edges and additional nodes. 
            // We do this for every neighbor pair. 
            let mut neighbors: Vec<(usize, usize)> = Vec::with_capacity(4);
            if x != 0 {
                neighbors.push((x-1, y));
            }
            if y != 0 {
                neighbors.push((x, y-1));
            }
            if x != w - 1 {
                neighbors.push((x+1, y));
            }
            if y != h + 1 {
                neighbors.push((x, y+1));
            }
            for (nx, ny) in neighbors {
                let neighbor_node = pixel_nodes[pixel_id(nx, ny)];
                let neighbor_label = current_labeling[nx][ny];
                if neighbor_label != current_label {
                    let aux_node = b.add_node();
                    let e3 = b.add_edge(node, aux_node);
                    edge_flows.insert(e3, energy.dual_energy(x, y, current_label, nx, ny, alpha_label));
                
                    let e4 = b.add_edge(aux_node, neighbor_node);
                    edge_flows.insert(e3, energy.dual_energy(x, y, alpha_label, nx, ny, neighbor_label));

                    let e5 = b.add_edge(current_label_node, aux_node);
                    edge_flows.insert(e3, energy.dual_energy(x, y, current_label, nx, ny, neighbor_label));
                } else {
                    let e6 = b.add_edge(node, neighbor_node);
                    edge_flows.insert(e6, energy.dual_energy(x, y, current_label, nx, ny, alpha_label));
                }
            }
        }
    }
    let g = b.into_graph();
    let (_energy, _flows, min_cut) = rs_graph::maxflow::dinic::dinic(&g, current_label_node, alpha_node, |edge| edge_flows[&edge]);
    let mut ret : Vec<(usize, usize)> = Vec::with_capacity(min_cut.len());
    for node in min_cut {
        ret.push(node_id_to_pixel[&g.node_id(node)]);
    }
    return ret;
}


fn main() {
    let img: image::DynamicImage = image::open("./resources/v1/0001.png").unwrap();    
    let mut n = 0;
    const K : u32 = 100;
    for i in 0..K {
        n += i;
    }
    let n2 = (K + 1) * (K / 2);
    println!("n2: {}, n1: {}", n2, n);
    println!("Hello, world!");
}
