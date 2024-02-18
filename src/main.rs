
use std::{borrow::BorrowMut, collections::HashMap};

use image::{self, GenericImageView, Pixel};
use rs_graph::{self, traits::DirectedEdge, Buildable, Builder, IndexGraph};
use rand::{self, seq::SliceRandom};

type LoopEnergyValue = u32;
type LoopLabel = (usize, usize);

pub trait ImageEnergy<L, E> {
    fn dimensions(&self) -> (usize, usize);
    fn single_energy(&self, x : usize, y : usize, label : L) -> E;
    fn dual_energy(&self, x1 : usize, y1: usize, label1 : L, x2 : usize, y2: usize, label2 : L) -> E;
}

pub fn img_alpha_expansion<L, E>(current_labeling : &Vec<L>, energy : &dyn ImageEnergy<L, E>, alpha_label : L) -> (E, Vec<usize>) where L: Eq + Copy, E : num_traits::NumAssign + Ord + Copy{
    let mut edge_flows: HashMap<rs_graph::vecgraph::Edge<usize>, E> = HashMap::new();
    let mut b: rs_graph::vecgraph::VecGraphBuilder<usize> = rs_graph::VecGraph::new_builder();
    let current_label_node = b.add_node();
    let alpha_node = b.add_node();
    let (w, h) = energy.dimensions();
    let pixel_id = |x : usize, y : usize| -> usize { (y * w) + x};
    let mut add_weighed_edge = |b : &mut rs_graph::vecgraph::VecGraphBuilder<usize>, n1, n2, weight| {
        let e1 = b.add_edge(n1, n2);
        let e2 = b.add_edge(n2, n1);
        edge_flows.insert(e1, weight);
        edge_flows.insert(e2, weight);
    };
    let pixel_nodes = b.add_nodes(w * h);
    println!("Building graph");
    for x in 0..w {
        for y in 0..h {
            let pid = pixel_id(x, y);
            let node = pixel_nodes[pid];
            let current_label = current_labeling[pixel_id(x, y)];
            // First, add the single energy edges. 
                    
            if current_label != alpha_label {
                add_weighed_edge(b.borrow_mut(),current_label_node, node, energy.single_energy(x, y, current_label));
            }

            add_weighed_edge(b.borrow_mut(),alpha_node, node, energy.single_energy(x, y, alpha_label));


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
            if y != h - 1 {
                neighbors.push((x, y+1));
            }
            for (nx, ny) in neighbors {
                let neighbor_node = pixel_nodes[pixel_id(nx, ny)];
                let neighbor_label = current_labeling[pixel_id(nx, ny)];
                if neighbor_label != current_label {
                    let aux_node = b.add_node();
                    add_weighed_edge(b.borrow_mut(),node, aux_node, energy.dual_energy(x, y, current_label, nx, ny, alpha_label));
                
                    add_weighed_edge(b.borrow_mut(),aux_node, neighbor_node, energy.dual_energy(x, y, alpha_label, nx, ny, neighbor_label));

                    add_weighed_edge(b.borrow_mut(),current_label_node, aux_node, energy.dual_energy(x, y, current_label, nx, ny, neighbor_label));
                } else {
                    add_weighed_edge(b.borrow_mut(),node, neighbor_node, energy.dual_energy(x, y, current_label, nx, ny, alpha_label));
                }
            }
        }
    }
    let g = b.into_graph();
    let (min_energy, _flows, min_cut) = rs_graph::maxflow::dinic::dinic(&g, current_label_node, alpha_node, |edge| edge_flows[&edge]);
    let mut ret : Vec<usize> = Vec::with_capacity(min_cut.len() - 1);
    for node in min_cut {
        let node_id = g.node_id(node);
        if node_id < 2 {
            // If it's source/sink node. 
            continue;
        }
        let pixel_id = node_id - 2;
        ret.push(pixel_id);
    }
    return (min_energy, ret);
}

struct LoopEnergy {
    static_cost : LoopEnergyValue,
    frames : Vec<image::DynamicImage>
}

fn pixel_dist(p1 : image::Rgba<u8>, p2 : image::Rgba<u8>) -> LoopEnergyValue {
    let c1 = p1.channels();
    let c2 = p2.channels();
    let mut dist = 0;
    for i in 0..3 {
        let diff = (c1[i] as i32) - (c2[i] as i32);
        dist += diff * diff;
    }
    return dist as LoopEnergyValue;
}

impl LoopEnergy {
    fn pixel_at_time(&self, x : usize, y : usize, time : usize) -> image::Rgba<u8> {
        let p = self.frames[time % self.frames.len()].get_pixel(x.try_into().unwrap(), y.try_into().unwrap());
        return p;
    }

    fn pixel_at_looped_time(&self, x : usize, y : usize, time : usize, label : LoopLabel) -> image::Rgba<u8> {
        let (start_time, period) = label;
        let num_frames = self.frames.len();
        let loop_time = start_time + ((num_frames + time - start_time) % period);
        return self.pixel_at_time(x, y, loop_time);
    }
}

impl ImageEnergy<LoopLabel, LoopEnergyValue> for LoopEnergy {

    // Labels here are period numbers. 
    fn dimensions(&self) -> (usize, usize) {
        let dim_uncast = self.frames[0].dimensions();
        return (dim_uncast.0.try_into().unwrap(), dim_uncast.1.try_into().unwrap());
    }

    fn single_energy(&self, x : usize, y : usize, label : LoopLabel) -> LoopEnergyValue {
        // temporal energy doesn't rely on neigbors in (x, y) space. 
        let (start_time, period) = label;
        let end_time = start_time + period;
        if period == 1 {
            // Use the static energy instead. 
            return self.static_cost;
        } else {
            let temporal = pixel_dist(self.pixel_at_time(x, y, start_time), self.pixel_at_time(x, y, end_time)) + pixel_dist(self.pixel_at_time(x, y, start_time - 1), self.pixel_at_time(x, y, end_time - 1));
            return temporal;    
        }
    }

    fn dual_energy(&self, x1 : usize, y1: usize, label1 : LoopLabel, x2 : usize, y2: usize, label2 : LoopLabel) -> LoopEnergyValue {
        // spatial terms: measure the difference between neigboring pixels. 
        let mut spatial : LoopEnergyValue = 0;
        for time in 0..self.frames.len() {
            spatial += pixel_dist(self.pixel_at_looped_time(x1, y1, time, label1), self.pixel_at_looped_time(x1, y1, time, label2));
            spatial += pixel_dist(self.pixel_at_looped_time(x2, y2, time, label1), self.pixel_at_looped_time(x2, y2, time, label2));
        }
        let ret = spatial / (self.frames.len() as LoopEnergyValue); 
        return ret
    }
}

fn main() {
    let paths = std::fs::read_dir("./resources/v1").unwrap();
    let mut frames = Vec::new();
    for path in paths {
        let img: image::DynamicImage = image::open(path.unwrap().path()).unwrap();
        let (w, h) = img.dimensions();
        frames.push(image::DynamicImage::ImageRgba8(image::imageops::resize(&img, w / 4, h / 4, image::imageops::FilterType::Nearest))); 
    }
    let num_frames = frames.len();
    let energy_container = LoopEnergy { frames : frames, static_cost : 20 };
    let (w, h) = energy_container.dimensions();
    let mut current_labeling: Vec<LoopLabel> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        current_labeling.push((2, 2));
    }
    let mut possible_labels : Vec<LoopLabel> = Vec::new();
    for p in 1..num_frames+1 {
        for s in 0..num_frames {
            possible_labels.push((p, s));
        }
    }
    let mut i = 0;
    loop {
        let alpha_label = *(possible_labels.choose(&mut rand::thread_rng()).unwrap());
        i += 1;
        let (new_energy, new_alpha_idxs) = img_alpha_expansion(&current_labeling, &energy_container, alpha_label);
        for alpha_idx in new_alpha_idxs {
            current_labeling[alpha_idx] = alpha_label;
            println!("{}", alpha_idx);
        }
        println!("Energy: {}", new_energy);
        if new_energy < 10 {
            break;
        }
    }

}
