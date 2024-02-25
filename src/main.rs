
use std::path::Path;
use std::{borrow::BorrowMut, collections::HashMap};

use image::{self, DynamicImage, GenericImageView, Pixel, Rgb};
use rs_graph::{self, Buildable, Builder, IndexGraph};
use rand::{self, seq::SliceRandom};
extern crate ffmpeg_next as ffmpeg;
use ffmpeg::media::Type;
use ffmpeg::util::frame::video::Video;

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
    let (w, h) = energy.dimensions();
    let pixel_nodes = b.add_nodes(w * h); // Add these first so their IDs are 0-indexed. 
    let current_label_node = b.add_node();
    let alpha_node = b.add_node();
    let pixel_id = |x : usize, y : usize| -> usize { (y * w) + x};
    let mut add_weighed_edge = |b : &mut rs_graph::vecgraph::VecGraphBuilder<usize>, n1, n2, weight| {
        let e1 = b.add_edge(n1, n2);
        let e2 = b.add_edge(n2, n1);
        edge_flows.insert(e1, weight);
        edge_flows.insert(e2, weight);
    };
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
        if node_id >= pixel_nodes.len() {
            // If it's source/sink node or one of the aux ones. 
            continue;
        }
        let pixel_id = node_id;
        ret.push(pixel_id);
    }
    return (min_energy, ret);
}

struct LoopEnergy {
    static_cost : LoopEnergyValue,
    temporal_mult : LoopEnergyValue,
    spatial_mult : LoopEnergyValue,
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
        let (period, start_time) = label;
        let loop_time = start_time as i32 + ((time as i32 - start_time as i32).rem_euclid(period as i32));
        return self.pixel_at_time(x, y, (loop_time as u32).try_into().unwrap());
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
        let (period, start_time) = label;
        let end_time = start_time + period;
        if period == 1 {
            // Use the static energy instead. 
            return self.static_cost;
        } else {
            let temporal = pixel_dist(self.pixel_at_time(x, y, start_time), self.pixel_at_time(x, y, end_time)) + pixel_dist(self.pixel_at_time(x, y, start_time + self.frames.len() - 1), self.pixel_at_time(x, y, end_time - 1));
            return temporal * self.temporal_mult;  
        }
    }

    fn dual_energy(&self, x1 : usize, y1: usize, label1 : LoopLabel, x2 : usize, y2: usize, label2 : LoopLabel) -> LoopEnergyValue {
        // spatial terms: measure the difference between neigboring pixels. 
        let mut spatial : LoopEnergyValue = 0;
        for time in 0..self.frames.len() {
            spatial += pixel_dist(self.pixel_at_looped_time(x1, y1, time, label1), self.pixel_at_looped_time(x1, y1, time, label2));
            spatial += pixel_dist(self.pixel_at_looped_time(x2, y2, time, label1), self.pixel_at_looped_time(x2, y2, time, label2));
        }
        let ret = spatial * self.spatial_mult / (self.frames.len() as LoopEnergyValue); 
        return ret
    }
}

fn save_debug_label_img(path : &std::path::Path, labels : &Vec<LoopLabel>, dimensions : (usize, usize)) {
    let (w, h) = dimensions;
    let mut period_img = image::RgbImage::new(w.try_into().unwrap(), h.try_into().unwrap());
    for i in 0..labels.len() {
        let (period, start) = labels[i];
        let x = i % w;
        let y = (i - x) / w;
        period_img.put_pixel(x.try_into().unwrap(), y.try_into().unwrap(), image::Rgb([period * 3, period * 3, period * 3].map(|p| p.try_into().unwrap())));
    }
    period_img.save(path);
}

fn load_video(path : &Path) -> Result<Vec<DynamicImage>, ffmpeg::Error> {
    ffmpeg::init().unwrap();
    let read_in = ffmpeg::format::input(&path);
    match read_in {
        Ok(mut ictx) => {
            let input = ictx
            .streams()
            .best(Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound)?;
            let video_stream_index = input.index();
            let mut context_decoder = ffmpeg::codec::context::Context::new();
            context_decoder.set_parameters(input.parameters())?;
            let mut decoder = context_decoder.decoder().video()?;

            let mut scaler = ffmpeg::software::scaling::context::Context::get(
                decoder.format(),
                decoder.width(),
                decoder.height(),
                ffmpeg::format::Pixel::RGB24,
                decoder.width() / 4,
                decoder.height() / 4,
                ffmpeg::software::scaling::Flags::BILINEAR,
            )?;

            let mut frames = Vec::with_capacity(input.frames().try_into().unwrap());    

            let mut receive_and_process_decoded_frames =
                |decoder: &mut ffmpeg::decoder::Video| -> Result<(), ffmpeg::Error> {
                    let mut decoded = Video::empty();
                    while decoder.receive_frame(&mut decoded).is_ok() {
                        let mut rgb_frame = Video::empty();
                        scaler.run(&decoded, &mut rgb_frame)?;
                        let w = rgb_frame.width();
                        let h = rgb_frame.height();
                        let mut frame_img = image::RgbImage::new(w, h);
                        let raw_data = rgb_frame.data(0);
                        for y in 0..h {
                            for x in 0..w {
                                let buffer_idx = ((y * rgb_frame.stride(0) as u32) + (x * 3)) as usize;
                                frame_img.put_pixel(x, y, Rgb([raw_data[buffer_idx], raw_data[buffer_idx + 1], raw_data[buffer_idx + 2]]));
                            }
                        }
                        frames.push(DynamicImage::ImageRgb8(frame_img)); 
                    }
                    Ok(())
                };

            for (stream, packet) in ictx.packets() {
                if stream.index() == video_stream_index {
                    decoder.send_packet(&packet)?;
                    receive_and_process_decoded_frames(&mut decoder)?;
                }
            }
            decoder.send_eof()?;
            receive_and_process_decoded_frames(&mut decoder)?;
            return Ok(frames);
        },
        Err(e) => {
            return Err(e);
        }
    }

}

fn main() {
    // Load the initial image frames into memory from the input video. 
    ffmpeg::init().unwrap();
    println!("Loading video...");
    let frames = load_video(Path::new("./resources/v2/original_loop.mp4")).unwrap();
    let num_frames = frames.len();
    
    // seed the initial parameters and labels. 
    println!("Generating labels");
    let energy_container = LoopEnergy { frames : frames, static_cost : 1000, temporal_mult : 2, spatial_mult : 1};
    let (w, h) = energy_container.dimensions();
    let mut current_labeling: Vec<LoopLabel> = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        current_labeling.push((1, 0));
    }
    // generate possible labels. 

    let mut possible_labels : Vec<LoopLabel> = Vec::new();
    let minimum_nonstatic_period = 8;
    for s in 0..num_frames {
        possible_labels.push((1, s));
        for p in minimum_nonstatic_period..num_frames+1 {
            possible_labels.push((p, s));
        }
    }
    // figure out optimal per-pixel labelings. 
    let max_iter = 15;
    for i in 0..max_iter {
        let alpha_label = *(possible_labels.choose(&mut rand::thread_rng()).unwrap());
        let (new_energy, new_alpha_idxs) = img_alpha_expansion(&current_labeling, &energy_container, alpha_label);
        for alpha_idx in new_alpha_idxs {
            current_labeling[alpha_idx] = alpha_label;
        }
        println!("Energy: {}", new_energy);
        if new_energy < 10 {
            break;
        }
        let path_str = format!("./periods_{}.png", i);
        save_debug_label_img(std::path::Path::new(&path_str), &current_labeling, (w, h));
    }

    // make the loop images themselves. 
    for i in 0..num_frames {
        let path_str = format!("./resources/v2/output/{}.png", i);
        let mut looped_frame = image::RgbImage::new(w.try_into().unwrap(), h.try_into().unwrap());
        for label_idx in 0..current_labeling.len() {
            let pixel_label = current_labeling[i];
            let x = label_idx % w;
            let y = (label_idx - x) / w;
            looped_frame.put_pixel(x.try_into().unwrap(), y.try_into().unwrap(), energy_container.pixel_at_looped_time(x, y, i, pixel_label).to_rgb());
        }
        looped_frame = image::imageops::resize(&looped_frame, (w * 4).try_into().unwrap(), (h * 4).try_into().unwrap(), image::imageops::FilterType::Gaussian);
        looped_frame.save(std::path::Path::new(&path_str));
    
    }

    // ffmpeg -framerate 30 -i resources/v1_output/%d.png resources/v1_loop.mp4

}
