use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::path::Path;

use image::{self, DynamicImage, GenericImageView, ImageError, Pixel, Rgb};
use rs_graph::{self, Buildable, Builder, IndexGraph};
use rand::{self, seq::SliceRandom};
extern crate ffmpeg_next as ffmpeg;
use ffmpeg::media::Type;
use ffmpeg::util::frame::video::Video;
use itertools::Itertools;

pub type LoopEnergyValue = u32;
pub type LoopLabel = (usize, usize);
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

pub struct LoopEnergy {
    pub static_cost : LoopEnergyValue,
    pub temporal_mult : LoopEnergyValue,
    pub spatial_mult : LoopEnergyValue,
    pub frames : Vec<image::DynamicImage>
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
    pub fn pixel_at_time(&self, x : usize, y : usize, time : usize) -> image::Rgba<u8> {
        let p = self.frames[time % self.frames.len()].get_pixel(x.try_into().unwrap(), y.try_into().unwrap());
        return p;
    }

    pub fn pixel_at_looped_time(&self, x : usize, y : usize, time : usize, label : LoopLabel) -> image::Rgba<u8> {
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

pub fn save_debug_label_img(path : &std::path::Path, labels : &Vec<LoopLabel>, dimensions : (usize, usize)) -> Result<(), ImageError>{
    let (w, h) = dimensions;
    let mut period_img = image::RgbImage::new(w.try_into().unwrap(), h.try_into().unwrap());
    for i in 0..labels.len() {
        let (period, start) = labels[i];
        let x = i % w;
        let y = (i - x) / w;
        period_img.put_pixel(x.try_into().unwrap(), y.try_into().unwrap(), image::Rgb([period * 3, period * 3, period * 3].map(|p| p.try_into().unwrap())));
    }
    return period_img.save(path);
}

pub fn load_video(path : &Path) -> Result<Vec<DynamicImage>, ffmpeg::Error> {
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
                decoder.width(),
                decoder.height(),
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

fn gcd(mut n: usize, mut m: usize) -> usize {
    assert!(n != 0 && m != 0);
    while m != 0 {
    if m < n {
        std::mem::swap(&mut m, &mut n);
    }
    m %= n;
    }
    n
}

fn lcm(n : usize, m : usize) -> usize {
    return n * m / gcd(n, m);
}

pub fn generate_loop(original_frames : Vec<DynamicImage>, orig_dimensions : (usize, usize), shrink_ratio : usize, labeling : Vec<(usize, usize)>) -> Vec<image::ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let num_output_frames : usize = std::cmp::min(300, labeling.clone().into_iter().map(|(period, start_time)| period).unique().reduce(lcm).unwrap_or(original_frames.len()));
    println!("Making {} output frames", num_output_frames);
    // make the loop images themselves. 
    let original_size_energy_container = LoopEnergy { frames : original_frames, static_cost : 0, temporal_mult : 0, spatial_mult : 0};
    let mut output_frames : Vec<image::ImageBuffer<Rgb<u8>, Vec<u8>>> = Vec::with_capacity(num_output_frames);
    let (orig_w, orig_h) = orig_dimensions;
    let w = (orig_w / shrink_ratio) as usize;
    let h = (orig_h / shrink_ratio) as usize;
    for i in 0..num_output_frames {
        let mut looped_frame = image::RgbImage::new(orig_w.try_into().unwrap(), orig_h.try_into().unwrap());
        for label_idx in 0..labeling.len() {
            let pixel_label: (usize, usize) = labeling[label_idx];
            let shrunk_x = label_idx % w;
            let shrunk_y = (label_idx - shrunk_x) / w;
            for dx in 0..shrink_ratio {
                for dy in 0..shrink_ratio {
                    let orig_x: usize = (shrunk_x * shrink_ratio) + dx;
                    let orig_y = (shrunk_y * shrink_ratio) + dy;
                    looped_frame.put_pixel(orig_x.try_into().unwrap(), orig_y.try_into().unwrap(), original_size_energy_container.pixel_at_looped_time(orig_x, orig_y, i, pixel_label).to_rgb());
                }
            }
        } 
        looped_frame = image::imageops::resize(&looped_frame, (w * 4).try_into().unwrap(), (h * 4).try_into().unwrap(), image::imageops::FilterType::Gaussian);
        output_frames.push(looped_frame);
    }
    return output_frames;
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb};

    use crate::LoopEnergy;


    fn single_frame(val : u8) -> DynamicImage{
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([val, val, val]));
        return DynamicImage::ImageRgb8(img);
    }
    #[test]
    fn looped_time_pixels() {
        let frames = vec![single_frame(0), single_frame(1), single_frame(2), single_frame(3)];
        let container = LoopEnergy{frames : frames, static_cost: 0, temporal_mult: 0, spatial_mult: 0 };
        assert_eq!(container.pixel_at_looped_time(0, 0, 0, (4, 0))[0], 0);
        assert_eq!(container.pixel_at_looped_time(0, 0, 1, (4, 0))[0], 1);
        assert_eq!(container.pixel_at_looped_time(0, 0, 2, (4, 0))[0], 2);
        assert_eq!(container.pixel_at_looped_time(0, 0, 3, (4, 0))[0], 3);
        assert_eq!(container.pixel_at_looped_time(0, 0, 4, (4, 0))[0], 0);
        assert_eq!(container.pixel_at_looped_time(0, 0, 5, (4, 0))[0], 1);

        assert_eq!(container.pixel_at_looped_time(0, 0, 0, (4, 1))[0], 0);
        assert_eq!(container.pixel_at_looped_time(0, 0, 1, (4, 1))[0], 1);
        assert_eq!(container.pixel_at_looped_time(0, 0, 2, (4, 1))[0], 2);
        assert_eq!(container.pixel_at_looped_time(0, 0, 3, (4, 1))[0], 3);
        assert_eq!(container.pixel_at_looped_time(0, 0, 4, (4, 1))[0], 0);


        assert_eq!(container.pixel_at_looped_time(0, 0, 0, (2, 1))[0], 2);
        assert_eq!(container.pixel_at_looped_time(0, 0, 1, (2, 1))[0], 1);
        assert_eq!(container.pixel_at_looped_time(0, 0, 2, (2, 1))[0], 2);

        assert_eq!(container.pixel_at_looped_time(0, 0, 0, (1, 3))[0], 3);
        assert_eq!(container.pixel_at_looped_time(0, 0, 1, (1, 3))[0], 3);
        assert_eq!(container.pixel_at_looped_time(0, 0, 2, (1, 3))[0], 3);
    }
}