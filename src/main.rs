
use std::path::Path;

use image::{DynamicImage, GenericImageView};
use loop_rust::{generate_loop, img_alpha_expansion, load_video, save_debug_label_img, ImageEnergy, LoopEnergy, LoopLabel};
use rand::{self, seq::SliceRandom};
extern crate ffmpeg_next as ffmpeg;

fn main() {
    // Load the initial image frames into memory from the input video. 
    ffmpeg::init().unwrap();
    println!("Loading video...");
    let shrink_ratio : usize = 4;
    let orig_frames = load_video(Path::new("./resources/v2/stable.mp4")).unwrap();
    let (orig_w, orig_h) = orig_frames[0].dimensions();
    let frames : Vec<DynamicImage> = orig_frames.clone().into_iter().map(|frame| {
        let shrunk = image::imageops::resize(&frame, frame.width() / (shrink_ratio as u32), frame.height() / (shrink_ratio as u32), image::imageops::FilterType::Gaussian);
        image::DynamicImage::ImageRgba8(shrunk)
    }).collect();
    let num_frames = frames.len();
    
    // seed the initial parameters and labels. 
    println!("Generating labels");
    let energy_container = LoopEnergy { frames : frames, static_cost : 10000, temporal_mult : 8, spatial_mult : 1};
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

    let output_frames = generate_loop(orig_frames, (orig_w.try_into().unwrap(), orig_h.try_into().unwrap()), shrink_ratio, current_labeling);
    for idx in 0..output_frames.len() {
        let frame = output_frames.get(idx).unwrap();
        let pathstr = format!("./resources/v2/output/{}.png", idx);
        frame.save(Path::new(&pathstr));
    }

    // ffmpeg -framerate 30 -i resources/v2/output/%d.png resources/v2/loop.mp4

}
