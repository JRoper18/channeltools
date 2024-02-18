use image;

trait Energy {
    fn num_variables() -> i32;
    fn single_energy(v : i32) -> i32;
    fn dual_energy(v1 : i32, v2: i32) -> i32;
}

fn img_to_graph(img : image::ImageReader) -> Graph {
    // img.dimensions();
}
