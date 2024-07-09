use std::error::Error;
use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};
use std::path::PathBuf;
use structopt::StructOpt;
use image::GenericImageView;


#[derive(StructOpt)]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(parse(from_os_str))]
    output: PathBuf,

    #[structopt(short, long, default_value = "42")]
    percentage: f32,


}

#[derive(Copy, Clone, Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

fn increase_bbox_size(bbox: &BBox, img_width: u32, img_height: u32, percentage: f32) -> (u32, u32, u32, u32) {
    let width = bbox.x2 - bbox.x1;
    let height = bbox.y2 - bbox.y1;

    let expand_x = width * percentage;
    let expand_y = height * percentage;

    let x1 = (bbox.x1 - expand_x).max(0.0) as u32;
    let y1 = (bbox.y1 - expand_y).max(0.0) as u32;
    let x2 = (bbox.x2 + expand_x).min(img_width as f32) as u32;
    let y2 = (bbox.y2 + expand_y).min(img_height as f32) as u32;

    (x1, y1, x2, y2)
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    // Load the graph as a byte array
    let model = include_bytes!("mtcnn.pb");

    // Create a TensorFlow graph from the model
    let mut graph = Graph::new();
    graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

    let input_image = image::open(&opt.input)?;
    let (img_width, img_height) = input_image.dimensions();

    let mut flattened: Vec<f32> = Vec::new();
    for (_x, _y, rgb) in input_image.pixels() {
        flattened.push(rgb[2] as f32);
        flattened.push(rgb[1] as f32);
        flattened.push(rgb[0] as f32);
    }

    // The `input` tensor expects BGR pixel data.
    let input = Tensor::new(&[img_height as u64, img_width as u64, 3])
        .with_values(&flattened)?;

    // Use input params from the existing module
    let min_size = Tensor::new(&[]).with_values(&[20f32])?;
    let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
    let factor = Tensor::new(&[]).with_values(&[0.709f32])?;

    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
    args.add_feed(&graph.operation_by_name_required("thresholds")?, 0, &thresholds);
    args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);
    args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);

    // Request the following outputs after the session runs
    let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
    let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);

    let session = Session::new(&SessionOptions::new(), &graph)?;
    session.run(&mut args)?;

    let bbox_res: Tensor<f32> = args.fetch(bbox)?;
    let prob_res: Tensor<f32> = args.fetch(prob)?;

    let bboxes: Vec<_> = bbox_res
        .chunks_exact(4)
        .zip(prob_res.iter())
        .map(|(bbox, &prob)| BBox {
            y1: bbox[0],
            x1: bbox[1],
            y2: bbox[2],
            x2: bbox[3],
            prob,
        })
        .collect();

    println!("BBox Length: {}, BBoxes:{:#?}", bboxes.len(), bboxes);

    // Change input_image since it is not needed.
    let mut output_image = input_image.to_rgba8();

    let extension = opt.output.extension().unwrap().to_str().unwrap();

    for (_i, bbox) in bboxes.iter().enumerate() {
        let (x1, y1, x2, y2) = increase_bbox_size(bbox, img_width, img_height, opt.percentage);

        let crop_width = x2 - x1;
        let crop_height = y2 - y1;

        // Crop the region inside the expanded bounding box
        let cropped_image = image::imageops::crop(&mut output_image, x1, y1, crop_width, crop_height).to_image();

        // Save the cropped image with a unique filename
        let cropped_output_path = opt.output.with_file_name(format!("{}.{}", opt.output.file_stem().unwrap().to_str().unwrap(), extension));
        cropped_image.save(cropped_output_path)?;
    }

    Ok(())
}
