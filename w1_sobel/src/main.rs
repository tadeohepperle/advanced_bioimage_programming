use image::{io::Reader as ImageReader, ImageBuffer, Luma};
use itertools::Itertools;
use rayon::prelude::*;
use std::{env, time::Instant};

const SOBEL_X: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
const SOBEL_Y: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

//  cargo run -- img/house.jpg img/ball.jpg img/lizard.jpg
fn main() {
    // collect commandline arguments:
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("no argument for the image file supplied.")
    }
    let image_file_paths: Vec<&String> = args.iter().skip(1).collect();
    println!("Images to read: {:?}", image_file_paths);
    for image_path in image_file_paths {
        let img = ImageReader::open(image_path)
            .unwrap()
            .decode()
            .unwrap()
            .grayscale()
            .to_luma8();
        println!("{} with size {}x{}", image_path, img.width(), img.height());

        let timer = Instant::now();
        // println!("    Sobel Operations start");
        let output = sobel(&img);
        print_elapsed(&timer, "    Sobel done");
        output.save(output_image_path(&image_path)).unwrap();
    }
    return;
    // let img = ImageReader::open(image_file_path)
    //     .unwrap()
    //     .decode()
    //     .unwrap()
    //     .grayscale()
    //     .to_luma8();

    // let timer = Instant::now();

    // let sobel_x: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    // let sobel_y: [[f64; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    // println!("Operations start");

    // let output = conv_3x3_with_sqrt(img, vec![sobel_x, sobel_y]);

    // print_elapsed(&timer, "Operations done");
    // output.save("output.jpg").unwrap();
}

fn sobel(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let output_sobel_x = conv_3x3(&img, SOBEL_X);
    let output_sobel_y = conv_3x3(&img, SOBEL_Y);
    let output = add_images(output_sobel_x, output_sobel_y);
    output
}

fn conv_3x3(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    mask: [[f64; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = img.dimensions();
    let mut output = ImageBuffer::new(w, h);

    for x in 1..(w - 1) {
        for y in 1..(h - 1) {
            let p1 = img.get_pixel(x - 1, y - 1).0[0] as f64 * mask[0][0];
            let p2 = img.get_pixel(x - 1, y).0[0] as f64 * mask[0][1];
            let p3 = img.get_pixel(x - 1, y + 1).0[0] as f64 * mask[0][2];
            let p4 = img.get_pixel(x, y - 1).0[0] as f64 * mask[1][0];
            let p5 = img.get_pixel(x, y).0[0] as f64 * mask[1][1];
            let p6 = img.get_pixel(x, y + 1).0[0] as f64 * mask[1][2];
            let p7 = img.get_pixel(x + 1, y - 1).0[0] as f64 * mask[2][0];
            let p8 = img.get_pixel(x + 1, y).0[0] as f64 * mask[2][1];
            let p9 = img.get_pixel(x + 1, y + 1).0[0] as f64 * mask[2][2];
            let val: u8 = ((p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 9.0).abs() as u8;
            output.put_pixel(x, y, Luma([val]));
        }
    }

    return output;
}

fn add_images(
    img1: ImageBuffer<Luma<u8>, Vec<u8>>,
    img2: ImageBuffer<Luma<u8>, Vec<u8>>,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    assert!(img1.dimensions() == img2.dimensions());
    let (w, h) = (img1.width(), img1.height());
    let mut output = ImageBuffer::new(w, h);
    for x in 0..w {
        for y in 0..h {
            let p1 = img1.get_pixel(x, y).0[0] as f64;
            let p2 = img2.get_pixel(x, y).0[0] as f64;

            let val = (p1 * p1 + p2 * p2).sqrt();

            output.put_pixel(x, y, Luma([val as u8]));
        }
    }

    return output;
}

fn output_image_path(input_imag_path: &str) -> String {
    let parts = input_imag_path.split(".").collect::<Vec<&str>>();
    if parts.len() != 2 {
        panic!("path {input_imag_path} is invalid, contains too many dots, 2 expected");
    }
    parts[0].to_owned() + "_sobel." + parts[1]
}

fn print_elapsed(instant: &Instant, title: &str) {
    let elapsed = instant.elapsed();
    println!("{title}: {:.3?}", elapsed);
}
