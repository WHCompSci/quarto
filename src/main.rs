use macroquad::prelude::*;
mod neural_net;

#[macroquad::main("Vector Racing")]
async fn main() {
    loop {
        clear_background(BLACK);
    }
}

#[derive(Debug, Clone, Copy)]
enum Tile {
    Empty { is_start: bool, is_finish: bool },
    Wall,
}

struct Map {
    height: usize,
    width: usize,
    tiles: Vec<Tile>,
}

fn new_map(height: usize, width: usize) -> Map {
    let tiles = vec![
        Tile::Empty {
            is_start: false,
            is_finish: false
        };
        height * width
    ];
    let mut m = Map {
        height,
        width,
        tiles,
    };
    generate_map(&mut m);
    m
}

fn generate_map(map: &mut Map) {
    //Step 1: Generate Random Points
    let margin = 10_usize;
    let N = 4;
    let mut random_points: Vec<(usize, usize)> = vec![];
    for quad_x in 0..N {
        for quad_y in 0..N {
            let x = rand::gen_range(
                margin + quad_x * (map.width - 2 * margin) / N,
                margin + (quad_x + 1) * (map.width - 2 * margin) / N,
            );
            let y = rand::gen_range(
                margin + quad_y * (map.height - 2 * margin) / N,
                margin + (quad_y + 1) * (map.height - 2 * margin) / N,
            );
            random_points.push((x, y));
        }
    }
    let mut distances: Vec<f32> = vec![];
    for (x1, y1) in random_points.iter() {
        for (x2, y2) in random_points.iter() {
            distances.push((((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) as f32).sqrt())
        }
    }
    let visited = vec![false; random_points.len()];
    let curr_point = random_points[0];
    for _ in 0..random_points.len() {
        
    }

}
