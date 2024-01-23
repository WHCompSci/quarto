use std::time::{Duration, SystemTime, UNIX_EPOCH};
use macroquad::prelude::*;
mod neural_net;

#[macroquad::main("Vector Racing")]
async fn main() {
    let map = new_map(100, 100);

    loop {
        clear_background(BLACK);
        draw_map(&map, 4.0);
        next_frame().await
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

fn set_pixel(map: &mut Map,x: usize, y: usize, tile: Tile) {
    map.tiles[y * map.width + x] = tile;
}

fn get_pixel(map: &Map,x: usize, y: usize) -> Tile {
    map.tiles[y * map.width + x]
}



fn generate_map(map: &mut Map) {
    let C0 = vec4(-1., 3., -3., 1.) * 0.5;
    let C1 = vec4(2., -5., 4., -1.) * 0.5;
    let C2 = vec4(-1., 0., 1., 0.) * 0.5;
    let C3 = vec4(0., 2., 0., 0.) * 0.5;
    //Step 1: Generate Random Points
    let margin = 10_usize;
    let N = 4;
    let mut random_points: Vec<Vec2> = vec![];
    let system_time = SystemTime::now();
    rand::srand(system_time.duration_since(UNIX_EPOCH)
    .expect("Time went backwards").as_millis() as u64);
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
            random_points.push(Vec2 {x: x as f32, y: y as f32});
        }
    }
    let mut visited = vec![false; random_points.len()];
    let mut ordered_points: Vec<Vec2> = vec![];
    let mut curr = random_points[0];
    visited[0] = true; //set the first point to visited
    ordered_points.push(curr);
    for _ in 0..N*N {
        //do a linear search through all the remaining points to find the shortest distance
        let (mut shortest, mut shortest_idx) = (f32::INFINITY, 0);
        for (i, next) in random_points.iter().enumerate() {
            if visited[i] {continue;}
            let dist = curr.distance(*next);
            if dist < shortest {
                shortest = dist;
                shortest_idx = i;
            }
        }
        ordered_points.push(random_points[shortest_idx]);
        curr = random_points[shortest_idx];
        visited[shortest_idx] = true;
    }
    //testing set all points in ordered points to walls
    // for(let i = 0; i < )
    //TODO http://alvyray.com/Memos/CG/Pixar/spline77.pdf page 3

}


fn draw_map(map: &Map, width: f32) {
    for y in 0..map.height {
        for x in 0..map.width {
            let color = match get_pixel(map, x, y) {
                Tile::Empty { is_start, is_finish } => WHITE,
                Tile::Wall => BLACK,
            };
            draw_rectangle(width * x as f32, width * y as f32, width, width, color);
        }
    }
    
} 