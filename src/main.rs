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
    let mut visited = vec![false; random_points.len()];
    let mut ordered_points: Vec<(usize, usize)> = vec![];
    let (mut curr_x, mut curr_y) = random_points[0];
    visited[0] = true; //set the first point to visited
    ordered_points.push((curr_x, curr_y));
    for _ in 0..N*N {
        //do a linear search through all the remaining points to find the shortest distance
        let (mut shortest, mut shortest_idx) = (f32::INFINITY, 0);
        for (i, (next_x, next_y)) in random_points.iter().enumerate() {
            if visited[i] {continue;}
            let (dx, dy) = (curr_x as f32 - *next_x as f32, curr_y as f32 - *next_y as f32);
            let dist = (dx*dx + dy*dy).sqrt();
            if dist < shortest {
                shortest = dist;
                shortest_idx = i;
            }
        }
        ordered_points.push(random_points[shortest_idx]);
        (curr_x, curr_y) = random_points[shortest_idx];
        visited[shortest_idx] = true;
    }
    //testing set all points in ordered points to walls
    for ((x1, y1),(x2, y2)) in ordered_points.iter().zip(ordered_points.iter().skip(1)) {
        for t in 0..100 {
            let (x1, x2, y1, y2) = (*x1 as f32, *x2 as f32, *y1 as f32, *y2 as f32);
            let t = t as f32 / 100.0;
            let new_x = (x1*t + x2*(1.0-t)) as usize;
            let new_y = (y1*t + y2*(1.0-t)) as usize;
            set_pixel(map, new_x, new_y, Tile::Wall);
        }
        
    }

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