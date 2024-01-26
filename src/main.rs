use macroquad::prelude::*;
use neural_net::NeuralNet;
use std::time::UNIX_EPOCH;
mod neural_net;
use std::f64::consts::FRAC_PI_2;

#[macroquad::main("Racetrack Generator")]
async fn main() {
    let mut map = generate_map(100, 100);

    loop {
        clear_background(BLACK);
        draw_map(&map);
        next_frame().await
    }
}
const RACER_NUM_RAYS: usize = 3;
const RACER_FOV_ANGLE: f64 = FRAC_PI_2;
#[derive(Debug, Clone)]
pub struct Racer {
    brain: NeuralNet,
    pos_x: u32,
    pos_y: u32,
    vel_x: i32,
    vel_y: i32,
    score: Option<f64>,
}

impl Racer {
    fn new() -> Self {
        let num_input_nodes = RACER_NUM_RAYS;
        let num_hidden_layers = 10;
        let mut layer_widths = Vec::new();
        // add the width of the input layer
        layer_widths.push(num_input_nodes);
        //add the hidden layers and output layer (9 wide)
        layer_widths.append(&mut vec![9; num_hidden_layers + 1]);
        Self {
            brain: NeuralNet::new(layer_widths),
            pos_x: 0,
            pos_y: 0,
            vel_x: 0,
            vel_y: 0,
            score: None,
        }
    }
}

struct Map {
    height: usize,
    width: usize,
    tiles: Vec<Tile>,
}

#[derive(Copy, Clone, PartialEq)]
enum Tile {
    Empty,
    Wall,
    Start,
    End,
}
fn get_system_time() -> std::time::SystemTime {
    std::time::SystemTime::now()
}
fn generate_map(height: usize, width: usize) -> Map {
    let mut tiles = vec![Tile::Wall; height * width];
    let mut random_points: Vec<Vec2> = Vec::new();
    //seed random number generator
    //get the current system time as a seed, get_time() doesnt work because it is always the same
    rand::srand(
        get_system_time()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u64,
    );
    //generate random points spread throughout the map, each point will be a certain distance away from the next
    let margin = 20;
    for _ in 0..20 {
        let x = rand::gen_range(margin, width - margin) as f32;
        let y = rand::gen_range(margin, height - margin) as f32;
        random_points.push(vec2(x, y));
    }
    let hull = convex_hull(&random_points);

    let rasterized_coords = get_rasterized_circle_coords(5);
    let mut m = Map {
        height,
        width,
        tiles,
    };
    let start_index_i = rand::gen_range(0, hull.len());
    let start_index_t = rand::gen_range(0, 100_usize);
    for i in 0..hull.len() {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % hull.len()];
        let m0 = vec2(
            (p1.x - hull[(i + hull.len() - 1) % hull.len()].x) / 2.0,
            (p1.y - hull[(i + hull.len() - 1) % hull.len()].y) / 2.0,
        );
        let m1 = vec2(
            (hull[(i + 2) % hull.len()].x - p0.x) / 2.0,
            (hull[(i + 2) % hull.len()].y - p0.y) / 2.0,
        );
        for t in 0..100 {
            let p = hermite(p0, p1, m0, m1, t as f32 / 100.0);
            let x = p.x.round() as usize;
            let y = p.y.round() as usize;
            if x >= width || y >= height {
                continue;
            }
            set_circle(&mut m, x, y, &rasterized_coords, Tile::Empty);
            if i != start_index_i || t != start_index_t {
                continue;
            }
            //place start

            let p_next = hermite(p0, p1, m0, m1, (t + 1) as f32 / 100.0);
            let a = p_next.x - p.x;
            let b = p_next.y - p.y;

            let c = -(1. - a * a / (a * a + b * b)).sqrt() * b.signum();
            let d = (a * a / (a * a + b * b)).sqrt() * a.signum();
            println!("a={} b={} c={} d={}", a, b, c, d);
            let mut mag = 0.0;
            loop {
                let x_coord = x as i32 + (c * mag as f32).round() as i32;
                let y_coord = y as i32 + (d * mag as f32).round() as i32;
                if x_coord < 0
                    || x_coord >= m.width as i32
                    || y_coord < 0
                    || y_coord >= m.width as i32
                    || *get_tile(&m, x_coord as u32, y_coord as u32) == Tile::Wall
                {
                    break;
                }
                println!(
                    "x-component={} y-component={}",
                    (c * mag as f32),
                    (d * mag as f32)
                );
                set_tile(&mut m, x_coord as u32, y_coord as u32, Tile::Start);
                mag += 1.0;
            }
            mag = 0.0;
            loop {
                let x_coord = x as i32 + (c * mag as f32).round() as i32;
                let y_coord = y as i32 + (d * mag as f32).round() as i32;
                if x_coord < 0
                    || x_coord >= m.width as i32
                    || y_coord < 0
                    || y_coord >= m.width as i32
                    || *get_tile(&m, x_coord as u32, y_coord as u32) == Tile::Wall
                {
                    break;
                }
                println!(
                    "x-component={} y-component={}",
                    (c * mag as f32),
                    (d * mag as f32)
                );
                set_tile(&mut m, x_coord as u32, y_coord as u32, Tile::Start);
                mag -= 1.0;
            }
            //set_tile(&mut m, x as u32, y as u32, Tile::End);
        }
    }

    return m;
}

fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    //given a list of points, return a list of points that make up the convex hull
    //gift wrapping algorithm

    //sort points by x value
    let mut points = points.to_vec();
    points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let mut hull: Vec<Vec2> = Vec::new();
    let mut p = points[0];
    let mut i = 0;
    loop {
        hull.push(p);
        let mut q = points[(i + 1) % points.len()];
        for j in 0..points.len() {
            if orientation(p, points[j], q) == 2 {
                q = points[j];
            }
        }
        i = points.iter().position(|&x| x == q).unwrap();
        p = q;
        if p == hull[0] {
            break;
        }
    }
    return hull;
}
fn orientation(p: Vec2, q: Vec2, r: Vec2) -> i32 {
    //given 3 points, return 0 if they are colinear, 1 if clockwise, 2 if counterclockwise
    let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if val == 0.0 {
        return 0;
    }
    if val > 0.0 {
        return 1;
    }
    return 2;
}

fn hermite(p0: Vec2, p1: Vec2, v0: Vec2, v1: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let a = 2.0 * p0 - 2.0 * p1 + v0 + v1;
    let b = -3.0 * p0 + 3.0 * p1 - 2.0 * v0 - v1;
    let c = v0;
    let d = p0;
    a * t3 + b * t2 + c * t + d
}

fn draw_map(map: &Map) {
    let tile_size = 5.0;
    for (i, tile) in map.tiles.iter().enumerate() {
        let x = (i % map.width) as f32 * tile_size;
        let y = (i / map.width) as f32 * tile_size;
        match tile {
            Tile::Empty => {
                draw_rectangle(x, y, tile_size, tile_size, WHITE);
                //draw grid dot
                draw_circle(x + tile_size / 2.0, y + tile_size / 2.0, 0.5, BLACK);
            }
            Tile::Wall => {
                draw_rectangle(x, y, tile_size, tile_size, BLACK);
            }
            Tile::Start => {
                draw_rectangle(x, y, tile_size, tile_size, GREEN);
            }
            Tile::End => {
                draw_rectangle(x, y, tile_size, tile_size, RED);
            }
        }
    }
}

fn set_circle(map: &mut Map, x: usize, y: usize, rasterized_coords: &Vec<Vec2>, tile: Tile) {
    for coord in rasterized_coords {
        for (a, b) in [(1, 1), (-1, 1), (1, -1), (-1, -1)] {
            let x = x as i32 + coord.x as i32 * a;
            let y = y as i32 + coord.y as i32 * b;
            if x < 0 || y < 0 || x >= map.width as i32 || y >= map.height as i32 {
                continue;
            }
            let curr_tile = *get_tile(&map, x as u32, y as u32);
            if curr_tile == Tile::Start || curr_tile == Tile::End {
                continue;
            }
            set_tile(map, x as u32, y as u32, tile);
        }
    }
}

fn set_tile(map: &mut Map, x: u32, y: u32, tile: Tile) {
    map.tiles[(y * map.width as u32 + x) as usize] = tile;
}
fn get_tile(map: &Map, x: u32, y: u32) -> &Tile {
    &map.tiles[(y * map.width as u32 + x) as usize]
}

fn get_rasterized_circle_coords(radius: usize) -> Vec<Vec2> {
    let mut coords = Vec::new();
    for i in 0..radius {
        for j in 0..radius {
            if i * i + j * j < radius * radius {
                coords.push(vec2(i as f32, j as f32));
            }
        }
    }
    return coords;
}

fn train_racers(population_size: u32) {
    let map = generate_map(100, 100);
    let racers = vec![Racer::new(); population_size as usize];
    let mut scores: Vec<i32> = racers.iter().map(|racer| test_racer(&map, racer)).collect();
}

fn test_racer(map: &Map, racer: &Racer) -> i32 {
    unimplemented!()
}
