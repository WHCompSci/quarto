use macroquad::math::{vec2, Vec2};
use macroquad::prelude::Color;
use macroquad::{color_u8, prelude as mq};
use neural_net::NeuralNet;
use rand::seq::SliceRandom;
use rand::Rng;
use std::f32::consts::FRAC_PI_4;
use std::time::UNIX_EPOCH;
const PX_SIZE: f32 = 8.0;
mod neural_net;
use std::f32::consts::FRAC_PI_2;

#[macroquad::main("Racetrack Generator")]
async fn main() {
    let population_size = 100;
    let num_generations = 1_000_000;
    let map = generate_map(100, 150);
    let mut racers = vec![Racer::new(); population_size as usize];
    let mut rng = rand::thread_rng();

    for generation in 0..num_generations {
        let do_drawing = generation % 10 == 0;
        if do_drawing {
            mq::clear_background(mq::BLACK);
            draw_map(&map);
        }
        
        for racer in racers.iter_mut() {
            racer.score = test_racer(&map, racer, do_drawing);
        }
        let avg_score = racers.iter().map(|racer| racer.score).sum::<i32>() as f32 / population_size as f32;
        
        //print scores
        for racer in racers.iter() {
            //print network weights avg
        }
        let mut new_racers = Vec::new();
        for _ in 0..population_size {
            
            let mut parent1 = racers
                .choose_weighted(&mut rng, |racer| racer.score)
                .unwrap()
                .clone();
            let mut parent2 = racers
                .choose_weighted(&mut rng, |racer| racer.score)
                .unwrap()
                .clone();
            let mut child = Racer::new();
            child.brain = NeuralNet::crossover(&mut parent1.brain, &mut parent2.brain);
            child.brain.mutate(0.1);
            new_racers.push(child);
        }
        //add the best racer from the previous generation
        let best_racer = racers.iter().max_by(|a, b| a.score.cmp(&b.score)).unwrap().clone();
        let second_best_racer = racers.iter().filter(|x| x.score != best_racer.score).max_by(|a, b| a.score.cmp(&b.score)).unwrap().clone();
        racers = new_racers;
        
        racers[0] = best_racer.clone();
        let best_racer_mutation_rate = (best_racer.score - second_best_racer.score) as f64 / (10000.0 * best_racer.score as f64);
        racers[0].brain.mutate(best_racer_mutation_rate);
        if do_drawing {
            println!(
            "Finished testing racers scores for generation: {}, Average Score={}",
            generation, 
            avg_score
            
        );
            mq::next_frame().await;
        }
        // mq::next_frame().await
    }
    // loop {
    //     mq::clear_background(mq::BLACK);

    //     mq::next_frame().await
    // }
}
const RACER_NUM_RAYS: usize = 4;
const RACER_FOV_ANGLE: f32 = FRAC_PI_2;
#[derive(Debug, Clone)]
pub struct Racer {
    brain: NeuralNet,
    pos_x: u32,
    pos_y: u32,
    vel_x: i32,
    vel_y: i32,
    score: i32,
}

impl Racer {
    fn new() -> Self {
        let num_input_nodes = RACER_NUM_RAYS + 2;
        let num_hidden_layers = 2;
        let mut layer_widths = Vec::new();
        // add the width of the input layer
        layer_widths.push(num_input_nodes);
        //add the hidden layers and output layer (9 wide)
        layer_widths.append(&mut vec![18; num_hidden_layers]);
        layer_widths.append(&mut vec![9; 1]);
        Self {
            brain: NeuralNet::new(layer_widths),
            pos_x: 0,
            pos_y: 0,
            vel_x: 0,
            vel_y: 0,
            score: 0,
        }
    }
}

struct Map {
    height: usize,
    width: usize,
    tiles: Vec<Tile>,
    tile_distances: Vec<u32>,
    start_tiles: Vec<(usize, usize)>,
    end_tiles: Vec<(usize, usize)>,
    starting_vec: Vec2,
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
    mq::rand::srand(
        get_system_time()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u64,
    );
    //generate random points spread throughout the map, each point will be a certain distance away from the next
    let margin = 20;
    for _ in 0..15 {
        let x = mq::rand::gen_range(margin, width - margin) as f32;
        let y = mq::rand::gen_range(margin, height - margin) as f32;
        random_points.push(vec2(x, y));
    }
    let mut hull = convex_hull(&random_points);
    let mut m = Map {
        height,
        width,
        tiles,
        tile_distances: vec![0; height * width],
        start_tiles: Vec::new(),
        end_tiles: Vec::new(),
        starting_vec: vec2(0.0, 0.0),
    };
    let center = vec2(width as f32 / 2.0, height as f32 / 2.0);
    //replace center point into hull in random location
    // hull.insert(mq::rand::gen_range(0, hull.len()), center);
    // let hl  = hull.len();
    // hull[mq::rand::gen_range(0,hl )] = center;

    let (p0, p1, m0, m1, t, p, x, y) = carve_out_path(hull, width, height, &mut m).unwrap();

    //place start

    let p_next = hermite(p0, p1, m0, m1, (t + 1) as f32 / 100.0);
    let a = p_next.x - p.x;
    let b = p_next.y - p.y;

    let c = -(1. - a * a / (a * a + b * b)).sqrt() * b.signum();
    let d = (a * a / (a * a + b * b)).sqrt() * a.signum();

    let starting_vec_x = if a.abs() > b.abs() {
        a.signum() as i32
    } else {
        0
    };
    let starting_vec_y = if a.abs() > b.abs() {
        0
    } else {
        b.signum() as i32
    };
    m.starting_vec = vec2(starting_vec_x as f32, starting_vec_y as f32);
    
    for inc in [-1.0, 1.0] {
        let mut mag = inc;
        loop {
            let x_coord = x as i32 + (c * mag as f32).round() as i32;
            let y_coord = y as i32 + (d * mag as f32).round() as i32;
            if x_coord < 0
                || x_coord >= m.width as i32
                || y_coord < 0
                || y_coord >= m.width as i32
                || (get_tile(&m, x_coord as u32, y_coord as u32) == &Tile::Wall)
            {
                break;
            }
            
            draw_start_line(&mut m, x_coord, y_coord, starting_vec_x, starting_vec_y);

            mag += inc;
        }
        draw_start_line(&mut m, x as i32, y as i32, starting_vec_x, starting_vec_y);
    }
    find_distances(&mut m);
    println!("distances: {:?}", m.tile_distances);
    m
    //set tile distances using BFS

    //set_tile(&mut m, x as u32, y as u32, Tile::End);
}

fn draw_start_line(m: &mut Map, x_coord: i32, y_coord: i32, starting_vec_x: i32, starting_vec_y: i32) {
    set_tile(
        m,
        (x_coord + starting_vec_x) as u32,
        (y_coord + starting_vec_y) as u32,
        Tile::Start,
    );
    m.start_tiles.push((
        (x_coord + starting_vec_x) as usize,
        (y_coord + starting_vec_y) as usize,
    ));
    set_tile(m, x_coord as u32, y_coord as u32, Tile::Wall);
    set_tile(
        m,
        (x_coord - starting_vec_x) as u32,
        (y_coord - starting_vec_y) as u32,
        Tile::Wall,
    );
    set_tile(
        m,
        (x_coord - 2*starting_vec_x) as u32,
        (y_coord - 2*starting_vec_y) as u32,
        Tile::End,
    );
    m.end_tiles.push((
        (x_coord - 2*starting_vec_x) as usize,
        (y_coord - 2*starting_vec_y) as usize,
    ));
}

fn carve_out_path(
    hull: Vec<Vec2>,
    width: usize,
    height: usize,
    m: &mut Map,
) -> Option<(Vec2, Vec2, Vec2, Vec2, usize, Vec2, usize, usize)> {
    let rasterized_coords = get_rasterized_circle_coords(6);
    let mut out = None;
    let start_index_i = mq::rand::gen_range(0, hull.len());
    let start_index_t = mq::rand::gen_range(25, 75_usize);
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
            set_circle(m, x, y, &rasterized_coords, Tile::Empty);
            if i == start_index_i && t != start_index_t {
                out = Some((p0, p1, m0, m1, t, p, x, y));
            }
        }
    }
    out
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

fn find_distances(m: &mut Map) {
    let mut queue: Vec<(usize, usize)> = Vec::new();
    let mut visited: Vec<bool> = vec![false; m.width * m.height];
    for (x, y) in m.end_tiles.iter() {
        queue.push((*x, *y));
        visited[y * m.width + x] = true;
    }
    while !queue.is_empty() {
        let (x, y) = queue.remove(0);
        let curr_dist = m.tile_distances[y * m.width + x];
        for (a, b) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let x = x as i32 + a;
            let y = y as i32 + b;
            if x < 0
                || y < 0
                || x >= m.width as i32
                || y >= m.height as i32
                || visited[y as usize * m.width + x as usize]
                || m.tiles[y as usize * m.width + x as usize] == Tile::Wall
            {
                continue;
            }
            visited[y as usize * m.width + x as usize] = true;
            m.tile_distances[y as usize * m.width + x as usize] = curr_dist + 1;
            queue.push((x as usize, y as usize));
        }
    }
    // subtract all distances from the max distance in the array
    let max_dist = m.tile_distances.iter().max().unwrap().clone();
    for (i, dist) in m.tile_distances.iter_mut().enumerate() {
        if m.tiles[i] == Tile::Wall
        {
            *dist = 0;
            continue;
        }
        *dist = max_dist - *dist + 1;
    }
    // println!("distances: {:?}", m.tile_distances);
}

fn draw_map(map: &Map) {
    let tile_size = PX_SIZE;
    for (i, tile) in map.tiles.iter().enumerate() {
        let x = (i % map.width) as f32 * tile_size;
        let y = (i / map.width) as f32 * tile_size;
        match tile {
            Tile::Empty => {
                mq::draw_rectangle(x, y, tile_size, tile_size, mq::WHITE);
                //draw grid dot
                mq::draw_circle(x + tile_size / 2.0, y + tile_size / 2.0, 1.0, color_u8!(map.tile_distances[i] as u8,  24, 255, 255));
            }
            Tile::Wall => {
                mq::draw_rectangle(x, y, tile_size, tile_size, mq::BLACK);
            }
            Tile::Start => {
                mq::draw_rectangle(x, y, tile_size, tile_size, mq::GREEN);
            }
            Tile::End => {
                mq::draw_rectangle(x, y, tile_size, tile_size, mq::RED);
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
            if curr_tile == Tile::Wall || curr_tile == Tile::Empty {
                set_tile(map, x as u32, y as u32, tile);
            }
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

fn train_racers(population_size: u32) -> Vec<Racer> {
    //genetic algorithm
    let map = generate_map(100, 100);
    let mut racers = vec![Racer::new(); population_size as usize];
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        mq::clear_background(mq::BLACK);
        for racer in racers.iter_mut() {
            racer.score = test_racer(&map, racer, false);
        }
        let mut new_racers = Vec::new();
        for _ in 0..population_size {
            let mut parent1 = racers
                .choose_weighted(&mut rng, |racer| racer.score)
                .unwrap()
                .clone();
            let mut parent2 = racers
                .choose_weighted(&mut rng, |racer| racer.score)
                .unwrap()
                .clone();
            let mut child = Racer::new();
            child.brain = NeuralNet::crossover(&mut parent1.brain, &mut parent2.brain);
            child.brain.mutate(0.1);
            new_racers.push(child);
        }
        racers = new_racers;
    }
    racers
}

fn test_racer(map: &Map, racer: &mut Racer, do_drawing: bool) -> i32 {

    let angle_between_rays = RACER_FOV_ANGLE / RACER_NUM_RAYS as f32;
    let ray_angles = (1..=RACER_NUM_RAYS)
        .map(|x| x as f32 * angle_between_rays - RACER_FOV_ANGLE / 2.0)
        .collect::<Vec<f32>>();
    //given a map and a racer, return the score of the racer
    //score is determined by how far the racer gets before hitting a wall
    //racer starts on a random start tile
    let mut rng = rand::thread_rng();
    let start_tile = map.start_tiles.choose(&mut rng).unwrap();
    // let start_tile = map.start_tiles[0];
    racer.pos_x = start_tile.0 as u32;
    racer.pos_y = start_tile.1 as u32;

    racer.vel_x = 2 * map.starting_vec.x as i32;
    racer.vel_y = 2 * map.starting_vec.y as i32;
    let mut score_bonus = 0;
    let mut curr_tile = get_tile(map, racer.pos_x, racer.pos_y);
    let mut has_made_it_to_end = true;
    'outer: while *curr_tile != Tile::End {
        if racer.pos_x as i32 + racer.vel_x < 0
            || racer.pos_x as i32 + racer.vel_x >= map.width as i32
            || racer.pos_y as i32 + racer.vel_y < 0
            || racer.pos_y as i32 + racer.vel_y >= map.height as i32
        {
            println!("I went out of bounds! my v was {}, {}", racer.vel_x, racer.vel_y);
            has_made_it_to_end = false;
            break;
        }
        //if the racer goes to a lower tile distance then before (it is going backwards), kill the racer!!!

        let curr_dist = map.tile_distances[racer.pos_y as usize * map.width + racer.pos_x as usize];
        let next_dist = map.tile_distances[(racer.pos_y as i32 + racer.vel_y) as usize * map.width + (racer.pos_x as i32 + racer.vel_x) as usize];
        if next_dist < curr_dist
        {
            // // println!("I went backwards! my v was {}, {}", racer.vel_x, racer.vel_y);
            // has_made_it_to_end = false;
            // break;
        }
        let vel_mag = ((racer.vel_x * racer.vel_x + racer.vel_y * racer.vel_y) as f32).sqrt() as usize;
        let vel_x_norm = racer.vel_x as f32 / vel_mag as f32;
        let vel_y_norm = racer.vel_y as f32 / vel_mag as f32;
        for mag in 1..=vel_mag+1 {
            
            let x =  (mag as f32 * vel_x_norm) as i32;
            let y =  (mag as f32 * vel_y_norm) as i32;
            //if the ray intersects a wall, kill the racer!!! (break out of the loop)
            let tile = *get_tile(
                map,
                (racer.pos_x as i32 + x) as u32,
                (racer.pos_y as i32 + y) as u32,
            );
            if tile == Tile::End {
                println!("I made it to the end! my v was {}, {}", racer.vel_x, racer.vel_y);
                break 'outer;
            }
            if tile == Tile::Wall
            {
                // println!("My velocity ray hit a wall! my v was {}, {}", racer.vel_x, racer.vel_y);
                has_made_it_to_end = false;
                break 'outer;
                
            }
            
        }
        if do_drawing {
            
        
        mq::draw_circle(
            racer.pos_x as f32 * PX_SIZE + 0.5 * PX_SIZE,
            racer.pos_y as f32 * PX_SIZE + 0.5 * PX_SIZE,
            0.5 * PX_SIZE,
            mq::BLUE,
        );
        //draw velocity vector
        mq::draw_line(
            racer.pos_x as f32 * PX_SIZE + 0.5 * PX_SIZE,
            racer.pos_y as f32 * PX_SIZE + 0.5 * PX_SIZE,
            (racer.pos_x as i32 + racer.vel_x) as f32 * PX_SIZE + 0.5 * PX_SIZE,
            (racer.pos_y as i32 + racer.vel_y) as f32 * PX_SIZE + 0.5 * PX_SIZE,
            1.0,
            mq::RED,
        );
        }
        racer.pos_x = (racer.pos_x as i32 + racer.vel_x) as u32;
        racer.pos_y = (racer.pos_y as i32 + racer.vel_y) as u32;
        curr_tile = get_tile(map, racer.pos_x, racer.pos_y);
        //prompt racer to make a decision
        let vel = mq::vec2(racer.vel_x as f32, racer.vel_y as f32).normalize();
        let mut inputs = Vec::new();
        for angle in ray_angles.iter() {
            let mut x = racer.pos_x as f32;
            let mut y = racer.pos_y as f32;

            let mut dist = 0;
            loop {
                //  println!("still going ({x}, {y})");
                x += (angle.cos() + vel.x) / 2.0;
                y += (angle.sin() + vel.y) / 2.0;
                if x < 0.0 || ((angle.cos() + vel.x).round() as i32 == 0 && (angle.sin() + vel.y).round() as i32 == 0)
                    || y < 0.0
                    || x >= map.width as f32
                    || y >= map.height as f32
                    || *get_tile(map, x.floor() as u32, y.floor() as u32) == Tile::Wall
                   
                {
                    
                    break;
                }
                dist += 1;
            
            }
            inputs.push(dist as f64);
            //add velocity to input
        }
        inputs.push(racer.vel_x as f64);
        inputs.push(racer.vel_y as f64);
        let outputs = racer.brain.feed_forward(inputs);
        let max_index = outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        // print!("mi={:?} ", outputs);
        // println!("weights: {:?}", &racer.brain.get_weights());
        // println!("biases: {:?}", &racer.brain.get_biases());
        // println!("layers: {:?}", &racer.brain.get_layers());
        let accel = match max_index {
            0 => (-1, -1),
            1 => (0, -1),
            2 => (1, -1),
            3 => (-1, 0),
            4 => (0, 0),
            5 => (1, 0),
            6 => (-1, 1),
            7 => (0, 1),
            8 => (1, 1),
            _ => panic!("Invalid output"),
        };
        //if velocity and acceleration are both 0, kill the racer!!!
        if racer.vel_x == 0 && racer.vel_y == 0 {
            // println!("I stopped moving! my v was {}, {}", racer.vel_x, racer.vel_y);
            has_made_it_to_end = false;
            break;
        }
        racer.vel_x += accel.0;
        racer.vel_y += accel.1;
    
        score_bonus+=1;
    }
    let score = map.tile_distances[racer.pos_y as usize * map.width + racer.pos_x as usize] as i32;
    if has_made_it_to_end {
        println!("Made it to the end!");
        (score - score_bonus * 2).max(1) * 3
    } else {
        (score).max(1)
    }
}
