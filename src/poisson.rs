use math::{dim2::Vec2, Vector};
use rand::prelude::Distribution;

use crate::distributions;
pub trait PoissonDiskAlgorithm {
    /// Shall be call only once
    fn init<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Option<Vec2>;

    /// Can be called as long None is not returned;
    /// Same semantic as next iter_next
    fn next<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Option<Vec2>;
}

struct Grid {
    inner: Vec<Option<Vec2>>,
    extent: [usize; 2],
    cell_size: f32,
    radius: f32,
    bottom_left: Vec2,
    top_right: Vec2,
}

impl Grid {
    fn new(radius: f32, extent: [Vec2; 2]) -> Self {
        let cell_size = radius / f32::sqrt(2.0);

        let [x1, y1] = extent[0].into_array();
        let [x2, y2] = extent[1].into_array();

        let grid_width = x2 - x1;
        let grid_height = y2 - y1;

        let width = (grid_width / cell_size).ceil() as usize;
        let height = (grid_height / cell_size).ceil() as usize;
        Self {
            inner: vec![None; width * height],
            bottom_left: extent[0],
            top_right: extent[1],
            extent: [width, height],
            cell_size,
            radius,
        }
    }

    fn get_cell(&self, pos: Vec2) -> Option<(usize, usize)> {
        let [x, y] = pos.into_array();
        let [bx, by] = self.bottom_left.into_array();
        let [tx, ty] = self.top_right.into_array();

        if x >= tx || x <= bx || y >= ty || y <= by {
            return None;
        }

        let [base_x, base_y] = self.bottom_left.into_array();

        let w = ((x - base_x) / self.cell_size).floor();
        let h = ((y - base_y) / self.cell_size).floor();

        Some((w as usize, h as usize))
    }

    fn get_index(&self, pos: Vec2) -> Option<usize> {
        self.get_cell(pos).map(|(w, h)| w + h * self.extent[0])
    }

    fn insert(&mut self, pos: Vec2) -> Option<usize> {
        self.get_index(pos).map(|index| {
            self.inner[index] = Some(pos);
            index
        })
    }

    fn get(&self, index: usize) -> Option<Vec2> {
        self.inner.get(index).and_then(|x| *x)
    }

    fn can_insert(&self, x: Vec2) -> bool {
        let m = 2 * ((1.0 / self.cell_size).ceil() as isize + 1);
        let Some((w, h)) = self.get_cell(x) else {return false};

        for i in -m..=m {
            let w = w as isize + i;
            if w < 0 {
                continue;
            }

            for j in -m..=m {
                let h = h as isize + j;
                if h < 0 {
                    continue;
                }

                let index = h as usize * self.extent[0] + w as usize;
                if let Some(pos) = self.get(index) {
                    let length = (pos - x).length_squared();

                    if length <= self.radius * self.radius {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// See https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
pub struct RobertBridson {
    grid: Grid,
    indices: Vec<usize>,
    annulus_distr: distributions::AnnulusDistribution,
    vec2_distr: rand::distributions::Uniform<Vec2>,
}

impl PoissonDiskAlgorithm for RobertBridson {
    fn init<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Option<Vec2> {
        let x0 = self.vec2_distr.sample(rng);

        //TODO: implement - in math lib
        let index = self.grid.insert(x0).unwrap();
        self.indices.push(index);
        Some(x0)
    }

    fn next<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Option<Vec2> {
        loop {
            if self.indices.is_empty() {
                return None;
            }

            let index = rng.gen_range(0..self.indices.len());
            let Some(xi) = self.grid.get(self.indices[index]) else {panic!()};

            for _ in 0..30 {
                let x = xi + self.annulus_distr.sample(rng);

                if self.grid.can_insert(x) {
                    self.indices.push(self.grid.insert(x).unwrap());

                    return Some(x);
                }
            }
            self.indices.swap_remove(index);
        }
    }
}

impl RobertBridson {
    pub fn new(radius: f32, extent: [Vec2; 2]) -> Self {
        Self {
            grid: Grid::new(radius, extent),
            indices: vec![],
            annulus_distr: distributions::AnnulusDistribution::new(radius, 2.0 * radius),
            vec2_distr: rand::distributions::Uniform::<Vec2>::new(extent[0], extent[1]),
        }
    }
}

pub struct PoissonDisk<R: rand::Rng, T: PoissonDiskAlgorithm> {
    rng: R,
    poisson_algo: T,
    init: bool,
}

impl<R: rand::Rng, T: PoissonDiskAlgorithm> PoissonDisk<R, T> {
    pub fn new(rng: R, poisson_algo: T) -> Self {
        Self {
            rng,
            poisson_algo,
            init: false,
        }
    }
}

impl<R: rand::Rng, T: PoissonDiskAlgorithm> Iterator for PoissonDisk<R, T> {
    type Item = Vec2;

    fn next(&mut self) -> Option<Vec2> {
        if !self.init {
            self.init = true;
            return self.poisson_algo.init(&mut self.rng);
        }
        self.poisson_algo.next(&mut self.rng)
    }
}

// TODO: GPU, based on compute shaders, maybe
// TODO: implement Ebeida algo
// TODO: implement Voronoi iteration, LLoyd's algorithm, relaxation
// TODO: implement Voronoi noise or something like that
