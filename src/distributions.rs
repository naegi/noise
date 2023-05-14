use math::dim2::Vec2;

/// Adapts  https://mathworld.wolfram.com/DiskPointPicking.html to an annulus
/// For a disk, the rejection method is faster, but thinner the annulus is, slower the rejection
/// method is
/// for an annulus of inner radius r, outer radius 0.5, the rejection method has a probability of
/// pi/4 - pi*r^2
/// thus the expected number samples needed of is 1/ pi / (1/4 - r^2) -> infty when r -> 1/2
/// Maybe do a split at r = 0.3 (expected samples count ~= 2)?
/// Need a benchmark for that
pub struct AnnulusDistribution {
    radius: rand::distributions::Uniform<f32>,
    angle: rand::distributions::Uniform<f32>,
}

impl AnnulusDistribution {
    pub fn new(low: f32, high: f32) -> Self {
        Self {
            angle: rand::distributions::Uniform::<f32>::new_inclusive(0.0, std::f32::consts::TAU),
            radius: rand::distributions::Uniform::<f32>::new_inclusive(low * low, high * high),
        }
    }
}

impl rand::distributions::Distribution<Vec2> for AnnulusDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec2 {
        let r = self.radius.sample(rng).sqrt();
        let (sin, cos) = self.angle.sample(rng).sin_cos();
        Vec2::from_components(r * cos, r * sin)
    }
}
