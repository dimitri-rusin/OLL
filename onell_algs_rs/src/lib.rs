use std::{
    iter::{repeat, repeat_with},
    ops::{Add, AddAssign},
};

use bitvec::prelude::*;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_mt::Mt64;
use statrs::distribution::{Binomial, Discrete};

struct FxLog {
    fxs: Vec<u64>,
    n_evalss: Vec<NEvals>,
}

impl FxLog {
    fn new() -> Self {
        Self {
            fxs: Vec::new(),
            n_evalss: Vec::new()
        }
    }
    fn record(&mut self, fx: u64, n_evals: &NEvals) {
        match self.fxs.last() {
            Some(p) => {
                if *p != fx {
                    self.fxs.push(fx);
                    self.n_evalss.push(n_evals.clone());
                }
            },
            None => {
                self.fxs.push(fx);
                self.n_evalss.push(n_evals.clone());
            }
        }
    }
    fn export(self) -> (Vec<u64>, Vec<u64>) {
        let n_evalss = self.n_evalss.into_iter().map(|x| x.export()).collect();
        (self.fxs, n_evalss)
    }
}

#[derive(Clone)]
enum NEvals {
    Inf,
    N(u64),
}

impl NEvals {
    fn new() -> NEvals {
        NEvals::N(0)
    }
    fn new_with_value(n: u64) -> NEvals {
        NEvals::N(n)
    }
    fn increament(&mut self) {
        *self = match self {
            Self::Inf => Self::Inf,
            Self::N(val) => Self::N(*val + 1),
        };
    }
    fn make_big(&mut self) {
        *self = Self::Inf;
    }
    fn export(self) -> u64 {
        match self {
            Self::Inf => u64::max_value(),
            Self::N(val) => match val.try_into() {
                Ok(val) => val,
                Err(_) => u64::max_value()
            }
        }
    }
}

impl Add for NEvals {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        match self {
            Self::Inf => Self::Inf,
            Self::N(val1) => {
                match other {
                    Self::Inf => Self::Inf,
                    Self::N(val2) => Self::N(val1 + val2)
                }
            }
        }
    }
}

impl AddAssign for NEvals {
    fn add_assign(&mut self, rhs: Self) {
        *self = match self {
            Self::Inf => Self::Inf,
            Self::N(val1) => {
                match rhs {
                    Self::Inf => Self::Inf,
                    Self::N(val2) => Self::N(*val1 + val2)
                }
            }
        }
    }
}

impl PartialOrd for NEvals {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self {
            Self::Inf => {
                match other {
                    Self::Inf => None,
                    Self::N(_) => Some(std::cmp::Ordering::Greater)
                }
            },
            Self::N(x) => {
                match other {
                    Self::Inf => Some(std::cmp::Ordering::Less),
                    Self::N(y) => Some(x.cmp(y))
                }
            }
        }
    }
}

impl PartialEq for NEvals {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Inf => false,
            Self::N(x) => match other {
                Self::Inf => false,
                Self::N(y) => x == y
            }
        }
    }
}

// New function with an additional probability parameter
fn random_bits_with_probability<R: rand::Rng>(rng: &mut R, length: usize, probability: f64) -> BitVec {
    let arch_len = std::mem::size_of::<usize>() * 8;
    let word_length = (length - 1) / arch_len + 1;

    let numbers = std::iter::repeat_with(|| {
        (0..arch_len).fold(0, |acc, _| (acc << 1) | rng.gen_bool(probability) as usize)
    }).take(word_length);

    let mut bv = bitvec![usize, Lsb0;];
    bv.extend(numbers);
    bv.truncate(length);
    bv
}

fn random_bits_with_ones<R: rand::Rng>(length: usize, amount: usize, rng: &mut R) -> BitVec {
    let bits_indices = (0..length).choose_multiple(rng, amount);
    let mut res = bitvec![usize, Lsb0;];
    res.extend(repeat(false).take(length));
    bits_indices.iter().for_each(|i| res.set(*i, true));
    res
}

fn random_ones_with_p<R: rand::Rng>(length: usize, p: f64, rng: &mut R) -> BitVec {
    let mut res = bitvec![usize, Lsb0;];
    res.extend(repeat_with(|| rng.gen_bool(p)).take(length));
    res
}

fn mutate<R: rand::Rng>(parent: &BitVec, p: f64, n_child: usize, rng: &mut R) -> (BitVec, NEvals) {
    let bi = Binomial::new(p, parent.len().try_into().unwrap()).unwrap();
    let l = *(0..=parent.len())
        .collect::<Vec<usize>>()
        .choose_weighted(rng, |i| {
            if *i == 0 {
                0f64
            } else {
                bi.pmf((*i).try_into().unwrap())
                    / (1f64 - ((1f64 - p) as f64).powi(parent.len().try_into().unwrap()))
            }
        })
        .unwrap();
    let children = repeat_with(|| {
        let mut child = random_bits_with_ones(parent.len(), l, rng);
        child ^= parent;
        child
    })
    .take(n_child);

    let x_prime = children
        .max_by(|x, y| x.count_ones().cmp(&y.count_ones()))
        .unwrap();

    (x_prime, NEvals::new_with_value(n_child.try_into().unwrap()))
}

fn crossover<R: rand::Rng>(
    parent: &BitVec,
    x_prime: BitVec,
    p: f64,
    n_child: usize,
    rng: &mut R,
) -> (BitVec, NEvals) {
    let mut n_evals = NEvals::new();
    let children = repeat_with(|| {
        let mut child = bitvec![usize, Lsb0;];
        child.extend(repeat(false).take(parent.len()));
        let mask = random_ones_with_p(parent.len(), p, rng);
        let parent_half = !mask.clone() & parent;
        let x_prime_half = mask & &x_prime;
        child = parent_half | x_prime_half;
        if child.as_bitslice() != parent && child.as_bitslice() != x_prime {
            n_evals.increament();
        }
        child
    })
    .take(n_child);
    let y = children
        .max_by(|x, y| x.count_ones().cmp(&y.count_ones()))
        .unwrap();

    // By design, this returns x_prime when the two have the same fitness score.
    let y = [y, x_prime]
        .into_iter()
        .max_by(|x, y| x.count_ones().cmp(&y.count_ones()))
        .unwrap();
    (y, n_evals)
}

// Do both rounds of (1+(lambda, lambda)).
fn generation_full(
    x: BitVec,
    p: f64,
    n_child_mutate: usize,
    c: f64,
    n_child_crossover: usize,
    generation_seed: u64,
) -> (BitVec, NEvals) {
    let mut rng: Mt64 = SeedableRng::seed_from_u64(generation_seed);
    let (x_prime, ne1) = mutate(&x, p, n_child_mutate, &mut rng);
    let (y, ne2) = crossover(&x, x_prime, c, n_child_crossover, &mut rng);
    let n_evals = ne1 + ne2;
    let x = [x, y]
        .into_iter()
        .max_by(|x, y| x.count_ones().cmp(&y.count_ones()))
        .unwrap();
    (x, n_evals)
}

// Do both rounds of (1+(lambda, lambda)) repeatedly until current fitness has been overcome.
fn generation_full_until_improving_fitness(
    x: BitVec,
    p: f64,
    n_child_mutate: usize,
    c: f64,
    n_child_crossover: usize,
    generation_seed: u64,
) -> (BitVec, NEvals) {
    let mut rng: Mt64 = SeedableRng::seed_from_u64(generation_seed);
    let mut x_current = x;
    let mut n_evals = NEvals::new();
    let current_fitness = x_current.count_ones();

    loop {
        let (x_prime, ne1) = mutate(&x_current, p, n_child_mutate, &mut rng);
        n_evals += ne1;
        let (y, ne2) = crossover(&x_current, x_prime, c, n_child_crossover, &mut rng);
        n_evals += ne2;
        let new_fitness = y.count_ones();

        if new_fitness > current_fitness {
            return (y, n_evals); // Improvement found, return the new configuration
        } else {
            x_current = y; // No improvement, repeat the process with the new result
        }
    }
}

fn onell_lambda_rs(n: usize, oll_parameters: Vec<(f64, usize, f64, usize)>, seed: u64, max_evals: NEvals, record_log: bool, probability: f64) -> (NEvals, u64, Option<FxLog>) {
    let mut rng: Mt64 = SeedableRng::seed_from_u64(seed);
    let mut x = random_bits_with_probability(&mut rng, n, probability);
    let mut n_evals = NEvals::new();
    let mut logs;
    if record_log {
        logs = Some(FxLog::new());
    } else {
        logs = None;
    }
    if record_log {
        let mut logs_i = logs.unwrap();
        logs_i.record(x.count_ones().try_into().unwrap(), &n_evals);
        logs = Some(logs_i);
    }

    let mut num_timesteps: u64 = 0;
    while x.count_ones() != n && n_evals < max_evals {
        let (mutation_rate, mutation_size, crossover_rate, crossover_size) = oll_parameters[x.count_ones()];
        let ne;
        let generation_seed = rng.gen::<u64>();
        (x, ne) = generation_full(x, mutation_rate, mutation_size, crossover_rate, crossover_size, generation_seed);
        n_evals += ne;
        num_timesteps += 1;

        if record_log {
            let mut logs_i = logs.unwrap();
            logs_i.record(x.count_ones().try_into().unwrap(), &n_evals);
            logs = Some(logs_i)
        }
    }

    if x.count_ones() != n {
        n_evals.make_big();
        if record_log {
            let mut logs_i = logs.unwrap();
            logs_i.record(x.count_ones().try_into().unwrap(), &n_evals);
            logs = Some(logs_i)
        }
    }
    if record_log {
        (n_evals, num_timesteps, logs)
    } else {
        (n_evals, num_timesteps, None)
    }
}

// the probability parameter represents the probability of getting a 1 in the generated bits
#[pyfunction]
fn onell_lambda(n: usize, oll_parameters: Vec<(f64, usize, f64, usize)>, seed: u64, max_evals: usize, probability: f64) -> PyResult<(u64, u64)> {
    let (n_evals, num_timesteps, _) = onell_lambda_rs(n, oll_parameters, seed, NEvals::new_with_value(max_evals.try_into().unwrap()), false, probability);
    Ok((n_evals.export(), num_timesteps))
}

#[pyfunction]
fn generation_full_py(
    x: Vec<bool>,
    p: f64,
    n_child_mutate: usize,
    c: f64,
    n_child_crossover: usize,
    generation_seed: u64,
) -> PyResult<(Vec<bool>, u64)> {
    let x_bitvec = x.iter().collect::<BitVec<_, Lsb0>>();
    let (y, n_evals) = generation_full(x_bitvec, p, n_child_mutate, c, n_child_crossover, generation_seed);

    let y_vec = y.into_iter().collect::<Vec<bool>>();
    let n_evals_export = n_evals.export(); // Assuming n_evals is a single NEvals value and needs export to convert to u64

    Ok((y_vec, n_evals_export))
}

#[pyfunction]
fn generation_full_until_improving_fitness_py(
    x: Vec<bool>,
    p: f64,
    n_child_mutate: usize,
    c: f64,
    n_child_crossover: usize,
    generation_seed: u64,
) -> PyResult<(Vec<bool>, u64)> {
    let x_bitvec = x.iter().collect::<BitVec<_, Lsb0>>();
    let (y, n_evals) = generation_full_until_improving_fitness(x_bitvec, p, n_child_mutate, c, n_child_crossover, generation_seed);

    let y_vec = y.into_iter().collect::<Vec<bool>>();
    let n_evals_export = n_evals.export(); // Assuming n_evals is a single NEvals value and needs export to convert to u64

    Ok((y_vec, n_evals_export))
}

#[pyfunction]
fn onell_lambda_with_log(n: usize, oll_parameters: Vec<(f64, usize, f64, usize)>, seed: u64, max_evals: usize) -> PyResult<(u64, Vec<u64>, Vec<u64>)> {
    let (n_evals, _, logs) = onell_lambda_rs(n, oll_parameters, seed, NEvals::new_with_value(max_evals.try_into().unwrap()), true, 0.5);
    let (a, b) = logs.unwrap().export();
    Ok((n_evals.export(), a, b))
}

/// A Python module implemented in Rust.
#[pymodule]
fn onell_algs_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(onell_lambda, m)?)?;
    m.add_function(wrap_pyfunction!(onell_lambda_with_log, m)?)?;
    m.add_function(wrap_pyfunction!(generation_full_py, m)?)?;
    m.add_function(wrap_pyfunction!(generation_full_until_improving_fitness_py, m)?)?;
    Ok(())
}
