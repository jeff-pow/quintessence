use std::ops::{Deref, DerefMut};

use self::network::Network;

pub mod accumulator;
pub mod network;
mod simd;

type Block = [i16; HIDDEN_SIZE];

pub const INPUT_SIZE: usize = 768;
const HIDDEN_SIZE: usize = 128;

static NET: Network = unsafe { std::mem::transmute(*include_bytes!("../../0.1alpha-quant.bin")) };

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub struct Align64<T>(pub T);

impl<T, const N: usize> Deref for Align64<[T; N]> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Align64<[T; N]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
