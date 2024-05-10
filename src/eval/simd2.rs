use super::Block;
use crate::eval::network::{RELU_MAX, RELU_MIN};
use core::arch::x86_64::*;

#[cfg(target_feature = "avx512bw")]
pub const REGS: usize = 8;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
pub const REGS: usize = 16;

#[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
pub const REGS: usize = 16;

#[derive(Clone, Copy)]
pub struct Vec16(
    #[cfg(target_feature = "avx512bw")] pub __m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))] pub __m256i,
    #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))] pub i16,
);

impl Vec16 {
    pub const POPULATION: usize = std::mem::size_of::<Self>() / std::mem::size_of::<i16>();
    pub const UNROLL: usize = Self::POPULATION * REGS;

    pub fn add(self, rhs: Self) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_add_epi16(self.0, rhs.0))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_add_epi16(self.0, rhs.0))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            self.0 + rhs.0
        }
    }

    pub fn sub(self, rhs: Self) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_sub_epi16(self.0, rhs.0))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_sub_epi16(self.0, rhs.0))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            self.0 - rhs.0
        }
    }

    pub fn mul_lo(self, rhs: Self) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_mullo_epi16(self.0, rhs.0))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_mullo_epi16(self.0, rhs.0))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            Self(self.0 * rhs.0)
        }
    }

    pub fn mul_accumulate(self, rhs: Self) -> Vec32 {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Vec32(_mm512_madd_epi16(self.0, rhs.0))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Vec32(_mm256_madd_epi16(self.0, rhs.0))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            Vec32(i32::from(self.0) * i32::from(rhs.0))
        }
    }

    pub fn clipped_relu(self) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            let min = _mm512_set1_epi16(RELU_MIN);
            let max = _mm512_set1_epi16(RELU_MAX);

            Self(_mm512_min_epi16(_mm512_max_epi16(self.0, min), max))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            let min = _mm256_set1_epi16(RELU_MIN);
            let max = _mm256_set1_epi16(RELU_MAX);

            _mm256_min_epi16(_mm256_max_epi16(i, min), max)
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            i.clamp(RELU_MIN, RELU_MAX)
        }
    }

    pub fn empty_regs() -> [Self; REGS] {
        unsafe { std::mem::MaybeUninit::uninit().assume_init() }
    }

    pub fn load(mem: &Block, offset: usize) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_load_si512(mem.as_ptr().add(offset).cast()))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_load_si256(mem.as_ptr().add(offset).cast()))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            Self(*mem.get_unchecked(offset))
        }
    }

    pub fn store(self, mem: &mut Block, offset: usize) {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            _mm512_store_si512(mem.as_mut_ptr().add(offset).cast(), self.0)
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            _mm256_store_si256(mem.as_mut_ptr().add(offset).cast(), self.0)
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            *mem.get_unchecked(offset) = self.0
        }
    }
}

#[derive(Clone, Copy)]
pub struct Vec32(
    #[cfg(target_feature = "avx512bw")] pub __m512i,
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))] pub __m256i,
    #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))] pub i32,
);

impl Vec32 {
    pub fn add(self, rhs: Self) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_add_epi32(self.0, rhs.0))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_add_epi32(self.0, rhs.0))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            self.0 + rhs.0
        }
    }

    pub fn sub(self, rhs: Self) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_sub_epi32(self.0, rhs.0))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_sub_epi32(self.0, rhs.0))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            self.0 - rhs.0
        }
    }

    pub fn reduce_add(self) -> i32 {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            _mm512_reduce_add_epi32(self.0)
        }

        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            let upper_128 = _mm256_extracti128_si256::<1>(self.0);
            let lower_128 = _mm256_castsi256_si128(self.0);
            let sum_128 = _mm_add_epi32(upper_128, lower_128);

            let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
            let sum_64 = _mm_add_epi32(upper_64, sum_128);

            let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
            let sum_32 = _mm_add_epi32(upper_32, sum_64);

            _mm_cvtsi128_si32(sum_32)
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            self.0
        }
    }

    pub fn zero() -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_setzero_si512())
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_setzero_si512())
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            Self(0)
        }
    }

    pub fn splat(x: i32) -> Self {
        #[cfg(target_feature = "avx512bw")]
        unsafe {
            Self(_mm512_set1_epi32(x))
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
        unsafe {
            Self(_mm256_set1_epi32(x))
        }
        #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
        {
            Self(x)
        }
    }
}
