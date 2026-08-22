// SPDX-License-Identifier: GPL-2.0-or-later
//! IEEE 754 half precision, in safe integer arithmetic.
//!
//! `f16` is still unstable in Rust, and this crate takes no dependencies, so the
//! conversions are done by hand. Both directions are exhaustively tested against
//! every one of the 65 536 half-precision bit patterns, which is cheap enough to
//! do in a unit test and removes any doubt about edge cases.

/// Decode an IEEE 754 binary16 bit pattern to `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let mant = u32::from(h & 0x03FF);

    match exp {
        0 => {
            if mant == 0 {
                // Signed zero.
                f32::from_bits(sign)
            } else {
                // Subnormal: renormalise into a normal f32.
                //
                // A half subnormal is `mant * 2^-24`. Shifting the mantissa left
                // until its implicit bit reaches position 10 costs one power of
                // two per shift, so the exponent bias works out as
                // (127 - 15) + 2 + e. Getting the constant wrong here is a
                // silent factor-of-two on the smallest weights, which is why the
                // round-trip test sweeps all 65 536 patterns.
                let mut e: i32 = 0;
                let mut m = mant;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                let exp32 = (127 - 15 + 1 + e) as u32;
                f32::from_bits(sign | (exp32 << 23) | ((m & 0x03FF) << 13))
            }
        }
        0x1F => {
            // Infinity or NaN.
            f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
        }
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            f32::from_bits(sign | (exp32 << 23) | (mant << 13))
        }
    }
}

/// Encode an `f32` as an IEEE 754 binary16 bit pattern, round-to-nearest-even.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = bits & 0x007F_FFFF;

    if ((bits >> 23) & 0xFF) == 0xFF {
        // Infinity or NaN. Preserve NaN-ness with a non-zero mantissa.
        let m = if mant != 0 { 0x0200 } else { 0 };
        return sign | 0x7C00 | m;
    }
    if exp >= 0x1F {
        // Overflow to infinity.
        return sign | 0x7C00;
    }
    if exp <= 0 {
        // Subnormal or zero.
        if exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let mut h = (m >> shift) as u16;
        // Round to nearest even.
        let rem_mask = (1u32 << shift) - 1;
        let rem = m & rem_mask;
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h_mant = (mant >> 13) as u16;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h_mant & 1) == 1) {
        h_mant += 1;
        if h_mant == 0x0400 {
            h_mant = 0;
            exp += 1;
            if exp >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp as u16) << 10) | h_mant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_bit_pattern_round_trips() {
        // f16 -> f32 -> f16 must be the identity for all finite values, and must
        // preserve infinities. NaN payloads are allowed to differ.
        for bits in 0u32..=0xFFFF {
            let h = bits as u16;
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7C00 == 0x7C00);
                assert!(f32_to_f16(f) & 0x03FF != 0);
                continue;
            }
            let back = f32_to_f16(f);
            assert_eq!(back, h, "0x{h:04x} -> {f} -> 0x{back:04x}");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
        // Smallest positive subnormal: 2^-24.
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
    }

    #[test]
    fn rounding_is_to_nearest_even() {
        // 1.0 + 2^-11 sits exactly halfway between 1.0 and the next half value;
        // round-half-to-even must pick 1.0 (even mantissa).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        // Just above the midpoint must round up.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3C01);
    }

    #[test]
    fn overflow_and_underflow_saturate() {
        assert_eq!(f32_to_f16(1e30), 0x7C00);
        assert_eq!(f32_to_f16(-1e30), 0xFC00);
        assert_eq!(f32_to_f16(1e-30), 0x0000);
        assert_eq!(f32_to_f16(-1e-30), 0x8000);
    }
}
