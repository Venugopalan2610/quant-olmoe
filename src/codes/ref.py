"""Reference implementations of the three QTIP code functions.

These are pure-Python references — slow but readable. The CUDA kernels in
external/qtip/qtip-kernels are bit-equivalent to these (modulo any kernel
bugs); these are the ground truth for testing.

All three take an L-bit state (as a uint32 with only the bottom L bits live)
and produce one or two approximately N(0,1) floats.
"""

import numpy as np

_1MAD_A = np.uint32(34038481)
_1MAD_B = np.uint32(76625530)
_1MAD_MEAN = 510.0   # 4 * 127.5
_1MAD_STD = 147.8    # ~ sqrt(4 * (256^2 - 1) / 12)

def decode_1mad(state):
    """1MAD decoder: state (uint32, L-bit) -> approx N(0,1) float.

    Hash via LCG, then sum the 4 bytes of the 32-bit result, then
    standardize. CLT with n=4 gives a Gaussian-shaped output.
    """
    x = np.uint32(state)
    x = (_1MAD_A * x + _1MAD_B)
    b0 = x & np.uint32(0xFF)
    b1 = (x >> np.uint32(8)) & np.uint32(0xFF)
    b2 = (x >> np.uint32(16)) & np.uint32(0xFF)
    b3 = (x >> np.uint32(24)) & np.uint32(0xFF)
    s = float(b0) + float(b1) + float(b2) + float(b3)
    return (s - _1MAD_MEAN) / _1MAD_STD
    
def decode_1mad_batch(states):
    """Vectorized 1MAD over a numpy array of states."""
    x = states.astype(np.uint32)
    x = _1MAD_A * x + _1MAD_B
    b0 = (x & np.uint32(0xFF)).astype(np.float32)
    b1 = ((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.float32)
    b2 = ((x >> np.uint32(16)) & np.uint32(0xFF)).astype(np.float32)
    b3 = ((x >> np.uint32(24)) & np.uint32(0xFF)).astype(np.float32)
    return (b0 + b1 + b2 + b3 - _1MAD_MEAN) / _1MAD_STD

# ============================================================================
# 3INST — fp16 reinterpretation of LCG output
# Paper: Algorithm 2, with constants from Section 3.1.1
# ============================================================================

_3INST_A = np.uint32(89226354)
_3INST_B = np.uint32(64248484)
_3INST_M_FP16 = np.float16(0.922)
# Mask: keep sign + low-2 exponent + mantissa, zero the top-3 exponent bits.
# Binary: 1000 1111 1111 1111 = 0x8FFF per fp16 half
_3INST_MASK = np.uint32(0x8FFF8FFF)
# Empirical std of unscaled 3INST output. Measured once, used to standardize.
# The paper's m=0.922 produces var ~ 1.55; we rescale by 1/sqrt(1.55) ~ 0.804.
_3INST_SCALE = np.float32(1.0 / 1.2447)


def decode_3inst(state):
    """3INST decoder: state (uint32, L-bit) -> approx N(0,1) float."""
    x = np.uint32(state)
    x = _3INST_A * x + _3INST_B

    m_bits = np.frombuffer(np.float16(_3INST_M_FP16).tobytes(), dtype=np.uint16)[0]
    m_packed = np.uint32(m_bits) | (np.uint32(m_bits) << np.uint32(16))

    y = (x & _3INST_MASK) ^ m_packed

    lo = np.uint16(y & np.uint32(0xFFFF))
    hi = np.uint16((y >> np.uint32(16)) & np.uint32(0xFFFF))
    f_lo = np.frombuffer(lo.tobytes(), dtype=np.float16)[0]
    f_hi = np.frombuffer(hi.tobytes(), dtype=np.float16)[0]
    return (float(f_lo) + float(f_hi)) * float(_3INST_SCALE)


def decode_3inst_batch(states):
    """Vectorized 3INST over a numpy array of states."""
    x = states.astype(np.uint32)
    x = _3INST_A * x + _3INST_B

    m_bits = np.frombuffer(np.float16(_3INST_M_FP16).tobytes(), dtype=np.uint16)[0]
    m_packed = np.uint32(m_bits) | (np.uint32(m_bits) << np.uint32(16))

    y = (x & _3INST_MASK) ^ m_packed

    lo = (y & np.uint32(0xFFFF)).astype(np.uint16)
    hi = ((y >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16)

    f_lo = np.frombuffer(lo.tobytes(), dtype=np.float16).astype(np.float32)
    f_hi = np.frombuffer(hi.tobytes(), dtype=np.float16).astype(np.float32)
    return (f_lo + f_hi) * _3INST_SCALE

# ============================================================================
# HYB — Klimov-Shamir hash + 2D LUT lookup + sign flip
# Paper: Algorithm 3, with Q=9 (HYB code's standard config)
# ============================================================================


def decode_hyb(state, lut, Q=9):
    """HYB decoder: state (uint32, L-bit) -> 2 approx N(0,1) floats.

    Klimov-Shamir hash (x*x + x), index into a 2^Q x 2 LUT, sign-flip the
    second component based on bit 15 of the hashed state.

    lut: numpy array of shape (2^Q, 2), float
    """
    assert lut.shape == (2 ** Q, 2), f"LUT shape mismatch: {lut.shape}"
    x = np.uint32(state)
    x = x * x + x  # mod 2^32 automatic
    idx = int((x >> np.uint32(15 - Q)) & np.uint32((1 << Q) - 1))
    a, b = float(lut[idx, 0]), float(lut[idx, 1])
    if x & np.uint32(1 << 15):
        b = -b
    return a, b

def decode_hyb_batch(states, lut, Q=9):
    """Vectorized HYB over a numpy array of states.

    Returns a (N, 2) float32 array.
    """
    x = states.astype(np.uint32)
    x = x * x + x
    idx = ((x >> np.uint32(15 - Q)) & np.uint32((1 << Q) - 1)).astype(np.int64)
    out = lut[idx].astype(np.float32).copy()  # shape (N, 2)
    sign_flip = (x & np.uint32(1 << 15)) != 0
    out[sign_flip, 1] = -out[sign_flip, 1]
    return out