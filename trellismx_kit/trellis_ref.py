"""Reference procedural NPK trellis packer and decoder for K2, P25, K3, and K4.

All four rates decode 256-weight tiles directly to FP8 (E4M3) operands using
the zero-storage procedural NPK (Neural Processing of K) 16-bit state law with
compander scale 2.0. Scalar rates (K2, K3, K4) advance one weight per step.
The pair rate (P25) advances 5 bits per step and emits two operands per 16-bit
state: NPK(state) and NPK((state * 1664525 + 1013904223) & 0xFFFF).
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

TILE_VALUES = 256
TRANSITIONS = 1 << 16
RATES = ("k2", "p25", "k3", "k4")
RATE_CODE = {"k2": 0, "p25": 1, "k3": 2, "k4": 3}
RATE_NAME = {code: name for name, code in RATE_CODE.items()}
STREAM_BITS = {"k2": 2, "p25": 5, "k3": 3, "k4": 4}
SYMBOLS_PER_TILE = {"k2": 256, "p25": 128, "k3": 256, "k4": 256}
TILE_I16 = {"k2": 32, "p25": 40, "k3": 48, "k4": 64}
TILE_U32 = {"k2": 16, "p25": 20, "k3": 24, "k4": 32}
BITS_PER_WEIGHT = {"k2": 2.0, "p25": 2.5, "k3": 3.0, "k4": 4.0}
PAIR_MIX = (1664525, 1013904223)
CODEC_LAW = "procedural-npk"
COMPANDER_SCALE = 2.0


def _validate_rate(rate: str) -> None:
    if rate not in RATES:
        raise ValueError(f"unsupported trellis rate {rate!r}; expected one of {RATES}")


def _half_from_bits(bits: torch.Tensor) -> torch.Tensor:
    signed = bits.to(torch.int64)
    signed = signed - 65536 * (signed >= 32768).to(torch.int64)
    return signed.to(torch.int16).view(torch.float16)


@lru_cache(maxsize=1)
def _state_bytes() -> torch.Tensor:
    states = torch.arange(TRANSITIONS, dtype=torch.int64)
    product = (states * 0xCBAC1FED) & 0xFFFFFFFF
    word = (product & 0x8FFF8FFF) ^ 0x3B603B60
    low = _half_from_bits(word & 0xFFFF)
    high = _half_from_bits((word >> 16) & 0xFFFF)
    combined = low + high
    values = combined.float() * COMPANDER_SCALE
    projected = values.to(torch.float8_e4m3fn).float()
    projected[projected == 0] = 0
    encoded = projected.to(torch.float8_e4m3fn)
    if not bool(torch.isfinite(encoded.float()).all()):
        raise RuntimeError("procedural NPK produced a non-finite E4M3 value")
    return encoded.view(torch.uint8).contiguous()


def state_lut() -> torch.Tensor:
    """Return the 65,536-byte procedural NPK E4M3 operand table."""
    return _state_bytes()


def pack_symbols(symbols: torch.Tensor, rate: str) -> torch.Tensor:
    """Pack branch symbols into little-endian int16 trellis words."""
    _validate_rate(rate)
    bits = STREAM_BITS[rate]
    expected = SYMBOLS_PER_TILE[rate]
    if symbols.shape[-1] != expected:
        raise ValueError(f"{rate} symbol tensor must end in {expected} values")
    values = symbols.to(torch.int64) & ((1 << bits) - 1)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=symbols.device)
    bit_stream = ((values[..., None] >> symbol_shifts) & 1).reshape(*values.shape[:-1], -1, 16)
    word_shifts = torch.arange(15, -1, -1, device=symbols.device)
    words_u16 = (bit_stream << word_shifts).sum(-1)
    words_u16 = words_u16.reshape(*words_u16.shape[:-1], -1, 2).flip(-1).reshape(words_u16.shape)
    signed = words_u16 - 65536 * (words_u16 >= 32768).to(torch.int64)
    return signed.to(torch.int16).contiguous()


def unpack_symbols(words: torch.Tensor, rate: str) -> torch.Tensor:
    """Unpack int16 trellis words to branch symbols in stream order."""
    _validate_rate(rate)
    bits = STREAM_BITS[rate]
    if words.dtype != torch.int16:
        raise TypeError("trellis words must be int16")
    if words.shape[-1] != TILE_I16[rate]:
        raise ValueError(f"{rate} tile must end in {TILE_I16[rate]} int16 words")
    values = words.to(torch.int64) & 0xFFFF
    values = values.reshape(*values.shape[:-1], -1, 2).flip(-1).reshape(values.shape)
    blocks = values.reshape(*values.shape[:-1], -1, bits)
    word_shifts = torch.arange(15, -1, -1, device=words.device)
    stream = ((blocks[..., None] >> word_shifts) & 1).flatten(-2)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=words.device)
    symbols = stream.reshape(*stream.shape[:-1], 16, bits) << symbol_shifts
    symbols = symbols.sum(-1)
    return symbols.reshape(*symbols.shape[:-2], -1).to(torch.int16).contiguous()


def _pair_states(states: torch.Tensor) -> torch.Tensor:
    multiplier, increment = PAIR_MIX
    mixed = states.to(torch.int64) * multiplier + increment
    return (mixed & 0xFFFF).to(torch.int16)


def tile_states(symbols: torch.Tensor, rate: str) -> torch.Tensor:
    """Return the 16-bit tail-biting state window for every symbol of one tile."""
    _validate_rate(rate)
    bits = STREAM_BITS[rate]
    expected = SYMBOLS_PER_TILE[rate]
    if symbols.shape[-1] != expected:
        raise ValueError(f"{rate} symbol tensor must end in {expected} values")
    if symbols.dtype == torch.bool or symbols.is_floating_point():
        raise TypeError("symbols must use an integer dtype")
    values = symbols.to(torch.int64) & ((1 << bits) - 1)
    states = torch.zeros_like(values)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(values, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(torch.int16).contiguous()


def decode_tile_operands(words: torch.Tensor, rate: str) -> torch.Tensor:
    """Decode one tile to raw E4M3 operand bytes in operand order."""
    _validate_rate(rate)
    symbols = unpack_symbols(words, rate)
    states = tile_states(symbols, rate)
    table = _state_bytes().to(words.device)
    operands = table[states.to(torch.int64)]
    if rate == "p25":
        pair_first = operands
        pair_second = table[_pair_states(states).to(torch.int64)]
        operands = torch.stack((pair_first, pair_second), dim=-1).flatten(-2)
    return operands.contiguous()


def decode_tile_values(words: torch.Tensor, rate: str) -> torch.Tensor:
    """Decode one tile to float32 values in operand order."""
    operands = decode_tile_operands(words, rate)
    return operands.view(torch.float8_e4m3fn).float().contiguous()
