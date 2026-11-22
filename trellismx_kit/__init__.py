"""Model-neutral per-expert sub-3-bit TrellisMX and Host-RAM Engram QEC toolkit."""

from .profile import ModelProfile, load_profile
from .trellis_ref import (
    BITS_PER_WEIGHT,
    PAIR_MIX,
    RATES,
    RATE_CODE,
    RATE_NAME,
    TILE_I16,
    TILE_U32,
    decode_tile_operands,
    decode_tile_values,
    pack_symbols,
    state_lut,
    unpack_symbols,
)

__all__ = [
    "BITS_PER_WEIGHT",
    "ModelProfile",
    "PAIR_MIX",
    "RATES",
    "RATE_CODE",
    "RATE_NAME",
    "TILE_I16",
    "TILE_U32",
    "decode_tile_operands",
    "decode_tile_values",
    "load_profile",
    "pack_symbols",
    "state_lut",
    "unpack_symbols",
]
