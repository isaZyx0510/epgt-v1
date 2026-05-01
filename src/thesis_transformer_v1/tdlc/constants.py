"""TDL profile constants."""

from __future__ import annotations

import numpy as np


# 3GPP TR 38.901 Table 7.7.2-3, TDL-C normalized delays and tap powers.
TDL_C_NORMALIZED_DELAYS = np.array(
    [
        0.0000,
        0.2099,
        0.2219,
        0.2329,
        0.2176,
        0.6366,
        0.6448,
        0.6560,
        0.6584,
        0.7935,
        0.8213,
        0.9336,
        1.2285,
        1.3083,
        2.1704,
        2.7105,
        4.2589,
        4.6003,
        5.4902,
        5.6077,
        6.3065,
        6.6374,
        7.0427,
        8.6523,
    ],
    dtype=np.float64,
)

TDL_C_POWERS_DB = np.array(
    [
        -4.4,
        -1.2,
        -3.5,
        -5.2,
        -2.5,
        0.0,
        -2.2,
        -3.9,
        -7.4,
        -7.1,
        -10.7,
        -11.1,
        -5.1,
        -6.8,
        -8.7,
        -13.2,
        -13.9,
        -13.9,
        -15.8,
        -17.1,
        -16.0,
        -15.7,
        -21.6,
        -22.8,
    ],
    dtype=np.float64,
)
