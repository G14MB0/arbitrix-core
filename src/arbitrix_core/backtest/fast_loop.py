"""Vectorized helpers for the backtest execution loop.

``check_stops`` uses numpy to evaluate stop-loss / take-profit hits for an
array of open trades against a single bar in one shot, avoiding per-trade
Python loops when the number of concurrent positions is large.

When *numba* is installed the hot inner loop is JIT-compiled for an extra
speed boost; otherwise a pure-numpy fallback is used transparently.
"""
from __future__ import annotations

import numpy as np

# ---- optional numba acceleration -------------------------------------------
try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

# intra-bar model constants
MODEL_SL_FIRST = 0
MODEL_TP_FIRST = 1
MODEL_NONE = 2


def check_stops(
    entry_prices: np.ndarray,
    stop_points: np.ndarray,
    take_points: np.ndarray,
    sides: np.ndarray,
    bar_high: float,
    bar_low: float,
    intra_bar_model: int = MODEL_SL_FIRST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised SL/TP check for *N* trades against a single OHLC bar.

    Parameters
    ----------
    entry_prices : float64 array (N,)
    stop_points  : float64 array (N,) – distance in price units
    take_points  : float64 array (N,) – distance (0 ⟹ no TP)
    sides        : int8 array (N,) – 0 = long, 1 = short
    bar_high     : scalar high of the bar
    bar_low      : scalar low  of the bar
    intra_bar_model : 0 = sl_first, 1 = tp_first, 2 = none (SL wins)

    Returns
    -------
    closed_mask  : bool array  – True where trade is closed
    exit_prices  : float64     – fill price (0 when not closed)
    stop_mask    : bool array  – True when closed by stop loss
    take_mask    : bool array  – True when closed by take profit
    """
    if len(entry_prices) == 0:
        empty_b = np.empty(0, dtype=np.bool_)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_b, empty_f, empty_b.copy(), empty_b.copy()

    if _HAS_NUMBA:
        return _check_stops_nb(
            entry_prices, stop_points, take_points, sides,
            bar_high, bar_low, intra_bar_model,
        )
    return _check_stops_np(
        entry_prices, stop_points, take_points, sides,
        bar_high, bar_low, intra_bar_model,
    )


# ---- numpy fallback --------------------------------------------------------

def _check_stops_np(
    entry_prices: np.ndarray,
    stop_points: np.ndarray,
    take_points: np.ndarray,
    sides: np.ndarray,
    bar_high: float,
    bar_low: float,
    model: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    long = sides == 0

    stop_price = np.where(long, entry_prices - stop_points, entry_prices + stop_points)
    take_price = np.where(long, entry_prices + take_points, entry_prices - take_points)
    has_tp = take_points > 0

    stop_hit = np.where(long, bar_low <= stop_price, bar_high >= stop_price)
    take_hit = has_tp & np.where(long, bar_high >= take_price, bar_low <= take_price)

    both = stop_hit & take_hit
    if model == MODEL_TP_FIRST:
        stop_hit = stop_hit & ~both
    else:  # sl_first or none → SL wins
        take_hit = take_hit & ~both

    closed = stop_hit | take_hit
    exit_prices = np.where(stop_hit, stop_price, np.where(take_hit, take_price, 0.0))

    return closed, exit_prices, stop_hit, take_hit


# ---- numba JIT (conditionally defined) ------------------------------------

if _HAS_NUMBA:

    @_njit(cache=True)
    def _check_stops_nb(
        entry_prices,
        stop_points,
        take_points,
        sides,
        bar_high,
        bar_low,
        model,
    ):  # pragma: no cover – JIT path tested when numba available
        n = len(entry_prices)
        closed = np.zeros(n, dtype=np.bool_)
        exit_prices = np.zeros(n, dtype=np.float64)
        is_stop = np.zeros(n, dtype=np.bool_)
        is_take = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            ep = entry_prices[i]
            sp = stop_points[i]
            tp = take_points[i]
            side = sides[i]

            if side == 0:  # long
                stop_price = ep - sp
                take_price = ep + tp
                s_hit = bar_low <= stop_price
                t_hit = tp > 0.0 and bar_high >= take_price
            else:  # short
                stop_price = ep + sp
                take_price = ep - tp
                s_hit = bar_high >= stop_price
                t_hit = tp > 0.0 and bar_low <= take_price

            if not s_hit and not t_hit:
                continue

            if s_hit and t_hit:
                if model == 1:  # tp_first
                    s_hit = False
                else:
                    t_hit = False

            closed[i] = True
            is_stop[i] = s_hit
            is_take[i] = t_hit
            exit_prices[i] = stop_price if s_hit else take_price

        return closed, exit_prices, is_stop, is_take
