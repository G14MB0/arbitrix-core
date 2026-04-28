from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Protocol
from uuid import uuid4

import pandas as pd

SignalAction = Literal[
    "buy",
    "sell",
    "exit",
    "close",
    "partial_close",
    "modify_sl",
    "modify_tp",
    "cancel_order",
]
OrderType = Literal["market", "limit", "stop"]
OrderStatus = Literal["new", "working", "filled", "cancelled", "expired"]
Side = Literal["buy", "sell"]
PositionSide = Literal["long", "short"]
TimeInForce = Literal["GTC", "GTD"]


@dataclass
class Signal:
    """Discrete trading intent emitted by a strategy.

    The additional order fields are optional so legacy strategies that only
    populate ``when``, ``action`` and ``price`` keep working without changes.
    """

    when: pd.Timestamp
    action: SignalAction
    price: float
    reason: str = ""
    order_type: OrderType = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    volume: Optional[float] = None
    tif: TimeInForce = "GTC"
    valid_until: Optional[pd.Timestamp] = None
    target_trade_id: Optional[str] = None
    target_order_id: Optional[str] = None
    close_volume: Optional[float] = None
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    risk_multiplier: float = 1.0
    magic: Optional[int] = None

    def is_entry(self) -> bool:
        return self.action in ("buy", "sell")


@dataclass
class Order:
    """Represents a pending or filled order in the backtest/live domain."""

    symbol: str
    side: Side
    type: OrderType
    volume: float
    id: str = field(default_factory=lambda: str(uuid4()))
    status: OrderStatus = "new"
    tif: TimeInForce = "GTC"
    price: Optional[float] = None
    created_at: Optional[pd.Timestamp] = None
    valid_until: Optional[pd.Timestamp] = None
    broker_ticket: Optional[int] = None
    strategy: Optional[str] = None
    magic: Optional[int] = None

    stop_points: float = 0.0
    take_points: float = 0.0
    trail_params: Dict[str, float] = field(default_factory=dict)
    parent_id: Optional[str] = None


@dataclass
class Trade:
    """Represents a completed or open trade linked to an entry order."""

    symbol: str
    side: PositionSide
    entry_time: pd.Timestamp
    entry_price: float
    id: str = field(default_factory=lambda: str(uuid4()))
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    volume: float = 0.0
    stop_points: float = 0.0
    take_points: float = 0.0
    pnl: float = 0.0
    commission_paid: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    swap_pnl: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    notes: Dict[str, float] = field(default_factory=dict)
    order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    broker_ticket: Optional[int] = None
    strategy: Optional[str] = None
    magic: Optional[int] = None
    _last_swap_day: Optional[pd.Timestamp] = None


@dataclass
class Position:
    """Aggregated view over multiple trades."""

    symbol: str
    side: PositionSide
    volume: float
    avg_price: float
    trades: list[Trade] = field(default_factory=list)


class ExecutionContext(Protocol):
    """Minimal protocol shared by backtest and live engines."""

    def submit_order(self, signal: Signal) -> Order:
        ...

    def modify_order(self, order_id: str, **kwargs: float) -> Order:
        ...

    def cancel_order(self, order_id: str) -> None:
        ...

    def close_position(self, symbol: str) -> None:
        ...
