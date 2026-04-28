from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from arbitrix_core.trading import Order, Trade


def _normalize_ts(value: Optional[datetime | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_day(value: Optional[datetime | pd.Timestamp]) -> Optional[pd.Timestamp]:
    ts = _normalize_ts(value)
    return ts.normalize() if ts is not None else None


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: Optional[pd.Timestamp]
    equity: float
    gross_equity: float
    equity_marked: float
    equity_source: str
    open_trades: Tuple[Trade, ...] = field(default_factory=tuple)
    closed_trades: Tuple[Trade, ...] = field(default_factory=tuple)
    pending_orders: Tuple[Order, ...] = field(default_factory=tuple)
    orders: Tuple[Order, ...] = field(default_factory=tuple)


class Portfolio:
    """Shared portfolio view used by backtest and live runtimes."""

    def __init__(self, *, initial_equity: float = 0.0, equity_source: str = "engine") -> None:
        self._equity = float(initial_equity)
        self._gross_equity = float(initial_equity)
        self._equity_marked = float(initial_equity)
        self._equity_source = str(equity_source)
        self._initial_equity = float(initial_equity)
        self._open_trades: List[Trade] = []
        self._closed_trades: List[Trade] = []
        self._pending_orders: List[Order] = []
        self._orders: List[Order] = []
        self._last_update: Optional[pd.Timestamp] = None
        self._order_id = 0
        self._lock = threading.RLock()
        self._version = 0
        self._exposure_cache: Dict[Tuple[str, Optional[str], Optional[pd.Timestamp], int], Dict[str, Any]] = {}
        self._last_prices: Dict[str, float] = {}

    def _bump(self) -> None:
        self._version += 1
        self._exposure_cache.clear()

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def gross_equity(self) -> float:
        return self._gross_equity

    @property
    def equity_marked(self) -> float:
        return self._equity_marked

    @property
    def equity_source(self) -> str:
        return self._equity_source

    @property
    def initial_equity(self) -> float:
        return self._initial_equity

    @property
    def last_update(self) -> Optional[pd.Timestamp]:
        return self._last_update

    @property
    def open_trades(self) -> List[Trade]:
        with self._lock:
            return list(self._open_trades)

    @property
    def closed_trades(self) -> List[Trade]:
        with self._lock:
            return list(self._closed_trades)

    @property
    def pending_orders(self) -> List[Order]:
        with self._lock:
            return list(self._pending_orders)

    @property
    def orders(self) -> List[Order]:
        with self._lock:
            return list(self._orders)

    def snapshot(self) -> PortfolioSnapshot:
        with self._lock:
            return PortfolioSnapshot(
                timestamp=self._last_update,
                equity=self._equity,
                gross_equity=self._gross_equity,
                equity_marked=self._equity_marked,
                equity_source=self._equity_source,
                open_trades=tuple(self._open_trades),
                closed_trades=tuple(self._closed_trades),
                pending_orders=tuple(self._pending_orders),
                orders=tuple(self._orders),
            )

    def update_equity(
        self,
        equity: float,
        *,
        gross_equity: Optional[float] = None,
        equity_marked: Optional[float] = None,
        equity_source: Optional[str] = None,
        timestamp: Optional[datetime | pd.Timestamp] = None,
    ) -> None:
        with self._lock:
            self._equity = float(equity)
            if gross_equity is not None:
                self._gross_equity = float(gross_equity)
            if equity_marked is not None:
                self._equity_marked = float(equity_marked)
            else:
                self._equity_marked = float(equity)
            if equity_source is not None:
                self._equity_source = str(equity_source)
            if timestamp is not None:
                self._last_update = _normalize_ts(timestamp)
            self._bump()

    def sync(
        self,
        *,
        timestamp: Optional[datetime | pd.Timestamp] = None,
        equity: Optional[float] = None,
        gross_equity: Optional[float] = None,
        equity_marked: Optional[float] = None,
        open_trades: Optional[Sequence[Trade]] = None,
        closed_trades: Optional[Sequence[Trade]] = None,
        pending_orders: Optional[Sequence[Order]] = None,
        orders: Optional[Sequence[Order]] = None,
    ) -> None:
        def _coerce_sequence(value: Sequence[Any]) -> List[Any]:
            # Backtests pass mutable lists that are already in-process and
            # repeatedly re-synced at every bar. Reusing list references avoids
            # quadratic copy overhead as trade history grows.
            if isinstance(value, list):
                return value
            return list(value)

        with self._lock:
            if equity is not None:
                self._equity = float(equity)
            if gross_equity is not None:
                self._gross_equity = float(gross_equity)
            if equity_marked is not None:
                self._equity_marked = float(equity_marked)
            if open_trades is not None:
                self._open_trades = _coerce_sequence(open_trades)
            if closed_trades is not None:
                self._closed_trades = _coerce_sequence(closed_trades)
            if pending_orders is not None:
                self._pending_orders = _coerce_sequence(pending_orders)
            if orders is not None:
                self._orders = _coerce_sequence(orders)
            if timestamp is not None:
                self._last_update = _normalize_ts(timestamp)
            self._bump()

    def next_order_id(self) -> int:
        with self._lock:
            self._order_id += 1
            return self._order_id

    def add_order(self, order: Order) -> None:
        with self._lock:
            self._orders.append(order)
            if order.status in ("new", "working"):
                self._pending_orders.append(order)
            self._bump()

    def cancel_order_by_id(self, order_id: str) -> bool:
        updated = False
        with self._lock:
            for order in self._orders:
                if order.id == order_id:
                    order.status = "cancelled"
                    updated = True
            if updated:
                self._pending_orders = [order for order in self._pending_orders if order.id != order_id]
                self._bump()
        return updated

    def add_trade(self, trade: Trade) -> None:
        with self._lock:
            self._open_trades.append(trade)
            self._bump()

    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        with self._lock:
            for trade in self._open_trades:
                if trade.id == trade_id:
                    return trade
        return None

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        with self._lock:
            for order in self._orders:
                if order.id == order_id:
                    return order
        return None

    def update_trade_broker_ticket(self, trade_id: str, ticket: Optional[int]) -> bool:
        with self._lock:
            for trade in self._open_trades:
                if trade.id == trade_id:
                    trade.broker_ticket = ticket
                    self._bump()
                    return True
        return False

    def update_order_broker_ticket(self, order_id: str, ticket: Optional[int]) -> bool:
        with self._lock:
            for order in self._orders:
                if order.id == order_id:
                    order.broker_ticket = ticket
                    self._bump()
                    return True
        return False

    def update_trade_stops(
        self,
        trade_id: str,
        *,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
    ) -> bool:
        with self._lock:
            for trade in self._open_trades:
                if trade.id != trade_id:
                    continue
                if new_sl is not None:
                    trade.stop_points = self._points_from_price(trade, float(new_sl), kind="sl")
                if new_tp is not None:
                    trade.take_points = self._points_from_price(trade, float(new_tp), kind="tp")
                self._bump()
                return True
        return False

    def update_order_stops(
        self,
        order_id: str,
        *,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
    ) -> bool:
        with self._lock:
            for order in self._orders:
                if order.id != order_id:
                    continue
                if order.price is None:
                    return False
                price = float(order.price)
                if new_sl is not None:
                    if order.side == "buy":
                        order.stop_points = max(price - float(new_sl), 0.0)
                    else:
                        order.stop_points = max(float(new_sl) - price, 0.0)
                if new_tp is not None:
                    if order.side == "buy":
                        order.take_points = max(float(new_tp) - price, 0.0)
                    else:
                        order.take_points = max(price - float(new_tp), 0.0)
                self._bump()
                return True
        return False

    def close_trade(self, trade: Trade, *, exit_price: float, exit_time: pd.Timestamp, reason: str) -> None:
        with self._lock:
            if trade in self._open_trades:
                self._open_trades.remove(trade)
            trade.exit_time = exit_time
            trade.exit_price = float(exit_price)
            trade.notes[f"exit_{reason}"] = 1.0
            pnl = self._calc_trade_pnl(trade, float(exit_price))
            trade.gross_pnl = pnl
            trade.pnl = pnl
            trade.net_pnl = pnl
            self._equity += pnl
            self._gross_equity += pnl
            self._closed_trades.append(trade)
            self._recalc_mark_to_market()
            self._bump()

    def close_trade_by_id(
        self,
        trade_id: str,
        *,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str = "signal_exit",
        close_volume: Optional[float] = None,
    ) -> Optional[Trade]:
        with self._lock:
            trade = self.get_trade_by_id(trade_id)
            if trade is None:
                return None
            if close_volume is None or close_volume >= trade.volume:
                self.close_trade(trade, exit_price=exit_price, exit_time=exit_time, reason=reason)
                return trade
            return self._partial_close_trade(trade, close_volume, exit_price, exit_time, reason)

    def apply_trade_outcome_overrides(
        self,
        trade_id: str,
        *,
        net_pnl: Optional[float] = None,
        gross_pnl: Optional[float] = None,
        commission_paid: Optional[float] = None,
        swap_pnl: Optional[float] = None,
        spread_cost: Optional[float] = None,
        slippage_cost: Optional[float] = None,
        notes: Optional[Dict[str, float]] = None,
    ) -> bool:
        with self._lock:
            trade = next((item for item in self._closed_trades if item.id == trade_id), None)
            if trade is None:
                return False
            previous_pnl = float(getattr(trade, "pnl", 0.0) or 0.0)
            previous_gross = float(getattr(trade, "gross_pnl", 0.0) or 0.0)
            changed = False

            if gross_pnl is not None:
                trade.gross_pnl = float(gross_pnl)
                changed = True
            if net_pnl is not None:
                parsed_net = float(net_pnl)
                trade.net_pnl = parsed_net
                trade.pnl = parsed_net
                changed = True
            if commission_paid is not None:
                trade.commission_paid = float(commission_paid)
                changed = True
            if swap_pnl is not None:
                trade.swap_pnl = float(swap_pnl)
                changed = True
            if spread_cost is not None:
                trade.spread_cost = float(spread_cost)
                changed = True
            if slippage_cost is not None:
                trade.slippage_cost = float(slippage_cost)
                changed = True
            if notes:
                for key, value in notes.items():
                    try:
                        trade.notes[str(key)] = float(value)
                    except Exception:
                        continue
                changed = True

            if not changed:
                return False

            self._equity += float(getattr(trade, "pnl", 0.0) or 0.0) - previous_pnl
            self._gross_equity += float(getattr(trade, "gross_pnl", 0.0) or 0.0) - previous_gross
            self._recalc_mark_to_market()
            self._bump()
            return True

    def close_positions(
        self,
        symbol: str,
        *,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str = "signal_exit",
        strategy: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> List[Trade]:
        closed: List[Trade] = []
        with self._lock:
            for trade in list(self._open_trades):
                if trade.symbol != symbol:
                    continue
                if strategy is not None and trade.strategy != strategy:
                    continue
                if magic is not None and trade.magic != magic:
                    continue
                self.close_trade(trade, exit_price=exit_price, exit_time=exit_time, reason=reason)
                closed.append(trade)
        self._cancel_pending_orders(symbol, strategy=strategy, magic=magic)
        return closed

    def purge_startup_hydration_state(self, *, owner: Optional[str] = None) -> Dict[str, int]:
        """Remove synthetic startup-hydration artifacts from portfolio state."""
        owner_note_key = f"startup_hydration_owner:{owner}" if owner else None
        owner_parent = f"startup_hydration:{owner}" if owner else None

        def _to_float(value: Any) -> float:
            try:
                return float(value)
            except Exception:
                return 0.0

        def _is_synthetic_trade(trade: Trade) -> bool:
            notes = dict(getattr(trade, "notes", {}) or {})
            if _to_float(notes.get("startup_hydration_synthetic")) <= 0.0:
                return False
            if owner_note_key is None:
                return True
            return _to_float(notes.get(owner_note_key)) > 0.0

        def _is_synthetic_order(order: Order) -> bool:
            parent_id = str(getattr(order, "parent_id", "") or "")
            if not parent_id.startswith("startup_hydration:"):
                return False
            if owner_parent is None:
                return True
            return parent_id == owner_parent

        with self._lock:
            removed_open = [trade for trade in self._open_trades if _is_synthetic_trade(trade)]
            removed_closed = [trade for trade in self._closed_trades if _is_synthetic_trade(trade)]
            removed_pending = [order for order in self._pending_orders if _is_synthetic_order(order)]
            removed_orders = [order for order in self._orders if _is_synthetic_order(order)]

            changed = bool(removed_open or removed_closed or removed_pending or removed_orders)
            if not changed:
                return {
                    "open_trades": 0,
                    "closed_trades": 0,
                    "pending_orders": 0,
                    "orders": 0,
                }

            if removed_open:
                self._open_trades = [trade for trade in self._open_trades if not _is_synthetic_trade(trade)]
            if removed_closed:
                closed_pnl = sum(float(getattr(trade, "pnl", 0.0) or 0.0) for trade in removed_closed)
                closed_gross = sum(
                    float(getattr(trade, "gross_pnl", getattr(trade, "pnl", 0.0)) or 0.0)
                    for trade in removed_closed
                )
                self._closed_trades = [trade for trade in self._closed_trades if not _is_synthetic_trade(trade)]
                self._equity -= closed_pnl
                self._gross_equity -= closed_gross
            if removed_pending:
                self._pending_orders = [order for order in self._pending_orders if not _is_synthetic_order(order)]
            if removed_orders:
                self._orders = [order for order in self._orders if not _is_synthetic_order(order)]

            self._recalc_mark_to_market()
            self._bump()
            return {
                "open_trades": len(removed_open),
                "closed_trades": len(removed_closed),
                "pending_orders": len(removed_pending),
                "orders": len(removed_orders),
            }

    def _cancel_pending_orders(
        self,
        symbol: str,
        *,
        strategy: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> None:
        remaining: List[Order] = []
        for order in self._pending_orders:
            if order.symbol == symbol:
                if strategy is not None and order.strategy != strategy:
                    remaining.append(order)
                    continue
                if magic is not None and order.magic != magic:
                    remaining.append(order)
                    continue
                order.status = "cancelled"
                continue
            remaining.append(order)
        if len(remaining) != len(self._pending_orders):
            self._pending_orders = remaining
            self._bump()

    def update_market(
        self,
        symbol: str,
        row: pd.Series,
        timestamp: pd.Timestamp,
        *,
        check_stops: bool = False,
        stop_priority: str = "sl_first",
    ) -> None:
        with self._lock:
            close = row.get("close")
            if close is not None:
                self._last_prices[symbol] = float(close)
            self._last_update = _normalize_ts(timestamp)
            self._recalc_mark_to_market()
        if check_stops:
            self._check_open_trade_stops(symbol, row, timestamp, stop_priority=stop_priority)
        self.process_pending_orders(symbol, row, timestamp)

    def _check_open_trade_stops(
        self,
        symbol: str,
        row: pd.Series,
        timestamp: pd.Timestamp,
        *,
        stop_priority: str = "sl_first",
    ) -> None:
        ts = _normalize_ts(timestamp)
        if ts is None:
            return
        high = float(row.get("high", row.get("close", 0.0)))
        low = float(row.get("low", row.get("close", 0.0)))
        with self._lock:
            for trade in list(self._open_trades):
                if trade.symbol != symbol:
                    continue
                entry_ts = _normalize_ts(getattr(trade, "entry_time", None))
                if entry_ts is not None and ts < entry_ts:
                    # Guard against out-of-order timestamps when event-based
                    # signals carry a later logical entry time than the
                    # current market bar being processed.
                    continue

                stop_points = float(trade.stop_points or 0.0)
                take_points = float(trade.take_points or 0.0)
                if stop_points <= 0.0 and take_points <= 0.0:
                    continue

                sl_price: Optional[float] = None
                tp_price: Optional[float] = None
                sl_hit = False
                tp_hit = False

                if trade.side == "long":
                    if stop_points > 0.0:
                        sl_price = float(trade.entry_price) - stop_points
                        sl_hit = low <= sl_price
                    if take_points > 0.0:
                        tp_price = float(trade.entry_price) + take_points
                        tp_hit = high >= tp_price
                else:
                    if stop_points > 0.0:
                        sl_price = float(trade.entry_price) + stop_points
                        sl_hit = high >= sl_price
                    if take_points > 0.0:
                        tp_price = float(trade.entry_price) - take_points
                        tp_hit = low <= tp_price

                if not (sl_hit or tp_hit):
                    continue

                if sl_hit and tp_hit:
                    tp_first = str(stop_priority or "").strip().lower() == "tp_first"
                    if tp_first and tp_price is not None:
                        self.close_trade(trade, exit_price=tp_price, exit_time=ts, reason="take_profit")
                    elif sl_price is not None:
                        self.close_trade(trade, exit_price=sl_price, exit_time=ts, reason="stop_loss")
                elif sl_hit and sl_price is not None:
                    self.close_trade(trade, exit_price=sl_price, exit_time=ts, reason="stop_loss")
                elif tp_hit and tp_price is not None:
                    self.close_trade(trade, exit_price=tp_price, exit_time=ts, reason="take_profit")

    def process_pending_orders(self, symbol: str, row: pd.Series, timestamp: pd.Timestamp) -> None:
        ts = _normalize_ts(timestamp)
        if ts is None:
            return
        high = float(row.get("high", row.get("close", 0.0)))
        low = float(row.get("low", row.get("close", 0.0)))
        close = float(row.get("close", 0.0))
        with self._lock:
            updated = False
            still_pending: List[Order] = []
            for order in self._pending_orders:
                if order.symbol != symbol:
                    still_pending.append(order)
                    continue
                if order.valid_until is not None and ts > order.valid_until:
                    order.status = "expired"
                    updated = True
                    continue
                price = order.price if order.price is not None else close
                filled = False
                fill_price = price
                if order.type == "market":
                    filled = True
                    fill_price = close
                elif order.type == "limit":
                    if order.side == "buy" and low <= price:
                        filled = True
                    elif order.side == "sell" and high >= price:
                        filled = True
                elif order.type == "stop":
                    if order.side == "buy" and high >= price:
                        filled = True
                    elif order.side == "sell" and low <= price:
                        filled = True
                if filled:
                    order.status = "filled"
                    trade = self._open_trade_from_order(order, fill_price=fill_price, fill_time=ts)
                    if trade is not None:
                        self._open_trades.append(trade)
                    updated = True
                    continue
                if order.status != "working":
                    updated = True
                order.status = "working"
                still_pending.append(order)
            if updated:
                self._pending_orders = still_pending
                self._last_update = ts
                self._recalc_mark_to_market()
                self._bump()

    def get_open_trades(
        self,
        symbol: Optional[str] = None,
        *,
        strategy: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> List[Trade]:
        with self._lock:
            trades = list(self._open_trades)
            if symbol:
                trades = [trade for trade in trades if trade.symbol == symbol]
            if strategy is not None:
                trades = [trade for trade in trades if trade.strategy == strategy]
            if magic is not None:
                trades = [trade for trade in trades if trade.magic == magic]
            return trades

    def get_closed_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        with self._lock:
            if not symbol:
                return list(self._closed_trades)
            return [trade for trade in self._closed_trades if trade.symbol == symbol]

    def get_pending_orders(
        self,
        symbol: Optional[str] = None,
        *,
        strategy: Optional[str] = None,
        magic: Optional[int] = None,
    ) -> List[Order]:
        with self._lock:
            orders = list(self._pending_orders)
            if symbol:
                orders = [order for order in orders if order.symbol == symbol]
            if strategy is not None:
                orders = [order for order in orders if order.strategy == strategy]
            if magic is not None:
                orders = [order for order in orders if order.magic == magic]
            return orders

    def get_exposure_per_symbol(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            cache_key = ("symbol", symbol, None, self._version)
            cached = self._exposure_cache.get(cache_key)
            if cached is not None:
                return cached
            if symbol:
                exposure = self._build_symbol_exposure(symbol)
            else:
                exposure = {sym: self._build_symbol_exposure(sym) for sym in self._all_symbols()}
            self._exposure_cache[cache_key] = exposure
            return exposure

    def get_exposure_per_day(self, day: Optional[datetime | pd.Timestamp] = None) -> Dict[str, Any]:
        with self._lock:
            target_day = _normalize_day(day or datetime.utcnow())
            cache_key = ("day", None, target_day, self._version)
            cached = self._exposure_cache.get(cache_key)
            if cached is not None:
                return cached
            exposure = self._build_day_exposure(target_day, symbol=None)
            self._exposure_cache[cache_key] = exposure
            return exposure

    def get_exposure_per_symbol_per_day(
        self,
        symbol: str,
        day: Optional[datetime | pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            target_day = _normalize_day(day) if day is not None else None
            cache_key = ("symbol_day", symbol, target_day, self._version)
            cached = self._exposure_cache.get(cache_key)
            if cached is not None:
                return cached
            if target_day is not None:
                exposure = self._build_day_exposure(target_day, symbol=symbol)
            else:
                exposure = {
                    day_key: self._build_day_exposure(day_key, symbol=symbol)
                    for day_key in self._all_days(symbol=symbol)
                }
            self._exposure_cache[cache_key] = exposure
            return exposure

    def _build_symbol_exposure(self, symbol: str) -> Dict[str, Any]:
        open_trades = [trade for trade in self._open_trades if trade.symbol == symbol]
        closed_trades = [trade for trade in self._closed_trades if trade.symbol == symbol]
        pending_orders = [order for order in self._pending_orders if order.symbol == symbol]
        open_volume = sum(trade.volume for trade in open_trades)
        closed_volume = sum(trade.volume for trade in closed_trades)
        open_notional = sum(trade.entry_price * trade.volume for trade in open_trades)
        closed_notional = sum(trade.entry_price * trade.volume for trade in closed_trades)
        pending_volume = sum(order.volume for order in pending_orders)
        return {
            "symbol": symbol,
            "open_trades": list(open_trades),
            "closed_trades": list(closed_trades),
            "pending_orders": list(pending_orders),
            "open_trade_count": len(open_trades),
            "closed_trade_count": len(closed_trades),
            "pending_order_count": len(pending_orders),
            "open_volume": float(open_volume),
            "closed_volume": float(closed_volume),
            "pending_volume": float(pending_volume),
            "open_notional": float(open_notional),
            "closed_notional": float(closed_notional),
        }

    def _build_day_exposure(self, day: pd.Timestamp, *, symbol: Optional[str]) -> Dict[str, Any]:
        opened_trades: List[Trade] = []
        closed_trades: List[Trade] = []
        open_trades: List[Trade] = []
        for trade in self._all_trades(symbol=symbol):
            entry_day = _normalize_day(trade.entry_time)
            exit_day = _normalize_day(trade.exit_time)
            if entry_day == day:
                opened_trades.append(trade)
            if exit_day == day:
                closed_trades.append(trade)
            if entry_day is not None and entry_day <= day:
                if exit_day is None or exit_day > day:
                    open_trades.append(trade)
        pending_orders = [
            order
            for order in self._pending_orders
            if (symbol is None or order.symbol == symbol)
            and _normalize_day(order.created_at) == day
        ]
        return {
            "day": day,
            "symbol": symbol,
            "open_trades": list(open_trades),
            "opened_trades": list(opened_trades),
            "closed_trades": list(closed_trades),
            "pending_orders": list(pending_orders),
            "open_count": len(open_trades),
            "opened_count": len(opened_trades),
            "closed_count": len(closed_trades),
            "pending_count": len(pending_orders),
            "open_volume": float(sum(trade.volume for trade in open_trades)),
            "opened_volume": float(sum(trade.volume for trade in opened_trades)),
            "closed_volume": float(sum(trade.volume for trade in closed_trades)),
        }

    def _all_trades(self, *, symbol: Optional[str] = None) -> Iterable[Trade]:
        if symbol is None:
            for trade in self._open_trades:
                yield trade
            for trade in self._closed_trades:
                yield trade
        else:
            for trade in self._open_trades:
                if trade.symbol == symbol:
                    yield trade
            for trade in self._closed_trades:
                if trade.symbol == symbol:
                    yield trade

    def _all_symbols(self) -> List[str]:
        symbols = {trade.symbol for trade in self._open_trades}
        symbols.update(trade.symbol for trade in self._closed_trades)
        symbols.update(order.symbol for order in self._pending_orders)
        return sorted(symbols)

    def _all_days(self, *, symbol: Optional[str] = None) -> List[pd.Timestamp]:
        days = set()
        for trade in self._all_trades(symbol=symbol):
            entry_day = _normalize_day(trade.entry_time)
            exit_day = _normalize_day(trade.exit_time)
            if entry_day is not None:
                days.add(entry_day)
            if exit_day is not None:
                days.add(exit_day)
        return sorted(days)

    def _open_trade_from_order(
        self,
        order: Order,
        *,
        fill_price: float,
        fill_time: pd.Timestamp,
    ) -> Optional[Trade]:
        trade = Trade(
            symbol=order.symbol,
            side="long" if order.side == "buy" else "short",
            entry_time=order.created_at or fill_time,
            entry_price=float(fill_price),
            volume=float(order.volume),
            stop_points=float(order.stop_points),
            take_points=float(order.take_points),
            order_id=order.id,
            broker_ticket=order.broker_ticket,
            strategy=order.strategy,
            magic=order.magic,
        )
        return trade

    def _calc_trade_pnl(self, trade: Trade, exit_price: float) -> float:
        import arbitrix_core.costs as costs
        try:
            point_value = costs.get_point_value(trade.symbol)
        except Exception:
            point_value = 1.0
        if not point_value:
            point_value = 1.0
        if trade.side == "long":
            return (exit_price - trade.entry_price) * point_value * trade.volume
        return (trade.entry_price - exit_price) * point_value * trade.volume

    def _calc_trade_pnl_volume(self, trade: Trade, exit_price: float, volume: float) -> float:
        import arbitrix_core.costs as costs
        try:
            point_value = costs.get_point_value(trade.symbol)
        except Exception:
            point_value = 1.0
        if not point_value:
            point_value = 1.0
        if trade.side == "long":
            return (exit_price - trade.entry_price) * point_value * volume
        return (trade.entry_price - exit_price) * point_value * volume

    def _partial_close_trade(
        self,
        trade: Trade,
        volume: float,
        exit_price: float,
        exit_time: pd.Timestamp,
        reason: str,
    ) -> Optional[Trade]:
        if volume <= 0:
            return None
        volume = min(float(volume), float(trade.volume))
        if volume <= 0:
            return None
        ratio = volume / float(trade.volume)
        closed_trade = Trade(
            symbol=trade.symbol,
            side=trade.side,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            volume=volume,
            stop_points=trade.stop_points,
            take_points=trade.take_points,
            commission_paid=trade.commission_paid * ratio,
            spread_cost=trade.spread_cost * ratio,
            slippage_cost=trade.slippage_cost * ratio,
            swap_pnl=trade.swap_pnl * ratio,
            order_id=trade.order_id,
            broker_ticket=trade.broker_ticket,
            strategy=trade.strategy,
            magic=trade.magic,
        )
        closed_trade.exit_time = exit_time
        closed_trade.exit_price = float(exit_price)
        closed_trade.notes[f"exit_{reason}"] = 1.0
        closed_trade.gross_pnl = self._calc_trade_pnl_volume(trade, float(exit_price), volume)
        total_costs = closed_trade.commission_paid + closed_trade.spread_cost + closed_trade.slippage_cost
        closed_trade.pnl = closed_trade.gross_pnl - total_costs
        closed_trade.net_pnl = closed_trade.gross_pnl - total_costs + closed_trade.swap_pnl

        trade.volume = float(trade.volume) - volume
        trade.commission_paid *= 1.0 - ratio
        trade.spread_cost *= 1.0 - ratio
        trade.slippage_cost *= 1.0 - ratio
        trade.swap_pnl *= 1.0 - ratio

        self._equity += closed_trade.pnl
        self._gross_equity += closed_trade.gross_pnl
        self._closed_trades.append(closed_trade)

        if trade.volume <= 0:
            if trade in self._open_trades:
                self._open_trades.remove(trade)
        self._recalc_mark_to_market()
        self._bump()
        return closed_trade

    @staticmethod
    def _points_from_price(trade: Trade, price: float, *, kind: str) -> float:
        if trade.side == "long":
            return (trade.entry_price - price) if kind == "sl" else (price - trade.entry_price)
        return (price - trade.entry_price) if kind == "sl" else (trade.entry_price - price)

    def _recalc_mark_to_market(self) -> None:
        unrealized = 0.0
        for trade in self._open_trades:
            last_price = self._last_prices.get(trade.symbol)
            if last_price is None:
                continue
            unrealized += self._calc_trade_pnl(trade, last_price)
        self._equity_marked = self._equity + unrealized
