from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd

from arbitrix_core.data.market import FeedKey, MarketBars, MarketFrames, PreparedMarket
from arbitrix_core.strategies.base import BaseStrategy
from arbitrix_core.trading import Order, Signal, Trade

EntryAction = Literal["buy", "sell"]
PendingType = Literal["limit", "stop"]
ManagementAction = Literal["close", "partial_close"]


class DeterministicMultiMarketParityStrategy(BaseStrategy):
    """Absolute-UTC 20-minute lifecycle scenario for live/backtest parity.

    It reads a data-only auxiliary provider and trades two symbols through the
    main provider. ``scenario_start_utc`` identifies the first closed M1 bar
    that emits signals; process start, warm-up length and backtest range do not
    affect the schedule. Stable magic slots resolve runtime-specific IDs.
    """

    name = "deterministic_multi_market_parity"
    requires_portfolio = True
    timeframe = "M1"
    duration_minutes = 20

    def __init__(
        self,
        *,
        scenario_start_utc: str,
        primary_symbol: str = "EURUSD",
        secondary_symbol: str = "GBPUSD",
        auxiliary_provider: str = "confirmation",
        auxiliary_symbol: str = "DXY",
        timeframe: str = "M1",
        primary_volume: float = 0.02,
        secondary_volume: float = 0.02,
        distance_fraction: float = 0.20,
        base_magic: int = 826_000,
    ) -> None:
        start = pd.Timestamp(scenario_start_utc)
        if start.tzinfo is None:
            raise ValueError("scenario_start_utc must include an explicit timezone")
        start = start.tz_convert("UTC")
        if start.second or start.microsecond or start.nanosecond:
            raise ValueError("scenario_start_utc must be aligned to an exact minute")
        if str(timeframe).upper() != "M1":
            raise ValueError("deterministic parity scenario requires timeframe='M1'")
        if primary_symbol == secondary_symbol:
            raise ValueError("primary_symbol and secondary_symbol must differ")
        if float(primary_volume) <= 0.0 or float(secondary_volume) <= 0.0:
            raise ValueError("scenario volumes must be positive")
        if not 0.0 < float(distance_fraction) < 0.5:
            raise ValueError("distance_fraction must be between 0 and 0.5")

        self.scenario_start_utc = start
        self.symbol = str(primary_symbol)
        self.secondary_symbol = str(secondary_symbol)
        self.auxiliary_provider = str(auxiliary_provider)
        self.auxiliary_symbol = str(auxiliary_symbol)
        self.timeframe = "M1"
        self.primary_volume = float(primary_volume)
        self.secondary_volume = float(secondary_volume)
        self.distance_fraction = float(distance_fraction)
        self.base_magic = int(base_magic)
        self._emitted_minutes: set[pd.Timestamp] = set()

    @property
    def _primary_key(self) -> FeedKey:
        return FeedKey("main", self.symbol, self.timeframe)

    @property
    def _secondary_key(self) -> FeedKey:
        return FeedKey("main", self.secondary_symbol, self.timeframe)

    @property
    def _auxiliary_key(self) -> FeedKey:
        return FeedKey(self.auxiliary_provider, self.auxiliary_symbol, self.timeframe)

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        data: MarketFrames | None = None,
    ) -> PreparedMarket:
        if data is None:
            raise ValueError("deterministic parity strategy requires multiprovider MarketFrames")
        required = {self._primary_key, self._secondary_key, self._auxiliary_key}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"deterministic parity strategy missing feeds: {sorted(map(str, missing))}")
        if df is not data.primary_frame or data.primary != self._primary_key:
            raise ValueError("primary frame does not match the configured main-provider symbol")

        auxiliary_close = data[self._auxiliary_key]["close"].to_numpy(copy=False)
        return PreparedMarket(
            {
                key: frame.assign(parity_aux_close=auxiliary_close)
                for key, frame in data.items()
            },
            primary=data.primary,
        )

    def stop_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str | None = None,
        data: MarketBars | None = None,
    ) -> float:
        del symbol, data
        return self._distance(float(row["close"]))

    def take_distance_points(
        self,
        row: pd.Series,
        *,
        symbol: str | None = None,
        data: MarketBars | None = None,
    ) -> float:
        del symbol, data
        return self._distance(float(row["close"]))

    def on_bar(
        self,
        row: pd.Series,
        portfolio: Any,
        regime_output: Any = None,
        ctx: Any = None,
        *,
        data: MarketBars | None = None,
    ) -> list[Signal]:
        del regime_output, ctx
        if getattr(self, "_startup_hydration_active", False):
            return []
        if data is None:
            raise ValueError("deterministic parity strategy requires synchronized MarketBars")
        self._validate_bars(data)

        ts = self._utc(row.name)
        elapsed = ts - self.scenario_start_utc
        if elapsed < pd.Timedelta(0) or elapsed >= pd.Timedelta(minutes=self.duration_minutes):
            return []
        if elapsed % pd.Timedelta(minutes=1) != pd.Timedelta(0) or ts in self._emitted_minutes:
            return []

        step = int(elapsed // pd.Timedelta(minutes=1))
        signals = self._build_step(step, ts, portfolio, data)
        self._emitted_minutes.add(ts)
        return signals

    def _build_step(
        self,
        step: int,
        ts: pd.Timestamp,
        portfolio: Any,
        data: MarketBars,
    ) -> list[Signal]:
        p, s = self.symbol, self.secondary_symbol
        pv, sv = self.primary_volume, self.secondary_volume

        if step == 0:
            return [self._entry(ts, data, p, "buy", pv, 1, "P1"), self._entry(ts, data, s, "sell", sv, 2, "S1")]
        if step == 1:
            return self._modify_trades(ts, portfolio, ((p, 1), (s, 2)))
        if step == 2:
            return self._manage_trades(ts, portfolio, "partial_close", ((p, 1, pv / 2), (s, 2, sv / 2)))
        if step == 3:
            return self._modify_trades(ts, portfolio, ((p, 1), (s, 2)))
        if step == 4:
            return [
                self._pending(ts, data, p, "buy", "limit", pv, 3, "PL1"),
                self._pending(ts, data, s, "sell", "stop", sv, 4, "SS1"),
            ]
        if step == 5:
            return self._reprice_orders(ts, portfolio, ((p, 3), (s, 4))) + self._modify_orders(
                ts, portfolio, ((p, 3), (s, 4)), kind="sl"
            )
        if step == 6:
            return self._cancel_orders(ts, portfolio, ((p, 3), (s, 4))) + self._manage_trades(
                ts, portfolio, "close", ((p, 1, None), (s, 2, None))
            )
        if step == 7:
            return [self._entry(ts, data, p, "sell", pv, 5, "P2"), self._entry(ts, data, s, "buy", sv, 6, "S2")]
        if step == 8:
            return self._modify_trades(ts, portfolio, ((p, 5), (s, 6)))
        if step == 9:
            return self._manage_trades(ts, portfolio, "partial_close", ((p, 5, pv / 2), (s, 6, sv / 2))) + self._modify_trades(
                ts, portfolio, ((p, 5), (s, 6)), kinds=("tp",)
            )
        if step == 10:
            return [
                self._pending(ts, data, p, "buy", "limit", pv, 7, "PL2"),
                self._pending(ts, data, s, "sell", "stop", sv, 8, "SS2"),
            ] + self._modify_trades(
                ts, portfolio, ((p, 5), (s, 6)), kinds=("sl",)
            )
        if step == 11:
            return self._cancel_orders(ts, portfolio, ((p, 7), (s, 8))) + self._manage_trades(
                ts, portfolio, "close", ((p, 5, None), (s, 6, None))
            )
        if step == 12:
            return [self._entry(ts, data, p, "buy", pv, 9, "P3"), self._entry(ts, data, s, "sell", sv, 10, "S3")]
        if step == 13:
            return self._modify_trades(ts, portfolio, ((p, 9), (s, 10)))
        if step == 14:
            return self._manage_trades(ts, portfolio, "partial_close", ((p, 9, pv / 2), (s, 10, sv / 2)))
        if step == 15:
            return self._modify_trades(ts, portfolio, ((p, 9), (s, 10)))
        if step == 16:
            return self._manage_trades(ts, portfolio, "close", ((p, 9, None), (s, 10, None))) + [
                self._entry(ts, data, p, "sell", pv, 11, "P4"),
                self._entry(ts, data, s, "buy", sv, 12, "S4"),
            ]
        if step == 17:
            return self._modify_trades(ts, portfolio, ((p, 11), (s, 12)))
        if step == 18:
            return self._manage_trades(ts, portfolio, "partial_close", ((p, 11, pv / 2), (s, 12, sv / 2))) + self._modify_trades(
                ts, portfolio, ((p, 11), (s, 12)), kinds=("tp",)
            )
        if step == 19:
            return self._manage_trades(ts, portfolio, "close", ((p, 11, None), (s, 12, None)))
        raise AssertionError(f"unhandled deterministic scenario step {step}")

    def _entry(
        self,
        ts: pd.Timestamp,
        data: MarketBars,
        symbol: str,
        action: EntryAction,
        volume: float,
        slot: int,
        label: str,
    ) -> Signal:
        return Signal(
            when=ts,
            action=action,
            price=self._close(data, symbol),
            volume=volume,
            magic=self.base_magic + slot,
            symbol=symbol,
            reason=self._reason(ts, label),
        )

    def _pending(
        self,
        ts: pd.Timestamp,
        data: MarketBars,
        symbol: str,
        action: EntryAction,
        order_type: PendingType,
        volume: float,
        slot: int,
        label: str,
    ) -> Signal:
        close = self._close(data, symbol)
        price = close - self._distance(close)
        return Signal(
            when=ts,
            action=action,
            price=price,
            order_type=order_type,
            limit_price=price if order_type == "limit" else None,
            stop_price=price if order_type == "stop" else None,
            volume=volume,
            magic=self.base_magic + slot,
            symbol=symbol,
            reason=self._reason(ts, label),
        )

    def _modify_trades(
        self,
        ts: pd.Timestamp,
        portfolio: Any,
        targets: tuple[tuple[str, int], ...],
        *,
        kinds: tuple[Literal["sl", "tp"], ...] = ("sl", "tp"),
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, slot in targets:
            trade = self._trade(portfolio, symbol, slot)
            if trade is None:
                continue
            for kind in kinds:
                value = self._protection_price(trade, kind)
                signals.append(
                    Signal(
                        when=ts,
                        action="modify_sl" if kind == "sl" else "modify_tp",
                        price=float(trade.entry_price),
                        target_trade_id=str(trade.id),
                        new_sl=value if kind == "sl" else None,
                        new_tp=value if kind == "tp" else None,
                        magic=self.base_magic + slot,
                        symbol=symbol,
                        reason=self._reason(ts, f"M{kind.upper()}{slot}"),
                    )
                )
        return signals

    def _manage_trades(
        self,
        ts: pd.Timestamp,
        portfolio: Any,
        action: ManagementAction,
        targets: tuple[tuple[str, int, float | None], ...],
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, slot, volume in targets:
            trade = self._trade(portfolio, symbol, slot)
            if trade is None:
                continue
            signals.append(
                Signal(
                    when=ts,
                    action=action,
                    price=float(trade.entry_price),
                    target_trade_id=str(trade.id),
                    close_volume=volume if action == "partial_close" else None,
                    magic=self.base_magic + slot,
                    symbol=symbol,
                    reason=self._reason(ts, f"{action[:2].upper()}{slot}"),
                )
            )
        return signals

    def _reprice_orders(
        self,
        ts: pd.Timestamp,
        portfolio: Any,
        targets: tuple[tuple[str, int], ...],
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, slot in targets:
            order = self._order(portfolio, symbol, slot)
            if order is None or order.price is None:
                continue
            new_price = float(order.price) - self._distance(float(order.price)) * 0.25
            signals.append(
                Signal(
                    when=ts,
                    action="modify_price",
                    price=float(order.price),
                    target_order_id=str(order.id),
                    new_price=new_price,
                    magic=self.base_magic + slot,
                    symbol=symbol,
                    reason=self._reason(ts, f"MP{slot}"),
                )
            )
        return signals

    def _modify_orders(
        self,
        ts: pd.Timestamp,
        portfolio: Any,
        targets: tuple[tuple[str, int], ...],
        *,
        kind: Literal["sl", "tp"],
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, slot in targets:
            order = self._order(portfolio, symbol, slot)
            if order is None or order.price is None:
                continue
            distance = self._distance(float(order.price))
            if kind == "sl":
                value = float(order.price) - distance if order.side == "buy" else float(order.price) + distance
            else:
                value = float(order.price) + distance if order.side == "buy" else float(order.price) - distance
            signals.append(
                Signal(
                    when=ts,
                    action="modify_sl" if kind == "sl" else "modify_tp",
                    price=float(order.price),
                    target_order_id=str(order.id),
                    new_sl=value if kind == "sl" else None,
                    new_tp=value if kind == "tp" else None,
                    magic=self.base_magic + slot,
                    symbol=symbol,
                    reason=self._reason(ts, f"PO{slot}"),
                )
            )
        return signals

    def _cancel_orders(
        self,
        ts: pd.Timestamp,
        portfolio: Any,
        targets: tuple[tuple[str, int], ...],
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, slot in targets:
            order = self._order(portfolio, symbol, slot)
            if order is None:
                continue
            signals.append(
                Signal(
                    when=ts,
                    action="cancel_order",
                    price=float(order.price or 0.0),
                    target_order_id=str(order.id),
                    magic=self.base_magic + slot,
                    symbol=symbol,
                    reason=self._reason(ts, f"CA{slot}"),
                )
            )
        return signals

    def _trade(self, portfolio: Any, symbol: str, slot: int) -> Trade | None:
        matches = portfolio.get_open_trades(symbol, magic=self.base_magic + slot)
        if not matches:
            return None
        return sorted(matches, key=lambda trade: (pd.Timestamp(trade.entry_time), str(trade.id)))[0]

    def _order(self, portfolio: Any, symbol: str, slot: int) -> Order | None:
        matches = portfolio.get_pending_orders(symbol, magic=self.base_magic + slot)
        if not matches:
            return None
        return sorted(matches, key=lambda order: (pd.Timestamp(order.created_at), str(order.id)))[0]

    def _protection_price(self, trade: Trade, kind: Literal["sl", "tp"]) -> float:
        distance = self._distance(float(trade.entry_price)) * (0.55 if kind == "sl" else 0.60)
        if trade.side == "long":
            return float(trade.entry_price) - distance if kind == "sl" else float(trade.entry_price) + distance
        return float(trade.entry_price) + distance if kind == "sl" else float(trade.entry_price) - distance

    def _validate_bars(self, data: MarketBars) -> None:
        if data.primary != self._primary_key:
            raise ValueError("MarketBars primary feed does not match the configured scenario")
        for key in (self._primary_key, self._secondary_key, self._auxiliary_key):
            if key not in data:
                raise ValueError(f"deterministic parity strategy missing bar for {key}")
            close = float(data[key]["close"])
            if not math.isfinite(close) or close <= 0.0:
                raise ValueError(f"deterministic parity strategy received invalid close for {key}")

    def _close(self, data: MarketBars, symbol: str) -> float:
        return float(data[FeedKey("main", symbol, self.timeframe)]["close"])

    def _distance(self, price: float) -> float:
        return max(abs(float(price)) * self.distance_fraction, 1e-9)

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            raise ValueError("deterministic parity scenario requires timezone-aware bars")
        return ts.tz_convert("UTC")

    @staticmethod
    def _reason(ts: pd.Timestamp, label: str) -> str:
        return f"DMP1:{ts.strftime('%Y%m%d%H%M')}:{label}"[:31]


__all__ = ["DeterministicMultiMarketParityStrategy"]
