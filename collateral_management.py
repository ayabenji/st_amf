"""Collateral management helpers for the AMF stress test workflow.

The historical development of the project mostly happened inside a
Jupyter notebook.  This module extracts the parts that are required to
run the collateral process in a reusable manner.  The central function
``process_pv_after_day_1`` aggregates the post‑Day‑1 trade values,
assesses margin calls, consumes available cash when Groupama needs to
post collateral and finally stores the daily results in a dedicated
history file.  A companion helper ``roll_balance_for_next_day`` updates
the collateral input so that the next run starts with the freshly
computed balances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Tuple, Any
import sys 
import inspect
import numpy as np
import pandas as pd
import re

@dataclass(frozen=True)
class LiabilityShockLookup:
    """Container holding normalised liability shock percentages."""

    by_portfolio: dict[object, dict[str, float]]
    by_normalised_portfolio: dict[str, dict[str, float]]
    day_columns: dict[str, str]

    def get_percentage(self, portfolio_key: object, day_key: str) -> float | None:
        """Return the liability percentage for ``portfolio_key`` and ``day_key``."""

        mapping = self.by_portfolio.get(portfolio_key)
        if mapping is not None and day_key in mapping:
            return mapping[day_key]

        normalised = _normalise_portfolio_value(portfolio_key)
        if normalised is not None:
            mapping = self.by_normalised_portfolio.get(normalised)
            if mapping is not None and day_key in mapping:
                return mapping[day_key]

        return None

@dataclass
class CollateralConfig:
    """Configuration of collateral related resources and column names."""

    collateral_input_path: Path = Path(r"C:\Users\abenjelloun\OneDrive - Cooperactions\GAM-E-Risk Perf - RMP\1.PROD\1.REGLEMENTAIRE\14.Stress Test AMF (JB)\Production\Périmètre et positions\Collat_Cash_MTM_20250401.csv")

    collateral_history_path: Path = Path("collateral_history.xlsx")
    monetary_fund_usage_history_path: Path = Path("monetary_fund_usage_history.xlsx")
    liability_shocks_path: Path = Path(r"C:\Users\abenjelloun\OneDrive - Cooperactions\GAM-E-Risk Perf - RMP\1.PROD\1.REGLEMENTAIRE\14.Stress Test AMF (JB)\Production\Périmètre et positions\Matrices correspondance_AB.xlsx")
    liability_shocks_sheet: str | None = "Liability_shocks"

    counterparty_col: str = "Counterparty"
    portfolio_col: str = "Portfolio"
    balance_prev_col: str = "Balance_J_1"
    threshold_col: str = "Seuil declenchement"
    cash_col: str = "Cash_disponible"

    def ensure_directories(self) -> None:
        """Create parent folders for the configured files if necessary."""
        for path in (
            self.collateral_input_path,
            self.collateral_history_path,
            self.monetary_fund_usage_history_path,
        ):
                if path.parent and path.parent != Path(""):
                    path.parent.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class LiabilityShockLookup:
    """Container holding normalised liability shock percentages."""

    by_portfolio: dict[object, dict[str, float]]
    by_normalised_portfolio: dict[str, dict[str, float]]
    day_columns: dict[str, str]

    def get_percentage(self, portfolio_key: object, day_key: str) -> float | None:
        """Return the liability percentage for ``portfolio_key`` and ``day_key``."""

        mapping = self.by_portfolio.get(portfolio_key)
        if mapping is not None and day_key in mapping:
            return mapping[day_key]

        normalised = _normalise_portfolio_value(portfolio_key)
        if normalised is not None:
            mapping = self.by_normalised_portfolio.get(normalised)
            if mapping is not None and day_key in mapping:
                return mapping[day_key]

        return None
    
def _normalise_asset_identifier(value: object) -> str | None:
    """Return a comparable representation of an asset identifier."""

    if not isinstance(value, str):
        return None
    simplified = re.sub(r"[\s\-]+", " ", value).strip().lower()
    return simplified or None


MONETARY_FUND_ASSET_IDS = {
    "GROUPAMA MONETAIRE- IC",
    "GROUPAMA TRESORERIE I",
    "GROUPAMA ENTREPRISES I",
    "GROUPAMA ULTRA SHORT TERM - IC"
}

_NORMALISED_MONETARY_FUND_IDS = {
    _normalise_asset_identifier(name) for name in MONETARY_FUND_ASSET_IDS
}

_MONETARY_FUND_PORTFOLIO_CODES = {
    _normalise_asset_identifier("GROUPAMA MONETAIRE- IC"): "300636",
    _normalise_asset_identifier("GROUPAMA ENTREPRISES I"): "300208",
    _normalise_asset_identifier("GROUPAMA TRESORERIE I"): "300203",
    _normalise_asset_identifier("GROUPAMA ULTRA SHORT TERM - IC"): "389038",
}


def _normalise_portfolio_value(value: object) -> str | None:
    """Return a normalised textual representation of a portfolio identifier."""

    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value}".strip()
    text = str(value).strip()
    return text or None

def build_liability_shock_lookup(
    liability_shocks: pd.DataFrame | Mapping[object, Mapping[object, object]]
) -> LiabilityShockLookup:
    """Normalise a liability shock table into a lookup structure.

    Parameters
    ----------
    liability_shocks:
        Either a dataframe exposing a portfolio column and one column per
        stress day or a nested mapping ``{portfolio: {day: percentage}}``.
        Percentages are expected in plain numeric form (``-1.42`` meaning a
        1.42% outflow).

    Returns
    -------
    LiabilityShockLookup
        Structured lookup ready to be consumed by
        :func:`process_pv_after_day_1`.
    """

    if isinstance(liability_shocks, LiabilityShockLookup):
        return liability_shocks

    day_columns: dict[str, str] = {}
    by_portfolio: dict[object, dict[str, float]] = {}
    by_normalised: dict[str, dict[str, float]] = {}

    def _update_mapping(portfolio_key: object, values: Mapping[str, float]) -> None:
        if not values:
            return
        by_portfolio[portfolio_key] = dict(values)
        normalised = _normalise_portfolio_value(portfolio_key)
        if normalised is not None:
            by_normalised[normalised] = dict(values)

    if isinstance(liability_shocks, Mapping):
        for portfolio_key, per_day in liability_shocks.items():
            if not isinstance(per_day, Mapping):
                continue
            normalised_values: dict[str, float] = {}
            for day_label, value in per_day.items():
                if day_label is None:
                    continue
                day_key = _normalise_column_name(str(day_label))
                if not day_key:
                    continue
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if not np.isfinite(numeric):
                    continue
                normalised_values[day_key] = float(numeric)
                day_columns.setdefault(day_key, str(day_label))
            _update_mapping(portfolio_key, normalised_values)

        return LiabilityShockLookup(by_portfolio, by_normalised, day_columns)

    if not isinstance(liability_shocks, pd.DataFrame):
        raise TypeError("liability_shocks must be a DataFrame or a mapping")

    df = liability_shocks.copy()
    if df.empty:
        return LiabilityShockLookup(by_portfolio, by_normalised, day_columns)

    portfolio_candidates = [
        col
        for col in df.columns
        if _normalise_column_name(col)
        in {"portfolio", "portefeuille", "code_portefeuille"}
    ]
    if not portfolio_candidates:
        raise KeyError(
            "Liability shocks table must expose a portfolio column (Portfolio/Portefeuille)."
        )

    portfolio_col = portfolio_candidates[0]
    day_cols = [col for col in df.columns if col != portfolio_col]
    day_lookup: dict[str, str] = {}
    for col in day_cols:
        day_key = _normalise_column_name(col)
        if day_key and day_key not in day_lookup:
            day_lookup[day_key] = col

    if not day_lookup:
        return LiabilityShockLookup(by_portfolio, by_normalised, day_columns)

    working = df.copy()
    for original_col in day_lookup.values():
        series = working[original_col]
        if series.dtype == object:
            series = series.astype(str).str.replace(",", ".")
        working[original_col] = pd.to_numeric(series, errors="coerce")

    for _, row in working.iterrows():
        portfolio_key = row[portfolio_col]
        key = portfolio_key if pd.notna(portfolio_key) else None
        normalised_values: dict[str, float] = {}
        for day_key, original_col in day_lookup.items():
            value = row[original_col]
            if pd.isna(value):
                continue
            numeric = float(value)
            if not np.isfinite(numeric):
                continue
            normalised_values[day_key] = numeric
            day_columns.setdefault(day_key, original_col)
        _update_mapping(key, normalised_values)

    return LiabilityShockLookup(by_portfolio, by_normalised, day_columns)

def _normalise_column_name(name: str) -> str:
    """Return a simplified representation used to identify columns."""

    normalized = (
        name.strip()
        .lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ç", "c")
        .replace(" ", "_")
        .replace("-", "_")
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _sanitise_snapshot_label(label: str | None) -> str:
    """Return a filesystem-friendly representation of a snapshot label."""

    if label is None:
        return ""
    text = str(label).strip()
    if not text:
        return ""
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", text.strip()).strip("_")
    return sanitized.lower()


def _write_collateral_frame(df: pd.DataFrame, path: Path) -> None:
    """Persist a collateral dataframe following the destination format."""

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False, sep=";", decimal=",", encoding="latin1")
    else:
        df.to_csv(path, index=False)

def _deduplicate_columns(
    df: pd.DataFrame,
    canonical_name: str,
) -> pd.DataFrame:
    """Ensure there is at most one column matching ``canonical_name``."""

    normalised_target = _normalise_column_name(canonical_name)
    matching_columns = [
        col for col in df.columns if _normalise_column_name(col) == normalised_target
    ]
    if not matching_columns:
        return df

    keep = canonical_name if canonical_name in matching_columns else matching_columns[0]
    if keep != canonical_name:
        df = df.rename(columns={keep: canonical_name})
        keep = canonical_name

    for col in matching_columns:
        if col == keep:
            continue
        if col in df.columns:
            df = df.drop(columns=col)
    return df

def _normalise_balance_columns(
    df: pd.DataFrame,
    config: CollateralConfig,
) -> pd.DataFrame:
    """Return a copy where balance columns use their canonical names."""

    frame = df
    frame = _deduplicate_columns(frame, "Balance_J")
    frame = _deduplicate_columns(frame, config.balance_prev_col)
    return frame

def _load_collateral_inputs(config: CollateralConfig) -> pd.DataFrame:
    """Load the collateral instructions for each counterparty/portfolio."""

    path = config.collateral_input_path
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            inputs = pd.read_excel(path)
        else:
            read_kwargs = {"sep": ";", "decimal": ",", "encoding": "latin1"}
            try:
                inputs = pd.read_csv(path, **read_kwargs)
            except pd.errors.ParserError:
                fallback_kwargs = {**read_kwargs, "sep": None, "engine": "python"}
                try:
                    inputs = pd.read_csv(path, **fallback_kwargs)
                except Exception as exc:  # pragma: no cover - safety net for unexpected formats
                    raise RuntimeError(
                        f"Unable to read collateral input file at '{path}'."
                    ) from exc
        inputs = inputs.rename(
            columns={
                "Code portefeuille": "Portfolio",
                "Contrepartie": "Counterparty",
            }
        )
    else:
        inputs = pd.DataFrame(
            columns=[
                config.portfolio_col,
                config.counterparty_col,
                config.balance_prev_col,
                config.threshold_col,
                config.cash_col,
            ]
        )

    rename_map: dict[str, str] = {}
    alt_map = {
        config.counterparty_col: {"counterparty", "contrepartie"},
        config.portfolio_col: {"portfolio", "code portefeuille"},
        config.balance_prev_col: {"balance_j_1", "balance_j1", "balance_jmoins1"},
        config.threshold_col: {
            "seuil_de_declenchement",
            "seuil_declenchement",
            "seuil",
            "threshold",
        },
        config.cash_col: {"cash_disponible"},
    }

    normalised = {_normalise_column_name(col): col for col in inputs.columns}
    for target, alternatives in alt_map.items():
        wanted = _normalise_column_name(target)
        if wanted in normalised:
            rename_map[normalised[wanted]] = target
            continue
        for alt in alternatives:
            if alt in normalised:
                rename_map[normalised[alt]] = target
                break

    if rename_map:
        inputs = inputs.rename(columns=rename_map)

    for col in (
        config.portfolio_col,
        config.counterparty_col,
        config.balance_prev_col,
        config.threshold_col,
        config.cash_col,
    ):
        if col not in inputs.columns:
            default = 0.0
            if col in (config.portfolio_col, config.counterparty_col):
                default = np.nan
            inputs[col] = default

    

    for numeric_col in (
        config.balance_prev_col,
        config.threshold_col,
        config.cash_col,
    ):
        inputs[numeric_col] = pd.to_numeric(inputs[numeric_col], errors="coerce").fillna(0.0)


    return inputs


def _format_alert(amount: float, *, reason: str | None = None) -> str:
    """Human readable representation of the missing cash amount."""

    formatted_amount = f"{amount:,.2f}".replace(",", " ")
    if reason:
        return f"cash insuffisant ({reason} : {formatted_amount})"
    return f"cash insuffisant ({formatted_amount})"

def _load_liability_shocks(config: CollateralConfig) -> pd.DataFrame | None:
    """Read the liability shock table defined in the configuration."""

    path = getattr(config, "liability_shocks_path", None)
    if path in (None, ""):
        return None

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Unable to read liability shocks at '{path}'.")

    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            sheet_name = getattr(config, "liability_shocks_sheet", None)
            try:
                return pd.read_excel(path, sheet_name=sheet_name)
            except ValueError as exc:
                raise RuntimeError(
                    f"Sheet '{sheet_name}' missing in liability shocks workbook '{path}'."
                ) from exc

        read_attempts = [
            {},
            {"sep": ";", "decimal": ","},
            {"sep": None, "engine": "python"},
            {"sep": ";", "decimal": ",", "encoding": "latin1"},
        ]
        for kwargs in read_attempts:
            try:
                return pd.read_csv(path, **kwargs)
            except pd.errors.ParserError:
                continue
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - safety net for unexpected formats
        raise RuntimeError(f"Unable to read liability shocks at '{path}'.") from exc

def _consume_monetary_funds(
    df: pd.DataFrame, indices: list[object], amount: float
) -> float:
    """Reduce the TV of monetary funds to free cash for collateral calls."""

    if amount <= 0 or not indices:
        return 0.0

    remaining = float(amount)
    consumed = 0.0
    for idx in indices:
        if remaining <= 1e-9:
            break
        tv_value = df.at[idx, "TV"]
        if pd.isna(tv_value):
            continue
        available = float(tv_value)
        if available <= 0:
            continue
        take = min(available, remaining)
        if take:
            df.at[idx, "TV"] = available - take
            remaining -= take
            consumed += take
    return consumed


def _consume_ranked_assets(
    df: pd.DataFrame, ordered_indices: list[object], amount: float
) -> float:
    """Consume asset TV following the provided priority order."""

    if amount <= 0 or not ordered_indices:
        return 0.0

    remaining = float(amount)
    consumed = 0.0
    for idx in ordered_indices:
        if remaining <= 1e-9:
            break
        tv_value = df.at[idx, "TV"]
        if pd.isna(tv_value):
            continue
        available = float(tv_value)
        if available <= 0:
            continue
        take = min(available, remaining)
        if take:
            df.at[idx, "TV"] = available - take
            remaining -= take
            consumed += take
    return consumed


def process_pv_after_day_1(
    exposures_next: pd.DataFrame,
    future_classes: Iterable[str] | Tuple[str, ...] = (
        "Bond Future",
        "Equity Index Future",
        "FX Future",
    ),
    option_classes: Iterable[str] | Tuple[str, ...] = (
        "Bond Future Option",
        "FX Option",
    ),
    cash_identifier: str = "CSH_EUR_DB",
    history_day_label: str | None = None,
    config: CollateralConfig | None = None,
    collateral_inputs: pd.DataFrame | None = None,
    history_date: str | pd.Timestamp | None = None,
    liability_shocks: pd.DataFrame
    | Mapping[object, Mapping[object, object]]
    | LiabilityShockLookup
    | None = None,
    liability_day_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate Day‑1 PVs and manage collateral balances.

    The workflow follows five steps, matching the specification provided
    in the request:

    1. Load the TV by counterparty/portfolio, compute ``Balance_J`` and
       read the latest collateral inputs (``Balance_J_1``, ``Seuil de
       déclenchement`` and cash availability) from Excel.
    2. Compute the variation, determine whether a call is triggered and
       identify its direction.
    3. When Groupama must post collateral, sell the designated monetary
       funds before decreasing the available cash and raise an alert in
       case of shortage.
    4. Update the balances whenever a call is executed and store the
       resulting data in the returned dataframe.
    5. Persist the information in the collateral history file so that the
       following day can reuse it.

    Parameters
    ----------
    exposures_next:
        DataFrame produced by the stress engine for the next day.
    future_classes:
        Asset classes for which TV should be aggregated and reset.
    cash_identifier:
        Identifier used to store the cash leg in ``exposures_next``.
    config:
        File paths and column names.  Defaults to :class:`CollateralConfig`.
    history_date:
        Processing date.  When ``None`` the current date (UTC) is used.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Updated exposures, futures TV sums, pre-collateral balances,
        collateral decisions dataframe and a table containing the alerts.
    """
    if config is None:
        config = CollateralConfig()
    config.ensure_directories()

    if liability_shocks is None:
        loaded_liability = _load_liability_shocks(config)
        if loaded_liability is not None:
            liability_shocks = loaded_liability

    liability_lookup: LiabilityShockLookup | None = None
    if liability_shocks is not None:
        liability_lookup = build_liability_shock_lookup(liability_shocks)

    liability_label_to_use = (
        liability_day_label if liability_day_label is not None else history_day_label
    )
    liability_day_key: str | None = None
    if liability_lookup is not None and liability_label_to_use is not None:
        candidate = _normalise_column_name(liability_label_to_use)
        if candidate in liability_lookup.day_columns:
            liability_day_key = candidate
        else:
            for key, original in liability_lookup.day_columns.items():
                if _normalise_column_name(original) == candidate:
                    liability_day_key = key
                    break

    df = exposures_next.copy()
    if "AssetClass" not in df.columns:
        raise KeyError("Column 'AssetClass' missing in exposures_next")

    asset_class_values = df["AssetClass"].astype(str).str.strip()
    asset_class_lower = asset_class_values.str.lower()
    futures_mask = asset_class_values.isin(tuple(future_classes))
    if option_classes is None:
        options_mask = asset_class_lower.str.contains("option", na=False)
    else:
        options_mask = asset_class_values.isin(tuple(option_classes))

    derivative_mask = futures_mask | options_mask

    if "TV" not in df.columns:
        raise KeyError("Column 'TV' missing in exposures_next")
    
    futures_tv = (
        df.loc[futures_mask]
        .groupby("AssetClass", as_index=False)["TV"].sum()
        .rename(columns={"TV": "TV_before_reset"})
        .sort_values("AssetClass")
    )


    original_tv_series = pd.to_numeric(df["TV"], errors="coerce").fillna(0.0)
    portfolio_nav_map: dict[object, float] = {}
    portfolio_nav_normalised: dict[str, float] = {}
    if config.portfolio_col in df.columns:
        grouped_nav = original_tv_series.groupby(df[config.portfolio_col], dropna=False).sum()
        for portfolio_value, nav in grouped_nav.items():
            key = portfolio_value if pd.notna(portfolio_value) else None
            nav_value = float(nav)
            portfolio_nav_map[key] = nav_value
            normalised_key = _normalise_portfolio_value(key)
            if normalised_key is not None:
                portfolio_nav_normalised[normalised_key] = nav_value
            


    if "TV_before_stress" not in df.columns:
        
        raise KeyError("Column 'TV_before_stress' is required to derive futures stress effects.")

    tv_before_stress_series = pd.to_numeric(df["TV_before_stress"], errors="coerce").fillna(0.0)

    normalised_cols = {
        _normalise_column_name(col): col for col in df.columns
    }
    #net_exposure_col = None
    #for candidate in ("NetExposure"):
    #    simplified = _normalise_column_name(candidate)
    #    if simplified in normalised_cols:
    #        net_exposure_col = normalised_cols[simplified]
    #        break
    #if net_exposure_col is None:
    #    raise KeyError(
     #       "Column 'NetExposure' (or equivalent) is required to process futures."
     #   )

    #net_exposure_series = pd.to_numeric(
    #    df[net_exposure_col], errors="coerce"
    #).fillna(0.0)

    derivative_effect_series = original_tv_series - tv_before_stress_series
    futures_effect_series = derivative_effect_series.where(futures_mask, 0.0)
    options_effect_series = derivative_effect_series.where(options_mask, 0.0)


    if futures_mask.any():
        #df.loc[futures_mask, "TV"] = net_exposure_series.loc[futures_mask]
        df.loc[futures_mask, "TV"] = 0

    if options_mask.any():
        df.loc[options_mask, "TV"] = tv_before_stress_series.loc[options_mask]

    #compute futures stress impacts by portfolio
    futures_by_portfolio: dict[object, float] = {}
    if config.portfolio_col in df.columns and futures_mask.any():
        futures_effect = futures_effect_series.where(futures_mask,0.0)
        grouped_futures = (
           futures_effect_series
            .groupby(df[config.portfolio_col], dropna=False).sum()
        )
        for portfolio_key, value in grouped_futures.items():
            key = portfolio_key if pd.notna(portfolio_key) else None
            futures_by_portfolio[key] = float(value)

    #compute options stress impacts by portfolio
    options_by_portfolio: dict[object, float] = {}
    if config.portfolio_col in df.columns and options_mask.any():
        options_effect = options_effect_series.where(options_mask, 0.0)
        grouped_options = options_effect.groupby(
            df[config.portfolio_col], dropna=False
        ).sum()
        for portfolio_key, value in grouped_options.items():
            key = portfolio_key if pd.notna(portfolio_key) else None
            options_by_portfolio[key] = float(value)

    # Recompute masks in case the dataframe structure changed after previous operations.
    asset_class_values = df["AssetClass"].astype(str).str.strip()
    asset_class_lower = asset_class_values.str.lower()
    futures_mask = asset_class_values.isin(tuple(future_classes))
    options_mask = asset_class_values.isin(tuple(option_classes))

    asset_id_col = "AssetID" if "AssetID" in df.columns else None
        
    
    if asset_id_col is None:
        raise KeyError(
            "Column identifying assets (AssetID) is missing "
            "from exposures_next"
        )

    asset_class_series = asset_class_lower
    asset_id_normalised = df[asset_id_col].map(_normalise_asset_identifier)
    monetary_fund_mask = asset_class_series.eq("fund unit") & asset_id_normalised.isin(
        _NORMALISED_MONETARY_FUND_IDS
    )

    monetary_indices_by_portfolio: dict[object, list[object]] = {}
    monetary_remaining_by_portfolio: dict[object, float] = {}
    monetary_index_to_key: dict[object, tuple[object, str]] = {}
    fund_usage_by_key: dict[tuple[object, str], float] = {}
    for idx in df.index[monetary_fund_mask]:
        portfolio_value = df.at[idx, config.portfolio_col]
        key = portfolio_value if pd.notna(portfolio_value) else None
        tv_value = df.at[idx, "TV"]
        if not np.isfinite(tv_value):
            continue
        monetary_indices_by_portfolio.setdefault(key, []).append(idx)
        monetary_remaining_by_portfolio[key] = (
            monetary_remaining_by_portfolio.get(key, 0.0) + float(tv_value)
        )
    
        asset_label = df.at[idx, asset_id_col]
        asset_label_str = "" if pd.isna(asset_label) else str(asset_label)
        monetary_index_to_key[idx] = (key, asset_label_str)

    def _snapshot_fund_tv(indices: list[object]) -> pd.Series:
        if not indices:
            return pd.Series(dtype=float)
        snapshot = pd.to_numeric(df.loc[indices, "TV"], errors="coerce").fillna(0.0)
        snapshot.index = indices
        return snapshot

    def _record_monetary_usage(before: pd.Series, reason: str) -> None:
        if before.empty:
            return
        after = pd.to_numeric(df.loc[before.index, "TV"], errors="coerce").fillna(0.0)
        for idx, before_val in before.items():
            after_val = after.at[idx]
            used = float(before_val) - float(after_val)
            if used <= 1e-9:
                continue
            portfolio_key, asset_label = monetary_index_to_key.get(idx, (None, ""))
            key = (portfolio_key, asset_label)
            entry = fund_usage_by_key.setdefault(
                key,
                {
                    "total": 0.0,
                    "cash_deficit": 0.0,
                    "futures": 0.0,
                    "options": 0.0,
                    "collateral": 0.0,
                     "liability": 0.0,
                },
            )
            entry["total"] += used
            if reason not in entry:
                raise ValueError(f"Unsupported monetary fund usage reason: {reason!r}")
            entry[reason] += used

    bond_mask = asset_class_series.eq("bond")
    bond_liquidity: pd.Series
    if bond_mask.any():
        if "LiquidityScore" not in df.columns:
            raise KeyError(
                "Column 'LiquidityScore' is required to rank bonds for collateral usage."
            )
        bond_liquidity = pd.to_numeric(df["LiquidityScore"], errors="coerce")
    else:
        bond_liquidity = pd.Series(np.nan, index=df.index)

    bond_indices_by_portfolio: dict[object, list[object]] = {}
    bond_remaining_by_portfolio: dict[object, float] = {}
    if bond_mask.any():
        for idx in df.index[bond_mask]:
            liquidity_score = bond_liquidity.at[idx]
            if not np.isfinite(liquidity_score):
                continue
            liquidity_score = float(liquidity_score)
            if (
                abs(liquidity_score - 9.0) > 1e-9
                and abs(liquidity_score - 10.0) > 1e-9
            ):
                continue
            portfolio_value = df.at[idx, config.portfolio_col]
            key = portfolio_value if pd.notna(portfolio_value) else None
            tv_value = df.at[idx, "TV"]
            if not np.isfinite(tv_value) or tv_value <= 0:
                continue
            bond_indices_by_portfolio.setdefault(key, []).append((float(liquidity_score), idx))
            bond_remaining_by_portfolio[key] = (
                bond_remaining_by_portfolio.get(key, 0.0) + float(tv_value)
            )

        for key, entries in list(bond_indices_by_portfolio.items()):
            entries.sort(
                key=lambda item: (-item[0], -float(df.at[item[1], "TV"]))
            )
            bond_indices_by_portfolio[key] = [idx for _, idx in entries]

    group_cols = [config.counterparty_col, config.portfolio_col]
    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in exposures_next: {', '.join(missing_cols)}")

    cp_port_tv = (
        df.loc[~derivative_mask & (df["Identifier"] != cash_identifier)]
        .groupby(group_cols, dropna=False)["TV"]
        .sum()
        .reset_index()
        .rename(columns={"TV": "TV_before_collat"})
        .sort_values(group_cols)
    )
    cp_port_tv = cp_port_tv[cp_port_tv['Counterparty'].notna()]

    cash_port_tv = (df.loc[df["Identifier"] == cash_identifier]
                    .groupby(config.portfolio_col,dropna= True)['TV']
                    .sum()
                    .reset_index(name=config.cash_col))
    

    balances = cp_port_tv.rename(columns={"TV_before_collat": "Balance_J"})
    

    if collateral_inputs is None:
        inputs = _load_collateral_inputs(config)
    else:
        inputs = _normalise_balance_columns(collateral_inputs.copy(), config)
    


    merged = balances.merge(
        inputs[[
            config.portfolio_col,
            config.counterparty_col,
            config.balance_prev_col,
            config.threshold_col,
        ]],
        on=[config.portfolio_col, config.counterparty_col],
        how="outer",
    )

    merged = merged.merge(cash_port_tv, on=config.portfolio_col, how="left")


    merged["Balance_J"] = merged["Balance_J"].fillna(0.0)
    merged[config.balance_prev_col] = merged[config.balance_prev_col].fillna(0.0)
    merged[config.threshold_col] = merged[config.threshold_col].fillna(0.0)
    merged[config.cash_col] = merged[config.cash_col].fillna(0.0)

    merged = merged.sort_values([config.portfolio_col, config.counterparty_col]).reset_index(drop=True)

    merged["Variation"] = merged["Balance_J"] - merged[config.balance_prev_col]
    merged["Seuil_respecte"] = merged["Variation"].abs() <= merged[config.threshold_col]
    merged["Appel_declenche"] = ~merged["Seuil_respecte"]

    merged["Sens_appel"] = np.select(
        [merged["Seuil_respecte"], merged["Variation"] < 0, merged["Variation"] > 0],
        ["Aucun appel", "Groupama poste", "Contrepartie poste"],
        default="Aucun appel",
    )

    merged["Balance_apres_appel"] = np.where(
        merged["Seuil_respecte"],
        merged[config.balance_prev_col],

        merged[config.balance_prev_col] + merged["Variation"],
    )

    merged["Cash_initial"] = merged[config.cash_col]
    merged["Cash_restant"] = merged[config.cash_col]
    merged["Cash_utilise"] = 0.0
    merged["Cash_deficit_couvert_par_fonds"] = 0.0
    merged["Cash_deficit_couvert_par_obligations"] = 0.0
    merged["Futures_augmentent_cash"] = 0.0
    merged["Futures_couverts_par_fonds"] = 0.0
    merged["Futures_couverts_par_cash"] = 0.0
    merged["Futures_couverts_par_obligations"] = 0.0
    merged["Options_augmentent_cash"] = 0.0
    merged["Options_couverts_par_fonds"] = 0.0
    merged["Options_couverts_par_cash"] = 0.0
    merged["Options_couverts_par_obligations"] = 0.0
    merged["Fonds_monetaires_initial"] = 0.0
    merged["Fonds_monetaires_utilises"] = 0.0
    merged["Fonds_monetaires_restant"] = 0.0
    merged["Obligations_liquides_initial"] = 0.0
    merged["Obligations_liquides_utilisees"] = 0.0
    merged["Obligations_liquides_restantes"] = 0.0
    merged["Liability_total"] = 0.0
    merged["Liability_couverts_par_fonds"] = 0.0
    merged["Liability_couverts_par_cash"] = 0.0
    merged["Liability_couverts_par_obligations"] = 0.0
    merged["Alerte"] = pd.Series([np.nan] * len(merged), dtype="object")

    groupama_mask = (~merged["Seuil_respecte"]) & (merged["Variation"] < 0)
    cash_adjustments_by_portfolio: dict[object, float] = {}

    for portfolio, portfolio_df in merged.groupby(config.portfolio_col, dropna=False):
        if portfolio_df.empty:
            continue

        portfolio_cash = portfolio_df[config.cash_col].iloc[0]
        if not np.isfinite(portfolio_cash):
            portfolio_cash = 0.0
        # ``available_cash_pool`` represents the portion of cash that can be mobilised
        # to satisfy collateral calls for the current portfolio.  The raw cash balance
        # may include negative amounts (for instance if positions already created an
        # overdraft).  Since only positive liquidity can be used to post collateral we
        # clamp the value at zero.
        available_cash_pool = float(portfolio_cash)
        key = portfolio if not pd.isna(portfolio) else None
        fund_pool = float(monetary_remaining_by_portfolio.get(key, 0.0))
        fund_indices = monetary_indices_by_portfolio.get(key, [])
        merged.loc[portfolio_df.index, "Fonds_monetaires_initial"] = fund_pool
        bond_pool = float(bond_remaining_by_portfolio.get(key, 0.0))
        bond_indices = bond_indices_by_portfolio.get(key, [])
        merged.loc[portfolio_df.index, "Obligations_liquides_initial"] = bond_pool

        cash_deficit_fund_used = 0.0
        cash_deficit_bond_used = 0.0
        cash_deficit_alert: str | None = None
        if available_cash_pool < -1e-9 and fund_pool > 1e-9 and fund_indices:
            deficit_to_cover = -available_cash_pool
            snapshot_before = _snapshot_fund_tv(fund_indices)
            converted = _consume_monetary_funds(df, fund_indices, deficit_to_cover)
            if converted > 0.0:
                fund_pool = max(fund_pool - converted, 0.0)
                available_cash_pool += converted
                cash_deficit_fund_used = converted
                cash_adjustments_by_portfolio[key] = (
                    cash_adjustments_by_portfolio.get(key, 0.0) + converted
                )
                _record_monetary_usage(snapshot_before, "cash_deficit")

        if available_cash_pool < -1e-9 and bond_pool > 1e-9 and bond_indices:
            deficit_to_cover = -available_cash_pool
            converted = _consume_ranked_assets(df, bond_indices, deficit_to_cover)
            if converted > 0.0:
                bond_pool = max(bond_pool - converted, 0.0)
                available_cash_pool += converted
                cash_deficit_bond_used = converted
                cash_adjustments_by_portfolio[key] = (
                    cash_adjustments_by_portfolio.get(key, 0.0) + converted
                )

        def _apply_liability_shock() -> tuple[float, float, float, float, str | None]:
            nonlocal available_cash_pool, fund_pool, bond_pool, total_used

            if liability_lookup is None or liability_day_key is None:
                return 0.0, 0.0, 0.0, 0.0, None

            pct_value = liability_lookup.get_percentage(key, liability_day_key)
            if pct_value is None:
                return 0.0, 0.0, 0.0, 0.0, None

            pct_value = float(pct_value)
            if abs(pct_value) <= 1e-9:
                return 0.0, 0.0, 0.0, 0.0, None

            if key is None:
                portfolio_mask = df[config.portfolio_col].isna()
            else:
                portfolio_mask = df[config.portfolio_col] == portfolio

            if not portfolio_mask.any():
                return 0.0, 0.0, 0.0, 0.0, None

            nav_series = pd.to_numeric(
                df.loc[portfolio_mask, "TV"], errors="coerce"
            ).fillna(0.0)
            if nav_series.empty:
                return 0.0, 0.0, 0.0, 0.0, None

            cash_mask_portfolio = (df["Identifier"] == cash_identifier) & portfolio_mask
            cash_series = pd.to_numeric(
                df.loc[cash_mask_portfolio, "TV"], errors="coerce"
            ).fillna(0.0)
            nav_without_cash = float(nav_series.sum() - cash_series.sum())
            nav_total = nav_without_cash + available_cash_pool

            if abs(nav_total) <= 1e-9:
                return 0.0, 0.0, 0.0, 0.0, None

            amount = -pct_value * nav_total / 100.0
            if amount <= 1e-9:
                return 0.0, 0.0, 0.0, 0.0, None

            remaining = float(amount)
            liability_fund_used = 0.0
            liability_cash_used = 0.0
            liability_bond_used = 0.0

            if fund_pool > 1e-9 and fund_indices:
                snapshot_before = _snapshot_fund_tv(fund_indices)
                consumed = _consume_monetary_funds(
                    df, fund_indices, min(remaining, fund_pool)
                )
                if consumed > 0.0:
                    fund_pool = max(fund_pool - consumed, 0.0)
                    remaining = max(remaining - consumed, 0.0)
                    liability_fund_used = consumed
                    _record_monetary_usage(snapshot_before, "liability")

            if remaining > 1e-9 and available_cash_pool > 0.0:
                cash_consumed = min(remaining, available_cash_pool)
                if cash_consumed > 0.0:
                    available_cash_pool = max(available_cash_pool - cash_consumed, 0.0)
                    remaining = max(remaining - cash_consumed, 0.0)
                    liability_cash_used = cash_consumed
                    total_used += cash_consumed
                    cash_adjustments_by_portfolio[key] = (
                        cash_adjustments_by_portfolio.get(key, 0.0) - cash_consumed
                    )

            if remaining > 1e-9 and bond_pool > 1e-9 and bond_indices:
                consumed_bonds = _consume_ranked_assets(df, bond_indices, remaining)
                if consumed_bonds > 0.0:
                    bond_pool = max(bond_pool - consumed_bonds, 0.0)
                    remaining = max(remaining - consumed_bonds, 0.0)
                    liability_bond_used = consumed_bonds

            alert = None
            if remaining > 1e-9:
                alert = _format_alert(remaining, reason="liability")

            return amount, liability_fund_used, liability_cash_used, liability_bond_used, alert

        if available_cash_pool < -1e-9:
            cash_deficit_alert = _format_alert(
                -available_cash_pool, reason="cash initial"
            )
            available_cash_pool = 0.0
        elif available_cash_pool < 0:
            available_cash_pool = 0.0

        futures_effect = futures_by_portfolio.get(key, 0.0)
        futures_cash_credit = 0.0
        futures_cash_used = 0.0
        futures_fund_used = 0.0
        futures_bond_used = 0.0
        futures_shortfall_alert: str | None = None

        options_effect = options_by_portfolio.get(key, 0.0)
        options_cash_credit = 0.0
        options_cash_used = 0.0
        options_fund_used = 0.0
        options_bond_used = 0.0
        options_shortfall_alert: str | None = None

        if futures_effect > 1e-9:
            available_cash_pool += futures_effect
            futures_cash_credit = futures_effect
            cash_adjustments_by_portfolio[key] = (
                cash_adjustments_by_portfolio.get(key, 0.0) + futures_effect
            )
        if options_effect > 1e-9:
            available_cash_pool += options_effect
            options_cash_credit = options_effect
            cash_adjustments_by_portfolio[key] = (
                cash_adjustments_by_portfolio.get(key, 0.0) + options_effect
            )

        if futures_effect < -1e-9:
            futures_need = -futures_effect
            if fund_pool > 1e-9 and fund_indices:
                snapshot_before = _snapshot_fund_tv(fund_indices)
                futures_fund_used = _consume_monetary_funds(df, fund_indices, futures_need)
                if futures_fund_used > 0.0:
                    fund_pool = max(fund_pool - futures_fund_used, 0.0)
                    futures_need = max(futures_need - futures_fund_used, 0.0)
                    _record_monetary_usage(snapshot_before, "futures")

            if futures_need > 1e-9 and available_cash_pool > 0.0:
                futures_cash_used = min(futures_need, available_cash_pool)
                available_cash_pool = max(available_cash_pool - futures_cash_used, 0.0)
                futures_need = max(futures_need - futures_cash_used, 0.0)
                cash_adjustments_by_portfolio[key] = (
                    cash_adjustments_by_portfolio.get(key, 0.0) - futures_cash_used
                )
            if futures_need > 1e-9 and bond_pool > 1e-9 and bond_indices:
                futures_bond_used = _consume_ranked_assets(df, bond_indices, futures_need)
                if futures_bond_used > 0.0:
                    bond_pool = max(bond_pool - futures_bond_used, 0.0)
                    futures_need = max(futures_need - futures_bond_used, 0.0)
            if futures_need > 1e-9:
                    futures_shortfall_alert = _format_alert(
                    futures_need, reason="futures"
                )

        if options_effect < -1e-9:
            options_need = -options_effect
            if fund_pool > 1e-9 and fund_indices:
                snapshot_before = _snapshot_fund_tv(fund_indices)
                options_fund_used = _consume_monetary_funds(df, fund_indices, options_need)
                if options_fund_used > 0.0:
                    fund_pool = max(fund_pool - options_fund_used, 0.0)
                    options_need = max(options_need - options_fund_used, 0.0)
                    _record_monetary_usage(snapshot_before, "options")

            if options_need > 1e-9 and available_cash_pool > 0.0:
                options_cash_used = min(options_need, available_cash_pool)
                available_cash_pool = max(available_cash_pool - options_cash_used, 0.0)
                options_need = max(options_need - options_cash_used, 0.0)
                cash_adjustments_by_portfolio[key] = (
                    cash_adjustments_by_portfolio.get(key, 0.0) - options_cash_used
                )
            if options_need > 1e-9 and bond_pool > 1e-9 and bond_indices:
                options_bond_used = _consume_ranked_assets(df, bond_indices, options_need)
                if options_bond_used > 0.0:
                    bond_pool = max(bond_pool - options_bond_used, 0.0)
                    options_need = max(options_need - options_bond_used, 0.0)
            if options_need > 1e-9:
                options_shortfall_alert = _format_alert(
                    options_need, reason="options"
                )

        initial_cash = available_cash_pool + futures_cash_used + options_cash_used
        total_used = futures_cash_used + options_cash_used

        merged.loc[portfolio_df.index, "Cash_initial"] = initial_cash
        merged.loc[portfolio_df.index, config.cash_col] = initial_cash

        needs_cash = groupama_mask.loc[portfolio_df.index].any()
        if not needs_cash:
            (
                liability_amount,
                liability_fund_used,
                liability_cash_used,
                liability_bond_used,
                liability_alert,
            ) = _apply_liability_shock()
            merged.loc[portfolio_df.index, "Liability_total"] = liability_amount
            merged.loc[portfolio_df.index, "Liability_total"] = liability_amount
            merged.loc[portfolio_df.index, "Cash_restant"] = available_cash_pool
            merged.loc[portfolio_df.index, "Fonds_monetaires_restant"] = fund_pool
            merged.loc[portfolio_df.index, "Obligations_liquides_restantes"] = bond_pool
            if portfolio_df.index.size:
                first_idx = portfolio_df.index[0]
                if cash_deficit_fund_used:
                    merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                        merged.at[first_idx, "Fonds_monetaires_utilises"]
                        + cash_deficit_fund_used
                    )
                if cash_deficit_bond_used:
                    merged.at[first_idx,"Obligations_liquides_utilisees"]=(
                        merged.at[first_idx, "Obligations_liquides_utilisees"]
                        + cash_deficit_bond_used
                    )
                if futures_fund_used:
                    merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                        merged.at[first_idx, "Fonds_monetaires_utilises"] + futures_fund_used
                    )
                if futures_cash_used:
                    merged.at[first_idx, "Cash_utilise"] = (
                        merged.at[first_idx, "Cash_utilise"] + futures_cash_used
                    )
                if futures_bond_used:
                    merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                        merged.at[first_idx, "Obligations_liquides_utilisees"] + futures_bond_used
                    )
                if options_fund_used:
                    merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                        merged.at[first_idx, "Fonds_monetaires_utilises"] + options_fund_used
                    )
                if options_cash_used:
                    merged.at[first_idx, "Cash_utilise"] = (
                        merged.at[first_idx, "Cash_utilise"] + options_cash_used
                    )
                if options_bond_used:
                    merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                        merged.at[first_idx, "Obligations_liquides_utilisees"] + options_bond_used
                    )
                if cash_deficit_fund_used:
                    merged.at[first_idx, "Cash_deficit_couvert_par_fonds"] = (
                        merged.at[first_idx, "Cash_deficit_couvert_par_fonds"]
                        + cash_deficit_fund_used
                    )
                if cash_deficit_bond_used:
                    merged.at[first_idx, "Cash_deficit_couvert_par_obligations"] = (
                        merged.at[first_idx, "Cash_deficit_couvert_par_obligations"]
                        + cash_deficit_bond_used
                    )
                if futures_cash_credit:
                    merged.at[first_idx, "Futures_augmentent_cash"] = (
                        merged.at[first_idx, "Futures_augmentent_cash"] + futures_cash_credit
                    )
                if futures_fund_used:
                    merged.at[first_idx, "Futures_couverts_par_fonds"] = (
                        merged.at[first_idx, "Futures_couverts_par_fonds"] + futures_fund_used
                    )
                if futures_cash_used:
                    merged.at[first_idx, "Futures_couverts_par_cash"] = (
                        merged.at[first_idx, "Futures_couverts_par_cash"] + futures_cash_used
                    )
                if futures_bond_used:
                    merged.at[first_idx, "Futures_couverts_par_obligations"] = (
                        merged.at[first_idx, "Futures_couverts_par_obligations"] + futures_bond_used
                    )
                if futures_shortfall_alert:
                    existing_alert = merged.at[first_idx, "Alerte"]
                    if pd.isna(existing_alert):
                        merged.at[first_idx, "Alerte"] = futures_shortfall_alert
                    else:
                        merged.at[first_idx, "Alerte"] = f"{existing_alert} ; {futures_shortfall_alert}"
                if cash_deficit_alert:
                    existing_alert = merged.at[first_idx, "Alerte"]
                    if pd.isna(existing_alert):
                        merged.at[first_idx, "Alerte"] = cash_deficit_alert
                    else:
                        merged.at[first_idx, "Alerte"] = f"{existing_alert} ; {cash_deficit_alert}"
                if options_cash_credit:
                    merged.at[first_idx, "Options_augmentent_cash"] = (
                        merged.at[first_idx, "Options_augmentent_cash"] + options_cash_credit
                    )
                if options_fund_used:
                    merged.at[first_idx, "Options_couverts_par_fonds"] = (
                        merged.at[first_idx, "Options_couverts_par_fonds"] + options_fund_used
                    )
                if options_cash_used:
                    merged.at[first_idx, "Options_couverts_par_cash"] = (
                        merged.at[first_idx, "Options_couverts_par_cash"] + options_cash_used
                    )
                if options_bond_used:
                    merged.at[first_idx, "Options_couverts_par_obligations"] = (
                        merged.at[first_idx, "Options_couverts_par_obligations"]
                        + options_bond_used
                    )
                if options_shortfall_alert:
                    existing_alert = merged.at[first_idx, "Alerte"]
                    if pd.isna(existing_alert):
                        merged.at[first_idx, "Alerte"] = options_shortfall_alert
                    else:
                        merged.at[first_idx, "Alerte"] = (
                            f"{existing_alert} ; {options_shortfall_alert}"
                        )
                if liability_amount:
                    merged.at[first_idx, "Liability_total"] = liability_amount
                if liability_fund_used:
                    merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                        merged.at[first_idx, "Fonds_monetaires_utilises"] + liability_fund_used
                    )
                    merged.at[first_idx, "Liability_couverts_par_fonds"] = (
                        merged.at[first_idx, "Liability_couverts_par_fonds"] + liability_fund_used
                    )
                if liability_cash_used:
                    merged.at[first_idx, "Cash_utilise"] = (
                        merged.at[first_idx, "Cash_utilise"] + liability_cash_used
                    )
                    merged.at[first_idx, "Liability_couverts_par_cash"] = (
                        merged.at[first_idx, "Liability_couverts_par_cash"] + liability_cash_used
                    )
                if liability_bond_used:
                    merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                        merged.at[first_idx, "Obligations_liquides_utilisees"] + liability_bond_used
                    )
                    merged.at[first_idx, "Liability_couverts_par_obligations"] = (
                        merged.at[first_idx, "Liability_couverts_par_obligations"]
                        + liability_bond_used
                    )
                if liability_alert:
                    existing_alert = merged.at[first_idx, "Alerte"]
                    if pd.isna(existing_alert):
                        merged.at[first_idx, "Alerte"] = liability_alert
                    else:
                        merged.at[first_idx, "Alerte"] = (
                            f"{existing_alert} ; {liability_alert}"
                        )
            monetary_remaining_by_portfolio[key] = fund_pool
            bond_remaining_by_portfolio[key]= bond_pool
            continue

        for idx in portfolio_df.index:
            variation = merged.at[idx, "Variation"]
            if pd.isna(variation) or merged.at[idx, "Seuil_respecte"] or variation >= 0:
                merged.at[idx, "Cash_utilise"] = 0.0
                merged.at[idx, "Cash_restant"] = available_cash_pool
                merged.at[idx, "Fonds_monetaires_utilises"] = 0.0
                merged.at[idx, "Fonds_monetaires_restant"] = fund_pool
                merged.at[idx, "Obligations_liquides_utilisees"] = 0.0
                merged.at[idx, "Obligations_liquides_restantes"] = bond_pool
                continue

            required = max(float(-variation), 0.0)
            fund_used = 0.0
            if fund_pool > 1e-9:
                fund_indices = monetary_indices_by_portfolio.get(key, [])
                remaining_need = min(required, fund_pool)
                while remaining_need > 1e-9 and fund_pool > 1e-9:
                    snapshot_before = _snapshot_fund_tv(fund_indices)
                    consumed = _consume_monetary_funds(df, fund_indices, remaining_need)
                    
                    if consumed <= 1e-9:
                        break
                    fund_used += consumed
                    fund_pool = max(fund_pool - consumed, 0.0)
                    required = max(required - consumed, 0.0)
                    remaining_need = min(required, fund_pool)
                    _record_monetary_usage(snapshot_before,"collateral")

                if required > 1e-6 and fund_pool > 1e-6:
                    raise ValueError(
                        "Unable to fully consume monetary funds before using cash for "
                        f"portfolio {portfolio!r}."
                    )

            merged.at[idx, "Fonds_monetaires_utilises"] = fund_used
            merged.at[idx, "Fonds_monetaires_restant"] = fund_pool

            if available_cash_pool <= 0.0 or required <= 1e-9:
                cash_used = 0.0
            else:
                cash_used = min(required, available_cash_pool)

            merged.at[idx, "Cash_utilise"] = cash_used
            available_cash_pool = max(available_cash_pool - cash_used, 0.0)
            merged.at[idx, "Cash_restant"] = available_cash_pool
            total_used += cash_used
            if cash_used:
                cash_adjustments_by_portfolio[key] = (
                    cash_adjustments_by_portfolio.get(key, 0.0) - cash_used
                )
            required = max(required - cash_used, 0.0)

            bond_used = 0.0
            if required > 1e-9 and bond_pool > 1e-9 and bond_indices:
                bond_used = _consume_ranked_assets(df, bond_indices, required)
                if bond_used > 0.0:
                    bond_pool = max(bond_pool - bond_used, 0.0)
                    required = max(required - bond_used, 0.0)

            merged.at[idx, "Obligations_liquides_utilisees"] = bond_used
            merged.at[idx, "Obligations_liquides_restantes"] = bond_pool

            shortfall = required
            if shortfall > 1e-9:
                merged.at[idx, "Alerte"] = _format_alert(
                    shortfall, reason="appel collateral"
                )

            (
            liability_amount,
            liability_fund_used,
            liability_cash_used,
            liability_bond_used,
            liability_alert,
        ) = _apply_liability_shock()
            
        merged.loc[portfolio_df.index, "Liability_total"] = liability_amount

        if portfolio_df.index.size:
            first_idx = portfolio_df.index[0]        
            if cash_deficit_fund_used:
                merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                    merged.at[first_idx, "Fonds_monetaires_utilises"] + cash_deficit_fund_used
                )
                merged.at[first_idx, "Cash_deficit_couvert_par_fonds"] = (
                    merged.at[first_idx, "Cash_deficit_couvert_par_fonds"] + cash_deficit_fund_used
                )
            if cash_deficit_bond_used: 
                merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                    merged.at[first_idx, "Obligations_liquides_utilisees"] + cash_deficit_bond_used
                )
                merged.at[first_idx, "Cash_deficit_couvert_par_obligations"] = (
                    merged.at[first_idx, "Cash_deficit_couvert_par_obligations"]
                    + cash_deficit_bond_used
                )
            if futures_fund_used:
                merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                    merged.at[first_idx, "Fonds_monetaires_utilises"] + futures_fund_used
                )   
                merged.at[first_idx, "Futures_couverts_par_fonds"] = (
                    merged.at[first_idx, "Futures_couverts_par_fonds"] + futures_fund_used
                )            
            if futures_cash_used:
                merged.at[first_idx, "Cash_utilise"] = (
                    merged.at[first_idx, "Cash_utilise"] + futures_cash_used
                )
                merged.at[first_idx, "Futures_couverts_par_cash"] = (
                    merged.at[first_idx, "Futures_couverts_par_cash"] + futures_cash_used
                )
            if futures_bond_used:
                merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                    merged.at[first_idx, "Obligations_liquides_utilisees"] + futures_bond_used
                )
                merged.at[first_idx, "Futures_couverts_par_obligations"] = (
                    merged.at[first_idx, "Futures_couverts_par_obligations"] + futures_bond_used
                )
            if futures_cash_credit:
                merged.at[first_idx, "Futures_augmentent_cash"] = (
                    merged.at[first_idx, "Futures_augmentent_cash"] + futures_cash_credit
                )
            if futures_shortfall_alert:
                existing_alert = merged.at[first_idx, "Alerte"]
                if pd.isna(existing_alert):
                    merged.at[first_idx, "Alerte"] = futures_shortfall_alert
                else:
                    merged.at[first_idx, "Alerte"] = f"{existing_alert} ; {futures_shortfall_alert}"


            if cash_deficit_alert:
                existing_alert = merged.at[first_idx, "Alerte"]
                if pd.isna(existing_alert):
                    merged.at[first_idx, "Alerte"] = cash_deficit_alert
                else:
                    merged.at[first_idx, "Alerte"] = f"{existing_alert} ; {cash_deficit_alert}"
            if options_fund_used:
                merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                    merged.at[first_idx, "Fonds_monetaires_utilises"] + options_fund_used
                )
                merged.at[first_idx, "Options_couverts_par_fonds"] = (
                    merged.at[first_idx, "Options_couverts_par_fonds"] + options_fund_used
                )
            if options_cash_used:
                merged.at[first_idx, "Cash_utilise"] = (
                    merged.at[first_idx, "Cash_utilise"] + options_cash_used
                )
                merged.at[first_idx, "Options_couverts_par_cash"] = (
                    merged.at[first_idx, "Options_couverts_par_cash"] + options_cash_used
                )
            if options_bond_used:
                merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                    merged.at[first_idx, "Obligations_liquides_utilisees"] + options_bond_used
                )
                merged.at[first_idx, "Options_couverts_par_obligations"] = (
                    merged.at[first_idx, "Options_couverts_par_obligations"] + options_bond_used
                )
            if options_cash_credit:
                merged.at[first_idx, "Options_augmentent_cash"] = (
                    merged.at[first_idx, "Options_augmentent_cash"] + options_cash_credit
                )
            if options_shortfall_alert:
                existing_alert = merged.at[first_idx, "Alerte"]
                if pd.isna(existing_alert):
                    merged.at[first_idx, "Alerte"] = options_shortfall_alert
                else:
                    merged.at[first_idx, "Alerte"] = (
                        f"{existing_alert} ; {options_shortfall_alert}"
          
                  )
                    
            if liability_amount:
                merged.at[first_idx, "Liability_total"] = liability_amount
            if liability_fund_used:
                merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                    merged.at[first_idx, "Fonds_monetaires_utilises"] + liability_fund_used
                )
                merged.at[first_idx, "Liability_couverts_par_fonds"] = (
                    merged.at[first_idx, "Liability_couverts_par_fonds"] + liability_fund_used
                )
            if liability_cash_used:
                merged.at[first_idx, "Cash_utilise"] = (
                    merged.at[first_idx, "Cash_utilise"] + liability_cash_used
                )
                merged.at[first_idx, "Liability_couverts_par_cash"] = (
                    merged.at[first_idx, "Liability_couverts_par_cash"] + liability_cash_used
                )
            if liability_bond_used:
                merged.at[first_idx, "Obligations_liquides_utilisees"] = (
                    merged.at[first_idx, "Obligations_liquides_utilisees"] + liability_bond_used
                )
                merged.at[first_idx, "Liability_couverts_par_obligations"] = (
                    merged.at[first_idx, "Liability_couverts_par_obligations"]
                    + liability_bond_used
                )
            if liability_alert:
                existing_alert = merged.at[first_idx, "Alerte"]
                if pd.isna(existing_alert):
                    merged.at[first_idx, "Alerte"] = liability_alert
                else:
                    merged.at[first_idx, "Alerte"] = (
                        f"{existing_alert} ; {liability_alert}"
                    )
                       

        if total_used > initial_cash + 1e-6:
            raise ValueError(
                "Cash utilisation exceeded the available pool for portfolio "
                f"{portfolio!r}."
            )

        if available_cash_pool < -1e-6:
            raise ValueError(
                "Cash remaining for portfolio "
                f"{portfolio!r} became negative despite the safeguard."
            )
        if available_cash_pool < 0:
            available_cash_pool = 0.0

        merged.loc[portfolio_df.index, "Cash_restant"] = available_cash_pool
        merged.loc[portfolio_df.index, "Fonds_monetaires_restant"] = fund_pool
        merged.loc[portfolio_df.index, "Obligations_liquides_restantes"] = bond_pool
        monetary_remaining_by_portfolio[key] = fund_pool
        bond_remaining_by_portfolio[key] = bond_pool

    if cash_adjustments_by_portfolio:
        cash_mask = df["Identifier"] == cash_identifier
        for portfolio_key, cash_delta in cash_adjustments_by_portfolio.items():
            if abs(cash_delta) <= 1e-9:
                continue

            if cash_mask.any():
                if portfolio_key is None:
                    portfolio_mask = df[config.portfolio_col].isna()
                else:
                    portfolio_mask = df[config.portfolio_col] == portfolio_key
                mask = cash_mask & portfolio_mask
                if not mask.any():
                    mask = cash_mask & df[config.portfolio_col].isna()
                if not mask.any():
                    mask = cash_mask
            else:
                mask = pd.Series(False, index=df.index)

            if not mask.any():
                new_row = {col: np.nan for col in df.columns}
                new_row.update({"Identifier": cash_identifier, "AssetClass": "Cash", "TV": 0.0})
                if portfolio_key is None:
                    new_row[config.portfolio_col] = np.nan
                else:
                    new_row[config.portfolio_col] = portfolio_key
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                cash_mask = df["Identifier"] == cash_identifier
                if portfolio_key is None:
                    portfolio_mask = df[config.portfolio_col].isna()
                else:
                    portfolio_mask = df[config.portfolio_col] == portfolio_key
                mask = cash_mask & portfolio_mask

            if not mask.any():
                continue

            first_idx = df.index[mask][0]
            current_tv = df.at[first_idx, "TV"]
            current_tv = 0.0 if pd.isna(current_tv) else float(current_tv)
            df.at[first_idx, "TV"] = current_tv + cash_delta

    history_dt = (
        pd.Timestamp.today().normalize()
        if history_date is None
        else pd.Timestamp(history_date).normalize()
    )

    history_df = merged[[
        config.portfolio_col,
        config.counterparty_col,
        "Balance_J",
        config.balance_prev_col,
        "Fonds_monetaires_restant",
        "Cash_restant",
        "Alerte",
    ]].copy()
    history_df.insert(0, "Date", history_dt)
    history_df = history_df.rename(
        columns={
            config.portfolio_col: "Portefeuille",
            config.counterparty_col: "Contrepartie",
            config.balance_prev_col: "Balance_J_1",
            "Fonds_monetaires_restant": "Fonds_monetaires_restants",
        }
    )
    history_df = history_df[[
        "Date",
        "Portefeuille",
        "Contrepartie",
        "Balance_J",
        "Balance_J_1",
        "Fonds_monetaires_restants",
        "Cash_restant",
        "Alerte",
    ]]

    def _lookup_portfolio_nav(key: object) -> float | None:
        normalised = _normalise_portfolio_value(key)
        if normalised is not None:
            nav_value = portfolio_nav_normalised.get(normalised)
            if nav_value is not None:
                return nav_value
        return portfolio_nav_map.get(key)

    usage_records: list[dict[str, object]] = []
    day_label_value = history_day_label if history_day_label is not None else np.nan

    for (portfolio_key, asset_label), usage in fund_usage_by_key.items():
        total_used = float(usage.get("total", 0.0))
        cash_deficit_used = float(usage.get("cash_deficit", 0.0))
        futures_used = float(usage.get("futures", 0.0))
        options_used = float(usage.get("options", 0.0))
        collateral_used = float(usage.get("collateral", 0.0))
        liability_used = float(usage.get("liability", 0.0))

        amount_signed = -total_used
        cash_deficit_signed = -cash_deficit_used
        futures_signed = -futures_used
        options_signed = -options_used
        collateral_signed = -collateral_used
        liability_signed = -liability_used
        nav_value = None
        normalised_asset = _normalise_asset_identifier(asset_label)
        if normalised_asset is not None:
            mapped_portfolio = _MONETARY_FUND_PORTFOLIO_CODES.get(normalised_asset)
            if mapped_portfolio is not None:
                nav_value = _lookup_portfolio_nav(mapped_portfolio)
        if nav_value is None:
            nav_value = _lookup_portfolio_nav(portfolio_key)

        nav_valid = nav_value is not None and abs(nav_value) > 1e-9
        nav_output = float(nav_value) if nav_valid else np.nan
        pct_total = amount_signed / nav_output if nav_valid else np.nan
        pct_cash_deficit = (
            cash_deficit_signed / nav_output if nav_valid else np.nan
        )
        pct_futures = futures_signed / nav_output if nav_valid else np.nan
        pct_options = options_signed / nav_output if nav_valid else np.nan
        pct_collateral = (
            collateral_signed / nav_output if nav_valid else np.nan
        )
        pct_liability = liability_signed / nav_output if nav_valid else np.nan

        usage_records.append(
            {
                "Date": history_dt,
                "Jour": day_label_value,
                "Portefeuille": portfolio_key if portfolio_key is not None else np.nan,
                "Fonds_monetaires": asset_label,
                "Montant_vendu": amount_signed,
                "Montant_vendu_deficit_cash": cash_deficit_signed,
                "Montant_vendu_futures": futures_signed,
                "Montant_vendu_options": options_signed,
                "Montant_vendu_appel_collat": collateral_signed,
                "Montant_vendu_passif": liability_signed,

                "Actif_net_portefeuille": nav_output,
                "Vente_pct_actif_net": pct_total,
                "Vente_pct_deficit_cash": pct_cash_deficit,
                "Vente_pct_futures": pct_futures,
                "Vente_pct_options": pct_options,
                "Vente_pct_appel_collat": pct_collateral,
                "Vente_pct_passif": pct_liability,
            }
        )

    usage_path = config.monetary_fund_usage_history_path
    usage_columns = [
        "Date",
        "Jour",
        "Portefeuille",
        "Fonds_monetaires",
        "Montant_vendu",
        "Montant_vendu_deficit_cash",
        "Montant_vendu_futures",
        "Montant_vendu_options",
        "Montant_vendu_appel_collat",
        "Montant_vendu_passif",
        "Actif_net_portefeuille",
        "Vente_pct_actif_net",
        "Vente_pct_deficit_cash",
        "Vente_pct_futures",
        "Vente_pct_options",
        "Vente_pct_appel_collat",
        "Vente_pct_passif",
        "Montant_vendu_cumule",
        "Montant_vendu_deficit_cash_cumule",
        "Montant_vendu_futures_cumule",
        "Montant_vendu_options_cumule",
        "Montant_vendu_appel_collat_cumule",
        "Vente_pct_passif_cumule",
        "Vente_pct_cumule_actif_net",
        "Vente_pct_cumule_deficit_cash",
        "Vente_pct_cumule_futures",
        "Vente_pct_cumule_options",
        "Vente_pct_cumule_appel_collat",
        "Vente_pct_cumule_passif",
    ]

    if usage_path.exists():
        existing_usage = pd.read_excel(usage_path)
    else:
        existing_usage = pd.DataFrame(columns=usage_columns)

    usage_df = pd.DataFrame(usage_records, columns=usage_columns)
    if not existing_usage.empty:
        missing_cols = [col for col in usage_columns if col not in existing_usage.columns]
        if missing_cols:
            for col in missing_cols:
                existing_usage[col] = np.nan
        existing_usage = existing_usage.reindex(columns=usage_columns)
        usage_df = pd.concat([existing_usage, usage_df], ignore_index=True, sort=False)

    if not usage_df.empty:
        usage_df = usage_df.sort_values(
            ["Portefeuille", "Fonds_monetaires", "Date", "Jour"], ignore_index=True
        )

        numeric_cols = [
            "Montant_vendu",
            "Montant_vendu_deficit_cash",
            "Montant_vendu_futures",
            "Montant_vendu_options",
            "Montant_vendu_appel_collat",
            "Actif_net_portefeuille",
            "Vente_pct_actif_net",
            "Vente_pct_deficit_cash",
            "Vente_pct_futures",
            "Vente_pct_options",
            "Vente_pct_appel_collat",
        ]
        for col in numeric_cols:
            usage_df[col] = pd.to_numeric(usage_df.get(col), errors="coerce")

        group_keys = ["Portefeuille", "Fonds_monetaires"]
        grouped = usage_df.groupby(group_keys, dropna=False)
        usage_df["Montant_vendu_cumule"] = grouped["Montant_vendu"].cumsum()
        usage_df["Montant_vendu_deficit_cash_cumule"] = grouped[
            "Montant_vendu_deficit_cash"
        ].cumsum()
        usage_df["Montant_vendu_futures_cumule"] = grouped[
            "Montant_vendu_futures"
        ].cumsum()
        usage_df["Montant_vendu_options_cumule"] = grouped[
            "Montant_vendu_options"
        ].cumsum()
        usage_df["Montant_vendu_appel_collat_cumule"] = grouped[
            "Montant_vendu_appel_collat"
        ].cumsum()
        usage_df["Montant_vendu_passif_cumule"] = grouped[
            "Montant_vendu_passif"
        ].cumsum()

        nav_abs = usage_df["Actif_net_portefeuille"].abs()
        safe_nav = nav_abs > 1e-9
        usage_df["Vente_pct_actif_net"] = np.where(
            safe_nav,
            usage_df["Montant_vendu"] / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_deficit_cash"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_deficit_cash"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_futures"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_futures"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_options"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_options"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_appel_collat"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_appel_collat"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_passif"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_passif"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_cumule_actif_net"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_cumule"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_cumule_deficit_cash"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_deficit_cash_cumule"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_cumule_futures"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_futures_cumule"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_cumule_options"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_options_cumule"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_cumule_appel_collat"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_appel_collat_cumule"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )
        usage_df["Vente_pct_cumule_passif"] = np.where(
            safe_nav,
            usage_df["Montant_vendu_passif_cumule"]
            / usage_df["Actif_net_portefeuille"],
            np.nan,
        )




    usage_df = usage_df[usage_columns]

    usage_df.to_excel(usage_path, index=False)

    normalised_cols = {_normalise_column_name(col): col for col in df.columns}

    tv_series = pd.to_numeric(df.get("TV"), errors="coerce").fillna(0.0)
    tv_prev_current = pd.to_numeric(df.get("TV_prev"), errors="coerce").fillna(0.0)
    ratio = np.divide(
        tv_series,
        tv_prev_current,
        out=np.zeros(len(df), dtype=float),
        where=tv_prev_current.abs() > 1e-9,
    )
    ratio_series = pd.Series(ratio, index=df.index)

    if "derivative_mask" in locals():
        non_derivative_mask = ~derivative_mask
    elif "futures_mask" in locals():
        non_derivative_mask = ~futures_mask
    else:
        non_derivative_mask = pd.Series(True, index=df.index)

 

    def _find_column(candidates: Iterable[str]) -> str | None:
        for candidate in candidates:
            simplified = _normalise_column_name(candidate)
            if simplified in normalised_cols:
                return normalised_cols[simplified]
        return None

    def _store_delta(values: pd.Series, target: str, *aliases: str) -> None:
        df[target] = values
        for alias in aliases:
            if alias and alias in df.columns:
                df[alias] = values

    fx_col = _find_column(["FX_delta", "FXDelta", "FX Delta"])
    if fx_col is not None:
        fx_base = pd.to_numeric(df[fx_col], errors="coerce").fillna(0.0)
        fx_updated = fx_base.copy()
        fx_updated.loc[non_derivative_mask] = (
            fx_base.loc[non_derivative_mask] * ratio_series.loc[non_derivative_mask]
        )
        _store_delta(fx_updated, "FXDelta", fx_col)

    equity_col = _find_column(["Equity_delta", "EquityDelta", "Equity Delta"])
    if equity_col is not None:
        equity_base = pd.to_numeric(df[equity_col], errors="coerce").fillna(0.0)
        equity_updated = equity_base.copy()
        equity_updated.loc[non_derivative_mask] = (
            equity_base.loc[non_derivative_mask]
            * ratio_series.loc[non_derivative_mask]
        )
        _store_delta(equity_updated, "EquityDelta", equity_col)

    duration_col = _find_column([
        "Duration",
    ])
    if duration_col is not None:
        duration_series = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0)
        rate_alias = _find_column(["Rate_delta", "RateDelta", "RateDelta1bp"])
        rate_source = rate_alias if rate_alias is not None else "Rate_delta"
        existing_rate = pd.to_numeric(
            df.get(rate_source, pd.Series(0.0, index=df.index)), errors="coerce"
        )
        if not isinstance(existing_rate, pd.Series):
            existing_rate = pd.Series(existing_rate, index=df.index)
        existing_rate = existing_rate.reindex(df.index).fillna(0.0)
        rate_delta = existing_rate.copy()
        computed_rate = duration_series * tv_series / 10000 * -1
    
        rate_delta.loc[non_derivative_mask] = computed_rate.loc[non_derivative_mask]
        _store_delta(rate_delta, "RateDelta1bp", rate_alias)

    spread_duration_col = _find_column([
        "SpreadDuration",
        "DurationSpread",
        "Spread_Duration",
    ])
    if spread_duration_col is not None:
        spread_duration_series = pd.to_numeric(
            df[spread_duration_col], errors="coerce"
        ).fillna(0.0)
        credit_alias = _find_column(["Credit_delta", "CreditDelta", "SpreadDelta1bp"])
        credit_source = credit_alias if credit_alias is not None else "Credit_delta"
        existing_credit = pd.to_numeric(
            df.get(credit_source, pd.Series(0.0, index=df.index)), errors="coerce"
        )
        if not isinstance(existing_credit, pd.Series):
            existing_credit = pd.Series(existing_credit, index=df.index)
        existing_credit = existing_credit.reindex(df.index).fillna(0.0)
        credit_delta = existing_credit.copy()
        computed_credit = spread_duration_series * tv_series / 10000 * -1
        
        credit_delta.loc[non_derivative_mask] = computed_credit.loc[non_derivative_mask]
        _store_delta(credit_delta, "SpreadDelta1bp", credit_alias)

    history_path = config.collateral_history_path
    if history_path.exists():
        existing_history = pd.read_excel(history_path)
        history_df = pd.concat([existing_history, history_df], ignore_index=True)
    history_df.to_excel(history_path, index=False)

    alerts_df = merged.loc[merged["Alerte"].notna(), [
        config.portfolio_col,
        config.counterparty_col,
        "Alerte",
    ]].reset_index(drop=True)

    merged = merged.rename(columns={config.cash_col: "Cash_disponible"})
    ordered_cols = [
        config.portfolio_col,
        config.counterparty_col,
        "Balance_J",
        config.balance_prev_col,
        "Variation",
        config.threshold_col,
        "Seuil_respecte",
        "Appel_declenche",
        "Sens_appel",
        "Balance_apres_appel",
        "Fonds_monetaires_initial",
        "Fonds_monetaires_utilises",
        "Fonds_monetaires_restant",
        "Obligations_liquides_initial",
        "Obligations_liquides_utilisees",
        "Obligations_liquides_restantes",
        "Cash_deficit_couvert_par_fonds",
        "Cash_deficit_couvert_par_obligations",
        "Futures_augmentent_cash",
        "Futures_couverts_par_fonds",
        "Futures_couverts_par_cash",
        "Futures_couverts_par_obligations",
        "Options_augmentent_cash",
        "Options_couverts_par_fonds",
        "Options_couverts_par_cash",
        "Options_couverts_par_obligations",
        "Liability_total",
        "Liability_couverts_par_fonds",
        "Liability_couverts_par_cash",
        "Liability_couverts_par_obligations",
        "Cash_initial",
        "Cash_disponible",
        "Cash_utilise",
        "Cash_restant",
        "Alerte",
    ]
    existing_cols = [col for col in ordered_cols if col in merged.columns]
    extra_cols = [col for col in merged.columns if col not in existing_cols]
    merged = merged[existing_cols + extra_cols]


    return df, futures_tv, cp_port_tv, merged, alerts_df


def roll_balance_for_next_day(
    processed_collateral: pd.DataFrame,
    config: CollateralConfig | None = None,
    current_inputs: pd.DataFrame | None = None,
    snapshot_label: str | None = None,
    snapshot_directory: Path | None = None,

) -> pd.DataFrame:
    """Update the collateral input so that ``Balance_J_1`` = ``Balance_J``.

    Parameters
    ----------
    processed_collateral:
        DataFrame returned by :func:`process_pv_after_day_1`.  It must
        contain the current balance column ``Balance_J`` as well as the
        counterparty and portfolio identifiers.
    config:
        Collateral configuration.  When omitted the default paths are
        used.

    Returns
    -------
    pd.DataFrame
        The refreshed collateral input dataframe.
    """

    if config is None:
        config = CollateralConfig()
    config.ensure_directories()

    required_cols = {
        config.portfolio_col,
        config.counterparty_col,
        "Balance_J",
    }
    missing = required_cols - set(processed_collateral.columns)
    if missing:
        raise KeyError(
            "Processed collateral is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if current_inputs is None:
        inputs = _load_collateral_inputs(config)
    else:
        inputs = _normalise_balance_columns(current_inputs.copy(), config)

    inputs = _normalise_balance_columns(inputs, config)


    update_cols = [config.portfolio_col, config.counterparty_col, "Balance_J"]

    updates = processed_collateral[update_cols].copy()

    refreshed = inputs.merge(
        updates,
        on=[config.portfolio_col, config.counterparty_col],
        how="outer",
        suffixes=("", "_new"),
    )
    if "Balance_J_new" in refreshed.columns:
        new_balance = refreshed["Balance_J_new"]
    else:
        new_balance = refreshed["Balance_J"]

    refreshed[config.balance_prev_col] = new_balance.combine_first(

        refreshed.get(config.balance_prev_col, pd.Series(dtype=float))
    )

    refreshed = refreshed.drop(columns=[col for col in refreshed.columns if col.endswith("_new")])
    refreshed = _normalise_balance_columns(refreshed, config)


    target_path = config.collateral_input_path
    #_write_collateral_frame(refreshed, target_path)

    snapshot_path: Path | None = None
    snapshot_slug = _sanitise_snapshot_label(snapshot_label)
    if snapshot_slug:
        snapshot_dir = snapshot_directory or target_path.parent
        if snapshot_dir:
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(snapshot_dir) / f"{target_path.stem}_{snapshot_slug}{target_path.suffix}"
        _write_collateral_frame(refreshed, snapshot_path)

    refreshed.attrs["input_path"] = target_path
    refreshed.attrs["snapshot_path"] = snapshot_path
    return refreshed

def run_stress_sequence(
    exposures: pd.DataFrame,
    merged_mapping: pd.DataFrame,
    day_columns: Sequence[str],
    *,
    day_step_apply_func: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] | None = None,
    day_step_kwargs: Mapping[str, Any] | None = None,
    process_kwargs: Mapping[str, Any] | None = None,
    history_dates: Sequence[str | pd.Timestamp | None] | None = None,
    history_day_labels: Sequence[str | None] | None = None,
    update_collateral_inputs: bool = True,
    snapshot_collateral_inputs: bool = True,
    collateral_snapshot_directory: Path | None = None,
) -> list[dict[str, Any]]:
    """Apply stress scenarios sequentially while managing collateral.

    Parameters
    ----------
    exposures:
        DataFrame containing the starting exposures (typically Day‑1
        results).
    merged_mapping:
        Output of :func:`merge_pos_scen` linking the position mapping to
        the scenario table.
    day_columns:
        Ordered list of scenario columns (``"Day 1"``, ``"Day 2"`` …)
        that will be fed into ``day_step_apply``.
    day_step_apply_func:
        Callable compatible with :func:`day_step_apply`.  When omitted the
        function attempts to reuse the helper defined in the calling
        environment (for example a notebook where ``day_step_apply`` has
        already been declared).
    day_step_kwargs:
        Additional keyword arguments forwarded to ``day_step_apply`` at
        each iteration.
    process_kwargs:
        Keyword arguments forwarded to :func:`process_pv_after_day_1`
        (``config``, ``future_classes`` …).
    history_dates:
        Optional sequence of dates associated with each day.  Missing
        entries fallback to ``None`` which triggers ``process_pv_after_day_1``
        default behaviour.
    history_day_labels:
        Optional textual labels (``"Day 1"`` …) mirrored into the
        monetary fund usage history.
    update_collateral_inputs:
        When ``True`` the helper calls :func:`roll_balance_for_next_day`
        to refresh the collateral input file after each day.

    Returns
    -------
    list[dict[str, Any]]
        A list containing the detailed artefacts for each processed day:
        deterministic results, per-identifier pivots, exposures before and
        after collateral processing, collateral decisions, alerts, the
        refreshed collateral input and, when available, the filesystem paths
        of both the overwritten base collateral file and the day-labelled
        snapshot.
    """
    

    if day_step_apply_func is None:
        potential = globals().get("day_step_apply")
        if potential is None:
            main_module = sys.modules.get("__main__")
            if main_module is not None:
                potential = getattr(main_module, "day_step_apply", None)
        if potential is None:
            stack = inspect.stack()
            try:
                for frame_info in stack:
                    candidate = frame_info.frame.f_locals.get("day_step_apply")
                    if callable(candidate):
                        potential = candidate
                        break
                # Explicitly drop references held by FrameInfo objects.
                for frame_info in stack:
                    del frame_info
                frame_info = None
            finally:
                del stack
        if potential is None:
            raise ValueError(
                "day_step_apply_func must be provided when the helper is not defined "
                "in the current namespace."
            )
        day_step_apply_func = potential

    def _normalise_liquidity_column(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        """Ensure the LiquidityScore information lives in a single column."""

        if frame is None:
            return None

        if "LiquidityScore" in frame.columns and not any(
            col.startswith("LiquidityScore") and col.endswith(("_x", "_y"))
            for col in frame.columns
        ):
            return frame

        frame = frame.copy()

        candidates = []
        for name in ("LiquidityScore", "LiquidityScore_x", "LiquidityScore_y"):
            if name in frame.columns:
                candidates.append(pd.to_numeric(frame[name], errors="coerce"))

        if candidates:
            consolidated = candidates[0]
            for extra in candidates[1:]:
                consolidated = consolidated.combine_first(extra)
            frame["LiquidityScore"] = consolidated

        for redundant in ("LiquidityScore_x", "LiquidityScore_y"):
            if redundant in frame.columns:
                frame = frame.drop(columns=redundant)

        return frame
    results: list[dict[str, Any]] = []
    exposures_current = _normalise_liquidity_column(exposures.copy())
    base_day_kwargs = dict(day_step_kwargs or {})
    base_process_kwargs = dict(process_kwargs or {})
    liability_obj = base_process_kwargs.get("liability_shocks")
    if liability_obj is not None and not isinstance(liability_obj, LiabilityShockLookup):
        base_process_kwargs["liability_shocks"] = build_liability_shock_lookup(liability_obj)

    collateral_inputs_current = base_process_kwargs.pop("collateral_inputs", None)

    liquidity_col = "LiquidityScore"
    key_col = base_day_kwargs.get("key_col") if base_day_kwargs else None
    if key_col is None:
        for candidate in ("Identifier", "identifier", "ID", "Id"):
            if candidate in exposures_current.columns:
                key_col = candidate
                break

    liquidity_lookup: pd.Series | None = None
    if (
        key_col is not None
        and key_col in exposures_current.columns
        and liquidity_col in exposures_current.columns
    ):
        base_lookup = (
            exposures_current[[key_col, liquidity_col]]
            .dropna(subset=[key_col])
            .drop_duplicates(subset=[key_col], keep="last")
            .set_index(key_col)[liquidity_col]
        )
        liquidity_lookup = base_lookup

    for idx, day_col in enumerate(day_columns, start=1):
        day_kwargs = dict(base_day_kwargs)
        day_kwargs["exposures"] = _normalise_liquidity_column(exposures_current)
        day_kwargs["merged_mapping"] = merged_mapping
        day_kwargs["day_col"] = day_col
        day_kwargs.setdefault("return_pivot", True)

        deterministic_df, per_id_df, exposures_next = day_step_apply_func(**day_kwargs)
        if (
            liquidity_lookup is not None
            and key_col is not None
            and key_col in exposures_next.columns
        ):
            exposures_next = _normalise_liquidity_column(exposures_next)
            if liquidity_col in exposures_next.columns:
                missing_liquidity = exposures_next[liquidity_col].isna()
                if missing_liquidity.any():
                    exposures_next.loc[missing_liquidity, liquidity_col] = (
                        exposures_next.loc[missing_liquidity, key_col]
                        .map(liquidity_lookup)
                        .values
                    )
            else:
                exposures_next[liquidity_col] = (
                    exposures_next[key_col].map(liquidity_lookup).values
                )

        history_date = None
        if history_dates is not None and idx - 1 < len(history_dates):
            history_date = history_dates[idx - 1]

        if history_day_labels is not None and idx - 1 < len(history_day_labels):
            day_label = history_day_labels[idx - 1]
        else:
            day_label = f"Day {idx}"

        process_args = dict(base_process_kwargs)
        process_args.setdefault("history_date", history_date)
        process_args["history_day_label"] = day_label
        process_args["liability_day_label"] = day_label
        process_args["collateral_inputs"] = collateral_inputs_current

        
        updated_exp, futures_tv, balances, decisions, alerts = process_pv_after_day_1(
            exposures_next, **process_args
        )

        refreshed_inputs = None
        snapshot_path = None
        input_path = None

        if update_collateral_inputs:
            config_obj = process_args.get("config")
            refreshed_inputs = roll_balance_for_next_day(
                decisions,
                config=config_obj,
                current_inputs=collateral_inputs_current,
                snapshot_label=day_label if snapshot_collateral_inputs else None,
                snapshot_directory=collateral_snapshot_directory,
            )
            collateral_inputs_current = refreshed_inputs
            if hasattr(refreshed_inputs, "attrs"):
                snapshot_path = refreshed_inputs.attrs.get("snapshot_path")
                input_path = refreshed_inputs.attrs.get("input_path")
        else:
            collateral_inputs_current = process_args.get("collateral_inputs")
            if hasattr(collateral_inputs_current, "attrs"):
                snapshot_path = collateral_inputs_current.attrs.get("snapshot_path")
                input_path = collateral_inputs_current.attrs.get("input_path")
            

        results.append(
            {
                "day_index": idx,
                "day_column": day_col,
                "day_label": day_label,
                "deterministic": deterministic_df,
                "per_identifier": per_id_df,
                "pre_collateral_exposures": exposures_next,
                "exposures": updated_exp,
                "futures_tv": futures_tv,
                "balances": balances,
                "decisions": decisions,
                "alerts": alerts,
                "collateral_input": refreshed_inputs,
                "collateral_input_path": input_path,
                "collateral_snapshot_path": snapshot_path,
            }
        )

        exposures_current = _normalise_liquidity_column(updated_exp.copy())
        if (
            liquidity_lookup is not None
            and key_col is not None
            and key_col in exposures_current.columns
            and liquidity_col in exposures_current.columns
        ):
            fresh_lookup = (
                exposures_current[[key_col, liquidity_col]]
                .dropna(subset=[key_col])
                .drop_duplicates(subset=[key_col], keep="last")
                .set_index(key_col)[liquidity_col]
            )
            liquidity_lookup = liquidity_lookup.combine_first(fresh_lookup)
        elif (
            key_col is not None
            and key_col in exposures_current.columns
            and liquidity_col in exposures_current.columns
        ):
            liquidity_lookup = (
                exposures_current[[key_col, liquidity_col]]
                .dropna(subset=[key_col])
                .drop_duplicates(subset=[key_col], keep="last")
                .set_index(key_col)[liquidity_col]
            )
    return results


# Backwards compatibility with the previous naming convention used in the
# notebook.
process_pv_after_day1 = process_pv_after_day_1

__all__ = [
    "CollateralConfig",
    "process_pv_after_day_1",
    "process_pv_after_day1",
    "roll_balance_for_next_day",
    "LiabilityShockLookup",
    "build_liability_shock_lookup",
    "run_stress_sequence", 
     "summarise_portfolio_sequence"]