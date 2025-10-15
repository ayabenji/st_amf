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
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import re



@dataclass
class CollateralConfig:
    """Configuration of collateral related resources and column names."""

    collateral_input_path: Path = Path(r"C:\Users\abenjelloun\OneDrive - Cooperactions\GAM-E-Risk Perf - RMP\1.PROD\1.REGLEMENTAIRE\14.Stress Test AMF (JB)\Production\Périmètre et positions\Collat_Cash_MTM_20250401.csv")

    collateral_history_path: Path = Path("collateral_history.xlsx")
    counterparty_col: str = "Counterparty"
    portfolio_col: str = "Portfolio"
    balance_prev_col: str = "Balance_J_1"
    threshold_col: str = "Seuil declenchement"
    cash_col: str = "Cash_disponible"

    def ensure_directories(self) -> None:
        """Create parent folders for the configured files if necessary."""

        for path in (self.collateral_input_path, self.collateral_history_path):
            if path.parent and path.parent != Path(""):
                path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_asset_identifier(value: object) -> str | None:
    """Return a comparable representation of an asset identifier."""

    if not isinstance(value, str):
        return None
    simplified = re.sub(r"[\s\-]+", " ", value).strip().lower()
    return simplified or None


MONETARY_FUND_ASSET_IDS = {
    "GROUPAMA MONETAIRE- IC",
    "GROUPAMA TRESORERIE I",
    "GROUPAMA ENTREPRISES I"
}

_NORMALISED_MONETARY_FUND_IDS = {
    _normalise_asset_identifier(name) for name in MONETARY_FUND_ASSET_IDS
}


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


def _load_collateral_inputs(config: CollateralConfig) -> pd.DataFrame:
    """Load the collateral instructions for each counterparty/portfolio."""

    path = config.collateral_input_path
    if path.exists():
        inputs = pd.read_csv(path, sep=';',decimal=',',encoding='latin1')
        inputs=inputs.rename(columns={
        "Code portefeuille": "Portfolio",
        "Contrepartie": "Counterparty"
   
    })
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


def _format_alert(amount: float) -> str:
    """Human readable representation of the missing cash amount."""

    return f"cash insuffisant ({amount:,.2f})".replace(",", " ")


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
    cash_identifier: str = "CSH_EUR_DB",
    config: CollateralConfig | None = None,
    history_date: str | pd.Timestamp | None = None,
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

    df = exposures_next.copy()
    if "AssetClass" not in df.columns:
        raise KeyError("Column 'AssetClass' missing in exposures_next")

    futures_mask = df["AssetClass"].isin(tuple(future_classes))
    futures_tv = (
        df.loc[futures_mask]
        .groupby("AssetClass", as_index=False)["TV"].sum()
        .rename(columns={"TV": "TV_before_reset"})
        .sort_values("AssetClass")
    )

    futures_by_portfolio: dict[object, float] = {}
    if config.portfolio_col in df.columns and futures_mask.any():
        grouped_futures = (
            df.loc[futures_mask]
            .groupby(config.portfolio_col, dropna=False)["TV"].sum()
        )
        for portfolio_key, value in grouped_futures.items():
            key = portfolio_key if pd.notna(portfolio_key) else None
            futures_by_portfolio[key] = float(value)


    futures_mask = df["AssetClass"].isin(tuple(future_classes))
    




    asset_id_col = next(
        (col for col in ("AssetID", "Asset Id", "Asset ID") if col in df.columns),
        None,
    )
    if asset_id_col is None:
        raise KeyError(
            "Column identifying assets (AssetID / Asset Id / Asset ID) is missing "
            "from exposures_next"
        )

    asset_class_series = df["AssetClass"].astype(str).str.strip().str.lower()
    asset_id_normalised = df[asset_id_col].map(_normalise_asset_identifier)
    monetary_fund_mask = asset_class_series.eq("fund unit") & asset_id_normalised.isin(
        _NORMALISED_MONETARY_FUND_IDS
    )

    monetary_indices_by_portfolio: dict[object, list[object]] = {}
    monetary_remaining_by_portfolio: dict[object, float] = {}
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
        df.loc[~futures_mask & (df["Identifier"] != cash_identifier)]
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
    
    #print(cash_port_tv)

    balances = cp_port_tv.rename(columns={"TV_before_collat": "Balance_J"})
    

    inputs = _load_collateral_inputs(config)

    mask_perimeter = inputs['Portfolio'].isin(balances['Portfolio'].unique())
    inputs = inputs.loc[mask_perimeter]


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
    merged["Fonds_monetaires_initial"] = 0.0
    merged["Fonds_monetaires_utilises"] = 0.0
    merged["Fonds_monetaires_restant"] = 0.0
    merged["Obligations_liquides_initial"] = 0.0
    merged["Obligations_liquides_utilisees"] = 0.0
    merged["Obligations_liquides_restantes"] = 0.0
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
        if available_cash_pool < -1e-9 and fund_pool > 1e-9 and fund_indices:
            deficit_to_cover = -available_cash_pool
            converted = _consume_monetary_funds(df, fund_indices, deficit_to_cover)
            if converted > 0.0:
                fund_pool = max(fund_pool - converted, 0.0)
                available_cash_pool += converted
                cash_deficit_fund_used = converted
                cash_adjustments_by_portfolio[key] = (
                    cash_adjustments_by_portfolio.get(key, 0.0) + converted
                )


        if available_cash_pool < 0:
            available_cash_pool = 0.0

        futures_effect = futures_by_portfolio.get(key, 0.0)
        futures_cash_used = 0.0
        futures_fund_used = 0.0
        futures_bond_used = 0.0
        futures_shortfall_alert: str | None = None

        if futures_effect > 1e-9:
            available_cash_pool += futures_effect
            cash_adjustments_by_portfolio[key] = (
                cash_adjustments_by_portfolio.get(key, 0.0) + futures_effect
            )
        elif futures_effect < -1e-9:
            futures_need = -futures_effect
            if fund_pool > 1e-9 and fund_indices:
                futures_fund_used = _consume_monetary_funds(df, fund_indices, futures_need)
                if futures_fund_used > 0.0:
                    fund_pool = max(fund_pool - futures_fund_used, 0.0)
                    futures_need = max(futures_need - futures_fund_used, 0.0)
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
                futures_shortfall_alert = _format_alert(futures_need)

        initial_cash = available_cash_pool + futures_cash_used
        total_used = futures_cash_used
        merged.loc[portfolio_df.index, "Cash_initial"] = initial_cash

        needs_cash = groupama_mask.loc[portfolio_df.index].any()
        if not needs_cash:
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
                if futures_shortfall_alert:
                    existing_alert = merged.at[first_idx, "Alerte"]
                    if pd.isna(existing_alert):
                        merged.at[first_idx, "Alerte"] = futures_shortfall_alert
                    else:
                        merged.at[first_idx, "Alerte"] = f"{existing_alert} ; {futures_shortfall_alert}"
            monetary_remaining_by_portfolio[key] = fund_pool
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
                    consumed = _consume_monetary_funds(df, fund_indices, remaining_need)
                    if consumed <= 1e-9:
                        break
                    fund_used += consumed
                    fund_pool = max(fund_pool - consumed, 0.0)
                    required = max(required - consumed, 0.0)
                    remaining_need = min(required, fund_pool)

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
                merged.at[idx, "Alerte"] = _format_alert(shortfall)

        if portfolio_df.index.size:
            first_idx = portfolio_df.index[0]        
            if cash_deficit_fund_used:
                merged.at[first_idx, "Fonds_monetaires_utilises"] = (
                    merged.at[first_idx, "Fonds_monetaires_utilises"] + cash_deficit_fund_used
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
            if futures_shortfall_alert:
                existing_alert = merged.at[first_idx, "Alerte"]
                if pd.isna(existing_alert):
                    merged.at[first_idx, "Alerte"] = futures_shortfall_alert
                else:
                    merged.at[first_idx, "Alerte"] = f"{existing_alert} ; {futures_shortfall_alert}"

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

    inputs = _load_collateral_inputs(config)

    update_cols = [config.portfolio_col, config.counterparty_col, "Balance_J"]

    updates = processed_collateral[update_cols].copy()

    refreshed = inputs.merge(
        updates,
        on=[config.portfolio_col, config.counterparty_col],
        how="outer",
        suffixes=("", "_new"),
    )
    refreshed[config.balance_prev_col] = refreshed["Balance_J_new"].combine_first(
        refreshed.get(config.balance_prev_col, pd.Series(dtype=float))
    )

    refreshed = refreshed.drop(columns=[col for col in refreshed.columns if col.endswith("_new")])

    refreshed.to_excel(config.collateral_input_path, index=False)
    return refreshed


# Backwards compatibility with the previous naming convention used in the
# notebook.
process_pv_after_day1 = process_pv_after_day_1

__all__ = [
    "CollateralConfig",
    "process_pv_after_day_1",
    "process_pv_after_day1",
    "roll_balance_for_next_day"]