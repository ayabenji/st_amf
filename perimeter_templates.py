from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Normalisation helpers shared by the perimeter utilities
# ---------------------------------------------------------------------------


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


def _match_column(df: pd.DataFrame, *candidates: str) -> str:
    """Return the column name matching one of ``candidates`` after normalisation."""

    normalised_map = {
        _normalise_column_name(col): col for col in df.columns if isinstance(col, str)
    }

    for candidate in candidates:
        key = _normalise_column_name(candidate)
        if key in normalised_map:
            return normalised_map[key]

    raise KeyError(
        "None of the expected columns were found. Tried: "
        + ", ".join(repr(candidate) for candidate in candidates)
    )


# ---------------------------------------------------------------------------
# Perimeter oriented structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerimeterLookup:
    """Mapping helper translating portfolios into perimeter labels."""

    by_portfolio: dict[object, str]
    by_normalised: dict[str, str]

    def resolve(self, portfolio: object) -> str | None:
        """Return the perimeter associated with ``portfolio`` if known."""

        if portfolio in self.by_portfolio:
            return self.by_portfolio[portfolio]

        normalised = _normalise_portfolio_value(portfolio)
        if normalised is not None:
            return self.by_normalised.get(normalised)

        return None

    def unique_perimeters(self) -> list[str]:
        """Return the sorted list of known perimeter labels."""

        values = set(self.by_portfolio.values()) | set(self.by_normalised.values())
        return sorted({value for value in values if value is not None})


@dataclass(frozen=True)
class NumericRange:
    """Descriptor used to express numeric selection bounds in filters."""

    minimum: float | None = None
    maximum: float | None = None
    inclusive_minimum: bool = True
    inclusive_maximum: bool = True

    def matches(self, series: pd.Series) -> pd.Series:
        """Return a boolean mask matching ``series`` against the range."""

        numeric = pd.to_numeric(series, errors="coerce")
        mask = pd.Series(True, index=series.index, dtype=bool)

        if self.minimum is not None:
            if self.inclusive_minimum:
                mask &= numeric >= self.minimum
            else:
                mask &= numeric > self.minimum

        if self.maximum is not None:
            if self.inclusive_maximum:
                mask &= numeric <= self.maximum
            else:
                mask &= numeric < self.maximum

        return mask & numeric.notna()


@dataclass(frozen=True)
class TemplateCategory:
    """Describe how to aggregate net exposures for a template row."""

    name: str
    criteria: Mapping[str, Any] | None = None
    filter_func: Callable[[pd.DataFrame], pd.Series] | None = None
    target_cell: str | None = None
    description: str | None = None

    def build_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return the boolean mask matching rows belonging to the category."""

        mask = pd.Series(True, index=df.index, dtype=bool)

        if self.criteria:
            for column, condition in self.criteria.items():
                if column not in df.columns:
                    raise KeyError(
                        f"Column {column!r} required for category {self.name!r}"
                    )

                series = df[column]

                if isinstance(condition, NumericRange):
                    mask &= condition.matches(series)
                elif callable(condition):
                    result = condition(series)
                    if isinstance(result, pd.Series):
                        candidate = result.reindex(df.index)
                    else:
                        candidate = pd.Series(result, index=df.index)
                    mask &= candidate.fillna(False).astype(bool)
                elif isinstance(condition, slice):
                    numeric = pd.to_numeric(series, errors="coerce")
                    candidate = pd.Series(True, index=df.index, dtype=bool)
                    if condition.start is not None:
                        candidate &= numeric >= condition.start
                    if condition.stop is not None:
                        candidate &= numeric < condition.stop
                    mask &= candidate.fillna(False)
                elif isinstance(condition, Iterable) and not isinstance(
                    condition, (str, bytes)
                ):
                    mask &= series.isin(list(condition)).fillna(False)
                elif condition is None:
                    mask &= series.isna()
                else:
                    mask &= (series == condition).fillna(False)

        if self.filter_func is not None:
            extra_mask = self.filter_func(df)
            if isinstance(extra_mask, pd.Series):
                candidate = extra_mask.reindex(df.index)
            else:
                candidate = pd.Series(extra_mask, index=df.index)
            mask &= candidate.fillna(False).astype(bool)

        return mask.fillna(False)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_perimeter_lookup(
    perimeter_table: pd.DataFrame | Mapping[object, object] | str | Path,
    *,
    sheet_name: str = "Perim",
    portfolio_column: str = "Portefeuille",
    perimeter_column: str = "Perimetre",
) -> PerimeterLookup:
    """Build a :class:`PerimeterLookup` from a dataframe or an Excel file."""

    if isinstance(perimeter_table, PerimeterLookup):
        return perimeter_table

    if isinstance(perimeter_table, (str, Path)):
        path = Path(perimeter_table)
        if not path.exists():
            raise FileNotFoundError(f"Perimeter file not found: {path}")
        perimeter_df = pd.read_excel(path, sheet_name=sheet_name)
    elif isinstance(perimeter_table, pd.DataFrame):
        perimeter_df = perimeter_table.copy()
    elif isinstance(perimeter_table, Mapping):
        by_portfolio = dict(perimeter_table)
        by_normalised = {}
        for key, value in by_portfolio.items():
            normalised = _normalise_portfolio_value(key)
            if normalised is not None:
                by_normalised[normalised] = value
        return PerimeterLookup(by_portfolio, by_normalised)
    else:
        raise TypeError(
            "perimeter_table must be a DataFrame, mapping or path to an Excel file"
        )

    if perimeter_df.empty:
        return PerimeterLookup({}, {})

    try:
        portfolio_col = _match_column(
            perimeter_df,
            portfolio_column,
            "Portfolio",
            "Code portefeuille",
            "Code_portefeuille",
        )
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError("Portfolio column not found in perimeter table") from exc

    try:
        perimeter_col = _match_column(
            perimeter_df,
            perimeter_column,
            "Perimeter",
            "Perimetre",
        )
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError("Perimeter column not found in perimeter table") from exc

    perimeter_df = perimeter_df[[portfolio_col, perimeter_col]].dropna(
        subset=[perimeter_col]
    )

    by_portfolio: dict[object, str] = {}
    by_normalised: dict[str, str] = {}

    for _, row in perimeter_df.iterrows():
        portfolio_value = row[portfolio_col]
        perimeter_value = row[perimeter_col]

        if perimeter_value is None:
            continue

        if isinstance(perimeter_value, float) and not np.isfinite(perimeter_value):
            continue

        by_portfolio[portfolio_value] = perimeter_value

        normalised = _normalise_portfolio_value(portfolio_value)
        if normalised is not None:
            by_normalised[normalised] = perimeter_value

    return PerimeterLookup(by_portfolio, by_normalised)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def prepare_perimeter_template_data(
    sequence_results: Sequence[Mapping[str, Any]],
    perimeter_lookup: PerimeterLookup,
    categories: Sequence[TemplateCategory],
    *,
    portfolio_col: str = "Portfolio",
    value_col: str = "NetExposure",
    fallback_value_cols: Sequence[str] = ("TV", "TV_day"),
    exposures_key: str = "exposures",
    include_all_perimeters: bool = True,
    include_zero_rows: bool = True,
    include_unmapped: bool = False,
    unmapped_label: str = "Unmapped",
) -> pd.DataFrame:
    """Aggregate sequence results to feed perimeter templates.

    Parameters
    ----------
    sequence_results:
        Iterable of dictionaries produced by :func:`run_stress_sequence`.
    perimeter_lookup:
        Mapping between portfolios and perimeter labels.
    categories:
        List of :class:`TemplateCategory` entries describing the
        aggregations to compute for each perimeter.
    portfolio_col:
        Column name identifying the portfolio in the exposure dataframes.
    value_col:
        Primary column containing the net exposures to aggregate.
    fallback_value_cols:
        Additional column names to try if ``value_col`` is absent from a
        given dataframe.
    exposures_key:
        Key used to retrieve the dataframe holding the exposures for each
        day from ``sequence_results``.
    include_all_perimeters:
        When ``True`` (default), the function emits zero-valued rows for
        perimeters defined in the lookup but absent from a given day.
    include_zero_rows:
        Emit rows where the aggregated value is zero.  Disabling this keeps
        the output compact by dropping zero rows.
    include_unmapped:
        When set, portfolios missing from the lookup are grouped under
        ``unmapped_label`` instead of being discarded.
    unmapped_label:
        Label used when ``include_unmapped`` is ``True``.

    Returns
    -------
    DataFrame
        Table with one row per (day, perimeter, category) containing the
        aggregated net exposure and metadata helpful to fill the official
        templates.
    """

    if not sequence_results:
        return pd.DataFrame(
            columns=[
                "day_index",
                "day_label",
                "Perimeter",
                "Category",
                "Value",
                "TargetCell",
                "Description",
                "ValueColumn",
            ]
        )

    if not isinstance(perimeter_lookup, PerimeterLookup):
        raise TypeError("perimeter_lookup must be a PerimeterLookup instance")

    normalised_categories: list[TemplateCategory] = []
    for category in categories:
        if isinstance(category, TemplateCategory):
            normalised_categories.append(category)
        else:
            raise TypeError("categories must be a sequence of TemplateCategory objects")

    if not normalised_categories:
        raise ValueError("At least one template category must be provided")

    records: list[dict[str, Any]] = []
    all_perimeters = perimeter_lookup.unique_perimeters()

    for day_index, result in enumerate(sequence_results):
        exposures_df = result.get(exposures_key)
        day_label = result.get("day_label")

        if exposures_df is None:
            continue
        if not isinstance(exposures_df, pd.DataFrame):
            raise TypeError("Each exposure entry must be a pandas DataFrame")

        working = exposures_df.copy()

        if portfolio_col not in working.columns:
            raise KeyError(
                f"Column {portfolio_col!r} missing from exposures dataframe for day {day_label!r}"
            )

        value_series = None
        value_source = None
        fallback_source = None
        fallback_series = None

        candidate_columns = [value_col, *fallback_value_cols]
        for candidate in candidate_columns:
            if candidate in working.columns:
                candidate_series = pd.to_numeric(working[candidate], errors="coerce")
                if candidate_series.notna().any():
                    value_source = candidate
                    value_series = candidate_series.fillna(0.0)
                    break
                if fallback_source is None:
                    fallback_source = candidate
                    fallback_series = candidate_series.fillna(0.0)

        if value_source is None:
            if fallback_source is not None:
                value_source = fallback_source
                value_series = fallback_series
            else:
                raise KeyError(
                    "None of the value columns were found in the exposures dataframe"
                )

        if value_series is None:
            raise KeyError(
                "Unable to determine a numeric series for the perimeter aggregation"
            )

        working["__value__"] = value_series.reindex(working.index).fillna(0.0)

        perimeter_values = working[portfolio_col].map(perimeter_lookup.resolve)

        if include_unmapped:
            perimeter_values = perimeter_values.fillna(unmapped_label)
        else:
            working = working.loc[perimeter_values.notna()].copy()
            perimeter_values = perimeter_values.loc[working.index]

        working["__perimeter__"] = perimeter_values

        seen_perimeters: set[str] = set()

        for perimeter_value, group in working.groupby("__perimeter__"):
            seen_perimeters.add(perimeter_value)
            group_values = group["__value__"]

            for category in normalised_categories:
                mask = category.build_mask(group)
                if mask is None or mask.empty:
                    aggregated = 0.0
                else:
                    aggregated = float(group_values.loc[mask].sum())

                if not include_zero_rows and abs(aggregated) < 1e-12:
                    continue

                records.append(
                    {
                        "day_index": day_index,
                        "day_label": day_label,
                        "Perimeter": perimeter_value,
                        "Category": category.name,
                        "Value": aggregated,
                        "TargetCell": category.target_cell,
                        "Description": category.description,
                        "ValueColumn": value_source,
                    }
                )

        if include_all_perimeters:
            for perimeter_value in all_perimeters:
                if perimeter_value in seen_perimeters:
                    continue

                for category in normalised_categories:
                    if not include_zero_rows:
                        continue

                    records.append(
                        {
                            "day_index": day_index,
                            "day_label": day_label,
                            "Perimeter": perimeter_value,
                            "Category": category.name,
                            "Value": 0.0,
                            "TargetCell": category.target_cell,
                            "Description": category.description,
                            "ValueColumn": value_source,
                        }
                    )

    result = pd.DataFrame.from_records(records)

    if result.empty:
        return result

    sort_columns = [
        col
        for col in ("day_index", "day_label", "Perimeter", "Category")
        if col in result.columns
    ]
    result = result.sort_values(sort_columns).reset_index(drop=True)

    return result
def prepare_perimeter_holdings_bridge(
    sequence_results: Sequence[Mapping[str, Any]],
    perimeter_lookup: PerimeterLookup,
    categories: Sequence[TemplateCategory],
    *,
    portfolio_col: str = "Portfolio",
    value_col: str = "NetExposure",
    fallback_value_cols: Sequence[str] = ("TV", "TV_day"),
    exposures_key: str = "exposures",
    pre_exposures_key: str = "pre_collateral_exposures",
    include_all_perimeters: bool = True,
    include_zero_rows: bool = True,
    include_unmapped: bool = False,
    unmapped_label: str = "Unmapped",
    opening_column: str = "Holdings Opening",
    purchases_column: str = "Purchases (+)",
    sales_column: str = "Sales (-)",
    closing_column: str = "Holdings end-of day",
) -> pd.DataFrame:
    """Return the daily holdings bridge for each perimeter/category pair.

    The helper aggregates the pre- and post-collateral exposures for the
    provided categories and computes a simple bridge showing how the
    holdings evolved throughout the collateral process.

    The ``opening`` value corresponds to the exposures immediately after the
    stress shocks (``pre_exposures_key`` in ``sequence_results``) while the
    ``closing`` value uses the post-collateral exposures (``exposures_key``).
    ``Purchases`` and ``Sales`` are inferred from the net change between the
    two snapshots and expressed as positive amounts.
    """

    pre_summary = prepare_perimeter_template_data(
        sequence_results,
        perimeter_lookup,
        categories,
        portfolio_col=portfolio_col,
        value_col=value_col,
        fallback_value_cols=fallback_value_cols,
        exposures_key=pre_exposures_key,
        include_all_perimeters=include_all_perimeters,
        include_zero_rows=include_zero_rows,
        include_unmapped=include_unmapped,
        unmapped_label=unmapped_label,
    )

    post_summary = prepare_perimeter_template_data(
        sequence_results,
        perimeter_lookup,
        categories,
        portfolio_col=portfolio_col,
        value_col=value_col,
        fallback_value_cols=fallback_value_cols,
        exposures_key=exposures_key,
        include_all_perimeters=include_all_perimeters,
        include_zero_rows=include_zero_rows,
        include_unmapped=include_unmapped,
        unmapped_label=unmapped_label,
    )

    join_cols = ["day_index", "day_label", "Perimeter", "Category"]

    if pre_summary.empty and post_summary.empty:
        columns = join_cols + [
            opening_column,
            purchases_column,
            sales_column,
            closing_column,
            "TargetCell",
            "Description",
            "ValueColumn_opening",
            "ValueColumn_closing",
        ]
        return pd.DataFrame(columns=columns)

    rename_pre = {
        "Value": opening_column,
        "ValueColumn": "ValueColumn_opening",
        "TargetCell": "TargetCell_opening",
        "Description": "Description_opening",
    }
    rename_post = {
        "Value": closing_column,
        "ValueColumn": "ValueColumn_closing",
        "TargetCell": "TargetCell_closing",
        "Description": "Description_closing",
    }

    pre_named = pre_summary.rename(columns=rename_pre)
    post_named = post_summary.rename(columns=rename_post)

    merged = pre_named.merge(post_named, on=join_cols, how="outer")

    for column in join_cols:
        if column in merged.columns:
            continue
        merged[column] = pd.NA

    merged[opening_column] = pd.to_numeric(
        merged.get(opening_column, 0.0), errors="coerce"
    ).fillna(0.0)
    merged[closing_column] = pd.to_numeric(
        merged.get(closing_column, 0.0), errors="coerce"
    ).fillna(0.0)

    delta = merged[closing_column] - merged[opening_column]
    delta = delta.where(np.abs(delta) >= 1e-12, 0.0)

    merged[purchases_column] = np.where(delta > 0, delta, 0.0)
    merged[sales_column] = np.where(delta < 0, -delta, 0.0)

    target_closing = merged.get("TargetCell_closing")
    target_opening = merged.get("TargetCell_opening")
    if not isinstance(target_closing, pd.Series):
        target_closing = pd.Series(pd.NA, index=merged.index)
    if not isinstance(target_opening, pd.Series):
        target_opening = pd.Series(pd.NA, index=merged.index)
    merged["TargetCell"] = target_closing.combine_first(target_opening)

    desc_closing = merged.get("Description_closing")
    desc_opening = merged.get("Description_opening")
    if not isinstance(desc_closing, pd.Series):
        desc_closing = pd.Series(pd.NA, index=merged.index)
    if not isinstance(desc_opening, pd.Series):
        desc_opening = pd.Series(pd.NA, index=merged.index)
    merged["Description"] = desc_closing.combine_first(desc_opening)

    val_closing = merged.get("ValueColumn_closing")
    if not isinstance(val_closing, pd.Series):
        val_closing = pd.Series(pd.NA, index=merged.index)
    merged["ValueColumn_closing"] = val_closing

    val_opening = merged.get("ValueColumn_opening")
    if not isinstance(val_opening, pd.Series):
        val_opening = pd.Series(pd.NA, index=merged.index)
    merged["ValueColumn_opening"] = val_opening

    merged = merged[join_cols + [
        opening_column,
        purchases_column,
        sales_column,
        closing_column,
        "TargetCell",
        "Description",
        "ValueColumn_opening",
        "ValueColumn_closing",
    ]]

    merged = merged.sort_values(join_cols).reset_index(drop=True)

    return merged

__all__ = [
    "PerimeterLookup",
    "NumericRange",
    "TemplateCategory",
    "build_perimeter_lookup",
    "prepare_perimeter_template_data",
    "prepare_perimeter_holdings_bridge",
    "_normalise_column_name",
    "_normalise_portfolio_value",
    "_match_column",
]