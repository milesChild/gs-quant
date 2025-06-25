"""
Copyright 2021 Goldman Sachs.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""
# Portions copyright Alta Fox Capital-Miles Child. Licensed under Apache 2.0 license
import datetime as dt
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from gs_quant.api.gs.portfolios import GsPortfolioApi
from gs_quant.errors import MqError, MqValueError
from gs_quant.markets.portfolio_manager import PortfolioManager
from gs_quant.markets.position_set import Position, PositionSet
from gs_quant.models.risk_model import ReturnFormat


def _validate_portfolio_creation_inputs(actual_portfolio_id: str,
                                        start: dt.date,
                                        end: dt.date,
                                        position_sets: bool,
                                        return_format: Union[ReturnFormat, None]) -> PortfolioManager:
    """Common argument checks; returns an initialised PortfolioManager if inputs are valid."""
    if start > end:
        raise MqValueError(f'start_date {start} must be on/ before end_date {end}')

    if position_sets and return_format is not None:
        raise MqValueError('When position_sets is True, return_format must be None')

    if not position_sets and return_format is None:
        raise MqValueError('One of position_sets or return_format must be specified')

    try:
        _ = GsPortfolioApi.get_portfolio(actual_portfolio_id)
    except Exception as err:
        raise MqError(f'Cannot retrieve portfolio {actual_portfolio_id}: {err}') from err

    pm = PortfolioManager(actual_portfolio_id)

    cal = pm.get_position_dates()
    print(cal)
    if start not in cal or end not in cal:
        raise MqError('start_date and/or end_date have no positions in the source portfolio')

    return pm


def _pull_constituents_for_date_range(pm: PortfolioManager,
                                      start: dt.date,
                                      end: dt.date) -> pd.DataFrame:
    """Download the full panel of daily constituents once."""
    rpt = pm.get_performance_report()
    raw = rpt.get_portfolio_constituents(
        start_date=start,
        end_date=end,
        fields=['netWeight', 'grossWeight', 'quantity', 'assetId'],
        return_format=ReturnFormat.JSON
    )

    if not raw:
        raise MqError('No constituent data returned – check permissions / date range')

    df = pd.DataFrame(raw)
    df['date'] = pd.to_datetime(df['date']).dt.date  # keep plain dates

    # choose the signed weight field
    if 'netWeight' in df and not df['netWeight'].isna().all():
        df['w_live'] = df['netWeight']
    else:
        df['w_live'] = df['grossWeight'] * np.sign(df['quantity'].fillna(0))

    df = df[['date', 'assetId', 'quantity', 'w_live']].dropna(subset=['w_live'])
    return df


def _add_equal_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add equal-weight column per calendar day.
    Longs get +1/N, shorts get -1/M where N=#longs, M=#shorts per day.
    Also handles duplicate rows by aggregating quantities and weights.
    """
    # drop duplicate rows first, keeping the *total* quantity
    df_clean = (df.groupby(['date', 'assetId'], as_index=False)
                  .agg({'quantity': 'sum', 'w_live': 'sum'}))

    # Calculate equal weights for each date
    result_list = []
    for _, day_df in df_clean.groupby('date'):
        # copy df to prevent SettingWithCopyWarning
        day_df = day_df.copy()
        longs = day_df.loc[day_df['w_live'] > 0]
        shorts = day_df.loc[day_df['w_live'] < 0]
        n_long, n_short = len(longs), len(shorts)

        day_df['w_xse'] = np.where(
            day_df['w_live'] > 0,
            1.0 / n_long if n_long else 0.0,
            -1.0 / n_short if n_short else 0.0,
        )
        result_list.append(day_df)

    return pd.concat(result_list, ignore_index=True)


# _generate_xse_quantities has been replaced by the more general _generate_equalised_quantities function


def _add_ts_equal_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every trading day, give all the non-zero positions (long / short)
    the same absolute weight  k_t = 1 / N_t  (N_t := # active positions that day).
    The original sign is preserved and dupes are aggregated first.
    """
    df_clean = (
        df.groupby(['date', 'assetId'], as_index=False)
          .agg({'quantity': 'sum', 'w_live': 'sum'})
    )

    out_rows: list[pd.DataFrame] = []
    for _, day_df in df_clean.groupby('date'):
        day_df = day_df.copy()
        n_active = len(day_df)
        if n_active == 0:
            continue

        k = 1.0 / n_active
        day_df['w_xstse'] = np.sign(day_df['w_live']) * k
        out_rows.append(day_df)

    return pd.concat(out_rows, ignore_index=True)


def _generate_equalised_quantities(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """
    `weight_col` tells us which equalised-weight column to use
    (either 'w_xse' or 'w_xstse').
    """
    if weight_col not in df.columns:
        raise MqError(f'Column {weight_col} not found in DataFrame')

    df = df.copy()
    df['spw'] = df['quantity'] / df['w_live'].replace(0, np.nan)

    # 1) shares‑per‑weight matrix (forward filled ⇒ no look‑ahead)
    spw_wide = (
        df.pivot(index='assetId', columns='date', values='spw')
          .sort_index(axis=1)
          .ffill(axis=1)
    )

    # 2) target weights matrix
    w_wide = (
        df.pivot(index='assetId', columns='date', values=weight_col)
          .sort_index(axis=1)
          .fillna(0.0)
    )

    # 3) target shares  =  weight × (shares / weight)_live
    qty_wide = (w_wide * spw_wide).where(w_wide != 0, 0)
    qty_wide = qty_wide.round().astype(int).fillna(0)

    # back to long form
    qty_long = qty_wide.stack().rename('quantity_eq').reset_index()

    tidy = (
        df[['date', 'assetId', weight_col]]
        .drop_duplicates(subset=['date', 'assetId'])
        .merge(qty_long, on=['assetId', 'date'], how='left')
        .rename(columns={weight_col: 'weight', 'quantity_eq': 'quantity'})
    )
    return tidy


def _build_position_sets(tidy: pd.DataFrame) -> List[PositionSet]:
    """Convert the tidy equal-weighted frame into PositionSet objects.
    Also resolves the position sets. If any single position set is unresolved, an error is raised.
    """
    ps_out: List[PositionSet] = []
    for day, day_df in tidy.groupby('date'):
        positions = [
            Position(asset_id=row.assetId,
                     identifier=row.assetId,
                     quantity=int(row.quantity),
                     weight=float(row.weight))
            for row in day_df.itertuples(index=False)
        ]
        ps_out.append(PositionSet(positions, date=day))
    # Resolve position sets
    for ps in ps_out:
        ps.resolve()
        if ps.unresolved_positions:
            raise MqError(f'Unresolved positions on {ps.date}: {ps.unresolved_positions}')
    return ps_out


def get_xse_portfolio(
        actual_portfolio_id: str,
        start_date: dt.date,
        end_date: dt.date,
        position_sets: bool = False,
        return_format: Union[ReturnFormat, None] = ReturnFormat.DATA_FRAME
) -> Union[Dict, pd.DataFrame, List[PositionSet]]:
    """
    Construct a *Cross-Sectionally Equalized* (“XSE”) clone of an existing Marquee portfolio.

    All live positions on any given day are sized such that **each long has identical +1/N exposure
    and each short has identical -1/M exposure** (N = # longs, M = # shorts).
    Notionals are therefore proportional to 1/|count|, not to the original weights.

    :param actual_portfolio_id: the portfolio ID of the actual, starting portfolio that you wish
    to convert to XSE
    :param start_date: start date, must have valid positions in the actual portfolio
    :param end_date: end date, must have valid positions in the actual portfolio
    :param position_sets: whether to return the position sets. Cannot be true if
    return_format is not None. Defaults to False
    :param return_format: return format, defaults to a Pandas DataFrame. Cannot be None if
    position_sets is False

    **Examples**

    >>> xse_portfolio = get_xse_portfolio(
    >>>     actual_portfolio_id='PORTFOLIOID',
    >>>     start_date=dt.date(2021, 1, 1),
    >>>     end_date=dt.date(2021, 1, 31)
    >>> )
    >>> You get back a pandas DataFrame with the XSE portfolio

    >>> xse_portfolio = get_xse_portfolio(
    >>>     actual_portfolio_id='PORTFOLIOID',
    >>>     start_date=dt.date(2021, 1, 1),
    >>>     end_date=dt.date(2021, 1, 31),
    >>>     position_sets=True
    >>> )
    >>> You get back a list of position sets that can be uploaded to a new portfolio via the
    PortfolioManager API
    """
    # VALIDATE INPUT
    pm = _validate_portfolio_creation_inputs(actual_portfolio_id, start_date, end_date,
                                             position_sets, return_format)

    # BUILD XSE PORTFOLIO
    live_df = _pull_constituents_for_date_range(pm, start_date, end_date)
    live_df = _add_equal_weights(live_df)
    equalised_df = _generate_equalised_quantities(live_df, weight_col='w_xse')

    # FORMAT OUTPUT
    if position_sets:
        return _build_position_sets(equalised_df)

    if return_format == ReturnFormat.DATA_FRAME:
        return equalised_df.reset_index(drop=True)

    if return_format == ReturnFormat.JSON:
        return equalised_df.to_dict(orient='records')

    raise MqValueError(f'Unsupported return_format {return_format}')


def get_xstse_portfolio(
        actual_portfolio_id: str,
        start_date: dt.date,
        end_date: dt.date,
        position_sets: bool = False,
        return_format: Union[ReturnFormat, None] = ReturnFormat.DATA_FRAME
) -> Union[Dict, pd.DataFrame, List[PositionSet]]:
    """
    Cross-Sectionally Time-Series Equalised clone of an existing GS Marquee
    portfolio. Extends the XSE logic by **also** fixing *gross* market
    value (GMV) across the entire horizon while still equal-weighting every
    active idea on every date.

    :param actual_portfolio_id: the portfolio ID of the actual, starting portfolio that you wish
    to convert to XSTSE
    :param start_date: start date, must have valid positions in the actual portfolio
    :param end_date: end date, must have valid positions in the actual portfolio
    :param position_sets: whether to return the position sets. Cannot be true if
    return_format is not None. Defaults to False
    :param return_format: return format, defaults to a Pandas DataFrame. Cannot be None if
    position_sets is False

    **Examples**

    >>> xstse_portfolio = get_xstse_portfolio(
    >>>     actual_portfolio_id='PORTFOLIOID',
    >>>     start_date=dt.date(2021, 1, 1),
    >>>     end_date=dt.date(2021, 1, 31)
    >>> )
    >>> You get back a pandas DataFrame with the XSTSE portfolio

    >>> xstse_portfolio = get_xstse_portfolio(
    >>>     actual_portfolio_id='PORTFOLIOID',
    >>>     start_date=dt.date(2021, 1, 1),
    >>>     end_date=dt.date(2021, 1, 31),
    >>>     position_sets=True
    >>> )
    >>> You get back a list of position sets that can be uploaded to a new portfolio via the
    PortfolioManager API
    """
    # VALIDATION
    pm = _validate_portfolio_creation_inputs(actual_portfolio_id, start_date, end_date,
                                             position_sets, return_format)

    # BUILD XSTSE PORTFOLIO --------------------------------------------------
    live_df = _pull_constituents_for_date_range(pm, start_date, end_date)

    # 1) equalise weights across *and* through time
    live_df = _add_ts_equal_weights(live_df)

    # 2) convert weights → integer share counts (same logic as XSE)
    equalised_df = _generate_equalised_quantities(live_df, weight_col='w_xstse')

    # FORMAT OUTPUT ----------------------------------------------------------
    if position_sets:
        return _build_position_sets(equalised_df)

    if return_format == ReturnFormat.DATA_FRAME:
        return equalised_df.reset_index(drop=True)

    if return_format == ReturnFormat.JSON:
        return equalised_df.to_dict(orient='records')

    raise MqValueError(f'Unsupported return_format {return_format}')
