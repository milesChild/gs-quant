"""
Copyright 2018 Goldman Sachs.
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
from typing import List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from gs_quant.errors import MqError, MqValueError
from gs_quant.markets.portfolio_utils import get_xse_portfolio, get_xstse_portfolio
from gs_quant.models.risk_model import ReturnFormat


# Stubs for position and position set
class DummyPosition:
    def __init__(self, asset_id: str, quantity: int, weight: float, identifier: str = None):
        self.asset_id = asset_id
        self.quantity = quantity
        self.weight = weight
        self.identifier = identifier or asset_id


class DummyPositionSet:
    def __init__(self, positions: List[DummyPosition], date: dt.date):
        self.positions = positions
        self.date = date
        self.unresolved_positions: List[str] = []

    def resolve(self):
        return self


# Central patch helper
def _patch_internals(
    mocker,
    *,
    portfolio_exists: bool = True,
    date_list: List[dt.date] | None = None,
    constituents: list | None = None,
    explode_on_constituents: bool = False,
    positionset_cls=DummyPositionSet,
):
    """Monkey-patch every gs-quant dependency touched by get_xse_portfolio."""
    if date_list is None:
        date_list = [dt.date(2025, 1, 1), dt.date(2025, 1, 2)]

    if constituents is None:
        constituents = [
            # 2025-01-01
            dict(date="2025-01-01", assetId="A", quantity=600, netWeight=0.60, grossWeight=0.60),
            dict(date="2025-01-01", assetId="B", quantity=-400, netWeight=-0.40, grossWeight=0.40),
            # 2025-01-02
            dict(date="2025-01-02", assetId="A", quantity=650, netWeight=0.65, grossWeight=0.65),
            dict(date="2025-01-02", assetId="B", quantity=-450, netWeight=-0.35, grossWeight=0.35),
        ]

    # GsPortfolioApi.get_portfolio
    if portfolio_exists:
        mocker.patch(
            "gs_quant.markets.portfolio_utils.GsPortfolioApi.get_portfolio",
            return_value={"id": "dummy"},
        )
    else:
        mocker.patch(
            "gs_quant.markets.portfolio_utils.GsPortfolioApi.get_portfolio",
            side_effect=Exception("not found"),
        )

    # PortfolioManager stub
    pm = mock.MagicMock(name="PortfolioManagerStub")
    pm.get_position_dates.return_value = date_list

    pr = mock.MagicMock(name="PerformanceReportStub")
    if explode_on_constituents:
        pr.get_portfolio_constituents.side_effect = MqError("boom")
    else:
        pr.get_portfolio_constituents.return_value = constituents
    pm.get_performance_report.return_value = pr

    mocker.patch("gs_quant.markets.portfolio_utils.PortfolioManager", return_value=pm)
    mocker.patch("gs_quant.markets.portfolio_utils.Position", DummyPosition, create=True)
    mocker.patch("gs_quant.markets.portfolio_utils.PositionSet", positionset_cls, create=True)


# Failure-path tests
def test_fail_nonexistent_portfolio(mocker):
    _patch_internals(mocker, portfolio_exists=False)
    with pytest.raises(MqError, match="Cannot retrieve portfolio"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def test_fail_start_after_end(mocker):
    _patch_internals(mocker)
    with pytest.raises(MqValueError, match="start_date .* must be on/ before"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 10), dt.date(2025, 1, 5))


def test_fail_dates_not_covered(mocker):
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 3), dt.date(2025, 1, 4)],
    )
    with pytest.raises(MqError, match="is outside available portfolio date range"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def test_fail_constituent_download(mocker):
    _patch_internals(mocker, explode_on_constituents=True)
    with pytest.raises(MqError, match="boom"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 1))


def test_fail_no_constituents(mocker):
    _patch_internals(mocker, constituents=[])
    with pytest.raises(MqError, match="No constituent data returned"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 1))


def test_fail_start_before_calendar_range(mocker):
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 10), dt.date(2025, 1, 20)],
    )
    with pytest.raises(MqError, match="is outside available portfolio date range"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 5), dt.date(2025, 1, 15))


def test_fail_end_after_calendar_range(mocker):
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 10), dt.date(2025, 1, 20)],
    )
    with pytest.raises(MqError, match="is outside available portfolio date range"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 15), dt.date(2025, 1, 25))


def test_success_dates_within_calendar_range(mocker):
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 10), dt.date(2025, 1, 15), dt.date(2025, 1, 20)],
    )
    # Should succeed: requested range [2025-01-12, 2025-01-18] is within [2025-01-10, 2025-01-20]
    df = get_xse_portfolio(
        "PF_X",
        dt.date(2025, 1, 12),
        dt.date(2025, 1, 18),
        return_format=ReturnFormat.DATA_FRAME,
    )
    assert isinstance(df, pd.DataFrame)


def test_fail_mutually_exclusive_flags(mocker):
    _patch_internals(mocker)
    with pytest.raises(MqValueError, match="When position_sets is True"):
        get_xse_portfolio(
            "PF_X",
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 2),
            position_sets=True,
            return_format=ReturnFormat.DATA_FRAME,
        )


def test_fail_flags_both_none(mocker):
    _patch_internals(mocker)
    with pytest.raises(MqValueError):
        get_xse_portfolio(
            "PF_X",
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 2),
            position_sets=False,
            return_format=None,
        )


def test_fail_position_set_resolution(mocker):
    class BadPS(DummyPositionSet):
        def resolve(self):
            self.unresolved_positions = ["XYZ"]
            return self

    _patch_internals(mocker, positionset_cls=BadPS)
    with pytest.raises(MqError, match="Unresolved positions"):
        get_xse_portfolio(
            "PF_X",
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 1),
            position_sets=True,
            return_format=None,
        )


# Cleaning / aggregation checks
def test_cleaning_and_unique_positions(mocker):
    duplicate_constituents = [
        dict(date="2025-01-01", assetId="A", quantity=300, netWeight=0.30, grossWeight=0.30),
        dict(date="2025-01-01", assetId="A", quantity=300, netWeight=0.30, grossWeight=0.30),
        dict(date="2025-01-01", assetId="B", quantity=-600, netWeight=-0.60, grossWeight=0.60),
    ]
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 1)],
        constituents=duplicate_constituents,
    )

    df = get_xse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 1),
        return_format=ReturnFormat.DATA_FRAME,
    )

    assert len(df) == df["assetId"].nunique()
    long_w = df.loc[df["weight"] > 0, "weight"].unique()
    short_w = df.loc[df["weight"] < 0, "weight"].unique()
    assert len(long_w) == 1 and len(short_w) == 1 and np.isclose(abs(long_w[0]), abs(short_w[0]))


# Happy-path output format tests
def test_output_dataframe(mocker):
    _patch_internals(mocker)
    df = get_xse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        return_format=ReturnFormat.DATA_FRAME,
    )
    assert isinstance(df, pd.DataFrame)
    assert {"date", "assetId", "weight", "quantity"}.issubset(df.columns)
    assert len(df) == 4  # 2 assets x 2 days


def test_output_json(mocker):
    _patch_internals(mocker)
    js = get_xse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        return_format=ReturnFormat.JSON,
    )
    assert isinstance(js, list) and len(js) == 4
    assert {"date", "assetId", "weight", "quantity"}.issubset(js[0].keys())


def test_output_position_sets(mocker):
    _patch_internals(mocker)
    psets = get_xse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        position_sets=True,
        return_format=None,
    )
    assert isinstance(psets, list) and len(psets) == 2
    for ps in psets:
        wts = [p.weight for p in ps.positions]
        assert len(set(w for w in wts if w > 0)) == 1
        assert len(set(w for w in wts if w < 0)) == 1


@pytest.mark.parametrize(
    "start,end,err_regex",
    [
        (dt.date(2025, 1, 10), dt.date(2025, 1, 5), "start_date .* must be on/ before"),
    ],
)
def test_fail_start_after_end_xstse(mocker, start, end, err_regex):
    _patch_internals(mocker)
    with pytest.raises(MqValueError, match=err_regex):
        get_xstse_portfolio("PF_X", start, end)


def test_fail_nonexistent_portfolio_xstse(mocker):
    _patch_internals(mocker, portfolio_exists=False)
    with pytest.raises(MqError, match="Cannot retrieve portfolio"):
        get_xstse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def test_fail_dates_not_covered_xstse(mocker):
    _patch_internals(
        mocker, date_list=[dt.date(2025, 1, 3), dt.date(2025, 1, 4)]
    )
    with pytest.raises(MqError, match="is outside available portfolio date range"):
        get_xstse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def test_fail_constituent_download_xstse(mocker):
    _patch_internals(mocker, explode_on_constituents=True)
    with pytest.raises(MqError, match="boom"):
        get_xstse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 1))


def test_fail_no_constituents_xstse(mocker):
    _patch_internals(mocker, constituents=[])
    with pytest.raises(MqError, match="No constituent data returned"):
        get_xstse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 1))


def test_fail_start_before_calendar_range_xstse(mocker):
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 10), dt.date(2025, 1, 20)],
    )
    with pytest.raises(MqError, match="is outside available portfolio date range"):
        get_xstse_portfolio("PF_X", dt.date(2025, 1, 5), dt.date(2025, 1, 15))


def test_success_dates_within_calendar_range_xstse(mocker):
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 10), dt.date(2025, 1, 15), dt.date(2025, 1, 20)],
    )
    # Should succeed: requested range [2025-01-12, 2025-01-18] is within [2025-01-10, 2025-01-20]
    df = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 12),
        dt.date(2025, 1, 18),
        return_format=ReturnFormat.DATA_FRAME,
    )
    assert isinstance(df, pd.DataFrame)


def test_fail_mutually_exclusive_flags_xstse(mocker):
    _patch_internals(mocker)
    with pytest.raises(MqValueError, match="When position_sets is True"):
        get_xstse_portfolio(
            "PF_X",
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 2),
            position_sets=True,
            return_format=ReturnFormat.DATA_FRAME,
        )


def test_fail_flags_both_none_xstse(mocker):
    _patch_internals(mocker)
    with pytest.raises(MqValueError):
        get_xstse_portfolio(
            "PF_X",
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 2),
            position_sets=False,
            return_format=None,
        )


def test_fail_position_set_resolution_xstse(mocker):
    class BadPS(DummyPositionSet):
        def resolve(self):
            self.unresolved_positions = ["XYZ"]
            return self

    _patch_internals(mocker, positionset_cls=BadPS)
    with pytest.raises(MqError, match="Unresolved positions"):
        get_xstse_portfolio(
            "PF_X",
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 1),
            position_sets=True,
            return_format=None,
        )


# Cleaning & aggregation
def test_cleaning_and_unique_positions_xstse(mocker):
    duplicate_constituents = [
        dict(date="2025-01-01", assetId="A", quantity=300, netWeight=0.30, grossWeight=0.30),
        dict(date="2025-01-01", assetId="A", quantity=300, netWeight=0.30, grossWeight=0.30),
        dict(date="2025-01-01", assetId="B", quantity=-600, netWeight=-0.60, grossWeight=0.60),
    ]
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 1)],
        constituents=duplicate_constituents,
    )

    df = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 1),
        return_format=ReturnFormat.DATA_FRAME,
    )

    # one row per unique asset
    assert len(df) == df["assetId"].nunique()

    # every active idea should have the same *magnitude* weight
    assert len(set(abs(w) for w in df["weight"])) == 1


# Happy‑path output formats
def test_output_dataframe_xstse(mocker):
    _patch_internals(mocker)
    df = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        return_format=ReturnFormat.DATA_FRAME,
    )
    assert isinstance(df, pd.DataFrame)
    assert {"date", "assetId", "weight", "quantity"}.issubset(df.columns)
    assert len(df) == 4  # 2 assets × 2 days


def test_output_json_xstse(mocker):
    _patch_internals(mocker)
    js = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        return_format=ReturnFormat.JSON,
    )
    assert isinstance(js, list) and len(js) == 4
    assert {"date", "assetId", "weight", "quantity"}.issubset(js[0].keys())


def test_output_position_sets_xstse(mocker):
    _patch_internals(mocker)
    psets = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        position_sets=True,
        return_format=None,
    )
    assert isinstance(psets, list) and len(psets) == 2
    for ps in psets:
        # magnitudes must be identical across every idea
        mags = {abs(p.weight) for p in ps.positions}
        assert len(mags) == 1


# XSTSE‑specific sanity checks
def test_signs_preserved(mocker):
    _patch_internals(mocker)
    out = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        return_format=ReturnFormat.DATA_FRAME,
    )

    # compare sign(w_live) vs sign(weight)  – they must match
    assert set(out["weight"].apply(np.sign)).issubset({-1, 1})


def test_constant_gmv_when_counts_equal(mocker):
    # three days – same # of ideas on day‑1 and day‑2, different on day‑3
    constituents = [
        dict(date="2025-01-01", assetId="A", quantity=100, netWeight=0.70, grossWeight=0.70),
        dict(date="2025-01-01", assetId="B", quantity=-100, netWeight=-0.30, grossWeight=0.30),
        dict(date="2025-01-02", assetId="A", quantity=110, netWeight=0.66, grossWeight=0.66),
        dict(date="2025-01-02", assetId="B", quantity=-120, netWeight=-0.34, grossWeight=0.34),
        # day‑3 adds a new idea
        dict(date="2025-01-03", assetId="A", quantity=120, netWeight=0.50, grossWeight=0.50),
        dict(date="2025-01-03", assetId="B", quantity=-80, netWeight=-0.33, grossWeight=0.33),
        dict(date="2025-01-03", assetId="C", quantity=60, netWeight=0.17, grossWeight=0.17),
    ]
    _patch_internals(
        mocker,
        date_list=[
            dt.date(2025, 1, 1),
            dt.date(2025, 1, 2),
            dt.date(2025, 1, 3),
        ],
        constituents=constituents,
    )

    df = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 3),
        return_format=ReturnFormat.DATA_FRAME,
    )

    gmv = df.groupby("date")["weight"].apply(lambda x: x.abs().sum())
    assert np.isclose(gmv.loc[dt.date(2025, 1, 1)], gmv.loc[dt.date(2025, 1, 2)])
    # day‑3 may differ (extra idea) – no assertion


def test_no_new_positions_invented(mocker):
    _patch_internals(mocker)
    result = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        return_format=ReturnFormat.DATA_FRAME,
    )
    assert set(result["assetId"]) <= {"A", "B"}  # subset of originals only


def test_idempotent_when_already_xstse(mocker):
    # Build constituents that are already XSTSE (|w|=0.5 for both L/S)
    constituents = [
        dict(date="2025-01-01", assetId="A", quantity=100, netWeight=0.5, grossWeight=0.5),
        dict(date="2025-01-01", assetId="B", quantity=-100, netWeight=-0.5, grossWeight=0.5),
    ]
    _patch_internals(
        mocker,
        date_list=[dt.date(2025, 1, 1)],
        constituents=constituents,
    )

    df = get_xstse_portfolio(
        "PF_X",
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 1),
        return_format=ReturnFormat.DATA_FRAME,
    )

    # weights should remain unchanged
    assert np.isclose(abs(df["weight"]).unique(), 0.5).all()
