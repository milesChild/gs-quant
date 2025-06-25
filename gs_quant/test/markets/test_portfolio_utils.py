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
from gs_quant.markets.portfolio_utils import get_xse_portfolio
from gs_quant.models.risk_model import ReturnFormat


# Stubs for position and position set
class DummyPosition:
    def __init__(self, asset_id: str, quantity: int, weight: float):
        self.asset_id = asset_id
        self.quantity = quantity
        self.weight = weight


class DummyPositionSet:
    def __init__(self, positions: List[DummyPosition], effective_date: dt.date):
        self.positions = positions
        self.effective_date = effective_date
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
    with pytest.raises(MqError, match="have no positions in the source portfolio"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 2))


def test_fail_constituent_download(mocker):
    _patch_internals(mocker, explode_on_constituents=True)
    with pytest.raises(MqError, match="boom"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 1))


def test_fail_no_constituents(mocker):
    _patch_internals(mocker, constituents=[])
    with pytest.raises(MqError, match="No constituent data returned"):
        get_xse_portfolio("PF_X", dt.date(2025, 1, 1), dt.date(2025, 1, 1))


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
