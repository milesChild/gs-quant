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
from typing import Dict, List, Union

import pandas as pd

from gs_quant.markets.report import ReturnFormat
from gs_quant.markets.portfolio import PositionSet


def get_xse_portfolio(
        actual_portfolio_id: str,
        start_date: dt.date,
        end_date: dt.date,
        position_sets: bool = False,
        return_format: Union[ReturnFormat, None] = ReturnFormat.DATA_FRAME
) -> Union[Dict, pd.DataFrame, List[PositionSet]]:
    """
    Get the Cross-Sectional Equalized (XSE) portfolio for a given portfolio

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
    # VALIDATION
    # validate that start_date is not > end_date
    # TODO
    # position_sets cannot be True & return_format
    # TODO
    # validate existence & access to the actual_portfolio_id
    # TODO
    # validate start_date and end_date in actual portfolio
    # TODO
    # DATA FETCHING AND CONVERSION
    # fetch positions from the OG portfolio
    # TODO
    # clean / aggregate by day, position
    # TODO
    # convert into XSE
    # TODO
    # FORMATTING AND RETURN
    # if position_sets:
    # convert and resolve positions sets on each day
    # TODO
    # else:
    # convert to return_format
    # TODO
    return
