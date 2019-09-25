"""
qu_index: Calculates price indexes from transaction data using the QU-index method

Author: Guido van den Heuvel
Date:   25 september 2019
(c) 2019 Centraal Bureau voor de Statistiek / Statistics Netherlands


This module implements the QU-index as described in the following reference:

[1] Antonio G. Chessa, A new methodology for processing scanner data in de Dutch CPI

It expects the microdata to be a CSV file, with columns separated by commas. Each row is
assumed to contain the total turnover and quantity sold for a single product during a 
single month. 

The following columns are expected:
* id              Unique identifier that identifies each product
* Verslagperiode  The month that each record refers to
* Productgroep    The product group that the product is part of.
* Omzet           Turnover for each product in the given month
* Aantal          Number of items sold for each product in the given month


The calculated QU indexes are returned as a pandas DataFrame, with one record for each
month / product group combination, with the calculated QU index for this month / product group.

Column names of this data frame are as follows:
* Verslagperiode  The month 
* Productgroep    The product group 
* qu_index        The value of the calculated QU index
"""

import pandas as pd


def calculate_shortindex(data, eps = 1e-6, baseindex = 100):
    """
    calculate_shortindex

    This calculates the QU index of a given month with respect to the base index, for a single product group.
    The code follows the method outlined in section 4.3 of paper [1].

    Input:
    data        microdata that the index is calculated from. All products in data must be from a
                single product group. The method calculates indexes for all months present in data,
                but only the final month's index is returned. Pandas DataFrame
    eps         accuracy that is required for the indexes. This is the absolute accuracy for indexes
                that start at 1, which, for indexes starting at 100 is approximately equal to the
                relative accuracy. Floating point value.
    baseindex   The given index of the starting month. Floating point value.

    The return value is a pandas DataFrame with a single row and two columns:
    * Verslagperiode    The month for which the index has been calculated (as a pandas Index)
    * qu_index          The value of the index (as an ordinary column)
    """

    periods = data.index.get_level_values("Verslagperiode").sort_values().unique()
    first_period = periods[0]
    final_period = periods[-1]

    workdata = data.copy()

    # Step 1: Choose initial values for the index. We choose $P_z = 1$ for all periods $z$.
    index_table = pd.DataFrame(data = {"qu_index": 1}, index = periods)

    # Step 2a: Calculate $phi_{i, z}$ for all products $i$ and all periods $z$.
    # Note that this step does not involve the price indexes. Therefore this step can be taken out of the loop
    somAantal = data.groupby("id")["Aantal"].sum()
    somAantal.rename("somAantal", inplace = True)
    workdata = pd.merge(data, somAantal, left_index = True, right_index = True)
    workdata["phi"] = workdata["Aantal"] / workdata["somAantal"]

    go_on = True
    n_iter = 0
    while (go_on):
        loopdata = workdata.copy()
        # Step 2b: Calculate $v_i$ for all products $i$.
        loopdata = pd.merge(loopdata, index_table, left_index = True, right_index = True)
        loopdata["vmicro"] = loopdata["phi"] * loopdata["Prijs"] / loopdata["qu_index"]
        v = loopdata.groupby("id")["vmicro"].sum()
        v.rename("v", inplace = True)
        loopdata = pd.merge(loopdata, v, left_index = True, right_index = True)

        # Step 3: Calculate new price indexes
        loopdata["pq"] = loopdata["Prijs"] * loopdata["Aantal"] 
        loopdata["vq"] = loopdata["v"] * loopdata["Aantal"]        
        new_index_table = loopdata.groupby("Verslagperiode")[["pq", "vq"]].sum()

        new_index_table["adjusted_unit_price"] = new_index_table["pq"] / new_index_table["vq"]
        adjusted_unit_price_0 = new_index_table.loc[first_period]["adjusted_unit_price"]
        new_index_table["new_qu_index"] = new_index_table["adjusted_unit_price"] / adjusted_unit_price_0

        # Step 4: Decide whether to stop. We stop if $\max_t |P_t^{(N)} - P_t^{(N-1)}| < \epsilon$, where $N$ is the iteration number
        diff_index = pd.merge(index_table, new_index_table, left_index = True, right_index = True)
        diff = (diff_index["new_qu_index"] - diff_index["qu_index"]).abs().max()
        go_on = diff > eps    
        n_iter += 1

        index_table["qu_index"] = new_index_table["new_qu_index"]

    result = index_table.loc[final_period].copy()
    result["qu_index"] *= baseindex
    
    return result


def calculate_productgroup(data, eps = 1e-6):
    """
    calculate_productgroup

    This calculates the QU index of all months in the data, starting from 100, for a single product group.
    The code follows the method outlined in section 4.3 of paper [1], chaining the indexes of each year into
    a long term index series.

    Input:
    data        microdata that the index is calculated from. All products in data must be from a
                single product group. The method calculates indexes for all months present in data. 
                Pandas DataFrame
    eps         accuracy that is required for the indexes. This is the absolute accuracy for indexes
                that start at 1, which, for indexes starting at 100 is approximately equal to the
                relative accuracy. Floating point value.

    The return value is a pandas DataFrame with two columns, with one row for each month:
    * Verslagperiode    The month for which the index has been calculated (as a pandas Index)
    * qu_index          The value of the index (as an ordinary column)
    """

    # At this point, data only contains turnovers and quantities for each product.
    # We derive the unit price for each product from these.
    data["Prijs"] = data["Omzet"] / data["Aantal"]
    
    periods = data.index.get_level_values("Verslagperiode").sort_values().unique()

    startperiod = periods[0]
    startindex = 100
    indexlist = pd.DataFrame(data = {"qu_index": startindex}, index = pd.Index([startperiod], name = "Verslagperiode"))

    for period in periods[1:]:
        period_data = data.loc[range(startperiod, period + 1)]
        new_index = calculate_shortindex(period_data, eps, startindex)
        indexlist = indexlist.append(new_index)

        # Start a new 13-month window each december
        # Use the index calculated for december as the start index of the next year
        if period % 100 == 12:
            startperiod = period
            startindex = float(new_index["qu_index"])

    return indexlist


def calculate(data, eps = 1e-6):
    """
    calculate

    This function, which is the main entry point of this module, calculates the QU index of all months in the data, 
    starting from 100, for all product groups. The code follows the method outlined in section 4.3 of paper [1], 
    chaining the indexes of each year into a long term index series.

    Input:
    data        microdata that the index is calculated from. All products in data must be from a
                single product group. The method calculates indexes for all months present in data.
                Pandas DataFrame
    eps         accuracy that is required for the indexes. This is the absolute accuracy for indexes
                that start at 1, which, for indexes starting at 100 is approximately equal to the
                relative accuracy. Floating point value.

    The return value is a pandas DataFrame with two columns, with one row for each month:
    * Verslagperiode    The month for which the index has been calculated
    * Productgroep      The product group
    * qu_index          The value of the index
    """

    groups = data.index.get_level_values("Productgroep").sort_values().unique()
    result = None

    for group in groups:
        group_data = data.loc[group]
        indexlist = calculate_productgroup(group_data, eps)
        indexlist["Productgroep"] = group
        if result is None:
            result = indexlist
        else:
            result = result.append(indexlist)

    return result.reset_index()


def load_microdata_csv(filename):
    """
    load_microdata_csv

    A convenience function for loading the microdata, as a CSV file, from disk.

    Input:
    filename        The name of the file as stored on disk

    Output:
    A pandas dataframe containing the microdata. No check is done on column names; names are
    expected to be as described in the module documentation. A pandas MultiIndex is constructed
    from the identifying columns (Iid, Verslagperiode and Productgroep).
    """

    data = pd.read_csv(filename, sep = ",", index_col = ("Productgroep", "Verslagperiode", "id"))
    return data
