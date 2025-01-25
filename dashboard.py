import streamlit as st
import pandas as pd
import numpy as np


def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["VIX Time"], index_col="VIX Time")
    return data


def apply_strategy(data, vix_type, vix_level, look_back_days):
    # Create a new column for the strategy and position value depending on type
    if vix_type == "Fixed":
        data["VIX Strategy Value"] = data["VIX Last"]
        data["Position"] = (data["VIX Last"] < vix_level).astype(int)
    elif vix_type == "Percent Change":
        data["VIX Strategy Value"] = (
            data["VIX Last"].pct_change(periods=look_back_days) * 100
        )
        data["Position"] = (data["VIX Strategy Value"] < vix_level).astype(int)

    # Calculate the overnight return for QQQ and the strategy
    data["QQQ Overnight Return"] = data["QQQ Open"] / data["QQQ Close"].shift(1) - 1
    data["Strategy Overnight Return"] = data["QQQ Overnight Return"] * data[
        "Position"
    ].shift(1)

    # Select only the necessary columns
    data = data[
        [
            "QQQ Open",
            "QQQ Close",
            "VIX Strategy Value",
            "Position",
            "QQQ Overnight Return",
            "Strategy Overnight Return",
        ]
    ]

    # Drop rows with NaN values
    data = data.dropna()
    return data


def get_stats(data):
    """
    1.	Percentage of Days Overnight we are in the Market vs Not in the Market
    2.	Average return per overnight in set A vs set B
    3.	Percentage of Days that we were long and the market was up
    4.	Overall cumulative return in set A vs set B vs buy and hold overnight
    5.  Sharpe Ratio for set A vs set B vs buy and hold overnight
    """

    # Percent in market overnight
    percent_in_market = data["Position"].mean()
    percent_not_in_market = 1 - percent_in_market

    # Average return per overnight if in market, dont count 0's
    avg_return_in_market = data["Strategy Overnight Return"].replace(0, np.nan).mean()

    # Percent of days long and market up
    data["Market Up"] = np.where(data["QQQ Overnight Return"] > 0, 1, 0)
    percent_long_up = (data["Position"] * data["Market Up"]).mean()

    # Overall cumulative return
    data["Strategy Overnight Cumulative Return"] = (
        data["Strategy Overnight Return"] + 1
    ).cumprod() - 1

    data["QQQ Overnight Cumulative Return"] = (
        data["QQQ Overnight Return"] + 1
    ).cumprod() - 1

    if data["Strategy Overnight Return"].std() == 0:
        sharpe_daily = 0
    else:
        sharpe_daily = (
            data["Strategy Overnight Return"].mean()
            / data["Strategy Overnight Return"].std()
            * np.sqrt(252)
        )

    if data["QQQ Overnight Return"].std() == 0:
        sharpe_buy_hold = 0
    else:
        sharpe_buy_hold = (
            data["QQQ Overnight Return"].mean()
            / data["QQQ Overnight Return"].std()
            * np.sqrt(252)
        )

    # cumulative return
    strategy_cumulative_return = data["Strategy Overnight Cumulative Return"][-1]
    buy_hold_cumulative_return = data["QQQ Overnight Cumulative Return"][-1]

    # Calculates the maximum drawdown in the returns series

    peak = data["Strategy Overnight Cumulative Return"].cummax()
    drawdown = (1 + peak) / (1 + data["Strategy Overnight Cumulative Return"]) - 1
    strategy_max_drawdown = drawdown.max()

    peak = data["QQQ Overnight Cumulative Return"].cummax()
    drawdown = (1 + peak) / (1 + data["QQQ Overnight Cumulative Return"]) - 1
    buy_hold_max_drawdown = drawdown.max()

    # format metrics
    percent_in_market = f"{percent_in_market:.2%}"
    percent_not_in_market = f"{percent_not_in_market:.2%}"
    avg_return_in_market = f"{avg_return_in_market:.2%}"
    percent_long_up = f"{percent_long_up:.2%}"
    sharpe_daily = f"{sharpe_daily:.2f}"
    sharpe_buy_hold = f"{sharpe_buy_hold:.2f}"
    strategy_cumulative_return = f"{strategy_cumulative_return:.2%}"
    buy_hold_cumulative_return = f"{buy_hold_cumulative_return:.2%}"
    strategy_max_drawdown = f"{strategy_max_drawdown:.2%}"
    buy_hold_max_drawdown = f"{buy_hold_max_drawdown:.2%}"

    return {
        "Percent in Market": percent_in_market,
        "Percent Not in Market": percent_not_in_market,
        "Average Return in Market": avg_return_in_market,
        "Percent Long and Market Up": percent_long_up,
        "Strategy Cumulative Return": strategy_cumulative_return,
        "Buy and Hold Cumulative Return": buy_hold_cumulative_return,
        "Strategy Sharpe Ratio": sharpe_daily,
        "Buy and Hold Sharpe Ratio": sharpe_buy_hold,
        "Strategy Max Drawdown": strategy_max_drawdown,
        "Buy and Hold Max Drawdown": buy_hold_max_drawdown,
    }


# Divides data into alternating months Set A and Set B
def divide_data(data, months_per_set=1):
    # Calculate the group by using floor division and modulo; the first n months should 0, the next should be 1, then the n months should be 0...
    data["Group"] = 0

    data.index = pd.to_datetime(data.index)
    # Get all month-years in data
    month_year = data.index.to_period("M").unique()

    # First n number of month years should be 0, the next n should be 1
    for i, my in enumerate(month_year):
        data.loc[data.index.to_period("M") == my, "Group"] = i // months_per_set % 2

    # If group is 0, then it is set A, if group is 1, then it is set B
    data["Group"] = np.where(data["Group"] == 0, "A", "B")
    data.index = data.index.date
    set_a = data[data["Group"] == "A"]
    set_b = data[data["Group"] == "B"]

    # Drop group from all
    data = data.drop(columns=["Group"])
    set_a = set_a.drop(columns=["Group"])
    set_b = set_b.drop(columns=["Group"])

    return set_a, set_b


def get_cumulative_returns(data):
    data["Strategy Overnight Cumulative Return"] = (
        data["Strategy Overnight Return"] + 1
    ).cumprod() - 1

    data["QQQ Overnight Cumulative Return"] = (
        data["QQQ Overnight Return"] + 1
    ).cumprod() - 1

    return data


def find_optimal_vix(
    data, vix_type, look_back_days, months_per_set, min_vix=15, max_vix=40
):
    returns_results = {}
    sharpe_results = {}

    # Try all whole number values
    for vix_threshold in range(int(min_vix), int(max_vix)):
        processed_data = apply_strategy(data, vix_type, vix_threshold, look_back_days)
        set_a, _ = divide_data(processed_data, months_per_set)

        returns_results[vix_threshold] = get_cumulative_returns(set_a)[
            "Strategy Overnight Cumulative Return"
        ][-1]
        sharpe_results[vix_threshold] = get_stats(set_a)["Sharpe Ratio Strategy"]
    best_vix_returns = max(returns_results, key=returns_results.get)
    best_vix_sharpe = max(sharpe_results, key=sharpe_results.get)

    return (
        best_vix_returns,
        returns_results[best_vix_returns],
        best_vix_sharpe,
        sharpe_results[best_vix_sharpe],
    )


def main():

    # Set up the sidebar and title
    st.title("Alternative Train and Test Sets - VIX")

    # Load data from local file
    file_path = "data.csv"
    data = load_data(file_path)

    # Get date values
    min_date, max_date = (
        data.index.min().to_pydatetime(),
        data.index.max().to_pydatetime(),
    )
    start_date = st.sidebar.slider(
        "Select Start Date", min_value=min_date, max_value=max_date, value=min_date
    )
    end_date = st.sidebar.slider(
        "Select End Date", min_value=min_date, max_value=max_date, value=max_date
    )

    # Confirm that start_date is less than end_date
    if start_date >= end_date:
        st.sidebar.error("End date must fall on or after start date.")
        return

    # Filtering data based on selected date range
    filtered_data = data.loc[start_date:end_date]

    # Get the VIX Type and VIX Level
    vix_type_options = ["Fixed", "Percent Change"]
    vix_type = st.sidebar.selectbox("VIX Type", options=vix_type_options)

    # Display look back days only for percent change
    if vix_type == "Percent Change":
        vix_level = st.sidebar.number_input("VIX Percent Change Threshold", value=1.0)
        look_back_days = st.sidebar.number_input("Look Back Days", value=1)
    else:
        vix_level = st.sidebar.number_input("VIX Value Threshold", value=20.0, step=0.1)
        look_back_days = 0

    # Get the number of months in the data
    total_months = len(filtered_data.resample("M").mean())

    # The maximum number of months per set is half of the total months in the data
    max_months_per_set = total_months // 2
    months_per_set = st.sidebar.number_input(
        f"Months per Set (Max: {max_months_per_set})",
        min_value=1,
        max_value=max_months_per_set,
        value=1,
    )

    # Display tuning options, they are different for fixed and percent change
    st.sidebar.write("Tuning Options:")
    if vix_type == "Fixed":
        # Input fields for VIX values to try
        vix_start = st.sidebar.number_input(
            "Start VIX Value", min_value=0.0, value=5.0, step=1.0
        )
        vix_end = st.sidebar.number_input(
            "End VIX Value", min_value=0.0, value=40.0, step=1.0
        )
    else:
        vix_start = st.sidebar.number_input(
            "Start VIX Value", min_value=-100.0, value=-2.0, step=0.1
        )
        vix_end = st.sidebar.number_input("End VIX Value", value=2.0, step=0.1)

    if vix_start >= vix_end:
        st.sidebar.error("End VIX Value must be greater than Start VIX Value.")

    # Run optimization
    if st.sidebar.button("Optimize VIX Threshold for Set A"):
        optimal_vix_return, max_return, optimal_vix_sharpe, max_sharpe = (
            find_optimal_vix(
                filtered_data,
                vix_type,
                look_back_days,
                months_per_set,
                vix_start,
                vix_end,
            )
        )
        st.sidebar.write(
            f"Optimal VIX Threshold for Max Return: {optimal_vix_return} with Return: {max_return:.2%}"
        )
        st.sidebar.write(
            f"Optimal VIX Threshold for Max Sharpe: {optimal_vix_sharpe} with Sharpe: {max_sharpe}"
        )

    ### Display the data and results on main panel ###

    processed_data = apply_strategy(filtered_data, vix_type, vix_level, look_back_days)
    set_a, set_b = divide_data(processed_data, months_per_set)

    # Show Data
    st.write("Processed Data", processed_data)
    st.write("Set A", set_a)
    st.write("Set B", set_b)

    # Get metrics
    set_a = get_cumulative_returns(set_a)
    set_b = get_cumulative_returns(set_b)
    processed_data = get_cumulative_returns(processed_data)

    plot_data = pd.concat(
        [
            set_a["Strategy Overnight Cumulative Return"],
            set_b["Strategy Overnight Cumulative Return"],
        ],
        axis=1,
        join="outer",
    )
    plot_data.columns = [
        "Set A Strategy Overnight Cumulative Return",
        "Set B Strategy Overnight Cumulative Return",
    ]

    processed_data.rename(
        columns={
            "Strategy Overnight Cumulative Return": "Overall Strategy Overnight Cumulative Return"
        },
        inplace=True,
    )

    plot_data.sort_index(inplace=True)

    # Plot QQQ vs Strategy
    st.line_chart(
        processed_data[
            [
                "Overall Strategy Overnight Cumulative Return",
                "QQQ Overnight Cumulative Return",
            ]
        ]
    )

    # Forward fill missing values
    plot_data.fillna(method="ffill", inplace=True)
    st.line_chart(plot_data)

    # Get stats
    stats = get_stats(processed_data)
    stats_a = get_stats(set_a)
    stats_b = get_stats(set_b)

    # Write as summary tables at the bottom
    stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Overall"])
    stats_df["Set A"] = stats_a.values()
    stats_df["Set B"] = stats_b.values()
    st.write(stats_df)


if __name__ == "__main__":
    main()
