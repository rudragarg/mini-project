import streamlit as st
import pandas as pd
import numpy as np


# Function to load data from a local CSV file
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["VIX Time"], index_col="VIX Time")
    return data


# Function to apply the trading strategy
def apply_strategy(data, vix_type, vix_level):
    if vix_type == "Fixed":
        data["VIX Strategy Value"] = data["VIX Last"]
        data["Position"] = (data["VIX Last"] < vix_level).astype(int)
    elif vix_type == "Percent Change":
        data["VIX Strategy Value"] = data["VIX Last"].pct_change() * 100
        data["Position"] = (data["VIX Change"] < vix_level).astype(int)
    else:
        # Add additional logic here for other VIX types
        pass

    # convert number % value to decimal
    data["Daily Return"] = data["QQQ Close"].pct_change()
    # current open / previous close
    data["Overnight Return"] = data["QQQ Open"] / data["QQQ Close"].shift(1) - 1
    data["Strategy Return"] = data["Daily Return"] * data["Position"].shift(1)
    data["Strategy Overnight Return"] = data["Overnight Return"] * data[
        "Position"
    ].shift(1)

    data = data[
        [
            "QQQ Close",
            "VIX Strategy Value",
            "Position",
            "Daily Return",
            "Overnight Return",
            "Strategy Return",
            "Strategy Overnight Return",
        ]
    ]
    return data


def get_stats(data):
    """
    1.	Percentage of Days Overnight we are in the Market vs Not in the Market
    2.	Average return per overnight in set A vs set B
    3.	Percentage of Days that we were long and the market was up
    4.	Overall cumulative return in set A vs set B vs buy and hold overnight

    """

    # percent in market overnight
    percent_in_market = data["Position"].mean()
    percent_not_in_market = 1 - percent_in_market

    # average return per overnight
    avg_return_in_market = data["Strategy Overnight Return"].mean()

    # percent of days long and market up
    data["Market Up"] = np.where(data["Overnight Return"] > 0, 1, 0)
    percent_long_up = (data["Position"] * data["Market Up"]).mean()

    # overall cumulative return
    data["Portfolio Overnight Return"] = (
        data["Strategy Overnight Return"] + 1
    ).cumprod()

    data["Buy and Hold Overnight Return"] = (data["QQQ Overnight Return"] + 1).cumprod()

    sharpe_daily = (
        data["Strategy Overnight Return"].mean()
        / data["Strategy Overnight Return"].std()
        * np.sqrt(252)
    )
    sharpe_buy_hold = (
        data["Overnight Return"].mean() / data["Overnight Return"].std() * np.sqrt(252)
    )

    return {
        "Percent in Market": percent_in_market,
        "Percent Not in Market": percent_not_in_market,
        "Average Return in Market": avg_return_in_market,
        "Percent Long and Market Up": percent_long_up,
        "Sharpe Ratio Strategy": sharpe_daily,
        "Sharpe Ratio Buy and Hold": sharpe_buy_hold,
    }


# Main function for the Streamlit app
def main():
    st.title("Alternative Train and Test Sets - VIX")

    # Load data from local file
    file_path = "data.csv"
    data = load_data(file_path)

    min_date, max_date = (
        data.index.min().to_pydatetime(),
        data.index.max().to_pydatetime(),
    )
    print(min_date, max_date)
    start_date = st.slider(
        "Select Start Date", min_value=min_date, max_value=max_date, value=min_date
    )
    end_date = st.slider(
        "Select End Date", min_value=min_date, max_value=max_date, value=max_date
    )

    # confirm that start_date is less than end_date
    if start_date > end_date:
        st.error("End date must fall after start date.")
        return

    # Filtering data based on selected date range
    filtered_data = data.loc[start_date:end_date]

    vix_type_options = ["Fixed", "Percent Change"]
    vix_type = st.selectbox("VIX Type", options=vix_type_options)
    vix_level = st.number_input("VIX Value", value=20.0)

    processed_data = apply_strategy(filtered_data, vix_type, vix_level)
    st.write("Processed Data", processed_data)

    # Additional visualizations and statistics
    # get portfolio return cum
    # processed_data["Portfolio Return"] = (
    #     processed_data["Strategy Return"] + 1
    # ).cumprod()
    processed_data["Portfolio Overnight Return"] = (
        processed_data["Strategy Overnight Return"] + 1
    ).cumprod()

    # get QQQ return
    # processed_data["QQQ Return"] = (processed_data["Daily Return"] + 1).cumprod()
    processed_data["QQQ Overnight Return"] = (
        processed_data["Overnight Return"] + 1
    ).cumprod()

    # plot, add axis labels
    # st.line_chart(processed_data[["Portfolio Return", "QQQ Return"]])
    st.line_chart(
        processed_data[["Portfolio Overnight Return", "QQQ Overnight Return"]]
    )

    # get stats
    stats = get_stats(processed_data)

    # write as table
    st.write(pd.DataFrame(stats.items(), columns=["Metric", "Value"]))


if __name__ == "__main__":
    main()
