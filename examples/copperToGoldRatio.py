import pandas as pd
import plotly.graph_objects as go
from openbb import obb
import start


# Set FRED API key
obb.user.credentials.fred_api_key = start.FRED_API_KEY
data = pd.DataFrame()

end_date = "2025-07-18"

cols_dict = {"GC=F": "Gold", "HG=F": "Copper"}
data = (
    obb.derivatives.futures.historical(
        ["GC", "HG"],
        start_date="2000-01-01",
        end_date=end_date,
        interval="1W",
    )
    .to_df()
    .pivot(columns="symbol", values="close")
)
data.columns = [cols_dict[symbol] for symbol in data.columns]
data.index = pd.to_datetime(data.index)
#%% md
# Let's inspect the results.
#%%
# print(data.head(2))
#%% md
# To get the copper-to-gold ratio, divide the two columns along each row.
#%%
data["Copper/Gold Ratio"] = data["Copper"] / data["Gold"]

# print(data.tail(2))
#%% md
# Because the numbers are so small, the ratio is often be presented as a % value.
# 0.2% is a popular way to display the value.  However, to plot it on the same y-axis as a Treasury yield, it needs to be multiplied by 1000.  Let's alter the block above to include this.
#%%
data["Copper/Gold Ratio"] = (data["Copper"] / data["Gold"]) * 1000

# print(data.tail(2))
#%% md
# Now let's add a column for the daily 10 Year US Treasury Yield.
# This can be requested using the `fred_series` function within the `economy` module.
# The first line in the block below requests the data, the second assigns it to a column in the target DataFrame.
"""
try:
    us10year = obb.economy.fred_series(
        "DGS10",
        frequency="d", # "wem",
        start_date="2000-08-28",
        end_date=end_date,
        provider="fred"
    ).to_df()[["DGS10"]]
    if us10year.empty:
        raise ValueError("No Treasury yield data returned for DGS10")
except Exception as e:
    print(f"Error fetching FRED data: {e}")
    exit()

us10year.index = pd.to_datetime(us10year.index)
us10year = us10year.resample("W-MON").mean()  # Weekly, ending Monday
data["US 10-Year Constant Maturity"] = us10year["DGS10"]
"""
try:
    us10year = obb.equity.price.historical(
        "^TNX",
        start_date="2000-08-28",
        end_date=end_date,
        interval="1W",
        provider="yfinance"
    ).to_df()[["close"]]
    data["US 10-Year Constant Maturity"] = us10year["close"]
except Exception as e:
    print(f"Error fetching Yahoo Finance data: {e}")
    exit()



data.head(2)
#%% md
# With all the data collected, let's draw the chart to visualize the relationship.
#%%
fig = go.Figure()
fig.add_scatter(
    x=data.index, y=data["Copper/Gold Ratio"], name="Copper/Gold Ratio (x1000) %"
)
fig.add_scatter(
    x=data.index,
    y=data["US 10-Year Constant Maturity"],
    name="US 10-Year Constant Maturity %",
)
fig.update(
    {
        "layout": {
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "%"},
            "title": "Copper/Gold Ratio vs. US 10-Year Constant Maturity",
            "title_y": 0.90,
            "title_x": 0.5,
        }
    }
)
fig.update_layout(legend=dict(yanchor="top", y=1, xanchor="right", x=1.0))
#%% md
# What we have currently is the price relationship between one Troy ounce of gold and one pound of copper.  As we described the copper-to-gold ratio as the price-per-ounce of each, some adjustments are required to be true to the definition.
# 
# - 1 ounce = 0.911458 Troy ounces
# - 1 pound = 16 ounces
#   
# To adjust the gold price as USD/ounce, multiply each row by 0.911458.  To adjust the copper price, divide each row by 16.
#%%
data["Copper/Gold Ratio per Ounce (x1000) %"] = (
    (data["Copper"] / 16) / (data["Gold"] * 0.911458)
) * 1000

data.tail(2)
#%% md
# Now let's draw it!
#%%
fig = go.Figure()

# Add the first scatter trace with its own y-axis
fig.add_scatter(
    x=data.index,
    y=data["Copper/Gold Ratio"],
    name="Copper/Gold Ratio (x1000) %",
    yaxis="y1",
)

# Add the second scatter trace with its own y-axis
fig.add_scatter(
    x=data.index,
    y=data["US 10-Year Constant Maturity"],
    name="US 10-Year Constant Maturity %",
    yaxis="y2",
)

# Update the layout to include the y-axes and their titles
fig.update_layout(
    yaxis=dict(
        title="Copper/Gold Ratio (x1000) %",
        side="left",
        position=0,
        title_font=dict(size=12),
        showgrid=False,
    ),
    yaxis2=dict(
        title="US 10-Year Constant Maturity %",
        side="right",
        overlaying="y",
        position=1,
        title_font=dict(size=12),
    ),
    xaxis=dict(title="Date"),
    title="Copper/Gold Ratio vs. US 10-Year Constant Maturity",
    title_y=0.90,
    title_x=0.5,
    legend=dict(yanchor="top", y=1, xanchor="right", x=1.0, font=dict(size=10)),
)



# Show the plot
fig.show()
#%% md
# There you have it, folks!  The OpenBB Platform provides endless possibilities for creating unique indicators and analysis with the wide variety of data available at your fingertips.  We love seeing the creations of users, so be sure to tag us on social media and show off your work.