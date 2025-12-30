import pandas as pd
import yfinance as yf

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


TICKERS = ["TSLA", "GME"]


# ---------- Helpers ----------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(x) for x in col if x is not None)
            for col in df.columns
        ]
    return df


def detect_close_column(df: pd.DataFrame) -> str:
    """
    Robustly detect a usable close price column from yfinance output
    """
    cols = list(df.columns)

    priority = [
        "Close",
        "Adj Close",
    ]

    for c in priority:
        if c in cols:
            return c

    for c in cols:
        if c.startswith("Close"):
            return c

    for c in cols:
        if "Adj Close" in c:
            return c

    raise KeyError(f"No close price column found. Columns: {cols}")


# ---------- Data ----------
def get_quarterly_net_income(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    qf = t.quarterly_financials

    if qf is None or qf.empty:
        raise ValueError(f"No quarterly financials for {ticker}")

    qf = qf.T
    qf.index = pd.to_datetime(qf.index)

    for label in ["Net Income", "Net Income Common Stockholders", "NetIncome"]:
        if label in qf.columns:
            break
    else:
        raise ValueError(f"No net income column for {ticker}")

    df = (
        qf[[label]]
        .rename(columns={label: "net_income"})
        .reset_index()
        .rename(columns={"index": "quarter_end"})
        .sort_values("quarter_end")
    )

    df["quarter_end"] = df["quarter_end"].dt.normalize()
    return df


def get_quarter_end_prices(ticker: str, min_q: pd.Timestamp) -> pd.DataFrame:
    start = (min_q - pd.Timedelta(days=365 * 6)).date()

    prices = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=False,
    )

    if prices.empty:
        raise ValueError(f"No price data for {ticker}")

    prices = flatten_columns(prices)
    prices = prices.reset_index().rename(columns={"Date": "date"})
    prices["date"] = pd.to_datetime(prices["date"])

    close_col = detect_close_column(prices)

    prices["quarter_end"] = (
        prices["date"]
        .dt.to_period("Q")
        .dt.end_time
        .dt.normalize()
    )

    last_per_quarter = (
        prices.sort_values("date")
        .groupby("quarter_end", as_index=False)
        .tail(1)
    )

    return (
        last_per_quarter[["quarter_end", close_col]]
        .rename(columns={close_col: "quarter_end_price"})
        .sort_values("quarter_end")
        .reset_index(drop=True)
    )


def build_quarterly_dataset(ticker: str) -> pd.DataFrame:
    profit = get_quarterly_net_income(ticker)
    prices = get_quarter_end_prices(ticker, profit["quarter_end"].min())

    df = pd.merge(profit, prices, on="quarter_end", how="inner")
    df["ticker"] = ticker
    return df


def build_all_data() -> pd.DataFrame:
    return pd.concat(
        [build_quarterly_dataset(t) for t in TICKERS],
        ignore_index=True,
    )


DATA = build_all_data()
print("Rows by ticker:\n", DATA.groupby("ticker").size())
print("\nDtypes:\n", DATA.dtypes)
print("\nGME sample:\n", DATA[DATA["ticker"]=="GME"].tail(10))

# --- Clean / enforce numeric types (important for GME) ---
DATA["net_income"] = pd.to_numeric(DATA["net_income"], errors="coerce")
DATA["quarter_end_price"] = pd.to_numeric(DATA["quarter_end_price"], errors="coerce")

# Drop rows where either metric is missing (prevents callback crashes)
DATA = DATA.dropna(subset=["quarter_end", "net_income", "quarter_end_price"]).copy()


# ---------- Dash ----------
app = Dash(__name__)
app.title = "TSLA vs GME — Price vs Profit"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "40px auto"},
    children=[
        html.H2("Stock Price vs Profit (Quarterly)"),

        dcc.Dropdown(
            id="ticker",
            options=[{"label": t, "value": t} for t in TICKERS],
            value="TSLA",
            clearable=False,
            style={"width": "200px"},
        ),

        dcc.RadioItems(
            id="view",
            options=[
                {"label": "Time series", "value": "ts"},
                {"label": "Scatter", "value": "sc"},
            ],
            value="ts",
            inline=True,
        ),

        dcc.Graph(id="chart"),
        html.Div(id="stats"),
    ],
)


@app.callback(
    Output("chart", "figure"),
    Output("stats", "children"),
    Input("ticker", "value"),
    Input("view", "value"),
)
def update_chart(ticker, view):
    try:
        df = DATA[DATA["ticker"] == ticker].sort_values("quarter_end").copy()

        # If GME has no rows after cleaning/merge, show a friendly message instead of crashing
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title=f"{ticker}: No overlapping quarterly profit+price data after cleaning",
            )
            return fig, (
                f"No data points available for {ticker}. "
                f"This usually happens when yfinance returns missing quarterly net income, "
                f"or quarter dates don't align after the merge."
            )

        if view == "ts":
            fig = go.Figure()

            fig.add_bar(
                x=df["quarter_end"],
                y=df["net_income"],
                name="Net Income",
            )

            fig.add_scatter(
                x=df["quarter_end"],
                y=df["quarter_end_price"],
                name="Price",
                yaxis="y2",
                mode="lines+markers",
            )

            fig.update_layout(
                title=f"{ticker}: Net Income vs Price (Quarterly)",
                xaxis_title="Quarter End",
                yaxis=dict(title="Net Income"),
                yaxis2=dict(title="Price", overlaying="y", side="right"),
                legend=dict(orientation="h"),
            )

            latest = df.iloc[-1]
            return fig, (
                f"Points: {len(df)} | "
                f"Latest quarter: {latest['quarter_end'].date()} | "
                f"Net income: {latest['net_income']:,.0f} | "
                f"Price: {latest['quarter_end_price']:,.2f}"
            )

        # Scatter view
        # Correlation can be NaN if net_income is constant or too few points
        corr = df["net_income"].corr(df["quarter_end_price"])
        corr_text = "N/A" if pd.isna(corr) else f"{corr:.3f}"

        fig = go.Figure(
            go.Scatter(
                x=df["net_income"],
                y=df["quarter_end_price"],
                mode="markers",
                text=df["quarter_end"].dt.strftime("%Y-%m-%d"),
            )
        )
        fig.update_layout(
            title=f"{ticker}: Price vs Net Income (corr={corr_text})",
            xaxis_title="Net Income",
            yaxis_title="Price",
        )

        return fig, f"Points: {len(df)} | Correlation: {corr_text}"

    except Exception as e:
        # If anything unexpected happens, don't crash Dash—show the error on screen
        fig = go.Figure()
        fig.update_layout(title=f"{ticker}: Callback crashed")
        return fig, f"Callback error for {ticker}: {type(e).__name__}: {e}"


# ✅ Dash 2.14+
if __name__ == "__main__":
    app.run(debug=True)