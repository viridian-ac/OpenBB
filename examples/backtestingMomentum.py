# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openbb import obb  # OpenBB SDK for financial data access

# %%
class MomentumTrader:
    """A class to implement and backtest a momentum trading strategy using OpenBB data."""

    def __init__(self, symbols, start_date, initial_capital=1000, short_window=40, long_window=100):
        """
        Initialize the MomentumTrader with trading parameters.

        Parameters:
        - symbols: List of stock symbols to backtest (e.g., ['AAPL', 'GOOG'])
        - start_date: Start date for fetching historical data (format: 'YYYY-MM-DD')
        - initial_capital: Starting portfolio value for the backtest
        - short_window: Number of days for the short-term moving average
        - long_window: Number of days for the long-term moving average
        """
        self.symbols = symbols
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.short_window = short_window
        self.long_window = long_window
        self.data = {}  # Dictionary to store stock data for each symbol

    def fetch_data(self, provider="yfinance"):
        """
        Fetch historical stock price data for each symbol using OpenBB.

        Parameters:
        - provider: Data provider, default is 'yfinance' (Yahoo Finance)

        This function retrieves the historical prices and stores them in the self.data dictionary,
        where each key is a stock symbol and the value is a DataFrame of prices.
        """
        dataframes = []

        for symbol in self.symbols:
            try:
                # Use OpenBB to fetch historical data for each symbol
                data = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=self.start_date,
                    provider=provider
                ).to_df()

                data['Symbol'] = symbol  # Tag each row with the symbol
                dataframes.append(data)

            except Exception as e:
                # Handle errors if data fetch fails
                print(f"Failed to fetch data for {symbol}: {str(e)}")

        # Combine all symbol data into a single DataFrame
        combined_data = pd.concat(dataframes).reset_index()

        # Separate combined data back into a dictionary of DataFrames by symbol
        self.data = {
            symbol: combined_data[combined_data['Symbol'] == symbol].copy()
            for symbol in self.symbols
        }

    def momentum_strategy(self, data):
        """
        Apply a simple momentum strategy based on moving average crossovers.

        Parameters:
        - data: A DataFrame containing stock price data

        Returns:
        - The modified DataFrame with strategy signals and positions

        Strategy:
        - Buy signal when short MA crosses above long MA
        - Sell signal when short MA crosses below long MA
        """
        # Calculate moving averages
        data['Short MA'] = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        data['Long MA'] = data['close'].rolling(window=self.long_window, min_periods=1).mean()

        # Initialize signal column
        data['Signal'] = 0

        # Generate signal: 1 if Short MA > Long MA, else -1
        signal_values = np.where(
            data['Short MA'][self.short_window:] > data['Long MA'][self.short_window:], 1, -1
        )

        # Apply signal values starting from the short window index
        data.loc[data.index[self.short_window:], 'Signal'] = signal_values

        # Shift signal to avoid look-ahead bias (we act on yesterday's signal)
        data['Position'] = data['Signal'].shift(1)

        return data

    def backtest(self, data):
        """
        Backtest the strategy by calculating returns and portfolio value over time.

        Parameters:
        - data: DataFrame with strategy signals and positions

        Returns:
        - The updated DataFrame with performance metrics
        """
        # Calculate daily returns
        data['Daily Return'] = data['close'].pct_change()

        # Apply the trading position to the daily return to simulate strategy returns
        data['Strategy Return'] = data['Position'] * data['Daily Return']

        # Calculate cumulative returns for both market and strategy
        data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
        data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

        # Simulate portfolio value over time using strategy returns
        data['Portfolio Value'] = self.initial_capital * data['Cumulative Strategy Return']

        return data

    def visualize_backtest(self, data, symbol):
        """
        Generate a plot comparing the momentum strategy to a buy-and-hold strategy.

        Parameters:
        - data: DataFrame with backtest results
        - symbol: The stock symbol (used in plot title)
        """
        plt.figure(figsize=(12, 7))

        # Plot cumulative market return
        plt.plot(data['date'], data['Cumulative Market Return'],
                 label='Market Return (Buy & Hold)', color='blue')

        # Plot cumulative strategy return
        plt.plot(data['date'], data['Cumulative Strategy Return'],
                 label='Momentum Strategy Return', color='green')

        # Add title and axis labels
        plt.title(f'{symbol} Backtest: Momentum Strategy vs Buy & Hold',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)

        # Improve readability of x-axis
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run the full backtesting pipeline for all symbols:
        - Fetch data
        - Apply strategy
        - Backtest performance
        - Visualize results
        - Print summary statistics
        """
        self.fetch_data()

        for symbol in self.symbols:
            if symbol in self.data:
                # Get data for current symbol
                stock_data = self.data[symbol]

                # Apply the momentum strategy
                stock_data = self.momentum_strategy(stock_data)

                # Backtest strategy performance
                stock_data = self.backtest(stock_data)

                # Store updated data
                self.data[symbol] = stock_data

                # Generate comparison plot
                self.visualize_backtest(stock_data, symbol)

                # Output final results
                final_portfolio_value = stock_data['Portfolio Value'].iloc[-1]
                total_market_return = stock_data['Cumulative Market Return'].iloc[-1] - 1
                total_strategy_return = stock_data['Cumulative Strategy Return'].iloc[-1] - 1

                print(f"Final portfolio value for {symbol}: ${final_portfolio_value:.2f}")
                print(f"Total market return for {symbol}: {total_market_return * 100:.2f}%")
                print(f"Total strategy return for {symbol}: {total_strategy_return * 100:.2f}%")
                print("=" * 40)


# Example usage of the MomentumTrader class
# Define the parameters for the backtest

symbols = ['TSLA', 'GOOG', 'MSFT', 'NVDA']        # List of stock symbols to analyze
start_date = '2015-01-01'                         # Backtest start date
initial_capital = 10000                           # Initial capital for each strategy
short_window = 10                                 # Short-term moving average window
long_window = 100                                 # Long-term moving average window

# Instantiate the trader object
trader = MomentumTrader(symbols, start_date, initial_capital, short_window, long_window)

# Run the backtesting pipeline
trader.run()
