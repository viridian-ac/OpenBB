# # **Backtesting Momentum Trading Strategies using OpenBB**
#
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openbb import obb


# %%
class MomentumTrader:
    """A class to implement and backtest a momentum trading strategy using OpenBB data."""

    def __init__(self, symbols, start_date, initial_capital=1000, short_window=40, long_window=100):
        """
        Initialize the MomentumTrader with trading parameters.

        Parameters:
        - symbols: List of stock symbols (e.g., ['AAPL', 'GOOG'])
        - start_date: Start date for historical data (e.g., '2015-01-01')
        - initial_capital: Initial portfolio value (default: 10000)
        - short_window: Short moving average window (default: 40)
        - long_window: Long moving average window (default: 100)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.short_window = short_window
        self.long_window = long_window
        self.data = {}

    def fetch_data(self, provider="yfinance"):
        """Fetch historical stock data for all symbols using OpenBB."""
        dataframes = []
        for symbol in self.symbols:
            try:
                data = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=self.start_date,
                    provider=provider
                ).to_df()
                data['Symbol'] = symbol
                dataframes.append(data)
            except Exception as e:
                print(f"Failed to fetch data for {symbol}: {str(e)}")

        combined_data = pd.concat(dataframes).reset_index()
        self.data = {symbol: combined_data[combined_data['Symbol'] == symbol].copy()
                     for symbol in self.symbols}

    def momentum_strategy(self, data):
        """Apply momentum strategy based on moving average crossovers."""
        data['Short MA'] = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        data['Long MA'] = data['close'].rolling(window=self.long_window, min_periods=1).mean()
        data['Signal'] = 0
        signal_values = np.where(
            data['Short MA'][self.short_window:] > data['Long MA'][self.short_window:], 1, -1
        )
        data.loc[data.index[self.short_window:], 'Signal'] = signal_values
        data['Position'] = data['Signal'].shift(1)
        return data

    def backtest(self, data):
        """Backtest the momentum strategy and calculate returns."""
        data['Daily Return'] = data['close'].pct_change()
        data['Strategy Return'] = data['Position'] * data['Daily Return']
        data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
        data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()
        data['Portfolio Value'] = self.initial_capital * data['Cumulative Strategy Return']
        return data

    def visualize_backtest(self, data, symbol):
        """Visualize strategy vs. buy-and-hold returns."""
        plt.figure(figsize=(12, 7))
        plt.plot(data['date'], data['Cumulative Market Return'],
                 label='Market Return (Buy & Hold)', color='blue')
        plt.plot(data['date'], data['Cumulative Strategy Return'],
                 label='Momentum Strategy Return', color='green')
        plt.title(f'{symbol} Backtest: Momentum Strategy vs Buy & Hold',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def run(self):
        """Execute the full backtesting process for all symbols."""
        self.fetch_data()
        for symbol in self.symbols:
            if symbol in self.data:
                stock_data = self.momentum_strategy(self.data[symbol])
                stock_data = self.backtest(stock_data)
                self.data[symbol] = stock_data

                self.visualize_backtest(stock_data, symbol)

                final_portfolio_value = stock_data['Portfolio Value'].iloc[-1]
                total_market_return = stock_data['Cumulative Market Return'].iloc[-1] - 1
                total_strategy_return = stock_data['Cumulative Strategy Return'].iloc[-1] - 1

                print(f"Final portfolio value for {symbol}: ${final_portfolio_value:.2f}")
                print(f"Total market return for {symbol}: {total_market_return * 100:.2f}%")
                print(f"Total strategy return for {symbol}: {total_strategy_return * 100:.2f}%")
                print("=" * 40)


# %%
# Example usage
symbols = ['TSLA', 'GOOG', 'MSFT', 'NVDA']
start_date = '2015-01-01'
initial_capital = 10000
short_window = 10
long_window = 100

trader = MomentumTrader(symbols, start_date, initial_capital, short_window, long_window)
trader.run()