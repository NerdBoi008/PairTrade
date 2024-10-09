from yfinance import download as yfDownload
from pandas import DataFrame
from datetime import date
import numpy as np
import math

class TradePair:
    """
    Class to analyze and perform pair trading between two stocks.
    """
    
    def __init__(self, stockA: str, stockB: str, startDate: str = "", endDate: str = "", period: str = '1mo', 
                 analysing_price: str = 'Close', float_precision: int = 6):
        """
        Initializes the TradePair class with stock tickers, time range, and other parameters.

        Args:
            stockA: Ticker of the first stock.
            stockB: Ticker of the second stock.
            startDate: Start date for fetching historical data.
            endDate: End date for fetching historical data.
            period: Period to fetch the data (e.g., '1mo').
            analysing_price: Type of price to analyze (e.g., 'Close').
            float_precision: Precision for floating point numbers in results.
        """
        self.stockATicker = stockA
        self.stockBTicker = stockB
        self.startDate = startDate
        self.endDate = endDate
        self.period = period
        self.analysing_price = analysing_price
        self.float_precision_int = float_precision
        self.get_data()

    def get_data(self):
        """Fetches historical data for both stocks."""
        self.stockA = self.request_data(self.stockATicker)
        self.stockB = self.request_data(self.stockBTicker)

    def request_data(self, stock: str) -> np.ndarray:
        """
        Requests historical data for a given stock.

        Args:
            stock: Stock ticker symbol.

        Returns:
            A NumPy array of the stock prices.
        """
        try:
            currentDate = date.today()
            stock_data = yfDownload(
                stock,
                start=date(currentDate.year - 2, currentDate.month, currentDate.day) if self.startDate == "" else self.startDate,
                end=currentDate if self.endDate == "" else self.endDate,
                period=self.period,
                progress=False
            )
            return stock_data[self.analysing_price].values
        except Exception as e:
            raise ValueError(f"Failed to retrieve data for {stock}: {e}")

    def daily_returns(self) -> np.ndarray:
        """Calculates daily returns as percentage change."""
        return np.diff(self.stockA) / self.stockA[:-1] * 100

    def spread(self) -> np.ndarray:
        """Calculates the absolute difference (spread) between the two stock prices."""
        return np.abs(self.stockA - self.stockB)

    def differential(self) -> np.ndarray:
        """Calculates the signed difference between the two stock prices."""
        return self.stockA - self.stockB

    def ratio(self) -> np.ndarray:
        """Calculates the ratio between the two stock prices."""
        return np.divide(self.stockA, self.stockB, out=np.zeros_like(self.stockA), where=self.stockB != 0)

    def mean(self, arr: np.ndarray) -> float:
        """Calculates the mean of the given array."""
        return np.mean(arr)

    def density_curve(self, arr: np.ndarray) -> np.ndarray:
        """
        Calculates the density curve (CDF) for the given array based on a normal distribution.

        Args:
            arr: Input array.

        Returns:
            NumPy array representing the density curve values.
        """
        mean_value = self.mean(arr)
        std_dev = self.standard_deviation(arr)
        return np.array([self.normal_cdf(x, mean_value, std_dev) for x in arr])

    def normal_cdf(self, x: float, mean: float, std_dev: float) -> float:
        """
        Calculates the cumulative distribution function (CDF) for a normal distribution.

        Args:
            x: Value to evaluate the CDF for.
            mean: Mean of the distribution.
            std_dev: Standard deviation of the distribution.

        Returns:
            CDF value.
        """
        z = (x - mean) / (std_dev * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

    def standard_deviation(self, arr: np.ndarray) -> float:
        """Calculates the standard deviation of the given array."""
        return np.std(arr)

    def correlation(self) -> float:
        """
        Calculates the correlation between the two stocks.

        Returns:
            Correlation value rounded to the specified precision.
        """
        return self.float_precision(np.corrcoef(self.stockA, self.stockB)[0, 1])

    def is_data_size_same(self) -> bool:
        """Checks if the data sizes of both stocks are the same."""
        return self.stockA.size == self.stockB.size

    def float_precision(self, x: float) -> float:
        """Rounds a float to the specified precision."""
        return round(x, self.float_precision_int)

    def decision(self):
        """
        Analyzes the trade pair and prints the trade decision and statistics.
        """
        if not self.is_data_size_same():
            raise ValueError(f"Data size mismatch: {self.stockATicker} has {self.stockA.size} data points, "
                             f"and {self.stockBTicker} has {self.stockB.size}.")

        correlation = self.correlation()
        ratio = self.ratio()
        standard_deviation = self.standard_deviation(ratio)
        density_curve = self.density_curve(ratio)
        last_density_curve = self.float_precision(density_curve[-1])

        sd_range, tradeA, tradeB = self.determine_trade_action(last_density_curve)
        
        stats = {
            'Correlation': [self.float_precision(correlation)],
            'Last Ratio': [self.float_precision(ratio[-1])],
            'Mean (Avg)': [self.float_precision(self.mean(ratio))],
            'Standard deviation': [self.float_precision(standard_deviation)],
            'Density Curve': [last_density_curve],
            'SD range': [f'{sd_range[0]} {sd_range[1]}% Chance'],
        }

        trade_setup = {
            'Instrument': [tradeA[0], tradeB[0]],
            'Trade Type': [tradeA[1], tradeB[1]],
            'Target': [tradeA[2], tradeB[2]],
            'Stoploss': [tradeA[3], tradeB[3]],
        }

        print('\nPair Instruments Stats\n')
        print(DataFrame(stats))
        print('\nTrade Setup\n')
        print(DataFrame(trade_setup))

    def determine_trade_action(self, last_density_curve: float):
        """
        Determines the trade action based on the density curve value.

        Args:
            last_density_curve: The last value of the density curve.

        Returns:
            A tuple containing the SD range and trade instructions for both assets.
        """
        if last_density_curve >= 0.997:
            return ('+3', 99.7), [self.stockATicker, 'Short (Sell)', '0.974 or lower', '0.997 or higher'], [self.stockBTicker, 'Long (Buy)', '0.025 or lower', '0.003 or higher']
        elif last_density_curve >= 0.974:
            return ('+3', 99.7), [self.stockATicker, 'Short (Sell)', '0.974 or lower', '0.997 or higher'], [self.stockBTicker, 'Long (Buy)', '0.025 or lower', '0.003 or higher']
        elif last_density_curve >= 0.84:
            return ('+2', 95), [self.stockATicker, 'Short (Sell)', '0.84 or lower', '0.947 or lower'], [self.stockBTicker, 'Long (Buy)', '0.16 or lower', '0.025 or higher']
        elif last_density_curve >= 0.16:
            return ('+-1', 65), [self.stockATicker, 'None', 'None', 'None'], [self.stockBTicker, 'None', 'None', 'None']
        elif last_density_curve >= 0.025:
            return ('-2', 95), [self.stockATicker, 'Long (Buy)', '0.16 or higher', '0.025 or lower'], [self.stockBTicker, 'Short (Sell)', '0.84 or higher', '0.16 or lower']
        else:
            return ('-3', 99.7), [self.stockATicker, 'Long (Buy)', '0.025 or higher', '0.003 or lower'], [self.stockBTicker, 'Short (Sell)', '0.974 or higher', '0.997 or lower']
