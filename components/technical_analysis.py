# components/technical_analysis.py
import pandas as pd
import numpy as np
import logging
from alpaca_trade_api.rest import TimeFrame
from datetime import datetime, timedelta
import yfinance as yf  # Import yfinance for backup data source

class TechnicalAnalyzer:
    def __init__(self, api):
        """Initialize the technical analyzer."""
        self.api = api
        self.logger = logging.getLogger('TechnicalAnalyzer')
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            self.logger.warning(f"Not enough data to calculate RSI. Need at least {period + 1} data points.")
            return None  # Return None instead of a default value
            
        # Ensure prices is 1D
        if len(prices.shape) > 1:
            self.logger.info(f"Flattening prices array from shape {prices.shape}")
            prices = prices.ravel()
            
        deltas = np.diff(prices)
        
        # Check if deltas array is empty
        if deltas.size == 0:
            self.logger.warning("No price changes detected for RSI calculation")
            return None
            
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else float('inf')
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else float('inf')
            rsi[i] = 100. - 100./(1. + rs)
        
        return rsi[-1]
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate Moving Average Convergence Divergence."""
        if len(prices) < slow_period + signal_period:
            self.logger.warning(f"Not enough data to calculate MACD. Need at least {slow_period + signal_period} data points.")
            return None  # Return None instead of default values
            
        # Ensure prices is 1D
        if len(prices.shape) > 1:
            self.logger.info(f"Flattening prices array from shape {prices.shape}")
            prices = prices.ravel()
            
        # Calculate EMAs
        ema_fast = pd.Series(prices).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line.iloc[-1],
            'signal_line': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def get_historical_data_yfinance(self, ticker, start_date, end_date):
        """Get historical data using yfinance as a fallback."""
        try:
            self.logger.info(f"Attempting to fetch data for {ticker} from yfinance")
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if data.size > 0:  # Using size instead of direct boolean evaluation
                self.logger.info(f"Successfully retrieved {len(data)} bars from yfinance for {ticker}")
                return data
            else:
                self.logger.warning(f"yfinance returned empty data for {ticker}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching data from yfinance for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, ticker, indicators_config):
        """Calculate technical indicators for a ticker."""
        results = {}
        
        try:
            # Increase the default max_period to ensure enough data for calculations
            max_period = 250  # Increase from 50 to 250 as recommended by StockCharts
            
            for indicator in indicators_config:
                if indicator['type'] == 'RSI':
                    # Double the period for more accurate RSI calculation
                    max_period = max(max_period, indicator['period'] * 3)
                elif indicator['type'] == 'MACD':
                    # For MACD, use at least twice the sum of slow period and signal period
                    max_period = max(max_period, (indicator['slow_period'] + indicator['signal_period']) * 3)
            
            # Calculate date range - go back far enough to get sufficient data
            end_date = datetime.now()
            # Go back 365 days to ensure we have enough historical data
            start_date = end_date - timedelta(days=365)
            
            # Format dates as simple YYYY-MM-DD strings
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            self.logger.info(f"Fetching historical data for {ticker} from {start_date_str} to {end_date_str}")
            
            # Try to get data from Alpaca first
            bars = pd.DataFrame()
            try:
                bars = self.api.get_bars(
                    ticker, 
                    TimeFrame.Day,
                    start=start_date_str,
                    end=end_date_str,
                    adjustment='raw'  # Get raw (unadjusted) data
                ).df
                
                if bars.size == 0:  # Using size instead of empty
                    self.logger.warning(f"No historical data available from Alpaca for {ticker}, trying yfinance")
                    bars = self.get_historical_data_yfinance(ticker, start_date_str, end_date_str)
                else:
                    self.logger.info(f"Retrieved {len(bars)} historical bars from Alpaca for {ticker}")
            
            except Exception as e:
                self.logger.warning(f"Error fetching data from Alpaca for {ticker}: {e}. Trying yfinance instead.")
                bars = self.get_historical_data_yfinance(ticker, start_date_str, end_date_str)
            
            # If we still don't have data, return empty results
            if bars.size == 0:  # Using size instead of empty
                self.logger.warning(f"Could not retrieve any historical data for {ticker} from either source")
                return results
            
            # Check if we have enough data
            min_required = 0
            for indicator in indicators_config:
                if indicator['type'] == 'RSI':
                    min_required = max(min_required, indicator['period'] + 1)
                elif indicator['type'] == 'MACD':
                    min_required = max(min_required, indicator['slow_period'] + indicator['signal_period'])
            
            if len(bars) < min_required:
                self.logger.warning(f"Insufficient data for {ticker}. Got {len(bars)} bars, need at least {min_required}.")
                return results
            
            # Extract closing prices - handle different column names between Alpaca and yfinance
            closes = None
            if 'close' in bars.columns:
                closes = bars['close'].values
            elif 'Close' in bars.columns:  # yfinance uses capitalized column names
                closes = bars['Close'].values
            else:
                self.logger.error(f"Could not find closing price column in data for {ticker}")
                return results
                
            # Check if closes array is empty
            if closes is None or closes.size == 0:
                self.logger.warning(f"No closing prices found for {ticker}")
                return results
                
            # Ensure closes is 1D
            if len(closes.shape) > 1:
                self.logger.info(f"Flattening closes array from shape {closes.shape}")
                closes = closes.ravel()
            
            # Calculate indicators
            for indicator in indicators_config:
                if indicator['type'] == 'RSI':
                    rsi = self.calculate_rsi(closes, period=indicator['period'])
                    if rsi is not None:
                        results['RSI'] = {
                            'value': rsi,
                            'overbought': indicator['overbought'],
                            'oversold': indicator['oversold']
                        }
                elif indicator['type'] == 'MACD':
                    macd = self.calculate_macd(
                        closes, 
                        fast_period=indicator['fast_period'],
                        slow_period=indicator['slow_period'],
                        signal_period=indicator['signal_period']
                    )
                    if macd is not None:
                        results['MACD'] = macd
            
            if len(results) > 0:
                self.logger.info(f"Successfully calculated technical indicators for {ticker}")
            else:
                self.logger.warning(f"Failed to calculate any technical indicators for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {ticker}: {e}")
        
        return results
