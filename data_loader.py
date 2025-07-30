import yfinance as yf
import pandas as pd
import numpy as np
import ta
from typing import Optional, List, Tuple
from datetime import datetime, timedelta


class DataLoader:
    """
    Data loader for fetching and preprocessing stock market data.
    """
    
    def __init__(self, symbol: str = "AAPL", period: str = "2y", interval: str = "1d", start_date: str = None, end_date: str = None):
        """
        Initialize data loader.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date for fetching data (YYYY-MM-DD)
            end_date: End date for fetching data (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.processed_data = None
    
        def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Returns:
            Raw OHLCV data
        """
        # Fetch data using yfinance
        try:
            print(f"Fetching data for {self.symbol}...")
            ticker = yf.Ticker(self.symbol)
            # Use history() as it can be more robust
            if self.start_date and self.end_date:
                data = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            else:
                data = ticker.history(period=self.period, auto_adjust=True)

            if data.empty:
                # Attempt to download again as a fallback
                print("history() returned no data, trying download()...")
                data = yf.download(self.symbol, start=self.start_date, end=self.end_date, ignore_tz=True)
                if data.empty:
                    raise ValueError(f"No data found for symbol {self.symbol} with either history() or download().")
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            print(f"Fetched {len(data)} data points for {self.symbol}")
            return data
            
        except Exception as e:
            print(f"An error occurred while fetching data for {self.symbol}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            data: OHLCV data
            
        Returns:
            Data with technical indicators
        """
        df = data.copy()
        
        # Price-based indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Volume-based indicator (using pandas, not ta.volume)
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        
        # Support and resistance levels (simplified)
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        return df
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize the data for better RL training.
        
        Args:
            data: Data to normalize
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            Normalized data
        """
        df = data.copy()
        
        # Columns to normalize (exclude date and categorical columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        elif method == 'zscore':
            df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        
        return df
    
    def prepare_data(self, normalize: bool = True, lookback_window: int = 10) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            normalize: Whether to normalize the data
            lookback_window: Number of previous days to include as features
            
        Returns:
            Processed data ready for RL training
        """
        if self.data is None:
            self.data = self.fetch_data()
        
        # Add technical indicators
        processed = self.add_technical_indicators(self.data)
        
        # Drop NaN values
        processed = processed.dropna()
        
        # Add lookback features
        for i in range(1, lookback_window + 1):
            for col in ['close', 'volume', 'rsi', 'macd']:
                if col in processed.columns:
                    processed[f'{col}_lag_{i}'] = processed[col].shift(i)
        
        # Drop NaN values again after adding lags
        processed = processed.dropna()
        
        # Normalize if requested
        if normalize:
            processed = self.normalize_data(processed)
        
        self.processed_data = processed
        print(f"Processed data shape: {processed.shape}")
        return processed
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns for the RL agent.
        
        Returns:
            List of feature column names
        """
        if self.processed_data is None:
            raise ValueError("Data not processed yet. Call prepare_data() first.")
        
        # Exclude basic OHLCV columns, keep technical indicators and lags
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        return feature_cols
    
    def split_data(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            train_ratio: Ratio of data to use for training
            
        Returns:
            Training and testing dataframes
        """
        if self.processed_data is None:
            raise ValueError("Data not processed yet. Call prepare_data() first.")
        
        split_idx = int(len(self.processed_data) * train_ratio)
        train_data = self.processed_data.iloc[:split_idx]
        test_data = self.processed_data.iloc[split_idx:]
        
        print(f"Training data: {len(train_data)} samples")
        print(f"Testing data: {len(test_data)} samples")
        
        return train_data, test_data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader(symbol="AAPL", period="2y", interval="1d")
    data = loader.prepare_data()
    print("\nFeature columns:")
    print(loader.get_feature_columns())
    
    train_data, test_data = loader.split_data()
    print(f"\nData preparation complete!")
