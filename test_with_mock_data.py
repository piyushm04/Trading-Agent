import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch, MagicMock

# Import your modules
from data_loader import DataLoader
from trading_env import TradingEnvironment
from agent import TradingAgent

class TestTradingAgentWithMockData(unittest.TestCase):
    """Test the RL trading agent with mock data."""
    
    @classmethod
    def setUpClass(cls):
        """Create test data before running any tests."""
        # Generate synthetic price data
        np.random.seed(42)
        n_days = 365
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate random walk for prices
        price_changes = 0.01 * np.random.randn(n_days)
        price_changes[0] = 150  # Start price
        prices = np.cumsum(price_changes)
        prices = np.maximum(10, prices)  # Ensure prices stay positive
        
        # Create DataFrame with OHLCV data
        cls.mock_data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + 0.01 * np.random.rand(n_days)),
            'low': prices * (1 - 0.01 * np.random.rand(n_days)),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, size=n_days)
        }, index=pd.DatetimeIndex(dates))
        
        # Add some technical indicators
        cls.mock_data['sma_10'] = cls.mock_data['close'].rolling(10).mean()
        cls.mock_data['rsi'] = 50 + 20 * np.random.randn(n_days)  # Random RSI around 50
        
    def test_data_loader(self):
        """Test DataLoader with mock data."""
        # Patch yfinance to return our mock data
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = self.mock_data
            mock_ticker.return_value = mock_instance
            
            # Test DataLoader
            loader = DataLoader(symbol="MOCK", period="1y")
            data = loader.fetch_data()
            
            # Verify data was loaded and processed
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            print("✅ DataLoader test passed with mock data")
    
    def test_training_loop(self):
        """Test the training loop with mock data."""
        # Create mock environment with correct parameter name
        env = TradingEnvironment(
            data=self.mock_data,
            initial_balance=10000,
            commission_rate=0.001,
            lookback_window=10
        )
        
        # Create agent
        agent = TradingAgent(env=env, algorithm="DQN")
        
        # Test training with more detailed error reporting
        try:
            print("Starting training...")
            print(f"Agent type: {type(agent).__name__}")
            print(f"Agent train method: {agent.train}")
            print(f"Agent train method signature: {agent.train.__code__.co_varnames[:agent.train.__code__.co_argcount]}")
            
            # Try calling with just required parameters
            agent.train(total_timesteps=1000)
            print("✅ Training test passed with mock data")
        except Exception as e:
            import traceback
            error_msg = f"Training failed with mock data: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
            self.fail(error_msg)

if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
