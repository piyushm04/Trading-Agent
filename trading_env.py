import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from data_loader import DataLoader


class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    
    Action Space: 0 = Hold, 1 = Buy, 2 = Sell
    State Space: Market features + Portfolio state
    Reward: Profit/Loss - Commission - Drawdown Penalty
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,
        max_position: float = 1.0,
        lookback_window: int = 10,
        reward_scaling: float = 1000.0
    ):
        """
        Initialize trading environment.
        
        Args:
            data: Market data with features
            initial_balance: Starting cash balance
            commission_rate: Commission rate per trade (0.001 = 0.1%)
            max_position: Maximum position size (1.0 = 100% of balance)
            lookback_window: Number of previous steps to include in state
            reward_scaling: Scaling factor for rewards
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        
        # Get feature columns (exclude price columns)
        self.feature_columns = [col for col in data.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # State space: market features + portfolio state
        market_features = len(self.feature_columns) * lookback_window
        portfolio_features = 4  # cash, position, portfolio_value, unrealized_pnl
        state_size = market_features + portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # Number of shares held
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_portfolio_value = self.initial_balance
        
        # Trading history
        self.trade_history = []
        self.portfolio_history = [self.initial_balance]
        self.action_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            observation, reward, done, info
        """
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update step
        self.current_step += 1
        self.action_history.append(action)
        
        # Calculate portfolio value and update history
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append(current_portfolio_value)
        
        # Update max portfolio value for drawdown calculation
        if current_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = current_portfolio_value
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                current_portfolio_value <= self.initial_balance * 0.1)  # Stop if 90% loss
        
        # Prepare info dictionary
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'current_price': self._get_current_price(),
            'action': action
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """
        Execute trading action and calculate reward.
        
        Args:
            action: Trading action
            
        Returns:
            Reward for the action
        """
        current_price = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value()
        
        reward = 0.0
        commission = 0.0
        
        if action == 1:  # Buy
            if self.position == 0 and self.balance > 0:
                # Calculate maximum shares we can buy
                max_shares = (self.balance * self.max_position) / current_price
                shares_to_buy = max_shares * 0.95  # Leave some buffer
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    commission = cost * self.commission_rate
                    
                    if self.balance >= cost + commission:
                        self.position = shares_to_buy
                        self.balance -= (cost + commission)
                        self.entry_price = current_price
                        self.total_trades += 1
                        
                        self.trade_history.append({
                            'step': self.current_step,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'commission': commission
                        })
                        
                        # Small positive reward for taking action
                        reward = 0.01
        
        elif action == 2:  # Sell
            if self.position > 0:
                revenue = self.position * current_price
                commission = revenue * self.commission_rate
                net_revenue = revenue - commission
                
                # Calculate profit/loss
                cost_basis = self.position * self.entry_price
                pnl = net_revenue - cost_basis
                
                self.balance += net_revenue
                
                # Track winning trades
                if pnl > 0:
                    self.winning_trades += 1
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': self.position,
                    'commission': commission,
                    'pnl': pnl
                })
                
                # Reward based on profit/loss
                reward = pnl / self.initial_balance * self.reward_scaling
                
                self.position = 0.0
                self.entry_price = 0.0
                self.total_trades += 1
        
        # Calculate portfolio value change
        current_portfolio_value = self._get_portfolio_value()
        portfolio_change = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Add portfolio performance to reward
        reward += portfolio_change * self.reward_scaling
        
        # Penalty for drawdown
        drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value
        if drawdown > 0.1:  # Penalty for >10% drawdown
            reward -= drawdown * 10
        
        # Small penalty for commission
        reward -= commission / self.initial_balance * self.reward_scaling
        
        return reward
    
    def _get_current_price(self) -> float:
        """Get current stock price."""
        return self.data.iloc[self.current_step]['close']
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        if self.position > 0:
            stock_value = self.position * self._get_current_price()
            return self.balance + stock_value
        return self.balance
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector containing market features and portfolio state
        """
        # Market features (lookback window)
        market_features = []
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        for i in range(start_idx, end_idx):
            if i < len(self.data):
                row_features = self.data.iloc[i][self.feature_columns].values
            else:
                row_features = np.zeros(len(self.feature_columns))
            market_features.extend(row_features)
        
        # Pad if necessary
        while len(market_features) < len(self.feature_columns) * self.lookback_window:
            market_features = [0.0] + market_features
        
        # Portfolio features
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized cash
            self.position * current_price / self.initial_balance,  # Normalized position value
            portfolio_value / self.initial_balance,  # Normalized portfolio value
            (portfolio_value - self.initial_balance) / self.initial_balance  # Normalized unrealized PnL
        ]
        
        # Combine features
        observation = np.array(market_features + portfolio_features, dtype=np.float32)
        
        return observation
    
    def get_portfolio_stats(self) -> Dict:
        """
        Calculate portfolio performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.portfolio_history) < 2:
            return {}
        
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Sharpe ratio (assuming 252 trading days per year)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        stats = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'volatility': np.std(returns) if len(returns) > 1 else 0.0
        }
        
        return stats
    
    def render(self, mode='human'):
        """Render environment state."""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.2f} shares")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Total Return: {(portfolio_value - self.initial_balance) / self.initial_balance * 100:.2f}%")
        print("-" * 50)


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load and prepare data
    loader = DataLoader(symbol="AAPL", period="1y")
    data = loader.prepare_data()
    
    # Create environment
    env = TradingEnvironment(data, initial_balance=10000)
    
    # Test random actions
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if done:
            break
    
    # Print final stats
    stats = env.get_portfolio_stats()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
