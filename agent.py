import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Union
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from trading_env import TradingEnvironment


class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.
    """
    
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.episode_rewards.extend(self.locals['rewards'])
        
        # Evaluate model periodically
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_model()
        
        return True
    
    def _evaluate_model(self):
        """Evaluate the model on the evaluation environment."""
        obs = self.eval_env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(100):  # Run for 100 steps
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        if self.verbose > 0:
            print(f"Eval Episode - Reward: {episode_reward:.2f}, Length: {episode_length}")
        
        # Save best model
        if episode_reward > self.best_mean_reward:
            self.best_mean_reward = episode_reward
            if hasattr(self.model, 'save'):
                self.model.save("models/best_model")


class TradingAgent:
    """
    Reinforcement Learning Trading Agent using Stable-Baselines3.
    """
    
    def __init__(
        self,
        env: TradingEnvironment,
        algorithm: str = "DQN",
        model_params: Optional[Dict] = None,
        verbose: int = 1
    ):
        """
        Initialize trading agent.
        
        Args:
            env: Trading environment
            algorithm: RL algorithm ('DQN' or 'PPO')
            model_params: Parameters for the RL model
            verbose: Verbosity level
        """
        self.env = env
        self.algorithm = algorithm
        self.verbose = verbose
        self.model = None
        self.training_history = []
        
        # Default model parameters
        default_params = {
            'DQN': {
                'learning_rate': 1e-4,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 32,
                'tau': 1.0,
                'gamma': 0.99,
                'train_freq': 4,
                'gradient_steps': 1,
                'target_update_interval': 1000,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'policy_kwargs': dict(net_arch=[256, 256])
            },
            'PPO': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'policy_kwargs': dict(net_arch=[256, 256])
            }
        }
        
        self.model_params = model_params or default_params.get(algorithm, {})
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Initialize model
        self._create_model()
    
    def _create_model(self):
        """Create the RL model based on the specified algorithm."""
        if self.algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                self.env,
                verbose=self.verbose,
                **self.model_params
            )
        elif self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=self.verbose,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[TradingEnvironment] = None,
        eval_freq: int = 5000,
        save_freq: int = 10000
    ):
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation during training
            eval_freq: Frequency of evaluation
            save_freq: Frequency of model saving
        """
        print(f"Starting training with {self.algorithm} for {total_timesteps} timesteps...")
        
        # Setup callback
        callback = None
        if eval_env is not None:
            callback = TradingCallback(eval_env, eval_freq=eval_freq, verbose=self.verbose)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        self.save_model("models/final_model")
        print("Training completed!")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action for given observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Predicted action
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def evaluate(
        self,
        env: TradingEnvironment,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_values.append(info['portfolio_value'])
            
            if self.verbose > 0:
                print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                      f"Portfolio=${info['portfolio_value']:.2f}")
        
        # Calculate statistics
        stats = env.get_portfolio_stats()
        eval_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values),
            'portfolio_stats': stats
        }
        
        return eval_results
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        if self.algorithm == "DQN":
            self.model = DQN.load(path, env=self.env)
        elif self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f"Model loaded from {path}")
    
    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """
        Get Q-values for DQN (for analysis).
        
        Args:
            observation: Current state observation
            
        Returns:
            Q-values for each action
        """
        if self.algorithm != "DQN":
            raise ValueError("Q-values only available for DQN")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Get Q-values from the model
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model.q_net(obs_tensor)
        
        return q_values.numpy().flatten()
    
    def analyze_actions(self, env: TradingEnvironment, n_steps: int = 100) -> Dict:
        """
        Analyze agent's action distribution.
        
        Args:
            env: Environment for analysis
            n_steps: Number of steps to analyze
            
        Returns:
            Action analysis results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        obs = env.reset()
        actions = []
        q_values_history = []
        
        for _ in range(n_steps):
            action = self.predict(obs, deterministic=True)
            actions.append(action)
            
            # Get Q-values if DQN
            if self.algorithm == "DQN":
                try:
                    q_values = self.get_q_values(obs)
                    q_values_history.append(q_values)
                except:
                    pass
            
            obs, _, done, _ = env.step(action)
            
            if done:
                obs = env.reset()
        
        # Calculate action distribution
        action_counts = np.bincount(actions, minlength=3)
        action_distribution = action_counts / len(actions)
        
        analysis = {
            'action_distribution': {
                'hold': action_distribution[0],
                'buy': action_distribution[1],
                'sell': action_distribution[2]
            },
            'total_actions': len(actions),
            'q_values_history': q_values_history if q_values_history else None
        }
        
        return analysis


def create_agent(
    train_env: TradingEnvironment,
    algorithm: str = "DQN",
    **kwargs
) -> TradingAgent:
    """
    Factory function to create a trading agent.
    
    Args:
        train_env: Training environment
        algorithm: RL algorithm to use
        **kwargs: Additional parameters for the agent
        
    Returns:
        Configured trading agent
    """
    return TradingAgent(train_env, algorithm=algorithm, **kwargs)


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(symbol="AAPL", period="2y")
    data = loader.prepare_data()
    train_data, test_data = loader.split_data()
    
    # Create environments
    train_env = TradingEnvironment(train_data, initial_balance=10000)
    test_env = TradingEnvironment(test_data, initial_balance=10000)
    
    # Create and train agent
    agent = create_agent(train_env, algorithm="DQN")
    
    print("Training agent...")
    agent.train(total_timesteps=10000, eval_env=test_env)
    
    print("\nEvaluating agent...")
    results = agent.evaluate(test_env, n_episodes=5)
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        if key != 'portfolio_stats':
            print(f"{key}: {value}")
    
    print("\nPortfolio Statistics:")
    for key, value in results['portfolio_stats'].items():
        print(f"{key}: {value}")
    
    # Analyze actions
    print("\nAction Analysis:")
    action_analysis = agent.analyze_actions(test_env, n_steps=50)
    print(f"Action Distribution: {action_analysis['action_distribution']}")
