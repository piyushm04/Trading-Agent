import os
import argparse
import json
from datetime import datetime
from data_loader import DataLoader
from trading_env import TradingEnvironment
from agent import TradingAgent, create_agent
import matplotlib.pyplot as plt
import numpy as np
from utils import ensure_directories
ensure_directories(['models', 'logs', 'plots', 'results'])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RL Trading Agent')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to trade (default: AAPL)')
    parser.add_argument('--period', type=str, default='2y',
                       help='Data period (default: 2y)')
    parser.add_argument('--algorithm', type=str, default='DQN', choices=['DQN', 'PPO'],
                       help='RL algorithm to use (default: DQN)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps (default: 100000)')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='Initial trading balance (default: 10000)')
    parser.add_argument('--commission_rate', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    parser.add_argument('--eval_freq', type=int, default=5000,
                       help='Evaluation frequency (default: 5000)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save training plots')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    
    return parser.parse_args()


def setup_directories():
    """Create necessary directories."""
    directories = ['models', 'logs', 'plots', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_training_config(args, filename='logs/training_config.json'):
    """Save training configuration."""
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to {filename}")


def plot_training_progress(agent, env, save_path='plots/training_progress.png'):
    """Plot training progress and analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Trading Agent Training Analysis', fontsize=16)
    
    # Portfolio value over time
    portfolio_history = env.portfolio_history
    axes[0, 0].plot(portfolio_history)
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)
    
    # Action distribution
    if hasattr(env, 'action_history') and env.action_history:
        action_counts = np.bincount(env.action_history, minlength=3)
        action_labels = ['Hold', 'Buy', 'Sell']
        axes[0, 1].bar(action_labels, action_counts)
        axes[0, 1].set_title('Action Distribution')
        axes[0, 1].set_ylabel('Count')
        
        # Add percentage labels
        total_actions = sum(action_counts)
        for i, count in enumerate(action_counts):
            percentage = count / total_actions * 100
            axes[0, 1].text(i, count + max(action_counts) * 0.01, 
                           f'{percentage:.1f}%', ha='center')
    
    # Returns distribution
    if len(portfolio_history) > 1:
        returns = np.diff(portfolio_history) / portfolio_history[:-1] * 100
        axes[1, 0].hist(returns, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Returns (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(returns), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(returns):.2f}%')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative returns
    if len(portfolio_history) > 1:
        cumulative_returns = (np.array(portfolio_history) / portfolio_history[0] - 1) * 100
        axes[1, 1].plot(cumulative_returns)
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Cumulative Returns (%)')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved to {save_path}")
    plt.close()


def train_agent(args):
    """Main training function."""
    print("=" * 60)
    print("RL TRADING AGENT TRAINING")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Training timesteps: {args.timesteps:,}")
    print(f"Initial balance: ${args.initial_balance:,.2f}")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Save training configuration
    save_training_config(args)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    loader = DataLoader(symbol=args.symbol, period=args.period, interval='1d', start_date=args.start_date, end_date=args.end_date)
    data = loader.prepare_data()
    
    print(f"   - Total data points: {len(data)}")
    print(f"   - Features: {len(loader.get_feature_columns())}")
    
    # Split data
    train_data, test_data = loader.split_data(train_ratio=args.train_ratio)
    
    # Create environments
    print("\n2. Creating trading environments...")
    train_env = TradingEnvironment(
        train_data, 
        initial_balance=args.initial_balance,
        commission_rate=args.commission_rate
    )
    
    test_env = TradingEnvironment(
        test_data, 
        initial_balance=args.initial_balance,
        commission_rate=args.commission_rate
    )
    
    print(f"   - Training environment: {len(train_data)} steps")
    print(f"   - Testing environment: {len(test_data)} steps")
    print(f"   - State space: {train_env.observation_space.shape}")
    print(f"   - Action space: {train_env.action_space.n}")
    
    # Create agent
    print(f"\n3. Creating {args.algorithm} agent...")
    agent = create_agent(train_env, algorithm=args.algorithm, verbose=args.verbose)
    
    # Train agent
    print(f"\n4. Training agent...")
    start_time = datetime.now()
    
    agent.train(
        total_timesteps=args.timesteps,
        eval_env=test_env,
        eval_freq=args.eval_freq
    )
    
    training_time = datetime.now() - start_time
    print(f"   - Training completed in: {training_time}")
    
    # Evaluate on test set
    print("\n5. Evaluating agent on test set...")
    test_results = agent.evaluate(test_env, n_episodes=5, deterministic=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    
    print(f"Training time: {training_time}")
    print(f"Mean test reward: {test_results['mean_reward']:.2f}")
    print(f"Mean portfolio value: ${test_results['mean_portfolio_value']:.2f}")
    
    if 'portfolio_stats' in test_results:
        stats = test_results['portfolio_stats']
        print(f"\nPortfolio Statistics:")
        print(f"  Total return: {stats.get('total_return', 0):.2%}")
        print(f"  Sharpe ratio: {stats.get('sharpe_ratio', 0):.2f}")
        print(f"  Max drawdown: {stats.get('max_drawdown', 0):.2%}")
        print(f"  Win rate: {stats.get('win_rate', 0):.2%}")
        print(f"  Total trades: {stats.get('total_trades', 0)}")
        print(f"  Volatility: {stats.get('volatility', 0):.2%}")
    
    # Action analysis
    print(f"\n6. Analyzing agent behavior...")
    action_analysis = agent.analyze_actions(test_env, n_steps=100)
    print(f"Action distribution:")
    for action, prob in action_analysis['action_distribution'].items():
        print(f"  {action}: {prob:.2%}")
    
    # Save results
    results_file = f'results/training_results_{args.symbol}_{args.algorithm}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_data = {
        'config': vars(args),
        'training_time': str(training_time),
        'test_results': {k: v for k, v in test_results.items() if k != 'portfolio_stats'},
        'portfolio_stats': test_results.get('portfolio_stats', {}),
        'action_analysis': action_analysis
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    if args.save_plots:
        print(f"\n7. Generating plots...")
        plot_training_progress(agent, test_env, 
                             f'plots/training_{args.symbol}_{args.algorithm}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return agent, test_results


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        agent, results = train_agent(args)
        
        # Optional: Run additional analysis
        print(f"\nModel saved in: models/")
        print(f"Logs saved in: logs/")
        print(f"Results saved in: results/")
        
        if args.save_plots:
            print(f"Plots saved in: plots/")
        
        print(f"\nTo evaluate the trained model, run:")
        print(f"python evaluate.py --model models/final_model --symbol {args.symbol}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
