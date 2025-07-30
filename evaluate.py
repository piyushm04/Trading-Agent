import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from data_loader import DataLoader
from trading_env import TradingEnvironment
from agent import TradingAgent, create_agent
from utils import ensure_directories
ensure_directories(['models', 'logs', 'plots', 'results'])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate RL Trading Agent')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to evaluate (default: AAPL)')
    parser.add_argument('--period', type=str, default='1y',
                       help='Evaluation period (default: 1y)')
    parser.add_argument('--algorithm', type=str, default='DQN', choices=['DQN', 'PPO'],
                       help='RL algorithm used (default: DQN)')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                       help='Initial trading balance (default: 10000)')
    parser.add_argument('--commission_rate', type=float, default=0.001,
                       help='Commission rate per trade (default: 0.001)')
    parser.add_argument('--n_episodes', type=int, default=1,
                       help='Number of evaluation episodes (default: 1)')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--interactive_plots', action='store_true',
                       help='Generate interactive Plotly charts')
    parser.add_argument('--benchmark', type=str, default='SPY',
                       help='Benchmark symbol for comparison (default: SPY)')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    
    return parser.parse_args()


def calculate_performance_metrics(portfolio_values, benchmark_values=None):
    """Calculate comprehensive performance metrics."""
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Basic metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    # Win rate
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'final_value': portfolio_values[-1],
        'total_days': len(portfolio_values)
    }
    
    # Benchmark comparison
    if benchmark_values is not None:
        benchmark_values = np.array(benchmark_values)
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        benchmark_total_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
        
        # Beta calculation
        if len(returns) == len(benchmark_returns):
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha calculation
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_total_return - risk_free_rate))
            
            metrics.update({
                'benchmark_return': benchmark_total_return,
                'alpha': alpha,
                'beta': beta,
                'excess_return': total_return - benchmark_total_return
            })
    
    return metrics


def create_performance_plots(env, agent, benchmark_data=None, save_path='plots/evaluation.png'):
    """Create comprehensive performance visualization."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('RL Trading Agent Performance Analysis', fontsize=16)
    
    # Portfolio value over time
    portfolio_history = env.portfolio_history
    dates = range(len(portfolio_history))
    
    axes[0, 0].plot(dates, portfolio_history, label='Portfolio', linewidth=2)
    if benchmark_data is not None:
        # Normalize benchmark to same starting value
        normalized_benchmark = benchmark_data * (portfolio_history[0] / benchmark_data[0])
        axes[0, 0].plot(dates[:len(normalized_benchmark)], normalized_benchmark, 
                       label='Benchmark', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative returns
    returns = (np.array(portfolio_history) / portfolio_history[0] - 1) * 100
    axes[0, 1].plot(dates, returns, linewidth=2, color='green')
    axes[0, 1].set_title('Cumulative Returns')
    axes[0, 1].set_xlabel('Days')
    axes[0, 1].set_ylabel('Cumulative Returns (%)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Drawdown
    peak = np.maximum.accumulate(portfolio_history)
    drawdown = (peak - portfolio_history) / peak * 100
    axes[1, 0].fill_between(dates, drawdown, 0, alpha=0.3, color='red')
    axes[1, 0].plot(dates, drawdown, color='red', linewidth=1)
    axes[1, 0].set_title('Drawdown')
    axes[1, 0].set_xlabel('Days')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Action distribution
    if hasattr(env, 'action_history') and env.action_history:
        action_counts = np.bincount(env.action_history, minlength=3)
        action_labels = ['Hold', 'Buy', 'Sell']
        colors = ['gray', 'green', 'red']
        
        wedges, texts, autotexts = axes[1, 1].pie(action_counts, labels=action_labels, 
                                                 colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Action Distribution')
    
    # Daily returns distribution
    if len(portfolio_history) > 1:
        daily_returns = np.diff(portfolio_history) / portfolio_history[:-1] * 100
        axes[2, 0].hist(daily_returns, bins=30, alpha=0.7, edgecolor='black', density=True)
        axes[2, 0].axvline(np.mean(daily_returns), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(daily_returns):.2f}%')
        axes[2, 0].axvline(0, color='black', linestyle='-', alpha=0.3)
        axes[2, 0].set_title('Daily Returns Distribution')
        axes[2, 0].set_xlabel('Daily Returns (%)')
        axes[2, 0].set_ylabel('Density')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # Trade analysis
    if hasattr(env, 'trade_history') and env.trade_history:
        trades_df = pd.DataFrame(env.trade_history)
        if 'pnl' in trades_df.columns:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            axes[2, 1].bar(['Profitable', 'Losing'], 
                          [len(profitable_trades), len(losing_trades)],
                          color=['green', 'red'], alpha=0.7)
            axes[2, 1].set_title('Trade Outcomes')
            axes[2, 1].set_ylabel('Number of Trades')
            
            # Add profit/loss amounts as text
            if len(profitable_trades) > 0:
                profit_sum = profitable_trades['pnl'].sum()
                axes[2, 1].text(0, len(profitable_trades), f'${profit_sum:.0f}', 
                               ha='center', va='bottom')
            if len(losing_trades) > 0:
                loss_sum = losing_trades['pnl'].sum()
                axes[2, 1].text(1, len(losing_trades), f'${loss_sum:.0f}', 
                               ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to {save_path}")
    plt.close()


def create_interactive_plots(env, price_data, save_path='plots/interactive_evaluation.html'):
    """Create interactive Plotly charts."""
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Portfolio Value & Stock Price', 'Actions', 'Cumulative Returns', 'Drawdown'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
    )
    
    dates = pd.date_range(start='2023-01-01', periods=len(env.portfolio_history), freq='D')
    
    # Portfolio value and stock price
    fig.add_trace(
        go.Scatter(x=dates, y=env.portfolio_history, name='Portfolio Value', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    if len(price_data) >= len(dates):
        fig.add_trace(
            go.Scatter(x=dates, y=price_data[:len(dates)], name='Stock Price',
                      line=dict(color='orange', width=1), yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
    
    # Actions
    if hasattr(env, 'action_history') and env.action_history:
        action_names = ['Hold', 'Buy', 'Sell']
        action_colors = ['gray', 'green', 'red']
        
        for i, action in enumerate(env.action_history):
            if action > 0:  # Only show Buy/Sell actions
                fig.add_trace(
                    go.Scatter(x=[dates[i]], y=[env.portfolio_history[i]], 
                             mode='markers', name=action_names[action],
                             marker=dict(color=action_colors[action], size=8),
                             showlegend=False),
                    row=2, col=1
                )
    
    # Cumulative returns
    returns = (np.array(env.portfolio_history) / env.portfolio_history[0] - 1) * 100
    fig.add_trace(
        go.Scatter(x=dates, y=returns, name='Cumulative Returns (%)',
                  line=dict(color='green', width=2), fill='tonexty'),
        row=3, col=1
    )
    
    # Drawdown
    peak = np.maximum.accumulate(env.portfolio_history)
    drawdown = (peak - env.portfolio_history) / peak * 100
    fig.add_trace(
        go.Scatter(x=dates, y=-drawdown, name='Drawdown (%)',
                  line=dict(color='red', width=2), fill='tozeroy'),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='RL Trading Agent Interactive Performance Dashboard',
        height=1000,
        showlegend=True
    )
    
    # Save interactive plot
    fig.write_html(save_path)
    print(f"Interactive plots saved to {save_path}")


def evaluate_agent(args):
    """Main evaluation function."""
    print("=" * 60)
    print("RL TRADING AGENT EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading evaluation data...")
    loader = DataLoader(symbol=args.symbol, period=args.period, interval='1d', start_date=args.start_date, end_date=args.end_date)
    data = loader.prepare_data()
    print(f"   - Data points: {len(data)}")
    
    # Load benchmark data if specified
    benchmark_data = None
    if args.benchmark:
        try:
            benchmark_loader = DataLoader(symbol=args.benchmark, period=args.period, interval='1d', start_date=args.start_date, end_date=args.end_date)
            benchmark_raw = benchmark_loader.prepare_data()
            benchmark_data = benchmark_raw['close'].values
            print(f"   - Benchmark data loaded: {args.benchmark}")
        except Exception as e:
            print(f"   - Warning: Could not load benchmark data: {e}")
    
    # Create environment
    print("\n2. Creating evaluation environment...")
    env = TradingEnvironment(
        data, 
        initial_balance=args.initial_balance,
        commission_rate=args.commission_rate
    )
    
    # Create and load agent
    print(f"\n3. Loading trained {args.algorithm} agent...")
    agent = create_agent(env, algorithm=args.algorithm, verbose=0)
    agent.load_model(args.model)
    
    # Run evaluation
    print(f"\n4. Running evaluation ({args.n_episodes} episodes)...")
    results = agent.evaluate(env, n_episodes=args.n_episodes, deterministic=True)
    
    # Calculate detailed metrics
    print("\n5. Calculating performance metrics...")
    benchmark_values = None
    if benchmark_data is not None and len(benchmark_data) >= len(env.portfolio_history):
        # Normalize benchmark to same starting value
        benchmark_values = benchmark_data[:len(env.portfolio_history)]
        benchmark_values = benchmark_values * (env.portfolio_history[0] / benchmark_values[0])
    
    metrics = calculate_performance_metrics(env.portfolio_history, benchmark_values)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    if 'benchmark_return' in metrics:
        print(f"\nBenchmark Comparison ({args.benchmark}):")
        print(f"Benchmark Return: {metrics['benchmark_return']:.2%}")
        print(f"Excess Return: {metrics['excess_return']:.2%}")
        print(f"Alpha: {metrics['alpha']:.2%}")
        print(f"Beta: {metrics['beta']:.2f}")
    
    # Trading statistics
    if hasattr(env, 'trade_history') and env.trade_history:
        trades_df = pd.DataFrame(env.trade_history)
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {len(trades_df)}")
        if 'pnl' in trades_df.columns:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            print(f"Profitable Trades: {len(profitable_trades)}")
            print(f"Average Profit: ${profitable_trades['pnl'].mean():.2f}" if len(profitable_trades) > 0 else "Average Profit: $0.00")
            
            losing_trades = trades_df[trades_df['pnl'] < 0]
            print(f"Losing Trades: {len(losing_trades)}")
            print(f"Average Loss: ${losing_trades['pnl'].mean():.2f}" if len(losing_trades) > 0 else "Average Loss: $0.00")
    
    # Action analysis
    action_analysis = agent.analyze_actions(env, n_steps=len(data))
    print(f"\nAction Distribution:")
    for action, prob in action_analysis['action_distribution'].items():
        print(f"  {action}: {prob:.2%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_file = f'results/evaluation_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    evaluation_data = {
        'config': vars(args),
        'metrics': metrics,
        'action_analysis': action_analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    if args.save_plots:
        print(f"\n6. Generating plots...")
        os.makedirs('plots', exist_ok=True)
        
        plot_path = f'plots/evaluation_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        create_performance_plots(env, agent, benchmark_data, plot_path)
        
        if args.interactive_plots:
            interactive_path = f'plots/interactive_{args.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            create_interactive_plots(env, data['close'].values, interactive_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)
    
    return results, metrics


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        results, metrics = evaluate_agent(args)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved in: results/")
        
        if args.save_plots:
            print(f"Plots saved in: plots/")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
