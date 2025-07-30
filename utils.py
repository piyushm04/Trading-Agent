import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

# --- TECHNICAL INDICATOR WRAPPERS ---
def add_ta_indicators(df):
    """Add a comprehensive set of TA indicators using the ta library."""
    df = dropna(df)
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    return df

def add_custom_indicators(df):
    """Add custom indicators not covered by ta library."""
    df['Price_Change'] = df['Close'].pct_change().fillna(0)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    return df

# --- FEATURE IMPORTANCE & CORRELATION ---
def plot_correlation_matrix(df, output_path=None):
    """Plot and optionally save the correlation matrix for features."""
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    if output_path:
        plt.savefig(output_path)
    plt.show()

def compute_feature_importance(X, y, output_path=None):
    """Compute and plot feature importance using RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()
    return dict(zip(X.columns, importances))

# --- PLOTTING UTILITIES ---
def plot_equity_curve(portfolio_values, output_path=None):
    """Plot the equity curve (portfolio value over time)."""
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.title('Equity Curve')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_trades(prices, trades, output_path=None):
    """Plot buy/sell points on price series."""
    plt.figure(figsize=(14, 7))
    plt.plot(prices, label='Price')
    buys = [t for t in trades if t['type'] == 'buy']
    sells = [t for t in trades if t['type'] == 'sell']
    if buys:
        plt.scatter([b['step'] for b in buys], [b['price'] for b in buys], color='g', marker='^', label='Buy', alpha=0.7)
    if sells:
        plt.scatter([s['step'] for s in sells], [s['price'] for s in sells], color='r', marker='v', label='Sell', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.title('Trade Actions')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_action_distribution(actions, output_path=None):
    """Plot the distribution of actions taken by the agent."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=actions)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_interactive_equity_curve(portfolio_values, output_path=None):
    """Create an interactive equity curve plot using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=portfolio_values, mode='lines', name='Portfolio Value'))
    fig.update_layout(title='Interactive Equity Curve', xaxis_title='Step', yaxis_title='Portfolio Value')
    if output_path:
        fig.write_html(output_path)
    fig.show()

# --- REPORT & JSON UTILITIES ---
def save_json(data, path):
    """Save a dictionary as JSON to the given path."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    """Load a dictionary from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def generate_report(metrics, output_path=None):
    """Generate and optionally save a summary report of evaluation metrics."""
    report_lines = [f"{k}: {v}" for k, v in metrics.items()]
    report = '\n'.join(report_lines)
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    return report

# --- DIRECTORY UTILITIES ---
def ensure_directories(dirs):
    """Ensure that output directories exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
