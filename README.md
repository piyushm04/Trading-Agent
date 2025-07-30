# RL Trading Agent

This project provides a complete framework for developing, training, and evaluating a reinforcement learning (RL) agent for stock trading.

## Features

- **Modular Design**: Separate components for data loading, environment, agent, training, and evaluation.
- **RL Agent**: Uses Stable-Baselines3 with a Deep Q-Network (DQN) agent.
- **Custom Gym Environment**: A custom `TradingEnvironment` that simulates trading, including commissions and portfolio management.
- **Data Handling**: Fetches market data from Yahoo Finance using the `yfinance` library and enriches it with technical indicators.
- **Automated Testing**: Includes a robust test suite with both mock data tests (for offline validation) and a live data pipeline test.

## Project Structure

```
rl_trading_agent/
├── models/                 # Saved RL models
├── data_loader.py          # Fetches and preprocesses market data
├── trading_env.py          # Custom Gym-compatible trading environment
├── agent.py                # RL agent logic (wraps Stable-Baselines3)
├── train.py                # Main script for training the agent
├── evaluate.py             # Script for evaluating a trained agent
├── utils.py                # Utility functions
├── test_with_mock_data.py  # Tests core logic with synthetic data
├── test_pipeline.py        # End-to-end test with live yfinance data
└── README.md               # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rl_trading_agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing pandas, numpy, yfinance, stable-baselines3[extra], gymnasium, matplotlib, etc.)*

## Usage

### Training

To train a new agent:
```bash
python train.py --symbol MSFT --timesteps 50000
```
This will train a DQN agent on MSFT data for 50,000 timesteps and save the best model to `models/best_model.zip` and a final model to `models/final_model.zip`.

### Evaluation

To evaluate a trained agent:
```bash
python evaluate.py --symbol MSFT --model_path models/best_model.zip
```
This will run the agent on the evaluation data and generate a performance report and plot.

### Testing

To run the test suite:

-   **Mock Data Test (Recommended for CI/CD):**
    ```bash
    python test_with_mock_data.py
    ```
-   **Live Data Pipeline Test:**
    ```bash
    python test_pipeline.py
    ```

## Important Note on Data Fetching

The `yfinance` library is used to fetch live market data. However, it can sometimes be unreliable due to network issues, firewalls, or temporary API problems. This may result in a `"No timezone found"` error.

-   The project's data loader and test pipeline are designed to be resilient to these failures.
-   If you encounter persistent data fetching errors, please check your network connection.
-   The core logic of the agent can be reliably tested using the mock data test (`test_with_mock_data.py`), which does not depend on the `yfinance` API.
