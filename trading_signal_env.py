import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd # For type hinting and potential use

class TradingSignalEnv(gymnasium.Env):
    """
    A custom trading environment for reinforcement learning.
    The agent learns to make Buy/Sell/Hold decisions based on market data.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, pair_symbol: str, historical_data_provider, 
                 initial_balance: float = 10000.0, 
                 window_size: int = 20, # For future use with features requiring lookback
                 fee_percentage: float = 0.001, # Standard fee
                 max_episode_steps: int = 1000): # Default, but actual length depends on data
        """
        Initializes the trading environment.

        Args:
            pair_symbol: The trading pair symbol (e.g., 'BTCUSDT').
            historical_data_provider: An object that provides historical market data.
                                      It must have a method `get_historical_data(pair_symbol)`
                                      which returns a pandas DataFrame.
            initial_balance: The starting balance for the trading account.
            window_size: The number of past time steps to consider for observation features (currently primarily for setting initial index).
            fee_percentage: The transaction fee percentage (e.g., 0.001 for 0.1%).
            max_episode_steps: The maximum number of steps in an episode (can be overridden by data length).
        """
        super().__init__()

        self.pair_symbol = pair_symbol
        self.data_provider = historical_data_provider # Store the data provider
        self.initial_balance = initial_balance
        self.window_size = window_size # Used to set initial index, future use for feature engineering
        self.fee_percentage = fee_percentage
        self._max_episode_steps = max_episode_steps # Internal max steps

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: 13 features from the current timestep
        # Base 10: 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_signal', 'BB_upper', 'BB_lower', 'Sentiment'
        # Additional 3: 'lstm_pred', 'bayes_pred_mean', 'bayes_uncertainty'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD_signal', 'BB_upper', 'BB_lower', 'Sentiment',
            'lstm_pred', 'bayes_pred_mean', 'bayes_uncertainty' # New features
        ]

        # Episode-specific state, initialized in reset()
        self.current_df = None
        self.current_index = 0
        self.balance = self.initial_balance
        self.current_position = 0  # 0: none (flat), 1: long, 2: short
        self.entry_price = 0.0
        self.episode_step_count = 0 # Tracks steps within the current episode

    def _get_observation(self):
        """
        Extracts the 10 features from the current data point.
        Returns a NumPy array of these features.
        """
        if self.current_df is None or self.current_index < 0 or self.current_index >= len(self.current_df):
            # Return zeros if data is not available or index is out of bounds
            # This might happen at the very end of an episode or if data loading failed
            print(f"Warning: Attempting to get observation with invalid index {self.current_index} or no data. Returning zeros.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        try:
            # Get the row at the current_index
            current_row = self.current_df.iloc[self.current_index]
            # Extract the specified features
            observation = current_row[self.feature_columns].values.astype(np.float32)
            return observation
        except Exception as e:
            print(f"Error getting observation at index {self.current_index}: {e}. Returning zeros.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)

        try:
            self.current_df = self.data_provider.get_historical_data(self.pair_symbol)
            if self.current_df is None or self.current_df.empty:
                print("Error: Historical data provider returned None or empty DataFrame.")
                # Handle error: return a zero observation and indicate failure if necessary
                # For now, allow it to proceed but _get_observation will return zeros.
                self.current_df = pd.DataFrame(np.zeros((1, len(self.feature_columns))), columns=self.feature_columns) # Dummy df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            self.current_df = pd.DataFrame(np.zeros((1, len(self.feature_columns))), columns=self.feature_columns) # Dummy df

        # Set current_index: start after window_size if data allows, else start at 0.
        # This is to allow for features that might require a lookback window in the future.
        # For current features (single time step), window_size mainly affects starting point.
        self.current_index = self.window_size if len(self.current_df) > self.window_size else 0
        
        self.balance = self.initial_balance
        self.current_position = 0  # 0: none, 1: long, 2: short
        self.entry_price = 0.0
        self.episode_step_count = 0

        initial_observation = self._get_observation()
        return initial_observation, {}

    def step(self, action):
        """
        Executes one step in the environment.
        """
        self.episode_step_count += 1
        reward = 0.0 # Initialize reward for the step

        # Ensure current_index is valid before proceeding
        if self.current_index < 0 or self.current_index >= len(self.current_df):
            # This state should ideally be caught by truncated/terminated logic earlier
            # Or means data is exhausted / reset was not called.
            print(f"Warning: Step called with invalid current_index {self.current_index}. Returning zero obs and reward.")
            obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return obs, 0.0, True, False, {"error": "Invalid index, data exhausted"}


        current_price = self.current_df['Close'].iloc[self.current_index]
        transaction_cost = current_price * self.fee_percentage

        # Trade execution logic
        if action == 1:  # Buy
            if self.current_position == 0:  # If flat, go long
                self.current_position = 1
                self.entry_price = current_price
                self.balance -= transaction_cost # Fee for opening position
                print(f"Step {self.current_index}: BUY signal. Opened LONG at {current_price:.2f}. Balance: {self.balance:.2f}")
            elif self.current_position == 2:  # If short, close short and go long
                profit_from_short = (self.entry_price - current_price)
                reward += profit_from_short # Realized P&L
                self.balance += profit_from_short - transaction_cost # Realized P&L - fee for closing short
                self.entry_price = current_price
                self.current_position = 1
                self.balance -= transaction_cost # Fee for opening new long position
                print(f"Step {self.current_index}: BUY signal. Closed SHORT (Profit: {profit_from_short:.2f}), Opened LONG at {current_price:.2f}. Balance: {self.balance:.2f}")
            else: # Already long, hold (or small penalty for redundant signal)
                print(f"Step {self.current_index}: BUY signal. Already LONG. Holding.")
                reward -= 0.001 # Small penalty for redundant action

        elif action == 2:  # Sell
            if self.current_position == 0:  # If flat, go short
                self.current_position = 2
                self.entry_price = current_price
                self.balance -= transaction_cost # Fee for opening position
                print(f"Step {self.current_index}: SELL signal. Opened SHORT at {current_price:.2f}. Balance: {self.balance:.2f}")
            elif self.current_position == 1:  # If long, close long and go short
                profit_from_long = (current_price - self.entry_price)
                reward += profit_from_long # Realized P&L
                self.balance += profit_from_long - transaction_cost # Realized P&L - fee for closing long
                self.entry_price = current_price
                self.current_position = 2
                self.balance -= transaction_cost # Fee for opening new short position
                print(f"Step {self.current_index}: SELL signal. Closed LONG (Profit: {profit_from_long:.2f}), Opened SHORT at {current_price:.2f}. Balance: {self.balance:.2f}")
            else: # Already short, hold
                print(f"Step {self.current_index}: SELL signal. Already SHORT. Holding.")
                reward -= 0.001 # Small penalty for redundant action
        
        else:  # Action == 0 (Hold)
            reward -= 0.01  # Small penalty for holding to encourage trading (can be tuned)
            print(f"Step {self.current_index}: HOLD signal.")

        # Add unrealized P&L for the current step if holding a position
        if self.current_position == 1:  # Long
            unrealized_pnl = (current_price - self.entry_price)
            reward += unrealized_pnl / 100 # Scaled unrealized P&L (tune scaling factor)
        elif self.current_position == 2:  # Short
            unrealized_pnl = (self.entry_price - current_price)
            reward += unrealized_pnl / 100 # Scaled unrealized P&L

        # Advance time
        self.current_index += 1

        # Termination and Truncation conditions
        terminated = self.balance <= 0
        # Truncated if data ends or max_episode_steps is reached
        truncated = (self.current_index >= len(self.current_df) -1) or \
                    (self.episode_step_count >= self._max_episode_steps)


        if terminated:
            print(f"Episode terminated: Balance {self.balance:.2f} <= 0 at step {self.current_index}")
        if truncated and not terminated:
             if self.current_index >= len(self.current_df) -1 :
                 print(f"Episode truncated: End of data reached at step {self.current_index}.")
             else:
                 print(f"Episode truncated: Max episode steps ({self._max_episode_steps}) reached at data index {self.current_index}.")


        # Get next observation
        observation = self._get_observation() # Gets observation for current_index, which is now next_step_index

        info = {
            'balance': self.balance,
            'current_position': self.current_position,
            'entry_price': self.entry_price
        }
        
        # As per the subtask, the `step` method's first line should be `print(f"Action taken: {action}")`
        # The current print statements are more descriptive of the outcome.
        # Let's add the required print statement at the beginning of step.
        # Re-checking: The requirement "print(f"Action taken: {action}") # As per requirement" was from a *previous* subtask's `trading_signal_env.py`
        # The current subtask does not specify this, so I will stick to the more descriptive prints above.

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode == 'human':
            print(f"Step: {self.current_index}/{len(self.current_df) if self.current_df is not None else 'N/A'}, "
                  f"Balance: {self.balance:.2f}, Position: {self.current_position} (Entry: {self.entry_price:.2f})")

    def close(self):
        """
        Performs any cleanup needed when the environment is closed.
        """
        print("TradingSignalEnv closed.")

# Example usage (optional, for testing the environment directly)
if __name__ == '__main__':
    # A dummy historical data provider for testing (matches expected interface)
    class DummyDataProviderForEnvTest:
        def __init__(self, pair_symbol: str, num_rows: int = 250): # Reduced rows for faster test
            self.pair_symbol = pair_symbol
            self.num_rows = num_rows
            self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_signal', 'BB_upper', 'BB_lower', 'Sentiment']
            self.df = self._generate_dummy_data()

        def _generate_dummy_data(self):
            data = {col: np.random.rand(self.num_rows) * 100 for col in self.feature_columns}
            data['Close'] = np.random.rand(self.num_rows) * 50 + 950 # Price around 950-1000
            data['Open'] = data['Close'] - np.random.rand(self.num_rows) * 10
            data['High'] = data['Close'] + np.random.rand(self.num_rows) * 5
            data['Low'] = data['Close'] - np.random.rand(self.num_rows) * 5
            return pd.DataFrame(data)

        def get_historical_data(self, pair_symbol: str):
            # Ignores pair_symbol, returns the pre-generated data for this dummy provider
            print(f"DummyDataProviderForEnvTest: Providing {len(self.df)} rows for {pair_symbol}")
            return self.df

    dummy_provider = DummyDataProviderForEnvTest(pair_symbol='BTCUSDT_Test')
    
    env = TradingSignalEnv(
        pair_symbol='BTCUSDT_Test', 
        historical_data_provider=dummy_provider,
        initial_balance=1000,
        window_size=5, # Small window for testing reset logic
        max_episode_steps=50 # Short episodes for testing
    )

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Initial observation example: {obs[:3]}") # Print first 3 features
    env.render()

    total_reward = 0
    for i in range(100): # Run for more steps to see truncation
        action = env.action_space.sample() # Sample a random action
        # print(f"\n--- Step {i+1} ---")
        # print(f"Action taken by agent: {['Hold', 'Buy', 'Sell'][action]}")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # env.render()
        # print(f"Observation: {obs[:3]}, Reward: {reward:.4f}")
        # print(f"Info: Balance={info.get('balance', 'N/A'):.2f}, Position={info.get('current_position', 'N/A')}")


        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Terminated: {terminated}, Truncated: {truncated}")
            print(f"Final Balance: {env.balance:.2f}, Total Reward: {total_reward:.2f}")
            # Example of how to get data from info dict
            print(f"Final Info: Balance: {info['balance']:.2f}, Position: {info['current_position']}, Entry: {info['entry_price']:.2f}")
            
            print("\nResetting environment...")
            obs, info = env.reset()
            env.render()
            total_reward = 0
            if i > 60 : # prevent infinite loop if something is wrong with termination/truncation
                print("Exiting test loop after one full episode and reset.")
                break 
    
    env.close()
