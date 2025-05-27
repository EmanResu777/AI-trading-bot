#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trading bot for automated cryptocurrency trading with AI and enhanced indicators."""

import ast
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import aiohttp
import git
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from telegram import Bot
from torch import nn

# Assuming these will be created with content later or are stubs for now
# from data.processor import DataProcessor
# from sentiment.finbert import SentimentAnalyzer

# --- Imports for Trading Signal RL Integration ---
from trading_signal_env import TradingSignalEnv # For the new signal RL model's environment
from train_trading_signal_rl import StaticHistoricalDataProvider # For dummy env instantiation if needed for loading signal model
# --- End Imports for Trading Signal RL Integration ---

# Placeholder classes if the actual files are empty for now
class DataProcessor:
    """
    PLACEHOLDER: Processes raw market data into features for AI models.
    This class is a simplified placeholder. A full implementation should:
    - Perform scaling and normalization (e.g., MinMaxScaler, StandardScaler).
    - Create additional technical indicators or derived features if not already done.
    - Handle NaN values more robustly (e.g., imputation or careful dropping).
    - Ensure the output array shape is consistent and matches model input expectations.
    """
    def prepare_data(self, arr: np.ndarray) -> np.ndarray:
        """
        Prepares the data for AI models.
        Expected input `arr`: NumPy array of shape (num_samples, num_base_features).
                               For this bot, num_base_features is 15 (OHLCV + 10 indicators).
        Expected output: NumPy array of shape (num_samples, num_processed_features).
                         Currently, this placeholder does no processing, so output shape is same as input.
                         A real implementation might change the number of features.
        """
        logger.debug(f"DataProcessor.prepare_data called with array of shape {arr.shape}")
        # Placeholder: No actual processing is done here.
        # Real implementation would involve scaling, normalization, feature engineering.
        return arr

    def add_news_sentiment(self, data: np.ndarray, sentiment_score: float) -> np.ndarray:
        """
        Adds news sentiment score as an additional feature column to the data.
        Expected input `data`: NumPy array of shape (num_samples, num_features_before_sentiment).
        Expected output: NumPy array of shape (num_samples, num_features_before_sentiment + 1).
                         The sentiment score is broadcasted to all samples.
        """
        logger.debug(f"DataProcessor.add_news_sentiment called with data shape {data.shape}, sentiment: {sentiment_score}")
        # Assuming data is a numpy array and sentiment_score is a float
        # Add sentiment as a new column, broadcasted to all rows.
        sentiment_column = np.full((data.shape[0], 1), sentiment_score)
        return np.hstack((data, sentiment_column))

class SentimentAnalyzer:
    """
    PLACEHOLDER: Analyzes news sentiment for cryptocurrencies.
    This class is a simplified placeholder. A full implementation should:
    - Fetch news articles from various reliable sources (e.g., RSS feeds, news APIs).
    - Use a proper NLP model (like FinBERT, or a more general sentiment model)
      to score the sentiment of each article regarding specific cryptocurrencies.
    - Aggregate sentiment scores, potentially weighting by source or recency.
    - Implement robust error handling, rate limiting, and caching for external API calls.
    - Handle new coin discovery more dynamically.
    """
    def __init__(self):
        self.request_timeout = 10 # Default timeout for simulated async operations

    async def get_news_sentiment(self, coin_symbol: str, keys: dict) -> dict:
        """
        Gets the aggregated news sentiment score for a given coin symbol.
        Expected output: A dictionary, e.g., {"score": 0.75} where score is float (-1.0 to 1.0).
        """
        # Placeholder: Simulates fetching and analyzing news.
        logger.info(f"Fetching news sentiment for {coin_symbol} (placeholder behavior)")
        await asyncio.sleep(1) # Simulate async work (e.g., network request, NLP processing)
        # In a real implementation, `keys` might contain API keys for news sources.
        # Return a dummy neutral sentiment.
        return {"score": 0.5} 

    async def fetch_futures_listings(self, keys: dict) -> list[str]:
        # Placeholder
        logger.info("Fetching future listings (placeholder)")
        await asyncio.sleep(1)
        return ["BTCUSDT", "ETHUSDT"]

    async def monitor_news_for_new_coins(self, keys):
        # Placeholder
        logger.info("Monitoring news for new coins (placeholder)")
        await asyncio.sleep(1)
        return ["SOLUSDT"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
handler = logging.handlers.RotatingFileHandler(
    "logs/trading.log", maxBytes=2 * 1024 * 1024, backupCount=5 # Adjusted path from worker report
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
))
logger.addHandler(handler)

trade_data = {
    "reward_points": 0.0,
    "daily_pnl": [],
    "strategy_success": {},
    "active_positions": {},
}

class LSTMModel(nn.Module):
    """LSTM модель для предсказания временных рядов."""
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the LSTM model."""
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM model for time series prediction with uncertainty.
    Note: The current training implementation in `train_lstm_models` for BayesianLSTM
    is a placeholder and does not perform actual Bayesian learning (e.g., via VI or MC Dropout training).
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """Defines the forward pass of the Bayesian LSTM model."""
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

class TradingEnv(Env):
    """
    Custom Gym environment for simulating cryptocurrency trading.
    This environment is used by the PPO reinforcement learning agent
    to learn optimal trading parameters.
    """
    def __init__(self, bot: "TradingBot"):
        super().__init__()
        # Observation space: 15 features (see _get_trading_state method)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        # Action space: 3 continuous values for risk_factor, tp_vol_mult, sl_vol_mult
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.bot = bot

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        # Returns a zero vector matching the observation space shape.
        # A real implementation might fetch an actual initial state.
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        """
        Executes one time step within the environment.
        Placeholder: Returns dummy values. A real implementation would simulate a trade
        based on the action and update the environment's state, calculating a reward.
        """
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False # Whether the episode has ended (e.g., account wiped out)
        truncated = False  # Whether the episode was cut short (e.g., time limit)
        info = {}          # Auxiliary diagnostic information
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Renders the environment (e.g., for visualization). Placeholder."""
        pass

class KeyManager:
    """
    Manages API keys and trading pair configurations stored in a JSON file.
    Handles loading keys from and saving keys to `secure/keys.json`.
    """
    def __init__(self):
        self.keys_file = "secure/keys.json"
        os.makedirs(os.path.dirname(self.keys_file), exist_ok=True)
        self.default_news_config = { 
            "news_fetch_enabled": False,
            "sources": [
                {"name": "NewsAPI", "api_key": "YOUR_NEWSAPI_KEY_DEFAULT", "enabled": False, "coins_supported": ["BTC", "ETH"]},
                {"name": "CryptoPanic", "api_key": "YOUR_CRYPTOPANIC_KEY_DEFAULT", "enabled": False, "coins_supported": ["general"]}
            ]
        }
        self.default_keys = { 
            "api_key": "DEFAULT_API_KEY",
            "api_secret": "DEFAULT_API_SECRET",
            "telegram_token": "DEFAULT_TELEGRAM_TOKEN",
            "chat_id": "DEFAULT_CHAT_ID",
            "trading_pairs": ["BTCUSDT", "ETHUSDT"],
            "news_api_config": self.default_news_config, 
            "max_risk_per_trade": 0.01, 
            "retraining_cooldown_period_hours": 24,
            "consecutive_loss_threshold": 3,
            "cumulative_pnl_check_trades": 10,
            "cumulative_pnl_threshold": -50.0,
            "liquidity_threshold": 300,
            "volatility_threshold": 0.01,
            "signal_rl_training_steps": 50000,
            # Portfolio Drawdown Keys
            "max_portfolio_drawdown_limit": 0.10, # Max 10% drawdown
            "portfolio_drawdown_check_interval_hours": 24, # Check every 24 hours
            "portfolio_initial_value_source": "current_on_start", # "current_on_start" or "fixed_amount_from_keys"
            "portfolio_fixed_initial_value_usdt": 10000.0 # Used if source is "fixed_amount_from_keys"
        }

    def load_keys(self) -> dict:
        """
        Loads keys and configuration from the JSON file.
        Provides default values for any missing keys to ensure robustness.
        Ensures 'news_api_config' exists with a default structure if missing.

        Returns:
            dict: A dictionary containing the loaded keys and configuration.
        """
        try:
            with open(self.keys_file, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
                # Ensure all default keys are present using setdefault
                for key, default_value in self.default_keys.items():
                    if key == "news_api_config": # Special handling for nested dict
                        loaded_json.setdefault(key, default_value.copy()) # Use a copy for the nested dict
                    else:
                        loaded_json.setdefault(key, default_value)
                return loaded_json
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading keys from {self.keys_file}: {e}. Returning default structure.")
            # Return a deep copy of the default structure
            return {k: (v.copy() if isinstance(v, dict) else v) for k, v in self.default_keys.items()}

    def save_keys(self, keys: dict) -> None:
        try:
            with open(self.keys_file, "w", encoding="utf-8") as f:
                json.dump(keys, f, indent=4)
        except OSError as e:
            logger.error(f"Error saving keys to {self.keys_file}: {e}")

    def update_trading_pairs(self, pairs: list[str]) -> None:
        keys = self.load_keys()
        current_pairs = keys.get("trading_pairs", [])
        # Ensure current_pairs is a list
        if not isinstance(current_pairs, list):
            current_pairs = []
        keys["trading_pairs"] = list(set(current_pairs + pairs))
        self.save_keys(keys)

    def update_news_api_source(self, source_name: str, api_key: str, enabled: bool, coins_supported: list[str] = None) -> bool:
        """
        Updates the API key, enabled status, and optionally coins_supported for a news source.
        If the source doesn't exist, it can be added (though current logic focuses on updates).

        Args:
            source_name (str): The name of the news source (e.g., "NewsAPI").
            api_key (str): The new API key.
            enabled (bool): The new enabled status.
            coins_supported (list[str], optional): List of coins this source supports. If None, not updated.


        Returns:
            bool: True if update was successful, False otherwise.
        """
        keys = self.load_keys()
        news_config = keys.get("news_api_config") # Already defaults in load_keys

        if not news_config or "sources" not in news_config:
            logger.error("news_api_config or sources list is missing. Cannot update source.")
            return False # Should not happen if load_keys works correctly

        source_found = False
        for source in news_config["sources"]:
            if source.get("name") == source_name:
                source["api_key"] = api_key
                source["enabled"] = enabled
                if coins_supported is not None: # Only update if provided
                    source["coins_supported"] = coins_supported
                source_found = True
                logger.info(f"Updated news source '{source_name}': enabled={enabled}, coins_supported={coins_supported if coins_supported is not None else 'not changed'}.")
                break
        
        if not source_found:
            # Optionally add new source if not found
            logger.warning(f"News source '{source_name}' not found. To add, implement source addition logic.")
            # Example for adding:
            # news_config["sources"].append({
            #     "name": source_name, "api_key": api_key, "enabled": enabled, 
            #     "coins_supported": coins_supported if coins_supported is not None else ["general"]
            # })
            # logger.info(f"Added new news source '{source_name}'.")
            return False # For now, only update existing

        self.save_keys(keys)
        return True

class TradingBot:
    """Автономный торговый бот с самообучением и расширенными индикаторами."""
    def __init__(self):
        logger.info("Init TradingBot")
        self.running = False
        self.loop = asyncio.get_event_loop()
        self.client: Client = None  # type: ignore
        self.bot: Bot = None        # type: ignore
        self.keys: dict = {}
        self.key_manager = KeyManager()
        self.sentiment = SentimentAnalyzer() # Using placeholder
        self.processor = DataProcessor()     # Using placeholder
        self.models: dict[str, dict] = {}
        self.trading_pairs: list[str] = []
        self.protected_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        self.data_cache: dict[str, pd.DataFrame] = {}
        self.error_log: list[str] = []
        self.pnl_history: list[dict] = []
        self.training_processes: dict[str, subprocess.Popen] = {}
        self.max_concurrent_training = 6
        # self.liquidity_threshold = 300 # Moved to keys.json
        # self.volatility_threshold = 0.01 # Moved to keys.json
        self.atr_period = 14
        self.trailing_stop_percent = 1.5
        self.risk_factor = 2.0
        self.tp_vol_mult = 3.0
        self.sl_vol_mult = 0.8
        self.commission_rate = 0.0004
        self.risk_params = {"tp_percent": 30.0, "sl_percent": 8.0}
        self.reward_per_profit = 1.5
        self.penalty_per_loss = -0.3
        self.env: TradingEnv = TradingEnv(self) # Initialize env here
        self.ai_optimizer: PPO = None  # type: ignore
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.idle_notification_sent = False # Flag for idle notification

        # For Trading Signal RL Model
        self.signal_env: TradingSignalEnv = None # Will be initialized if needed for model loading

        # Sentiment Thresholds
        self.sentiment_strong_positive_threshold = 0.6
        self.sentiment_strong_negative_threshold = -0.6
        self.sentiment_neutral_upper_threshold = 0.2
        self.sentiment_neutral_lower_threshold = -0.2
        
        self._load_keys_and_models() # Loads self.keys

        # Performance-based retraining & Other Configurable Thresholds
        self.trade_history: dict[str, list[dict]] = {}
        self.retraining_cooldown: dict[str, datetime] = {}
        self.retraining_cooldown_period_hours = self.keys.get("retraining_cooldown_period_hours", 24)
        self.consecutive_loss_threshold = self.keys.get("consecutive_loss_threshold", 3)
        self.cumulative_pnl_check_trades = self.keys.get("cumulative_pnl_check_trades", 10)
        self.cumulative_pnl_threshold = self.keys.get("cumulative_pnl_threshold", -50.0)
        
        # Liquidity and Volatility Thresholds from keys.json
        self.liquidity_threshold = self.keys.get("liquidity_threshold", self.default_keys["liquidity_threshold"])
        self.volatility_threshold = self.keys.get("volatility_threshold", self.default_keys["volatility_threshold"])
        self.max_risk_per_trade = self.keys.get("max_risk_per_trade", self.default_keys["max_risk_per_trade"])
        
        # Portfolio Drawdown Config
        self.max_portfolio_drawdown_limit = self.keys.get("max_portfolio_drawdown_limit", self.default_keys["max_portfolio_drawdown_limit"])
        self.portfolio_drawdown_check_interval_hours = self.keys.get("portfolio_drawdown_check_interval_hours", self.default_keys["portfolio_drawdown_check_interval_hours"])
        self.portfolio_initial_value_source = self.keys.get("portfolio_initial_value_source", self.default_keys["portfolio_initial_value_source"])
        self.portfolio_fixed_initial_value_usdt = self.keys.get("portfolio_fixed_initial_value_usdt", self.default_keys["portfolio_fixed_initial_value_usdt"])

        # Portfolio Drawdown State Variables
        self.initial_portfolio_value_usdt = 0.0
        self.last_drawdown_check_time = None # Will be set after portfolio value initialization
        self.trading_paused_due_to_drawdown = False


    def _load_keys_and_models(self) -> None:
        """Загрузить API‑ключи, подключиться к Binance, Telegram и загрузить модели."""
        try:
            self.keys = self.key_manager.load_keys()
            self.client = Client(
                self.keys.get("api_key"), self.keys.get("api_secret"), tld="com"
            )
            self.bot = Bot(token=self.keys.get("telegram_token"))
            self.trading_pairs = self.keys.get(
                "trading_pairs", self.protected_pairs.copy()
            )
            # Ensure trading_pairs is a list
            if not isinstance(self.trading_pairs, list):
                self.trading_pairs = self.protected_pairs.copy()

        except KeyError as e: # Should be less likely with .get and defaults in load_keys
            logger.error("Missing key during setup (should have defaults): %s", e)
            # Consider how to handle this - maybe a critical notification and stop
            # For now, rely on defaults loaded by KeyManager
        except BinanceAPIException as e:
            logger.error("Binance init error: %s", e)
            # This is critical, bot cannot trade. Notify and potentially stop.
            # Consider not raising immediately to allow Telegram notification if possible
            # For now, keeping original 'raise'
            if self.loop and self.bot and self.keys.get("chat_id"): # Ensure chat_id exists
                 asyncio.run_coroutine_threadsafe(self.send_notification(f"🆘 CRITICAL: Binance API Error during _load_keys_and_models: {e}. Bot may not function."), self.loop)
            raise

        # Send notification about model incompatibility due to indicator changes
        # This should ideally be a one-time or configurable check.
        # For now, sending it if the loop is running.
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_notification("⚠️ INDICATOR SET CHANGED: All existing AI models (PPO, LSTM) are now incompatible and must be retrained. Please delete old model files from './best_models/' and './saved_models/' to allow fresh training."),
                self.loop
            )
        else: # If loop not running (e.g. during initial setup before main loop starts), log it.
            logger.warning("INDICATOR SET CHANGED: Models are incompatible. Loop not running yet to send Telegram notification; will be sent at bot start if possible.")


        logger.info("Init AI models")
        self._initialize_models() # self.env should be initialized by now
        # Ensure loop is running for this if called from __init__
        # Moved initial "Bot started" notification to the start of the run() method
        # to ensure all initial setup (including async loop) is more likely complete.


    def _initialize_models(self) -> None:
        """Загрузить существующие модели или запустить обучение новых."""
        if self.env is None: # Should have been initialized in __init__
            self.env = TradingEnv(self)
            logger.warning("TradingEnv was not initialized before _initialize_models. Initializing now.")

        for pair in self.trading_pairs:
            self.models[pair] = {}
            rl_path = f"./best_models/{pair}_ai_model.zip"
            lstm_path = f"./saved_models/{pair}_lstm.pth"
            bayes_path = f"./saved_models/{pair}_bayesian_lstm.pt"
            try:
                if os.path.exists(rl_path):
                    self.models[pair]["rl"] = PPO.load(rl_path, env=self.env) # Pass env here
                if os.path.exists(lstm_path):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Updated input_size to 16 features (15 base + 1 sentiment)
                    # This change makes existing saved models incompatible.
                    m = LSTMModel(input_size=16).to(device) 
                    m.load_state_dict(torch.load(lstm_path))
                    self.models[pair]["lstm"] = m
                if os.path.exists(bayes_path):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Updated input_size to 16 features
                    m2 = BayesianLSTM(input_size=16).to(device) 
                    m2.load_state_dict(torch.load(bayes_path))
                    self.models[pair]["bayesian_lstm"] = m2
                if not self.models[pair]:
                    if self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.train_new_pair(pair), self.loop)
                    else:
                        # This is problematic if loop isn't running (e.g. during __init__)
                        # For now, log it. Consider a list of pairs to train once loop starts.
                        logger.warning(f"Loop not running, cannot schedule training for {pair} from _initialize_models.")

            except Exception as e:
                logger.error("Error loading models for %s: %s", pair, e)
                if self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.train_new_pair(pair), self.loop)
                else:
                    logger.warning(f"Loop not running, cannot schedule training for {pair} after error.")

    async def send_notification(self, msg: str) -> None:
        """Послать сообщение в Telegram, разбивая на 4000‑символьные чанки."""
        if not self.bot or "chat_id" not in self.keys or not self.keys["chat_id"]: # check if chat_id has value
            logger.warning("Telegram bot or chat_id not configured. Cannot send notification.")
            return
        chat_id = self.keys["chat_id"]
        for i in range(0, len(msg), 4000):
            try:
                await self.bot.send_message(chat_id=chat_id, text=msg[i : i + 4000])
            except aiohttp.ClientError as e: # Catch specific aiohttp client errors
                logger.error("Telegram send message error: %s", e)
            except Exception as e: # Catch other potential errors from telegram lib
                logger.error("General error sending Telegram message: %s", e)


    async def fix_code(self, err: str) -> None:
        """Автоматический патч кода по тексту ошибки."""
        logger.info(f"fix_code called with error string: {err}")
        specific_handler_applied = False

        if isinstance(exception_obj, BinanceAPIException):
            code = exception_obj.code
            message = exception_obj.message
            logger.info(f"BinanceAPIException in fix_code: Code {code}, Message: {message}")
            
            pair_context = "" # Try to extract pair if possible from err string for better messages
            import re
            match_symbol = re.search(r"symbol '([^']*)'", err.lower()) # Example: "symbol 'BTCUSDT'"
            if not match_symbol: # Try another common pattern
                match_symbol = re.search(r"for (\w+USDT)", err) # Catches "for BTCUSDT"
            if match_symbol:
                pair_context = f" for {match_symbol.group(1).upper()}"


            if code == -2019: # MARGIN_IS_INSUFFICIENT
                await self.send_notification(f"🚨 CRITICAL: Insufficient margin{pair_context}. Code: {code}. Message: {message}. Check account balance immediately! Trading for this pair might be paused.")
                # Consider adding logic to pause trading for this pair: self.pause_pair(pair_context.strip().replace("for ", ""))
                specific_handler_applied = True
            elif code == -1013: # MIN_NOTIONAL
                await self.send_notification(f"⚠️ Order Error{pair_context}: Value less than min notional (Code: {code}). Message: {message}. Review order size/price or bot's quantity calculation logic.")
                specific_handler_applied = True
            elif code == -1111 or code == -1104: # PRECISION_IS_TOO_HIGH_FOR_PRICE or INVALID_QTY_PRECISION
                 precision_type = "quantity/price"
                 if "price" in message.lower(): precision_type = "price"
                 elif "qty" in message.lower() or "quantity" in message.lower(): precision_type = "quantity"
                 await self.send_notification(f"⚠️ Order Error{pair_context}: Invalid precision for {precision_type} (Code: {code}). Message: {message}. Check bot's rounding logic against exchange rules.")
                 specific_handler_applied = True
            elif code == -2010 or code == -4003: # NEW_ORDER_REJECTED or INVALID_ORDER_STATUS
                 await self.send_notification(f"⚠️ Order Rejected/Invalid Status{pair_context} (Code: {code}). Message: {message}. May be due to risk limits, filters, or other issues. Review position and order details.")
                 specific_handler_applied = True
            # Add more specific Binance error codes here as needed
        
        if not specific_handler_applied:
            # Try AST-based fixes or general analysis if no specific Binance code was handled above
            if "IndexError" in err:
                await self._fix_index_error()
            elif "Invalid symbol" in err: # This is often a Binance error but might not have a specific code if from client-side validation
                logger.info("Skip Invalid symbol errors for AST patching, may need manual pair list review.")
                await self.send_notification(f"⚠️ Encountered 'Invalid symbol' error. Pair might be delisted or incorrect: {err}")
            elif "rate limit" in err.lower(): # General rate limit not caught by BinanceAPIException codes
                await self._fix_rate_limit()
            elif "timeout" in err.lower():
                # Check if it's a FinBERT timeout specifically
                if "sentiment/finbert.py" in err: # This check is hypothetical
                     await self._fix_timeout()
                else: # General timeout
                    await self.send_notification(f"⏳ General timeout error: {err}. Suggestion: Consider increasing relevant timeout settings or check network.")
            else: # Fallback to generic AI analysis
                await self._ai_analyze_error(err, exception_obj)

    async def _ai_analyze_error(self, err_str: str, exception_obj: Exception = None) -> None:
        """Отправить предложение патча от AI, potentially with more context."""
        logger.info(f"AI analysis invoked for error: {err_str}")
        additional_details = ""
        if exception_obj:
            additional_details = f"Type: {type(exception_obj).__name__}."
            if isinstance(exception_obj, BinanceAPIException):
                additional_details += f" Binance Code: {exception_obj.code}, Msg: {exception_obj.message}."
            # Attempt to include a snippet of stack trace if available (difficult from exc object alone)
            # In a real scenario, you'd capture traceback.format_exc() where the exception is first caught.
            # For now, this is a conceptual placeholder.
            # if hasattr(exception_obj, '__traceback__'):
            #     try:
            #         import traceback
            #         # This gets the current stack, not necessarily the exception's original one if re-raised.
            #         # For true original stack, it needs to be captured at the point of origin.
            #         # tb_summary = traceback.format_tb(exception_obj.__traceback__, limit=2) 
            #         # additional_details += f" Trace (summary): {''.join(tb_summary)}"
            #         logger.warning("Stack trace extraction from exception object is limited here. Capture with traceback.format_exc() at source for full details.")
            #     except ImportError:
            #         pass # traceback module might not be available in all restricted envs

        suggestion = f"🛠️ AI analysis invoked for error: '{err_str}'. Details: {additional_details}. No automated patch. Manual review needed."
        await self.send_notification(suggestion)


    async def _fix_index_error(self) -> None:
        """Патч по IndexError через трансформацию AST."""
        path = "main.py" # Current file
        try:
            with open(path, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            transformer = IndexErrorTransformer()
            new_tree = transformer.visit(tree)
            ast.fix_missing_locations(new_tree)
            transformed_code = ast.unparse(new_tree)
            if source_code != transformed_code:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(transformed_code)
                await self.send_notification("Applied IndexError patch via AST. Restarting bot might be needed.")
                # Consider programmatic restart or signaling mechanism if safe
            else:
                await self.send_notification("IndexError patch attempted but no changes made to code.")
        except Exception as e:
            logger.error(f"Failed to apply IndexError patch: {e}")
            await self.send_notification(f"🚨 Error applying IndexError patch: {e}")


    async def _fix_rate_limit(self) -> None:
        """Патч по rate limit через AST."""
        path = "main.py" # Current file
        try:
            with open(path, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code)
            transformer = APIRateLimitTransformer()
            new_tree = transformer.visit(tree)
            ast.fix_missing_locations(new_tree)
            transformed_code = ast.unparse(new_tree)
            if source_code != transformed_code:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(transformed_code)
                await self.send_notification("Applied API rate limit patch via AST. Restarting bot might be needed.")
            else:
                await self.send_notification("Rate limit patch attempted but no changes made to code.")
        except Exception as e:
            logger.error(f"Failed to apply rate limit patch: {e}")
            await self.send_notification(f"🚨 Error applying rate limit patch: {e}")

    async def _fix_timeout(self) -> None:
        """Увеличить timeout в FinBERT (sentiment/finbert.py)."""
        # This function assumes finbert.py has a line "timeout=self.request_timeout"
        # This is a fragile way to patch code and might break if finbert.py changes.
        # A more robust solution would involve configuration or specific error handling in FinBERT.
        p = "sentiment/finbert.py"
        try:
            if not os.path.exists(p):
                await self.send_notification(f"File not found: {p}. Cannot apply timeout patch.")
                return

            with open(p, "r", encoding="utf-8") as f:
                c = f.read()
            
            original_line = "timeout=self.request_timeout"
            doubled_timeout_line = "timeout=self.request_timeout * 2" # More robust than just multiplying a number literal

            if original_line in c:
                new_c = c.replace(original_line, doubled_timeout_line)
                if new_c != c:
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(new_c)
                    await self.send_notification(f"✅ Applied timeout patch to {p}. Original: '{original_line}', New: '{doubled_timeout_line}'. Restart may be needed.")
                else:
                    await self.send_notification(f"ℹ️ Timeout patch: '{original_line}' not found or already patched in {p}.")
            else:
                # Fallback if the exact line isn't found, try to find "request_timeout"
                # This is even more speculative
                import re
                match = re.search(r"timeout\s*=\s*self\.request_timeout", c) # type: ignore
                if match:
                    new_c = c.replace(match.group(0), f"{match.group(0)} * 2") # Append *2
                    if new_c != c:
                         with open(p, "w", encoding="utf-8") as f:
                            f.write(new_c)
                         await self.send_notification(f"✅ Applied speculative timeout patch to {p}. Restart may be needed.")
                    else:
                        await self.send_notification(f"ℹ️ Speculative timeout patch for {p} made no changes.")
                else:
                    await self.send_notification(f"⚠️ Could not find timeout setting in {p} for patching.")

        except FileNotFoundError:
            await self.send_notification(f"❌ File {p} not found. Cannot apply timeout patch.")
        except Exception as e:
            logger.error(f"Failed to apply timeout patch to {p}: {e}")
            await self.send_notification(f"🚨 Error applying timeout patch to {p}: {e}")

    async def train_new_pair(self, pair: str) -> None:
        """Запустить процесс обучения RL‑модели для новой пары."""
        if pair in self.training_processes and self.training_processes[pair].poll() is None:
            logger.info(f"Training for {pair} is already in progress.")
            await self.send_notification(f"ℹ️ Training for {pair} is already running.")
            return

        if len(self.training_processes) >= self.max_concurrent_training:
            logger.warning("Max concurrent training limit reached. Cannot start new training.")
            await self.send_notification("⚠️ Max concurrent training limit reached. Training for {pair} deferred.")
            return
        
        # Ensure train_rl.py exists
        train_script = "train_rl.py"
        if not os.path.exists(train_script):
            logger.error(f"{train_script} not found. Cannot train new pair {pair}.")
            await self.send_notification(f"🚨 ERROR: {train_script} not found. Cannot train {pair}.")
            return

        try:
            proc = subprocess.Popen(
                [sys.executable, train_script, "--pair", pair, "--steps", "15000"], # Use sys.executable
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.training_processes[pair] = proc
            logger.info(f"Started training {pair} with PID {proc.pid}")
            await self.send_notification(f"🚀 Started training for {pair} (RL Model).")
        except Exception as e:
            logger.error(f"Failed to start training process for {pair}: {e}")
            await self.send_notification(f"❌ Error starting RL training for {pair}: {e}")


    async def check_training_status(self) -> None:
        """Отслеживать завершение RL‑процессов."""
        completed_pairs = []
        for pair, proc in list(self.training_processes.items()):
            if proc.poll() is not None: # Process has terminated
                stdout_bytes, stderr_bytes = proc.communicate()
                stdout = stdout_bytes.decode(errors='ignore')
                stderr = stderr_bytes.decode(errors='ignore')
                
                if proc.returncode == 0:
                    logger.info(f"Training for {pair} completed successfully.")
                    await self.send_notification(f"✅ Training for {pair} (RL) completed successfully.")
                    rl_model_path = f"./best_models/{pair}_ai_model.zip"
                    if os.path.exists(rl_model_path):
                        try:
                            self.models[pair]["rl"] = PPO.load(rl_model_path, env=self.env)
                            logger.info(f"Successfully loaded new RL model for {pair}.")
                            await self.send_notification(f"🧠 New RL model for {pair} loaded.")
                            # Now trigger LSTM/Bayesian model training for this pair
                            await self.train_lstm_models(pair)
                        except Exception as e:
                            logger.error(f"Error loading RL model for {pair} after training: {e}")
                            await self.send_notification(f"🚨 Error loading RL model for {pair} post-training: {e}")
                    else:
                        logger.warning(f"RL model file {rl_model_path} not found after successful training for {pair}.")
                            await self.send_notification(f"⚠️ RL model file not found for {pair} after successful training. Check {train_script}.")
                else:
                    logger.error(f"Training for {pair} failed. Return code: {proc.returncode}")
                    logger.error(f"Stderr for {pair} training: {stderr}")
                    await self.send_notification(
                        f"❌ Training for {pair} (RL) FAILED. Code: {proc.returncode}.\nError: {stderr[:1000]}" # Limit error length
                    )
                completed_pairs.append(pair)
        
        for pair in completed_pairs:
            del self.training_processes[pair]


    async def train_lstm_models(self, pair: str) -> None:
        """Обучить или дообучить LSTM и Bayesian LSTM."""
        logger.info(f"Starting LSTM and Bayesian LSTM training for {pair}.")
        await self.send_notification(f"🤖🧠 Starting LSTM/Bayesian model training for {pair}.")

        df = await self._get_historical_data(pair)
            if df is None or df.empty or len(df) < 120: # Ensure enough data for a window of 120
                logger.warning(f"Not enough historical data for {pair} to train LSTM/Bayesian models (need 120, got {len(df) if df is not None else 'None'}).")
            await self.send_notification(f"📉 Insufficient data for {pair} to train LSTM/Bayesian models.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device} for LSTM/Bayesian training.")

        # --- LSTM ---
        # Updated input_size to 16 features
        lstm_input_features = 16
        lstm_window_size = 60    # As per original prepare_model_input logic for lstm

        d1_full = await self.prepare_model_input(pair, df, "lstm") # This should return data shaped for LSTM

        if d1_full is not None and d1_full.shape[0] >= lstm_window_size and d1_full.shape[1] == lstm_input_features:
            d1_windowed = d1_full[-lstm_window_size:] # Take the last 'window_size' observations
            
            m = LSTMModel(input_size=lstm_input_features).to(device) # input_size = 16
            opt = torch.optim.Adam(m.parameters(), lr=0.001)
            loss_fn = torch.nn.MSELoss()
            
            # Prepare tensor for LSTM: (batch_size, seq_len, num_features)
            # Here, batch_size=1, seq_len=lstm_window_size
            t = torch.tensor(d1_windowed, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Target should be the 'close' price of the next period after the window
            # This requires careful handling of target variable.
            # The original code uses df["close"].values[-1] which is the last known close.
            # A more standard approach for forecasting is to predict the N+1 step.
            # For simplicity, let's assume we predict the last value in the sequence (less ideal for forecasting future)
            # Or, if df has future data, use that. The current `_get_historical_data` gets up to "now".
            # Let's stick to the original logic of using the last close price as target for simplicity.
            
            if len(df["close"]) > lstm_window_size:
                 # Target is the close price corresponding to the *end* of the input sequence.
                 # If d1_windowed represents data up to T-1, target is price at T.
                 # The original code uses df["close"].values[-1].
                 # Let's assume d1_windowed is data for [T-window ... T-1], target is price at T.
                 # So, if df has 120 rows, and window is 60, d1_windowed is from row 60 to 119.
                 # Target should be based on df["close"].iloc[some_index_after_window]
                 # The original code's target: torch.tensor(df["close"].values[-1], dtype=torch.float32).to(dev)
                 # This implies LSTM output is compared against the most recent close.
                target_idx = -1 # last element of the dataframe
                target_val = df["close"].values[target_idx]
                tgt = torch.tensor(target_val, dtype=torch.float32).unsqueeze(0).to(device) # Ensure target shape matches output

                logger.info(f"Training LSTM for {pair}. Input shape: {t.shape}, Target value: {target_val}")
                for epoch in range(100): # Number of epochs
                    opt.zero_grad()
                    out = m(t) # out shape should be [1, 1]
                    loss = loss_fn(out.squeeze(), tgt.squeeze()) # Squeeze to match shapes if necessary
                    loss.backward()
                    opt.step()
                    if (epoch + 1) % 20 == 0:
                        logger.debug(f"LSTM {pair} Epoch {epoch+1}, Loss: {loss.item()}")
                
                os.makedirs("./saved_models/", exist_ok=True)
                torch.save(m.state_dict(), f"./saved_models/{pair}_lstm.pth")
                self.models[pair]["lstm"] = m
                logger.info(f"LSTM model for {pair} trained and saved.")
                await self.send_notification(f"👍 LSTM model for {pair} trained and saved.")
            else:
                logger.warning(f"Not enough data points in df[close] for target selection for LSTM on {pair}")
                await self.send_notification(f"📉 Not enough data for LSTM target for {pair}.")

        else:
            logger.warning(f"Could not prepare LSTM input for {pair} or data shape mismatch. Expected features: {lstm_input_features}, window: {lstm_window_size}. Got: {d1_full.shape if d1_full is not None else 'None'}")
            await self.send_notification(f"📉 Failed to prepare/validate data for LSTM training for {pair}.")


        # --- Bayesian LSTM ---
        # Updated input_size to 16 features
        bayesian_input_features = 16
        bayesian_window_size = 120 # Keeping window size as originally distinct

        d2_full = await self.prepare_model_input(pair, df, "bayesian_lstm")

        if d2_full is not None and d2_full.shape[0] >= bayesian_window_size and d2_full.shape[1] == bayesian_input_features:
            d2_windowed = d2_full[-bayesian_window_size:]
            
            m2 = BayesianLSTM(input_size=bayesian_input_features).to(device) # input_size = 16
            # Bayesian LSTM training often involves sampling or variational inference,
            # the original code just runs forward passes: for _ in range(50): m2(t2)
            # This doesn't seem like a standard training loop (no optimizer, no loss).
            # It might be for "burning in" the model or if the forward pass itself has some learning mechanism (unlikely with standard nn.LSTM).
            # Replicating original behavior for now. This part might need significant review for correctness.
            
            t2 = torch.tensor(d2_windowed, dtype=torch.float32).unsqueeze(0).to(device)
            # logger.info(f"Processing Bayesian LSTM for {pair}. Input shape: {t2.shape}. (Note: Original training loop was minimal)") # Old log

            # --- Proper Training Loop for BayesianLSTM ---
            opt_bayes = torch.optim.Adam(m2.parameters(), lr=0.001)
            loss_fn_bayes = torch.nn.MSELoss()

            # Target preparation for BayesianLSTM
            # Assuming the target is the most recent 'close' price, similar to LSTMModel.
            # This should ideally align with the end of the 'd2_windowed' sequence.
            if len(df["close"]) > bayesian_window_size: # Ensure there's a valid target beyond the window if predicting next step
                                                       # Or, at least enough data to correspond to the window's end.
                                                       # For using df["close"].values[-1], this check is more about data availability.
                target_val_bayes = df["close"].values[-1] # Using the last known close as target
                tgt_bayes = torch.tensor(target_val_bayes, dtype=torch.float32).unsqueeze(0).to(device)

                logger.info(f"Training Bayesian LSTM for {pair}. Input shape: {t2.shape}, Target value: {target_val_bayes}")
                m2.train() # Set model to training mode
                for epoch in range(100): # Number of epochs (e.g., 100)
                    opt_bayes.zero_grad()
                    out_bayes = m2(t2) # out_bayes shape should be [1, 1]
                    loss_bayes = loss_fn_bayes(out_bayes.squeeze(), tgt_bayes.squeeze()) # Squeeze to match shapes
                    loss_bayes.backward()
                    opt_bayes.step()
                    if (epoch + 1) % 20 == 0:
                        logger.debug(f"Bayesian LSTM {pair} Epoch {epoch+1}, Loss: {loss_bayes.item()}")
                
                os.makedirs("./saved_models/", exist_ok=True)
                torch.save(m2.state_dict(), f"./saved_models/{pair}_bayesian_lstm.pt")
                logger.info(f"Bayesian LSTM model for {pair} trained and saved.")
                await self.send_notification(f"👍 Bayesian LSTM model for {pair} trained and saved.")
            else:
                logger.warning(f"Not enough data points in df[close] for target selection for Bayesian LSTM on {pair} (need > {bayesian_window_size}, have {len(df['close'])}). Skipping Bayesian LSTM training.")
                await self.send_notification(f"📉 Not enough data for Bayesian LSTM target for {pair}. Training skipped.")
                # Ensure model is not assigned if not trained
                if "bayesian_lstm" in self.models.get(pair, {}):
                    del self.models[pair]["bayesian_lstm"] # Remove if it existed from a previous load attempt
                return # Skip assigning this model if target prep failed

            # os.makedirs("./saved_models/", exist_ok=True) # Moved inside if block
            # torch.save(m2.state_dict(), f"./saved_models/{pair}_bayesian_lstm.pt") # Moved inside if block
            self.models[pair]["bayesian_lstm"] = m2 # This assignment happens if training was successful
            # logger.info(f"Bayesian LSTM model for {pair} processed and saved.") # Old log
            # await self.send_notification(f"👍 Bayesian LSTM model for {pair} 'trained' (processed) and saved.") # Old notification
        else: # This else corresponds to: if d2_full is not None and ...
            logger.warning(f"Could not prepare Bayesian LSTM input for {pair} or data shape mismatch. Expected features: {bayesian_input_features}, window: {bayesian_window_size}. Got: {d2_full.shape if d2_full is not None else 'None'}")
            await self.send_notification(f"📉 Failed to prepare/validate data for Bayesian LSTM training for {pair}.")
            # Ensure model is not assigned if data prep failed
            if "bayesian_lstm" in self.models.get(pair, {}):
                 del self.models[pair]["bayesian_lstm"]


    async def _notify_performance_retraining(self, pair: str, reason: str) -> None:
        """Placeholder for performance-based retraining notification."""
        message = f"♻️ Performance issues detected for {pair}: {reason}. Initiating model retraining."
        logger.info(message)
        await self.send_notification(message)


    async def _get_historical_data(self, pair: str) -> pd.DataFrame:
        """Загрузить исторические OHLCV + ATR, RSI, MACD."""
        if pair in self.data_cache and not self.data_cache[pair].empty: # Check if empty
            # Basic check for recency; invalidate if too old (e.g., > 1 hour for hourly data)
            # This is a simple cache validation. More sophisticated would check last timestamp.
            # For now, let's assume if it's in cache, it's recent enough for one cycle.
            # A better approach might be to store timestamp with cache and check age.
            return self.data_cache[pair]
        try:
            logger.info(f"Fetching historical klines for {pair} (1h, 7 days)")
            klines = await self.loop.run_in_executor(
                self.executor,
                lambda: self.client.get_historical_klines(
                    pair, Client.KLINE_INTERVAL_1HOUR, "7 days ago UTC"
                )
            )
            # klines = self.client.get_historical_klines(
            #     pair, Client.KLINE_INTERVAL_1HOUR, "7 days ago UTC"
            # )
            df = pd.DataFrame(klines, columns=[
                "ts", "open", "high", "low", "close", "volume",
                "ct", "qav", "ntr", "tbb", "tbq", "ig"
            ]).astype(float)
            
            # Ensure we have enough data for indicators
            # Bollinger Bands (20), ADX (14*2-1=27 for full ADX calc), Stochastic (14)
            # Minimum periods needed is roughly 30-40 for these.
            min_data_periods = 40 
            if len(df) < min_data_periods:
                logger.warning(f"Not enough klines data for {pair} to calculate all indicators (need ~{min_data_periods}, got {len(df)})")
                return pd.DataFrame()

            # Select initial columns
            df = df[["open", "high", "low", "close", "volume"]]
            
            # Calculate base indicators
            df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            df["RSI"] = ta.rsi(df["close"], length=14)
            macd_df = ta.macd(df["close"])
            if macd_df is not None and all(col in macd_df.columns for col in ["MACD_12_26_9", "MACDh_12_26_9"]):
                df["MACD"] = macd_df["MACD_12_26_9"]
                df["MACD_DIFF"] = macd_df["MACDh_12_26_9"]
            else:
                logger.warning(f"Could not calculate MACD for {pair}. Filling with NaNs.")
                df["MACD"] = np.nan
                df["MACD_DIFF"] = np.nan

            # Calculate new indicators
            # Bollinger Bands (20-period, 2 std dev)
            bbands_df = ta.bbands(df["close"], length=20, std=2)
            if bbands_df is not None and all(col in bbands_df.columns for col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]):
                df["BBL"] = bbands_df["BBL_20_2.0"]
                df["BBM"] = bbands_df["BBM_20_2.0"]
                df["BBU"] = bbands_df["BBU_20_2.0"]
            else:
                logger.warning(f"Could not calculate Bollinger Bands for {pair}. Filling with NaNs.")
                df["BBL"], df["BBM"], df["BBU"] = np.nan, np.nan, np.nan

            # Stochastic Oscillator (14,3,3)
            stoch_df = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
            if stoch_df is not None and all(col in stoch_df.columns for col in ["STOCHk_14_3_3", "STOCHd_14_3_3"]):
                df["STOCHk"] = stoch_df["STOCHk_14_3_3"]
                df["STOCHd"] = stoch_df["STOCHd_14_3_3"]
            else:
                logger.warning(f"Could not calculate Stochastic Oscillator for {pair}. Filling with NaNs.")
                df["STOCHk"], df["STOCHd"] = np.nan, np.nan
            
            # Average Directional Index (ADX, 14-period)
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx_df is not None and "ADX_14" in adx_df.columns:
                df["ADX"] = adx_df["ADX_14"]
            else:
                logger.warning(f"Could not calculate ADX for {pair}. Filling with NaNs.")
                df["ADX"] = np.nan

            # List all feature columns before dropna
            # open, high, low, close, volume, ATR, RSI, MACD, MACD_DIFF, BBL, BBM, BBU, STOCHk, STOCHd, ADX
            # Total 15 features from data itself.
            
            df = df.dropna() # Drop rows with NaN from indicator calculations
            
            # Ensure enough data after dropna for model window (e.g. 120)
            required_rows_after_dropna = 120 
            if len(df) < required_rows_after_dropna:
                 logger.warning(f"Not enough data for {pair} after calculating all indicators and dropna (need {required_rows_after_dropna}, got {len(df)})")
                 self.data_cache[pair] = df.copy() 
                 return df 
            
            df_final = df.tail(required_rows_after_dropna) # Now take the tail
            self.data_cache[pair] = df_final.copy()
            return df_final
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching historical data for {pair}: {e}")
            await self.fix_code(f"BinanceAPIException historical_data {pair}: {str(e)}", e)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data for {pair}: {e}")
            await self.fix_code(f"Unexpected error historical_data {pair}: {str(e)}", e)
            return pd.DataFrame()


    async def prepare_model_input(
        self, pair: str, df: pd.DataFrame, model_type: str
    ) -> np.ndarray | None:
        """Собрать входные признаки для LSTM или Bayesian LSTM."""
        try:
            if df.empty:
                logger.warning(f"DataFrame is empty for {pair} in prepare_model_input.")
                return None

            # Base features from DataFrame (now 15 features)
            required_cols = [
                "open", "high", "low", "close", "volume",  # OHLCV (5 features)
                "ATR",                                     # Average True Range (1 feature)
                "RSI",                                     # Relative Strength Index (1 feature)
                "MACD", "MACD_DIFF",                       # MACD and MACD Difference/Histogram (2 features)
                "BBL", "BBM", "BBU",                       # Bollinger Bands (Lower, Middle, Upper) (3 features)
                "STOCHk", "STOCHd",                        # Stochastic Oscillator (%K, %D) (2 features)
                "ADX"                                      # Average Directional Index (1 feature)
            ] # Total: 5 + 1 + 1 + 2 + 3 + 2 + 1 = 15 features
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                logger.warning(f"Missing one or more required columns for {pair} in prepare_model_input. Missing: {missing_cols}. Available: {df.columns.tolist()}")
                return None
            
            arr = df[required_cols].values # Shape: (num_samples, 15)
            
            # DataProcessor.prepare_data is currently a placeholder.
            # In a real implementation, it would scale/normalize these 15 features.
            data = self.processor.prepare_data(arr) # Output shape depends on DataProcessor, currently (num_samples, 15)

            # Add news sentiment score as the 16th feature.
            senti_result = await self.sentiment.get_news_sentiment(pair.replace("USDT", ""), self.keys)
            sentiment_score = senti_result.get("score", 0.0) 
            data_with_sentiment = self.processor.add_news_sentiment(data, sentiment_score) # Shape: (num_samples, 16)

            # Feature count consistency check for LSTM/Bayesian models
            current_num_features = data_with_sentiment.shape[1] # Should be 16
            
            if model_type == "lstm":
                expected_features, window_size = 16, 60 
            elif model_type == "bayesian_lstm":
                expected_features, window_size = 16, 120 
            else:
                logger.warning(f"Unknown model_type '{model_type}' in prepare_model_input for {pair}.")
                return None

            if current_num_features != expected_features:
                logger.critical(
                    f"CRITICAL FEATURE MISMATCH for {pair}, model_type '{model_type}': "
                    f"DataProcessor output {current_num_features} features, "
                    f"but model expects {expected_features} features. Ensure DataProcessor and model defs are aligned."
                )
                return None 

            if len(data_with_sentiment) < window_size:
                logger.warning(f"Not enough data rows for {pair} (model {model_type}). Need {window_size}, got {len(data_with_sentiment)}. Historical data might be too short after NaNs.")
                return None
            
            # Return the whole sequence, not just the last window, matching original return.
            # The windowing is done in the training/prediction methods.
            return data_with_sentiment # Shape: (num_samples, num_features)

        except Exception as e:
            logger.error(f"Error preparing model input for {pair} ({model_type}): {e}", exc_info=True)
            await self.fix_code(f"Error in prepare_model_input for {pair} ({model_type}): {str(e)}", e)
            return None


    async def predict_with_lstm(self, pair: str, data_full_sequence: np.ndarray | None) -> float | None:
        """Получить предсказание от LSTM модели."""
        if data_full_sequence is None or data_full_sequence.size == 0:
            logger.warning(f"No data provided for LSTM prediction for {pair}.")
            return None
        
        model_info = self.models.get(pair, {})
        if "lstm" not in model_info:
            logger.warning(f"LSTM model for {pair} not found.")
            return None

        lstm_model = model_info["lstm"]
        
        # LSTM expects input shape: (batch_size, seq_len, num_features)
        # Updated: input_size=16, window=60
        lstm_input_features = 16 # Must match model architecture
        lstm_window_size = 60    # Must match how model was trained

        if data_full_sequence.shape[1] != lstm_input_features:
            logger.error(f"LSTM prediction data for {pair} has {data_full_sequence.shape[1]} features, expected {lstm_input_features}. Model definitions need update.")
            return None
        if data_full_sequence.shape[0] < lstm_window_size:
            logger.warning(f"Not enough data for LSTM window for {pair}. Need {lstm_window_size}, got {data_full_sequence.shape[0]}.")
            return None
            
        # Take the most recent 'lstm_window_size' data points for prediction
        data_windowed = data_full_sequence[-lstm_window_size:]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_model.to(device) # Ensure model is on the correct device
        
        tensor_input = torch.tensor(data_windowed, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension
        
        try:
            lstm_model.eval() # Set model to evaluation mode
            with torch.no_grad():
                prediction = lstm_model(tensor_input)
            return float(prediction.item())
        except Exception as e:
            logger.error(f"Error during LSTM prediction for {pair}: {e}", exc_info=True)
            return None


    async def predict_with_bayesian_lstm(
        self, pair: str, data_full_sequence: np.ndarray | None
    ) -> float | None:
        """Получить предсказание от Bayesian LSTM модели."""
        if data_full_sequence is None or data_full_sequence.size == 0:
            logger.warning(f"No data provided for Bayesian LSTM prediction for {pair}.")
            return None

        model_info = self.models.get(pair, {})
        if "bayesian_lstm" not in model_info:
            logger.warning(f"Bayesian LSTM model for {pair} not found.")
            return None
            
        bayesian_model = model_info["bayesian_lstm"]

        # Bayesian LSTM: input_size=16, window=120
        bayesian_input_features = 16 # Must match model architecture
        bayesian_window_size = 120   # Must match how model was trained

        if data_full_sequence.shape[1] != bayesian_input_features:
            logger.error(f"Bayesian LSTM prediction data for {pair} has {data_full_sequence.shape[1]} features, expected {bayesian_input_features}. Model definitions need update.")
            return None
        if data_full_sequence.shape[0] < bayesian_window_size:
            logger.warning(f"Not enough data for Bayesian LSTM window for {pair}. Need {bayesian_window_size}, got {data_full_sequence.shape[0]}.")
            return None
            
        data_windowed = data_full_sequence[-bayesian_window_size:]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bayesian_model.to(device)
        
        tensor_input = torch.tensor(data_windowed, dtype=torch.float32).unsqueeze(0).to(device)
        
        try:
            bayesian_model.eval()
            with torch.no_grad():
                # Bayesian models might require multiple forward passes for uncertainty estimation (Monte Carlo dropout)
                # The original code just does one pass. Replicating that.
                prediction = bayesian_model(tensor_input)
            return float(prediction.item())
        except Exception as e:
            logger.error(f"Error during Bayesian LSTM prediction for {pair}: {e}", exc_info=True)
            return None


    async def initialize_ai_optimizer(self) -> None:
        """Инициализация оптимизатора RL, если ещё не создан."""
        if self.ai_optimizer is None:
            if self.env is None: # Should be TradingEnv(self)
                 self.env = TradingEnv(self) # Ensure env is initialized
                 logger.info("TradingEnv initialized in initialize_ai_optimizer")
            try:
                # Ensure the environment's observation space matches _get_trading_state output (9 features)
                # And action space matches PPO model's expected action (3 continuous values)
                self.ai_optimizer = PPO("MlpPolicy", self.env, verbose=0, n_steps=2048) # Default n_steps
                logger.info("🤖 AI optimizer (PPO) initialized with new TradingEnv instance.")
                await self.send_notification("🤖 AI optimizer (PPO) initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize PPO AI optimizer: {e}", exc_info=True)
                await self.send_notification(f"🚨 Failed to initialize PPO AI optimizer: {e}")
            await self.fix_code(f"PPO Init Error: {str(e)}", e) # Try to fix if PPO init fails
        else:
            logger.info("AI optimizer already initialized.")


    async def optimize_parameters(self, pair: str) -> None:
        """Оптимизация параметров стратегии с помощью RL."""
        try:
            # Ensure optimizer is initialized (can take a moment if it's the first time)
            if self.ai_optimizer is None:
                await self.initialize_ai_optimizer()
                if self.ai_optimizer is None: # Still None after attempt
                    logger.error(f"Cannot optimize parameters for {pair}: AI optimizer not available (init failed).")
                    return

            state = await self._get_trading_state(pair)
            if state is None or state.size == 0:
                 logger.warning(f"Cannot optimize parameters for {pair}: trading state is unavailable or empty.")
                 return

            # Ensure state shape matches environment's observation space
            if state.shape != self.env.observation_space.shape:
                logger.error(f"State shape mismatch for {pair}: state {state.shape}, space {self.env.observation_space.shape}. Cannot predict.")
                await self.send_notification(f"🚨 State shape mismatch for {pair}. PPO prediction skipped.")
                return

            action, _ = self.ai_optimizer.predict(state, deterministic=True) # Use deterministic for "best" action
            
            # action is expected to be a Box(3,) ranging from -1 to 1.
            # Scale these actions to meaningful parameter ranges.
            # Original scaling:
            # self.risk_factor = max(2.0, min(action[0] * 2.0, 4.0)) -> This seems off, action[0]*2 can be -2 to 2. min then makes it -2.
            # Let's assume action[0] should be mapped from [-1, 1] to [min_risk, max_risk]
            # Example: map action[0] from [-1, 1] to [1.0, 3.0] for risk_factor
            # ((action[0] + 1) / 2) maps [-1, 1] to [0, 1]. Then scale and shift.
            
            # Risk Factor: e.g., from 1.5 to 3.5
            # The original max(2.0, min(action[0] * 2.0, 4.0)) is problematic.
            # If action[0] is -1, action[0]*2.0 = -2. min(-2, 4.0) = -2. max(2.0, -2) = 2.0.
            # If action[0] is 1,  action[0]*2.0 =  2. min(2, 4.0)  =  2. max(2.0,  2) = 2.0.
            # This means risk_factor is always 2.0 with the original logic. This needs correction.
            # Let's assume the intent was to scale action[0] (from -1 to 1) to a range like [1.5, 3.5]
            min_risk, max_risk = 1.5, 3.5
            self.risk_factor = min_risk + ((action[0] + 1) / 2) * (max_risk - min_risk)
            self.risk_factor = max(1.0, min(self.risk_factor, 5.0)) # Clamp to a safe overall range

            # TP Volatility Multiplier: e.g., from 1.0 to 5.0
            # Original: self.tp_vol_mult = max(3.0, action[1] * 1.5) -> If action[1] is -1, action[1]*1.5 = -1.5. max(3.0, -1.5) = 3.0.
            # This is also problematic, always results in >= 3.0.
            min_tp_mult, max_tp_mult = 1.0, 5.0
            self.tp_vol_mult = min_tp_mult + ((action[1] + 1) / 2) * (max_tp_mult - min_tp_mult)
            self.tp_vol_mult = max(0.5, min(self.tp_vol_mult, 7.0)) # Clamp

            # SL Volatility Multiplier: e.g., from 0.5 to 2.0
            # Original: self.sl_vol_mult = max(0.6, action[2] * 0.6) -> Similar issue.
            min_sl_mult, max_sl_mult = 0.5, 2.0
            self.sl_vol_mult = min_sl_mult + ((action[2] + 1) / 2) * (max_sl_mult - min_sl_mult)
            self.sl_vol_mult = max(0.3, min(self.sl_vol_mult, 3.0)) # Clamp

            logger.info(
                f"Optimized params for {pair} using PPO: risk_factor={self.risk_factor:.2f}, "
                f"tp_vol_mult={self.tp_vol_mult:.2f}, sl_vol_mult={self.sl_vol_mult:.2f}"
            )
            # await self.send_notification(f"💡 Optimized params for {pair}: RiskF={self.risk_factor:.2f}, TP_VM={self.tp_vol_mult:.2f}, SL_VM={self.sl_vol_mult:.2f}")

        except asyncio.TimeoutError: # This was from a wait_for in original, not present here
            logger.error("Timeout occurred during optimize_parameters for %s (should not happen with current structure)", pair)
        except Exception as e:
            logger.error(f"Error optimizing parameters for {pair}: {e}", exc_info=True)
            await self.fix_code(f"OptimizeParams Error {pair}: {str(e)}", e)


    async def _get_trading_state(self, pair: str) -> np.ndarray | None:
        """Сформировать вектор состояния для RL (9 признаков)."""
        df = await self._get_historical_data(pair)
        if df.empty or len(df) < 1: # Need at least one row
            logger.warning(f"Cannot get trading state for {pair}: historical data is empty or too short.")
            return np.zeros(self.env.observation_space.shape, dtype=np.float32) # Return default zero state

        # Calculate features based on the latest data point in df
        # Ensure df has the necessary columns and at least one row after processing
        try:
            close_std = df["close"].std()
            close_mean = df["close"].mean()
            volatility = (close_std / close_mean) if close_mean != 0 else 0.0 # Market volatility
            
            atr = df["ATR"].iloc[-1] if "ATR" in df.columns and not df["ATR"].empty else 0.0
            rsi = df["RSI"].iloc[-1] if "RSI" in df.columns and not df["RSI"].empty else 50.0 # Default to neutral
            macd_diff = df["MACD_DIFF"].iloc[-1] if "MACD_DIFF" in df.columns and not df["MACD_DIFF"].empty else 0.0
            
            senti_result = await self.sentiment.get_news_sentiment(pair.replace("USDT", ""), self.keys)
            sentiment_score = senti_result.get("score", 0.0) # News sentiment score

            # Futures position data
            position_qty = 0.0
            unrealized_pnl = 0.0
            try:
                pos_info = await self.loop.run_in_executor(
                    self.executor,
                    lambda: self.client.futures_position_information(symbol=pair)
                )
                # pos_info = self.client.futures_position_information(symbol=pair)
                if pos_info and isinstance(pos_info, list) and len(pos_info) > 0:
                    position_qty = float(pos_info[0].get("positionAmt", 0.0))
                    unrealized_pnl = float(pos_info[0].get("unRealizedProfit", 0.0))
            except BinanceAPIException as e:
                logger.warning(f"API error getting position info for {pair} in _get_trading_state: {e}")
            except Exception as e: # Other errors like parsing
                 logger.warning(f"Error processing position info for {pair} in _get_trading_state: {e}")


            # New indicators
            bbl = df["BBL"].iloc[-1] if "BBL" in df.columns and not df["BBL"].empty else (df["close"].iloc[-1] * 0.98) 
            bbm = df["BBM"].iloc[-1] if "BBM" in df.columns and not df["BBM"].empty else df["close"].iloc[-1]       
            bbu = df["BBU"].iloc[-1] if "BBU" in df.columns and not df["BBU"].empty else (df["close"].iloc[-1] * 1.02) 
            stoch_k = df["STOCHk"].iloc[-1] if "STOCHk" in df.columns and not df["STOCHk"].empty else 50.0 
            stoch_d = df["STOCHd"].iloc[-1] if "STOCHd" in df.columns and not df["STOCHd"].empty else 50.0 
            adx = df["ADX"].iloc[-1] if "ADX" in df.columns and not df["ADX"].empty else 20.0 

            # PPO State vector: 15 features
            # Feature order:
            # 0: volatility (market volatility)
            # 1: atr (Average True Range)
            # 2: rsi (Relative Strength Index)
            # 3: macd_diff (MACD Difference/Histogram)
            # 4: bbl (Bollinger Bands Lower Band)
            # 5: bbm (Bollinger Bands Middle Band)
            # 6: bbu (Bollinger Bands Upper Band)
            # 7: stoch_k (Stochastic %K)
            # 8: stoch_d (Stochastic %D)
            # 9: adx (Average Directional Index)
            # 10: sentiment_score (News sentiment score)
            # 11: position_qty (Current position quantity)
            # 12: unrealized_pnl (Current unrealized PNL)
            # 13: self.risk_factor (Current bot risk factor parameter)
            # 14: self.tp_vol_mult (Current bot TP volatility multiplier parameter)
            state_array = np.array([
                volatility, atr, rsi, macd_diff, 
                bbl, bbm, bbu, stoch_k, stoch_d, adx,
                sentiment_score, position_qty, unrealized_pnl,
                self.risk_factor, 
                self.tp_vol_mult
            ], dtype=np.float32)

            if state_array.shape != self.env.observation_space.shape:
                logger.error(f"Critical state shape mismatch for {pair}! Generated: {state_array.shape}, Expected: {self.env.observation_space.shape} ({self.env.observation_space.shape[0]} features). RL will fail.")
                return np.zeros(self.env.observation_space.shape, dtype=np.float32)
            
            return state_array

        except IndexError: 
            logger.warning(f"IndexError generating trading state for {pair}. DF might be missing indicator columns or data after processing. Columns: {df.columns.tolist()}")
            return np.zeros(self.env.observation_space.shape, dtype=np.float32)
        except Exception as e:
            logger.error(f"Unexpected error generating trading state for {pair}: {e}", exc_info=True)
            return np.zeros(self.env.observation_space.shape, dtype=np.float32)


    async def execute_order(self, pair: str, side: str, entry_price: float, stop_loss_price: float, is_closing_order: bool = False, quantity_to_close: float = 0.0) -> dict | None:
        """
        Sends a market order to Binance Futures. 
        If is_closing_order is True, it closes the position with quantity_to_close.
        Otherwise, it calculates position size based on risk parameters for a new entry.

        Args:
            pair (str): The trading pair symbol.
            side (str): "BUY" or "SELL".
            entry_price (float): The estimated entry price for the trade (for new entries).
            stop_loss_price (float): The pre-calculated stop-loss price for this trade (for new entries).
            is_closing_order (bool): If True, this order is to close an existing position.
            quantity_to_close (float): The quantity to trade if this is a closing order.

        Returns:
            dict | None: The order response from Binance or None if order failed.
        """
        try:
            if side.upper() not in ["BUY", "SELL"]:
                logger.error(f"Invalid order side: {side} for {pair}.")
                return None

            if not is_closing_order: # Validations for new entry orders
                if entry_price <= 0:
                    logger.error(f"Invalid entry_price: {entry_price} for {pair} {side} order. Must be positive.")
                    return None
                if side.upper() == "BUY" and (stop_loss_price <= 0 or stop_loss_price >= entry_price):
                    logger.error(f"Invalid stop_loss_price: {stop_loss_price} for {pair} BUY order (entry: {entry_price}). SL must be below entry and positive.")
                    return None
                if side.upper() == "SELL" and (stop_loss_price <= 0 or stop_loss_price <= entry_price):
                    logger.error(f"Invalid stop_loss_price: {stop_loss_price} for {pair} SELL order (entry: {entry_price}). SL must be above entry and positive.")
                    return None
            elif quantity_to_close <=0: # Validation for closing order
                 logger.error(f"Invalid quantity_to_close: {quantity_to_close} for closing order on {pair} {side}.")
                 return None


            try:
                balance_info = await self.loop.run_in_executor(self.executor, self.client.futures_account_balance)
            except BinanceAPIException as e:
                logger.error(f"Binance API error fetching balance for order on {pair}: {e}")
                await self.send_notification(f"⚠️ Binance API error fetching balance for {pair} order: {e.message}")
                return None # Cannot proceed without balance
            except Exception as e: # Other unexpected errors
                logger.error(f"Unexpected error fetching balance for order on {pair}: {e}")
                await self.send_notification(f"⚠️ Unexpected error fetching balance for {pair} order: {str(e)}")
                return None


            usdt_balance = 0.0
            if balance_info: # Ensure balance_info is not empty
                for asset_balance in balance_info:
                    if asset_balance["asset"] == "USDT":
                        usdt_balance = float(asset_balance.get("balance", 0.0)) # Use .get for safety
                        break
            
            if usdt_balance < 10.0: # Minimum threshold to trade
                logger.warning(f"Insufficient USDT balance for {pair}: {usdt_balance:.2f} USDT. Minimum 10 USDT required.")
                # await self.send_notification(f"💸 Low USDT balance: {usdt_balance:.2f}. Cannot trade {pair}.") # Emoji for money/balance
                return None

            # Get account balance info
            balance_info = []
            try:
                balance_info = await self.loop.run_in_executor(self.executor, self.client.futures_account_balance)
            except BinanceAPIException as e:
                logger.error(f"Binance API error fetching balance for order on {pair}: {e}")
                await self.send_notification(f"⚠️ Binance API error fetching balance for {pair} order: {e.message}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching balance for order on {pair}: {e}")
                await self.send_notification(f"⚠️ Unexpected error fetching balance for {pair} order: {str(e)}")
                return None

            usdt_balance = 0.0
            if balance_info:
                for asset_balance in balance_info:
                    if asset_balance["asset"] == "USDT":
                        usdt_balance = float(asset_balance.get("balance", 0.0))
                        break
            
            if usdt_balance < 10.0: # Minimum threshold to trade
                logger.warning(f"Insufficient USDT balance for {pair}: {usdt_balance:.2f} USDT. Minimum 10 USDT required.")
                return None

            # Get account balance info (needed for new entries, not strictly for closing existing with known quantity)
            balance_info = []
            usdt_balance = 0.0
            if not is_closing_order: # Only fetch balance if it's a new entry requiring sizing
                try:
                    balance_info = await self.loop.run_in_executor(self.executor, self.client.futures_account_balance)
                    if balance_info:
                        for asset_balance in balance_info:
                            if asset_balance["asset"] == "USDT":
                                usdt_balance = float(asset_balance.get("balance", 0.0))
                                break
                    if usdt_balance < 10.0: 
                        logger.warning(f"Insufficient USDT balance for new entry on {pair}: {usdt_balance:.2f} USDT. Min 10 USDT required.")
                        return None
                except BinanceAPIException as e_bal:
                    logger.error(f"Binance API error fetching balance for order on {pair}: {e_bal}")
                    await self.send_notification(f"⚠️ Binance API error fetching balance for {pair} order: {e_bal.message}")
                    return None
                except Exception as e_bal_gen:
                    logger.error(f"Unexpected error fetching balance for order on {pair}: {e_bal_gen}")
                    await self.send_notification(f"⚠️ Unexpected error fetching balance for {pair} order: {str(e_bal_gen)}")
                    return None

            quantity_to_trade = 0.0
            log_price_sl_info = ""

            if is_closing_order:
                quantity_to_trade = quantity_to_close
                logger.info(f"Preparing CLOSING order for {pair} {side}, Quantity: {quantity_to_trade:.8f}")
            else: # New entry order, calculate size
                risk_per_unit_in_quote = abs(entry_price - stop_loss_price)
                if risk_per_unit_in_quote == 0:
                    logger.error(f"Risk per unit is zero for {pair} {side} (entry: {entry_price}, SL: {stop_loss_price}). Cannot size position.")
                    return None
                max_loss_for_trade_usdt = usdt_balance * self.max_risk_per_trade
                quantity_to_trade = max_loss_for_trade_usdt / risk_per_unit_in_quote
                log_price_sl_info = f"(Entry: ~{entry_price:.4f}, SL: {stop_loss_price:.4f})"
                logger.info(f"Position sizing for {pair} {side}: Balance={usdt_balance:.2f}, MaxRiskTrade={max_loss_for_trade_usdt:.2f}, RiskPerUnit={risk_per_unit_in_quote:.4f} -> Initial Quantity={quantity_to_trade:.8f}")

            # Apply exchange quantity precision rules
            try:
                exchange_info = await self.loop.run_in_executor(self.executor, lambda: self.client.get_exchange_info())
                symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == pair)
                lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
                step_size = float(lot_size_filter['stepSize'])
                min_qty = float(lot_size_filter['minQty'])

                quantity_to_trade = (quantity_to_trade // step_size) * step_size
                quantity_to_trade = round(quantity_to_trade, 8) 

                if quantity_to_trade < min_qty:
                    context = "calculated new entry" if not is_closing_order else "closing"
                    logger.warning(f"Quantity {quantity_to_trade:.8f} for {pair} ({context}) is below minQty {min_qty:.8f}. Cannot place order.")
                    if not is_closing_order: # Only send notification for new entries failing this
                        await self.send_notification(f"⚠️ Calc. qty {quantity_to_trade:.4f} for {pair} < min {min_qty:.4f}. Order skipped.")
                    return None
            except Exception as e:
                logger.error(f"Could not get exchange info or apply lot size for {pair}: {e}. Using rough rounding.")
                quantity_to_trade = round(quantity_to_trade, 3) 
                if quantity_to_trade < 0.001: 
                    logger.warning(f"Quantity too small for {pair} after fallback rounding: {quantity_to_trade}")
                    return None
            
            if quantity_to_trade <= 0:
                logger.warning(f"Final quantity is zero or negative for {pair} ({quantity_to_trade:.8f}). Order skipped.")
                return None

            logger.info(f"Attempting to {side} {quantity_to_trade:.8f} of {pair} {log_price_sl_info if not is_closing_order else ''}")
            
            order_params = {"symbol": pair, "side": side.upper(), "type": "MARKET", "quantity": quantity_to_trade}
            if is_closing_order:
                order_params["reduceOnly"] = "true" # Ensure closing orders only reduce position

            order = await self.loop.run_in_executor(self.executor, lambda: self.client.futures_create_order(**order_params))
            
            order_id = order.get('orderId', 'N/A')
            filled_price = float(order.get('avgPrice', entry_price if not is_closing_order else 0.0)) 
            logger.info(f"Executed {side} for {pair}: {order}")
            trade_emoji = "📈" if side.upper() == "BUY" else "📉"
            action_type_msg = "CLOSING" if is_closing_order else "NEW"
            
            if not is_closing_order:
                await self.send_notification(f"{trade_emoji} {action_type_msg} TRADE: {side.upper()} {quantity_to_trade:.4f} {pair} @ MKT ~{filled_price:.4f} {log_price_sl_info}. OrderID: {order_id}")
                trade_data["active_positions"][pair] = {
                    "side": side.upper(), "quantity": quantity_to_trade,
                    "entry_price": filled_price, "stop_loss_price": stop_loss_price,
                    "order_id": order_id, "timestamp": datetime.utcnow().isoformat()
                }
            else: # It's a closing order
                await self.send_notification(f"{trade_emoji} {action_type_msg} TRADE: {side.upper()} {quantity_to_trade:.4f} {pair} @ MKT ~{filled_price:.4f}. OrderID: {order_id}")
                if pair in trade_data["active_positions"]: # Clear position from bot's tracking
                    del trade_data["active_positions"][pair]
                    logger.info(f"Cleared active position data for {pair} after closing order.")
            return order

        except BinanceAPIException as e:
            logger.error(f"Binance API error executing order for {pair} ({side}): {e}")
            await self.send_notification(f"🚨 Binance API Error on {side} {pair}: {e.status_code} - {e.message}")
            await self.fix_code(f"BinanceAPIException execute_order {pair} {side}: {str(e)}", e)
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing order for {pair} ({side}): {e}", exc_info=True)
            await self.send_notification(f"🚨 Unexpected Error on {side} {pair}: {str(e)}")
            await self.fix_code(f"Unexpected error execute_order {pair} {side}: {str(e)}", e)
            return None

    async def monitor_pnl_and_adjust_stops(self, pair: str) -> None:
        """Контролировать PnL, двигать трейлинг‑стоп + TP/SL, and check for retraining."""
        position_closed_by_this_logic = False
        pnl_for_this_trade = 0.0

        try:
            pos_info_list = await self.loop.run_in_executor(
                self.executor,
                lambda: self.client.futures_position_information(symbol=pair)
            )
            if not pos_info_list or not isinstance(pos_info_list, list) or len(pos_info_list) == 0:
                if pair in trade_data["active_positions"]:
                    logger.info(f"Position for {pair} (was active) now not found in futures_position_information. Assuming closed.")
                    del trade_data["active_positions"][pair]
                return

            pos_info = pos_info_list[0]
            
            position_amt = float(pos_info.get("positionAmt", 0.0))
            entry_price = float(pos_info.get("entryPrice", 0.0))
            unrealized_pnl = float(pos_info.get("unRealizedProfit", 0.0))
            # Use markPrice for calculations as it's generally recommended for PNL and liquidation checks
            current_price = float(pos_info.get("markPrice", entry_price if entry_price != 0 else pos_info.get("lastPrice", 0.0) ))
            
            if current_price == 0 and entry_price != 0: # Fallback if markPrice is somehow zero but we have entry
                current_price = entry_price


            if position_amt == 0:
                if pair in trade_data["active_positions"]:
                    logger.info(f"Position for {pair} is now zero (checked at start of monitor_pnl). Clearing active trade data.")
                    # PNL should be recorded at the moment of closure by TP/SL logic below.
                    # If it's closed by other means (e.g. manual, or exchange liquidation not caught by SL),
                    # this PNL might be missed by the bot's own history.
                    del trade_data["active_positions"][pair]
                return

            # Make sure entry_price is valid for PNL calculations
            if entry_price == 0 and position_amt != 0: # Position exists but entry price is zero
                logger.warning(f"Entry price for active position {pair} is 0. Cannot calculate PNL accurately or set stops based on entry. Current Price: {current_price}")
                # Attempt to update entry price from trade_data if available (less accurate over time)
                if pair in trade_data["active_positions"] and "entry_price" in trade_data["active_positions"][pair]:
                    entry_price = trade_data["active_positions"][pair]["entry_price"]
                    logger.info(f"Using cached entry price {entry_price} for {pair} as API returned 0.")
                else:
                    # If still zero, we cannot proceed with PNL-based logic.
                    # This might indicate an issue with how the position was opened or Binance API.
                    await self.send_notification(f"⚠️ Critical issue: Entry price for active position {pair} is 0. PNL monitoring and SL/TP based on entry price are compromised.")
                    return # Cannot proceed with PNL logic

            # --- Aggressive Trailing Stop (Original Logic) ---
            # "агрессивный трейлинг: стоп = entry + 75% от прибыли/количества"
            # This means locking in 25% of current profit (or trailing by 75% of profit amount from entry)
            if unrealized_pnl > 0:
                # Calculate the price that would realize 25% of the current unrealized PNL
                # For a LONG: stop_price = entry_price + (unrealized_pnl * 0.25 / abs(position_amt))
                # For a SHORT: stop_price = entry_price - (unrealized_pnl * 0.25 / abs(position_amt))
                # The original comment implies stop_price = entry + (pnl * 0.75 / abs(qty))
                # This means if PNL is 100, and entry is 1000, qty 1. stop = 1000 + 75 = 1075. This moves stop further into profit.
                # Let's reinterpret the original: trailing stop moves to lock in a portion of profit.
                # If current price is C, entry E, profit P = C-E (for long). Trail by X% of P.
                # Stop = E + P * (1-Trail%) = E + (C-E)*(1-Trail%)
                # Original: stop_price = entry + (pnl * 0.75 / abs(qty))
                # This is equivalent to: stop_price for LONG = entry + ( (current_price - entry) * qty * 0.75 / qty )
                #                                          = entry + (current_price - entry) * 0.75
                #                                          = entry * 0.25 + current_price * 0.75
                # This means the stop is 75% of the way from entry to current price.
                
                # Let's use a simpler ATR based trailing stop or a percentage of current price.
                # For now, disabling the original complex trailing stop as it might be too aggressive or misinterepreted.
                # A proper trailing stop usually trails below current price for long, above for short.
                pass # Original aggressive trailing stop logic commented out for review.

            # --- TP / SL based on fixed percentages (Original Logic) ---
            # These are percentage of entry price * initial size.
            # This is a fixed TP/SL, not dynamic based on volatility multipliers from PPO (self.tp_vol_mult, self.sl_vol_mult)
            # This section should ideally use the PPO optimized parameters.

            atr_val = (await self._get_historical_data(pair))["ATR"].iloc[-1] if pair in self.data_cache and not self.data_cache[pair].empty and "ATR" in self.data_cache[pair].columns else current_price * 0.01 # fallback ATR

            # Calculate TP/SL prices based on PPO-optimized volatility multipliers
            side_multiplier = 1 if position_amt > 0 else -1 # 1 for long, -1 for short

            take_profit_price = entry_price + (side_multiplier * atr_val * self.tp_vol_mult)
            stop_loss_price = entry_price - (side_multiplier * atr_val * self.sl_vol_mult)

            # Check for TP
            if (position_amt > 0 and current_price >= take_profit_price) or \
               (position_amt < 0 and current_price <= take_profit_price):
                logger.info(f"Take Profit condition met for {pair} at {current_price:.2f} (TP target: {take_profit_price:.2f}). Closing position.")
                await self.send_notification(f"💰 TAKE PROFIT for {pair} at {current_price:.2f} (Target: {take_profit_price:.2f}).")
                close_order = await self.loop.run_in_executor(
                    self.executor,
                    lambda: self.client.futures_create_order(
                        symbol=pair,
                        side="SELL" if position_amt > 0 else "BUY",
                        type="MARKET",
                        quantity=abs(position_amt),
                        reduceOnly=True,
                    )
                )
                logger.info(f"TP Close order for {pair}: {close_order}")
                trade_data["reward_points"] += self.reward_per_profit
                pnl_for_this_trade = unrealized_pnl # Record PNL at closure
                trade_data["daily_pnl"].append({"pair": pair, "pnl": pnl_for_this_trade, "type": "TP", "timestamp": datetime.utcnow().isoformat()})
                if pair in trade_data["active_positions"]: del trade_data["active_positions"][pair]
                position_closed_by_this_logic = True
                # return # Position closed # Don't return yet, proceed to retraining check

            # Check for SL (only if not already closed by TP)
            if not position_closed_by_this_logic and \
               ((position_amt > 0 and current_price <= stop_loss_price) or \
                (position_amt < 0 and current_price >= stop_loss_price)):
                logger.info(f"Stop Loss condition met for {pair} at {current_price:.2f} (SL target: {stop_loss_price:.2f}). Closing position.")
                await self.send_notification(f"🛑 STOP LOSS for {pair} at {current_price:.2f} (Target: {stop_loss_price:.2f}). PNL: {unrealized_pnl:.2f}")
                close_order = await self.loop.run_in_executor(
                    self.executor,
                    lambda: self.client.futures_create_order(
                        symbol=pair,
                        side="SELL" if position_amt > 0 else "BUY",
                        type="MARKET",
                        quantity=abs(position_amt),
                        reduceOnly=True,
                    )
                )
                logger.info(f"SL Close order for {pair}: {close_order}")
                trade_data["reward_points"] += self.penalty_per_loss
                pnl_for_this_trade = unrealized_pnl # Record PNL at closure
                trade_data["daily_pnl"].append({"pair": pair, "pnl": pnl_for_this_trade, "type": "SL", "timestamp": datetime.utcnow().isoformat()})
                if pair in trade_data["active_positions"]: del trade_data["active_positions"][pair]
                position_closed_by_this_logic = True
                # return # Position closed # Don't return yet

        except BinanceAPIException as e:
            logger.error(f"Binance API error in monitor_pnl_and_adjust_stops for {pair}: {e}")
            if "order" not in str(e).lower():
                 await self.fix_code(f"BinanceAPIException monitor_pnl {pair}: {str(e)}", e)
        except Exception as e:
            logger.error(f"Unexpected error in monitor_pnl_and_adjust_stops for {pair}: {e}", exc_info=True)
            await self.fix_code(f"Unexpected error monitor_pnl {pair}: {str(e)}", e)
        
        finally: # This block will execute even if errors occur above, or if returns are hit
            if position_closed_by_this_logic:
                logger.info(f"Trade closed for {pair} by TP/SL logic. PNL: {pnl_for_this_trade:.2f}. Triggering retraining check.")
                if pair not in self.trade_history:
                    self.trade_history[pair] = []
                self.trade_history[pair].append({'pnl': pnl_for_this_trade, 'timestamp': datetime.utcnow()})
                await self._check_and_trigger_retraining(pair)


    async def _check_and_trigger_retraining(self, pair: str) -> None:
        """Check performance metrics and trigger retraining if thresholds are met."""
        logger.info(f"Checking performance for {pair} to decide on retraining.")

        # Cooldown check
        if pair in self.retraining_cooldown:
            cooldown_end_time = self.retraining_cooldown[pair] + pd.Timedelta(hours=self.retraining_cooldown_period_hours)
            if datetime.utcnow() < cooldown_end_time:
                logger.info(f"Retraining for {pair} is on cooldown. Last retrain: {self.retraining_cooldown[pair]}. Next possible: {cooldown_end_time}.")
                return

        history = self.trade_history.get(pair, [])
        if not history:
            logger.info(f"No trade history for {pair} to check for retraining.")
            return

        retrain_reason = None

        # 1. Consecutive Losses Check
        if len(history) >= self.consecutive_loss_threshold:
            last_n_trades = history[-self.consecutive_loss_threshold:]
            if all(trade['pnl'] < 0 for trade in last_n_trades):
                retrain_reason = f"{self.consecutive_loss_threshold} consecutive losses."
                logger.warning(f"Retraining condition met for {pair}: {retrain_reason}")

        # 2. Cumulative PNL Drop Check (only if no consecutive loss trigger)
        if not retrain_reason and len(history) >= self.cumulative_pnl_check_trades:
            relevant_trades = history[-self.cumulative_pnl_check_trades:]
            cumulative_pnl = sum(trade['pnl'] for trade in relevant_trades)
            if cumulative_pnl < self.cumulative_pnl_threshold:
                retrain_reason = f"Cumulative PNL of last {self.cumulative_pnl_check_trades} trades ({cumulative_pnl:.2f}) is below threshold ({self.cumulative_pnl_threshold:.2f})."
                logger.warning(f"Retraining condition met for {pair}: {retrain_reason}")
        
        if retrain_reason:
            await self._notify_performance_retraining(pair, retrain_reason)
            logger.info(f"Triggering retraining for {pair} due to: {retrain_reason}.")
            
            # Trigger RL model training
            await self.train_new_pair(pair) 
            
            # Optionally, trigger LSTM/Bayesian model training
            # Consider if these should always be retrained or based on different/more specific criteria
            logger.info(f"Also triggering LSTM/Bayesian model retraining for {pair} as part of performance recovery.")
            await self.train_lstm_models(pair)

            self.retraining_cooldown[pair] = datetime.utcnow()
            logger.info(f"Retraining cooldown started for {pair} at {self.retraining_cooldown[pair]}.")
        else:
            logger.info(f"No retraining criteria met for {pair} based on current history (last {len(history)} trades).")


    async def monitor_new_pairs(self) -> None:
        """Авто‑добавление новых фьючерсных пар через API и новости (placeholders)."""
        try:
            # These sentiment methods are placeholders and need real implementation
            # The timeout here is for the placeholder's asyncio.sleep
            newly_listed_futures = await asyncio.wait_for(self.sentiment.fetch_futures_listings(self.keys), timeout=5)
            coins_from_news = await asyncio.wait_for(self.sentiment.monitor_news_for_new_coins(self.keys), timeout=5)
            
            potential_new_pairs = set(newly_listed_futures + coins_from_news)
            added_pairs_count = 0

            for p_base in potential_new_pairs:
                # Assuming pairs are like "BTCUSDT", "ETHUSDT". News might give "BTC".
                pair_symbol = p_base if "USDT" in p_base.upper() else f"{p_base.upper()}USDT"

                if pair_symbol not in self.trading_pairs and pair_symbol not in self.protected_pairs:
                    # Further validation: Check if symbol is actually tradable on Binance Futures
                    try:
                        logger.info(f"Checking validity of potential new pair: {pair_symbol}")
                        exchange_info = await self.loop.run_in_executor(self.executor, self.client.get_exchange_info)
                        # futures_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['contractType'] == "PERPETUAL"] # This check might be slow
                        is_valid_future = any(s['symbol'] == pair_symbol and s.get('contractType') == "PERPETUAL" for s in exchange_info['symbols'])

                        if is_valid_future:
                            logger.info(f"Found new valid futures pair: {pair_symbol}. Adding to trading list and initiating training.")
                            self.trading_pairs.append(pair_symbol)
                self.idle_notification_sent = False # Reset idle flag as we have a new pair
                await self.send_notification(f"➕ New pair detected and added: {pair_symbol}. Starting RL model training.")
                            await self.train_new_pair(pair_symbol) # Start training RL model
                            added_pairs_count += 1
                        else:
                logger.info(f"Potential pair {pair_symbol} is not a valid perpetual futures contract on Binance or already present.")

                    except BinanceAPIException as e:
                        logger.warning(f"Binance API error when validating new pair {pair_symbol}: {e}")
                    except Exception as e:
                        logger.warning(f"Error validating new pair {pair_symbol}: {e}")
            
            if added_pairs_count > 0:
                self.key_manager.update_trading_pairs(self.trading_pairs) # Save updated list
                logger.info(f"Updated trading pairs in keys.json with {added_pairs_count} new pair(s).")

        except asyncio.TimeoutError:
            logger.warning("Timeout in monitor_new_pairs (likely due to placeholder sentiment methods).")
        except Exception as e:
            logger.error(f"Error in monitor_new_pairs: {e}", exc_info=True)
            # Avoid fix_code here as it might be related to external sentiment analysis logic


    async def manage_liquidity(self) -> None:
        """Удалять пары с низкой ликвидностью или малой волатильностью."""
        to_remove: list[str] = []
        logger.info("Starting liquidity management scan...")
        original_pair_count = len(self.trading_pairs)
        pairs_to_check = [p for p in self.trading_pairs if p not in self.protected_pairs]

        if not pairs_to_check:
            logger.info("No non-protected pairs to check for liquidity management.")
            return

        for p in pairs_to_check:
            df = await self._get_historical_data(p) # Gets last 120 1-hour candles
            if df.empty or len(df) < 24: # Need at least 24 hours of data
                logger.warning(f"Not enough data for liquidity check on {p} (has {len(df)} rows). Skipping removal check for now.")
                continue

            try:
                # Liquidity: Average daily volume in USDT (using last 24h as proxy for daily)
                # Ensure 'volume' and 'close' columns exist and are numeric
                if not all(col in df.columns for col in ["volume", "close"]) or not pd.api.types.is_numeric_dtype(df["volume"]) or not pd.api.types.is_numeric_dtype(df["close"]):
                    logger.warning(f"Volume or Close column missing or not numeric for {p}. Skipping liquidity check.")
                    continue

                avg_volume_24h = df["volume"].tail(24).mean()
                last_close_price = df["close"].iloc[-1]
                avg_volume_usdt_24h = avg_volume_24h * last_close_price
                
                # Volatility: Standard deviation of close prices / mean close price (coefficient of variation) over the period
                # Using the full df period (max 120 hours) for volatility calculation
                price_std = df["close"].std()
                price_mean = df["close"].mean()
                volatility_coeff = (price_std / price_mean) if price_mean != 0 else 0.0

                logger.debug(f"Liquidity check for {p}: AvgVolUSDT_24h={avg_volume_usdt_24h:.0f} (Threshold: {self.liquidity_threshold}), VolatilityCoeff={volatility_coeff:.4f} (Threshold: {self.volatility_threshold})")

                if avg_volume_usdt_24h < self.liquidity_threshold or volatility_coeff < self.volatility_threshold:
                    reason = []
                    if avg_volume_usdt_24h < self.liquidity_threshold: reason.append(f"Liquidity {avg_volume_usdt_24h:.0f} < {self.liquidity_threshold}")
                    if volatility_coeff < self.volatility_threshold: reason.append(f"Volatility {volatility_coeff:.4f} < {self.volatility_threshold}")
                    
                    logger.info(f"Pair {p} marked for removal. Reason: {'; '.join(reason)}.")
                    to_remove.append(p)
                    await self.send_notification(f"➖ Pair {p} to be removed: {'; '.join(reason)}.")
                else:
                    logger.info(f"Pair {p} passed liquidity/volatility checks.")


            except Exception as e:
                logger.error(f"Error during liquidity check for pair {p}: {e}", exc_info=True)
        
        if to_remove:
            for p_rem in to_remove:
                if p_rem in self.trading_pairs:
                    self.trading_pairs.remove(p_rem)
                if p_rem in self.models:
                    self.models.pop(p_rem, None)
                    logger.info(f"Removed models for delisted pair {p_rem}.")
                if p_rem in self.data_cache:
                    del self.data_cache[p_rem]
                if p_rem in self.training_processes: # Stop ongoing training if any
                    proc = self.training_processes.pop(p_rem) # Remove and get process
                    if proc.poll() is None: # If still running
                        try:
                            proc.terminate() # Send SIGTERM
                            proc.wait(timeout=5) # Wait for termination
                            logger.info(f"Terminated training for removed pair {p_rem}.")
                        except Exception as e:
                            logger.warning(f"Could not terminate training for removed pair {p_rem}: {e}. Killing.")
                            proc.kill() # Force kill if terminate fails
                logger.info(f"Removed pair {p_rem} from trading list.")
            
            self.key_manager.update_trading_pairs(self.trading_pairs) # Save the pruned list
            logger.info(f"Updated trading_pairs.json after removing {len(to_remove)} illiquid/low-volatility pair(s).")
            await self.send_notification(f"🗑️ Removed {len(to_remove)} pair(s) due to low liquidity/volatility. Current active pairs: {len(self.trading_pairs)}.")
        
        if original_pair_count > 0 and not self.trading_pairs and not any(p in self.protected_pairs for p in to_remove) : # Check if all tradable pairs were removed
             logger.warning("All non-protected trading pairs have been removed by liquidity management. Check thresholds or market conditions.")
             await self.send_notification("⚠️ All non-protected trading pairs removed by liquidity management! Bot may become idle if no protected pairs are active.")


    async def run(self) -> None:
        """Основной бесконечный цикл работы бота."""
        self.running = True
        logger.info("Bot main loop starting...")
        # Send initial "Bot started" notification here, after loop is confirmed running
        await self.send_notification("🚀 Bot instance started and main loop is running.")


        # Initialize PPO optimizer once at the start of the run loop if not already done
        if self.ai_optimizer is None:
            await self.initialize_ai_optimizer()

        last_daily_report_time = datetime.utcnow().date() # For daily PNL report

        while self.running:
            try:
                current_time = datetime.utcnow()
                logger.info(f"Main loop iteration started at {current_time.isoformat()}")

                # --- Housekeeping Tasks (less frequent) ---
                loop_count = getattr(self, 'loop_count', 0) + 1 # Basic counter
                self.loop_count = loop_count

                if loop_count == 1 or loop_count % 10 == 0: # Approx every 5 minutes (30s * 10), also on first loop
                    logger.info("Running periodic housekeeping tasks (training status, new pairs, liquidity)...")
                    await asyncio.wait_for(self.check_training_status(), timeout=45)
                    await asyncio.wait_for(self.monitor_new_pairs(), timeout=45)
                    await asyncio.wait_for(self.manage_liquidity(), timeout=75)
                
                # --- Daily PNL Report ---
                if current_time.date() > last_daily_report_time:
                    await self.send_daily_pnl_report()
                    last_daily_report_time = current_time.date()


                # --- Trading Logic for each pair ---
                if not self.trading_pairs:
                    if not self.idle_notification_sent:
                        logger.info("No trading pairs configured or all removed. Bot is idle. Will check for new pairs periodically.")
                        await self.send_notification("😴 Bot is idle: No trading pairs available. Monitoring for new pairs.")
                        self.idle_notification_sent = True
                    await asyncio.sleep(60) # Sleep longer if no pairs
                    continue
                else: # If there are pairs, reset the idle notification flag
                    if self.idle_notification_sent: # Only log/reset if it was previously set
                        logger.info("Trading pairs are now available. Bot is active.")
                        self.idle_notification_sent = False


                logger.info(f"Processing {len(self.trading_pairs)} trading pairs: {self.trading_pairs}")
                active_pairs_processed_this_cycle = 0
                for p in list(self.trading_pairs): # Iterate on a copy in case list changes
                    if not self.running: break # Exit if bot stopped during pair processing
                    
                    logger.info(f"--- Processing pair: {p} ---")
                    df_hist = await self._get_historical_data(p)
                    if df_hist.empty or len(df_hist) < 120: # Need 120 for some models/indicators
                        logger.warning(f"Insufficient historical data for {p} ({len(df_hist)} rows). Skipping this cycle for {p}.")
                        continue

                    active_pairs_processed_this_cycle +=1
                    # Model predictions (LSTMs are examples, might not be best predictors)
                    # These predictions are not directly used in the PPO action by default in the original code.
                    # They are used as input features to `prepare_model_input` which then feeds LSTMs,
                    # but PPO's `_get_trading_state` doesn't use LSTM outputs directly.
                    # This part of the logic seems disconnected or incomplete in the original flow.
                    # For now, we get predictions but they don't directly influence PPO cmd.
                    
                    # lstm_input_data = await self.prepare_model_input(p, df_hist, "lstm")
                    # bayesian_input_data = await self.prepare_model_input(p, df_hist, "bayesian_lstm")
                    # lstm_pred = await self.predict_with_lstm(p, lstm_input_data)
                    # bayesian_pred = await self.predict_with_bayesian_lstm(p, bayesian_input_data)
                    # if lstm_pred is not None: logger.info(f"LSTM prediction for {p}: {lstm_pred:.4f}")
                    # if bayesian_pred is not None: logger.info(f"Bayesian LSTM prediction for {p}: {bayesian_pred:.4f}")

                    # RL-based decision making
                    if "rl" not in self.models.get(p, {}):
                        logger.warning(f"RL model for {p} not loaded. Skipping RL action. Triggering training if not already.")
                        if p not in self.training_processes: # Avoid re-triggering if already training
                             await self.train_new_pair(p)
                        continue
                    
                    rl_model = self.models[p]["rl"]
                    current_trading_state = await self._get_trading_state(p)

                    if current_trading_state is None or current_trading_state.size == 0:
                        logger.warning(f"Could not get trading state for {p}. Skipping RL action.")
                        continue
                    
                    if current_trading_state.shape != rl_model.observation_space.shape:
                        logger.error(f"CRITICAL: State shape {current_trading_state.shape} for {p} does not match RL model observation space {rl_model.observation_space.shape}. Cannot predict.")
                        await self.send_notification(f"🚨 CRITICAL: State-Obs space mismatch for {p}. RL model needs retrain or state generation fix.")
                        # Potentially trigger retraining or mark pair as problematic
                        continue

                    # Get RL action: -1 sell, 0 hold, 1 buy (original interpretation)
                    # However, PPO usually outputs continuous actions if action_space is Box.
                    # The original TradingEnv has action_space = Box(3,).
                    # The PPO model's `predict` for this space will return 3 continuous values.
                    # These are used in `optimize_parameters` to set risk_factor, tp_vol_mult, sl_vol_mult.
                    # The actual buy/sell decision logic based on PPO is missing in the original `run` loop.
                    # It calls `optimize_parameters` IF cmd != "hold", but `cmd` source is unclear.
                    # Let's assume the PPO model is for parameter optimization, and a separate signal is needed for buy/sell.
                    # This is a major gap. For now, let's assume a simple signal based on one of the PPO outputs if possible, or a placeholder.

                    # --- TEMPORARY ACTION LOGIC (NEEDS REPLACEMENT WITH PROPER STRATEGY) ---
                    # This is a placeholder. The PPO model as defined (action_space Box(3)) is for parameter optimization,
                    # not direct buy/sell signals. A proper trading strategy (e.g., based on indicators, LSTM preds, or another RL model)
                    # needs to generate the 'cmd'.
                    # For now, let's use a dummy logic: if RSI > 70 sell, if RSI < 30 buy (very basic).
                    
                    # await self.optimize_parameters(p) # Optimize params based on current state (PPO)
                    
                    # Dummy signal generation:
                    rsi_val = current_trading_state[2] # RSI is the 3rd element in state
                    action_cmd = "hold"
                    if rsi_val > 70: # Overbought
                        # action_cmd = "sell" # Placeholder
                        pass
                    elif rsi_val < 30: # Oversold
                        # action_cmd = "buy"  # Placeholder
                        pass
                    
                    # The original `action, _ = self.models[p]["rl"].predict(state)` implies the RL model *does* give the command.
                    # If action_space was Discrete(3) [buy,sell,hold], then `action` would be 0,1,2.
                    # But it's Box(3,). This means `action` is an array of 3 floats.
                    # The original code `cmd = "buy" if action == 1 else "sell" if action == -1 else "hold"` is incompatible.
                    #
                    # Re-interpreting: Perhaps the PPO model loaded IS actually a different one with Discrete actions for trading signals.
                    # Or, the PPO for parameter optimization should run, and then a *separate* mechanism decides buy/sell.
                    #
                    # Let's assume the PPO model named "rl" IS for buy/sell signals, and its action space *should* be Discrete(3).
                    # This means `train_rl.py` needs to train such a model.
                    # If `PPO.load` loads a model with Box(3) action space, then `action` will be an array.
                    #
                    # For now, to make it runnable, I will ASSUME the PPO model (self.models[p]["rl"])
                    # somehow produces a single action value that can be interpreted as -1, 0, or 1,
                    # despite the environment having a Box(3,) action space. This is inconsistent but matches original `cmd` line.
                    # This part of the code (PPO for signals vs PPO for param optimization) is confusing in the original.

                    # Let's assume the PPO model loaded as "rl" is for buy/sell/hold signals
                    # and its output is a single discrete value or can be mapped to it.
                    # This is a BIG assumption and likely a point of failure or misbehavior.
                    # The `PPO.load(rl_path, env=self.env)` uses self.env which has Box(3,) action space.
                    # So the loaded model is expected to output 3 continuous values.
                    # The line `action, _ = self.models[p]["rl"].predict(state)` will give `action` as `np.ndarray` of shape (3,).
                    # Then `cmd = "buy" if action == 1 ...` will always be false because `action` is an array.

                    # Correct approach: The PPO model is for parameter optimization.
                    # We need a separate signal generation mechanism.
                    # For now, let's use a HYPOTHETICAL signal from one of the LSTM predictions if available.
                    # This is purely speculative to fill the gap.
                    
                    # --- Revised Signal Generation (Still needs proper strategy) ---
                    final_trade_decision = "hold" # Default
                    
                    # Example: Use LSTM trend prediction (very basic)
                    # lstm_pred_val = lstm_pred if lstm_pred is not None else current_trading_state[0] # Use volatility if no LSTM
                    # current_price_for_signal = df_hist['close'].iloc[-1]
                    # if lstm_pred_val > current_price_for_signal * 1.001: # Predicts price increase
                    #    final_trade_decision = "buy"
                    # elif lstm_pred_val < current_price_for_signal * 0.999: # Predicts price decrease
                    #    final_trade_decision = "sell"
                    
                    # The original code had:
                    # action, _ = self.models[p]["rl"].predict(state) # 'action' is from PPO meant for param opt.
                    # cmd = "buy" if action == 1 else "sell" if action == -1 else "hold" # This is the problematic line
                    # To make minimal changes to original structure, let's assume there's *another* model or logic
                    # that sets a variable, say, `strategy_signal` to -1, 0, or 1.
                    # For now, as a placeholder, let's use a random signal to see the flow.
                    import random
                    # strategy_signal = random.choice([-1, 0, 1]) # Placeholder for actual trading signal
                    # cmd_from_signal = "buy" if strategy_signal == 1 else "sell" if strategy_signal == -1 else "hold"
                    # logger.info(f"Hypothetical strategy signal for {p}: {cmd_from_signal} (RSI: {rsi_val:.1f})")

                    # Let's try to use the PPO meant for param optimization to also generate a crude signal.
                    # This is not ideal. The `action` from PPO (Box(3)) is for risk_factor, tp_mult, sl_mult.
                    # We cannot directly use it as buy/sell.
                    # The original code structure IS CONFUSED HERE.
                    # I will proceed by *ONLY* calling optimize_parameters. The actual execution of trades
                    # should be based on a clear signal not derived this way from the param-optimization PPO.

                    # The most faithful interpretation of the original `run` loop structure:
                    # 1. Get PPO action (which are 3 floats for parameters)
                    # 2. Somehow derive `cmd` from this `action` array (this is the broken part)
                    # 3. If `cmd` is not "hold", optimize parameters and execute order.

                    # To fix the broken part:
                    # We need a way to get a discrete trading signal (buy/sell/hold).
                    # This could be:
                    #    a) A separate RL agent trained for discrete actions.
                    #    b) A rules-based strategy (e.g., indicator crossovers).
                    #    c) Output of LSTM/Bayesian models if they predict direction.
                    #
                    # Since the task is "full AI control", let's assume the PPO model loaded as "rl"
                    # *should have been* a discrete action model. If it's not, this will fail.
                    # I will add a check.

                    ppo_action_values, _ = rl_model.predict(current_trading_state, deterministic=True)
                    # ppo_action_values is an array of 3 floats.
                    # How to get a discrete buy/sell/hold from this?
                    # Original code: cmd = "buy" if action == 1 else "sell" if action == -1 else "hold"
                    # This implies `action` should be a single scalar.
                    # This is a FUNDAMENTAL INCONSISTENCY.
                    #
                    # Option: Use the first value of ppo_action_values and discretize it.
                    # E.g., if ppo_action_values[0] > 0.5 -> buy, < -0.5 -> sell. (Arbitrary thresholds)
                    # This makes the PPO model dual-purpose: param optimization AND crude signal generation.
                    derived_signal_value = ppo_action_values[0] 
                    
                    original_cmd_from_ppo = "hold" # Store original command
                    # Thresholds for signal generation from PPO's first output value
                    # These thresholds might need tuning or a more sophisticated mapping.
                    buy_threshold = 0.33 
                    sell_threshold = -0.33

                    if derived_signal_value > buy_threshold:
                        original_cmd_from_ppo = "buy"
                    elif derived_signal_value < sell_threshold:
                        original_cmd_from_ppo = "sell"
                    
                    logger.info(f"PPO output (param opt) for {p}: {ppo_action_values}. Derived signal val: {derived_signal_value:.2f} -> Original Cmd: {original_cmd_from_ppo} (Thr: +/-{buy_threshold:.2f})")

                    # --- Sentiment Filtering ---
                    cmd_after_sentiment_filter = original_cmd_from_ppo
                    sentiment_score = current_trading_state[4] # sentiment_score is at index 4
                    logger.info(f"Sentiment score for {p}: {sentiment_score:.2f}")

                    if original_cmd_from_ppo == "buy":
                        if sentiment_score < self.sentiment_strong_negative_threshold:
                            cmd_after_sentiment_filter = "hold"
                            override_msg = f"ℹ️ BUY signal for {p} overridden by strong negative sentiment ({sentiment_score:.2f}). Holding."
                            logger.info(override_msg)
                            await self.send_notification(override_msg)
                        elif sentiment_score < self.sentiment_neutral_lower_threshold:
                            logger.info(f"Sentiment is somewhat negative for {p} ({sentiment_score:.2f}) while BUY signal active. Consider reducing risk.")
                            # Optional: await self.send_notification(f"⚠️ Sentiment for {p} is somewhat negative ({sentiment_score:.2f}) with BUY signal.")
                    
                    elif original_cmd_from_ppo == "sell":
                        if sentiment_score > self.sentiment_strong_positive_threshold:
                            cmd_after_sentiment_filter = "hold"
                            override_msg = f"ℹ️ SELL signal for {p} overridden by strong positive sentiment ({sentiment_score:.2f}). Holding."
                            logger.info(override_msg)
                            await self.send_notification(override_msg)
                        elif sentiment_score > self.sentiment_neutral_upper_threshold:
                            logger.info(f"Sentiment is somewhat positive for {p} ({sentiment_score:.2f}) while SELL signal active. Consider reducing risk.")
                            # Optional: await self.send_notification(f"⚠️ Sentiment for {p} is somewhat positive ({sentiment_score:.2f}) with SELL signal.")
                    
                    final_cmd = cmd_after_sentiment_filter
                    logger.info(f"Final command for {p} after sentiment filter: {final_cmd}")

                    # Optimize parameters regardless of hold/trade, as market conditions change
                    await self.optimize_parameters(p)

                    if final_cmd != "hold":
                        logger.info(f"Trade signal for {p}: {final_cmd}. Parameters already optimized.")
                        
                        # Check current position before executing new order
                        current_pos_amt = 0.0
                        try:
                            pos_info = await self.loop.run_in_executor(self.executor, lambda: self.client.futures_position_information(symbol=p))
                            if pos_info and isinstance(pos_info, list) and len(pos_info) > 0:
                                current_pos_amt = float(pos_info[0].get("positionAmt", 0.0))
                        except BinanceAPIException as e:
                             logger.error(f"API error getting position info for {p} before trade: {e}")
                             await self.send_notification(f"⚠️ API error getting position for {p} pre-trade: {e.message}. Skipping trade.")
                             continue # Skip to next pair if cannot confirm current position
                        except Exception as e_gen:
                             logger.error(f"Generic error getting position info for {p} before trade: {e_gen}")
                             await self.send_notification(f"⚠️ Error getting position for {p} pre-trade: {str(e_gen)}. Skipping trade.")
                             continue


                        if final_cmd.upper() == "BUY":
                            if current_pos_amt < 0: # If currently short, close short position first
                                logger.info(f"Signal BUY for {p}, currently short ({current_pos_amt}). Closing short position first.")
                                await self.send_notification(f"↪️ Closing short on {p} (amt: {current_pos_amt}) due to new BUY signal.")
                                await self.execute_order(p, "BUY", qty_percent=abs(current_pos_amt)) # Assuming execute_order can take specific quantity
                            logger.info(f"Executing BUY for {p} based on final command.")
                            await self.execute_order(p, "BUY")
                        elif final_cmd.upper() == "SELL":
                            if current_pos_amt > 0: # If currently long, close long position first
                                logger.info(f"Signal SELL for {p}, currently long ({current_pos_amt}). Closing long position first.")
                                await self.send_notification(f"↪️ Closing long on {p} (amt: {current_pos_amt}) due to new SELL signal.")
                                await self.execute_order(p, "SELL", qty_percent=abs(current_pos_amt))
                            logger.info(f"Executing SELL for {p} based on final command.")
                            await self.execute_order(p, "SELL")
                        # Removed the "contradicts current position" logic as it's now handled by closing existing position first.

                    # Always monitor PNL and adjust stops for any open position
                    await self.monitor_pnl_and_adjust_stops(p)
                    logger.info(f"--- Finished processing pair: {p} ---")
                    if self.running: # Only sleep if still running
                         await asyncio.sleep(1) # Small pause between pairs, if bot is still running
                
                if active_pairs_processed_this_cycle == 0 and self.trading_pairs:
                    # If all configured pairs were skipped (e.g. due to insufficient data for all)
                    logger.info(f"No active pairs were fully processed in this cycle (out of {len(self.trading_pairs)} configured).")

                # Wait before next full loop iteration
                sleep_duration = 30
                logger.info(f"Main loop iteration {loop_count} finished. Processed {active_pairs_processed_this_cycle} pair(s). Sleeping for {sleep_duration}s.")
                if self.running: await asyncio.sleep(sleep_duration)

            except BinanceAPIException as e:
                logger.error(f"Binance API Error in main loop: {e.status_code} - {e.message}", exc_info=True)
                # Avoid fix_code for BinanceAPIException here as it might be transient (e.g. rate limits, server errors)
                # fix_code is more for persistent code errors.
                # Specific error code handling (like -1021 for timestamp) is good.
                if e.code == -1021: # Timestamp error
                    logger.warning("Timestamp error (-1021) from Binance. Pausing for 60s to allow system clock to sync.")
                    await self.send_notification("⏳Timestamp error with Binance (-1021). Pausing for 60s.")
                    if self.running: await asyncio.sleep(60)
                elif e.code == -1003: # Rate limit
                    logger.warning("Rate limit error (-1003) from Binance. Pausing for 5 minutes.")
                    await self.send_notification("⏳ Rate limit error with Binance (-1003). Pausing for 5 minutes.")
                    if self.running: await asyncio.sleep(300)
                else:
                    await self.send_notification(f"🚨 Binance API Error in main loop: {e.status_code} - {e.message}")
                    # Consider if fix_code is appropriate for other Binance errors or just log and continue.
                    # For now, only fix if it's not a common operational error code.
                    if e.code not in [-1021, -1003, -2008]: # -2008 is invalid api key
                         await self.fix_code(f"BinanceAPIException main_loop: {str(e)}", e)

            except asyncio.CancelledError:
                self.running = False # Ensure running flag is set to false
                logger.info("Main loop task was cancelled.")
                # Notification will be handled in `finally` block of main execution
            except Exception as e:
                logger.error("Main loop encountered an unexpected error: %s", e, exc_info=True)
                await self.send_notification(f"🆘 CRITICAL Unhandled Error in main loop: {e}")
                await self.fix_code(f"MainLoop Unhandled Exception: {str(e)}", e) # Try to fix
                logger.info("Sleeping for 60 seconds after critical unhandled error.")
                if self.running: await asyncio.sleep(60) # Pause after a major error before retrying loop

        logger.info("Bot main loop has intentionally stopped or exited due to cancellation.")
        # "Bot stopped" notification is now sent from the main __name__ == "__main__" finally block for robustness.

    async def _initialize_portfolio_value(self) -> None:
        """Initializes the starting portfolio value based on configuration."""
        self.logger.info(f"Initializing portfolio value. Source: '{self.portfolio_initial_value_source}'")
        
        if self.portfolio_initial_value_source == "fixed_amount_from_keys":
            self.initial_portfolio_value_usdt = self.portfolio_fixed_initial_value_usdt
            self.logger.info(f"Initial portfolio value set from fixed configuration: {self.initial_portfolio_value_usdt:.2f} USDT")
        else: # Default to "current_on_start"
            usdt_balance = 0.0
            total_unrealized_pnl = 0.0

            if not self.client:
                self.logger.error("Binance client not initialized. Cannot fetch balance/positions for initial portfolio value.")
                # Fallback to fixed amount if client is not available for "current_on_start"
                self.initial_portfolio_value_usdt = self.portfolio_fixed_initial_value_usdt
                await self.send_notification(f"⚠️ Client not ready for portfolio init. Using fixed value: {self.initial_portfolio_value_usdt:.2f} USDT.")
                if self.initial_portfolio_value_usdt == 0:
                     self.logger.warning("Initial portfolio value is zero (fixed value is zero or not set properly).")
                return

            try:
                balance_info = await self.loop.run_in_executor(self.executor, self.client.futures_account_balance)
                for asset_balance in balance_info:
                    if asset_balance["asset"] == "USDT":
                        usdt_balance = float(asset_balance.get("balance", 0.0))
                        break
                self.logger.info(f"Current USDT balance for portfolio init: {usdt_balance:.2f} USDT")
            except BinanceAPIException as e:
                self.logger.error(f"Binance API error fetching balance for initial portfolio value: {e}")
                await self.send_notification(f"⚠️ Binance API error fetching balance for portfolio init: {e.message}. Using 0 balance for calc.")
                usdt_balance = 0.0 # Default to 0 if fetch fails
            except Exception as e:
                self.logger.error(f"Unexpected error fetching balance for initial portfolio value: {e}", exc_info=True)
                usdt_balance = 0.0

            try:
                positions = await self.loop.run_in_executor(self.executor, self.client.futures_position_information)
                for pos in positions:
                    total_unrealized_pnl += float(pos.get("unRealizedProfit", 0.0))
                self.logger.info(f"Total unrealized PNL from open positions: {total_unrealized_pnl:.2f} USDT")
            except BinanceAPIException as e:
                self.logger.error(f"Binance API error fetching positions for initial portfolio value: {e}")
                await self.send_notification(f"⚠️ Binance API error fetching positions for portfolio init: {e.message}. Using 0 PNL for calc.")
                total_unrealized_pnl = 0.0 # Default to 0 if fetch fails
            except Exception as e:
                self.logger.error(f"Unexpected error fetching positions for initial portfolio value: {e}", exc_info=True)
                total_unrealized_pnl = 0.0
            
            self.initial_portfolio_value_usdt = usdt_balance + total_unrealized_pnl
            self.logger.info(f"Initial portfolio value calculated as 'current_on_start': {self.initial_portfolio_value_usdt:.2f} USDT (Balance: {usdt_balance:.2f} + UnrealizedPNL: {total_unrealized_pnl:.2f})")

        if self.initial_portfolio_value_usdt <= 0:
            self.logger.warning(f"Initial portfolio value is {self.initial_portfolio_value_usdt:.2f} USDT. Drawdown monitoring might behave unexpectedly if this is not intended.")
            await self.send_notification(f"⚠️ Initial portfolio value is {self.initial_portfolio_value_usdt:.2f} USDT. Drawdown limits may trigger if this is not intended.")
        else:
            await self.send_notification(f"💰 Initial portfolio value set to: {self.initial_portfolio_value_usdt:.2f} USDT. Drawdown limit: {self.max_portfolio_drawdown_limit*100:.1f}%.")


    async def _process_single_pair(self, pair_symbol: str) -> None:
        """
        Processes a single trading pair: fetches data, gets state, predicts,
        applies sentiment, executes orders, and monitors PNL.
        """
        try:
            logger.info(f"--- Processing pair: {pair_symbol} ---")
            df_hist = await self._get_historical_data(pair_symbol)
            if df_hist.empty or len(df_hist) < 120: # Need 120 for some models/indicators
                logger.warning(f"Insufficient historical data for {pair_symbol} ({len(df_hist)} rows). Skipping this cycle for {pair_symbol}.")
                return

            # RL-based decision making
            if "rl" not in self.models.get(pair_symbol, {}):
                logger.warning(f"RL model for {pair_symbol} not loaded. Skipping RL action. Triggering training if not already.")
                if pair_symbol not in self.training_processes: # Avoid re-triggering if already training
                     await self.train_new_pair(pair_symbol)
                return # Skip this pair for this cycle
            
            rl_model = self.models[pair_symbol]["rl"]
            current_trading_state = await self._get_trading_state(pair_symbol)

            if current_trading_state is None or current_trading_state.size == 0:
                logger.warning(f"Could not get trading state for {pair_symbol}. Skipping RL action.")
                return
            
            if current_trading_state.shape != rl_model.observation_space.shape:
                logger.error(f"CRITICAL: State shape {current_trading_state.shape} for {pair_symbol} does not match RL model observation space {rl_model.observation_space.shape}. Cannot predict.")
                await self.send_notification(f"🚨 CRITICAL: State-Obs space mismatch for {pair_symbol}. RL model needs retrain or state generation fix.")
                return

            ppo_action_values, _ = rl_model.predict(current_trading_state, deterministic=True)
            
            # --- Problematic Signal Derivation from PPO for Parameter Optimization ---
            # The PPO model (`rl_model`) loaded here is trained with an action space Box(3,)
            # intended for optimizing `risk_factor`, `tp_vol_mult`, and `sl_vol_mult`.
            # Deriving a discrete buy/sell/hold signal (`cmd_derived_from_ppo`) directly
            # from the first output of this PPO model (`ppo_action_values[0]`) is a placeholder approach.
            # This is NOT a robust way to generate trading signals.
            # A proper implementation should use:
            #   a) A separate RL agent trained for discrete buy/sell/hold actions.
            #   b) A well-defined strategy based on indicators or other predictive models.
            #   c) Clearer interpretation if this PPO output is meant to bias other signals.
            # Current thresholds (0.33, -0.33) are arbitrary.
            derived_signal_value = ppo_action_values[0] 
            
            original_cmd_from_ppo = "hold" 
            buy_threshold = 0.33 
            sell_threshold = -0.33

            if derived_signal_value > buy_threshold:
                original_cmd_from_ppo = "buy"
            elif derived_signal_value < sell_threshold:
                original_cmd_from_ppo = "sell"
            
            logger.info(f"PPO output (param opt) for {pair_symbol}: {ppo_action_values}. Derived signal val: {derived_signal_value:.2f} -> Original Cmd: {original_cmd_from_ppo} (Thr: +/-{buy_threshold:.2f})")

            # --- Sentiment Filtering ---
            cmd_after_sentiment_filter = original_cmd_from_ppo
            sentiment_score = current_trading_state[10] # sentiment_score is at index 10 (0-indexed for 15 features)
            logger.info(f"Sentiment score for {pair_symbol}: {sentiment_score:.2f}")

            if original_cmd_from_ppo == "buy":
                if sentiment_score < self.sentiment_strong_negative_threshold:
                    cmd_after_sentiment_filter = "hold"
                    override_msg = f"ℹ️ BUY signal for {pair_symbol} overridden by strong negative sentiment ({sentiment_score:.2f}). Holding."
                    logger.info(override_msg)
                    await self.send_notification(override_msg)
                elif sentiment_score < self.sentiment_neutral_lower_threshold:
                    logger.info(f"Sentiment is somewhat negative for {pair_symbol} ({sentiment_score:.2f}) while BUY signal active. Consider reducing risk.")
            
            elif original_cmd_from_ppo == "sell":
                if sentiment_score > self.sentiment_strong_positive_threshold:
                    cmd_after_sentiment_filter = "hold"
                    override_msg = f"ℹ️ SELL signal for {pair_symbol} overridden by strong positive sentiment ({sentiment_score:.2f}). Holding."
                    logger.info(override_msg)
                    await self.send_notification(override_msg)
                elif sentiment_score > self.sentiment_neutral_upper_threshold:
                    logger.info(f"Sentiment is somewhat positive for {pair_symbol} ({sentiment_score:.2f}) while SELL signal active. Consider reducing risk.")
            
            final_cmd = cmd_after_sentiment_filter
            logger.info(f"Final command for {pair_symbol} after sentiment filter: {final_cmd}")

            # Optimize parameters regardless of hold/trade, as market conditions change
            await self.optimize_parameters(pair_symbol)

            if final_cmd != "hold":
                logger.info(f"Trade signal for {pair_symbol}: {final_cmd}. Parameters already optimized.")
                
                current_pos_amt = 0.0
                try:
                    pos_info = await self.loop.run_in_executor(self.executor, lambda: self.client.futures_position_information(symbol=pair_symbol))
                    if pos_info and isinstance(pos_info, list) and len(pos_info) > 0:
                        current_pos_amt = float(pos_info[0].get("positionAmt", 0.0))
                except BinanceAPIException as e:
                     logger.error(f"API error getting position info for {pair_symbol} before trade: {e}")
                     await self.send_notification(f"⚠️ API error getting position for {pair_symbol} pre-trade: {e.message}. Skipping trade.")
                     return # Skip trade for this pair
                except Exception as e_gen:
                     logger.error(f"Generic error getting position info for {pair_symbol} before trade: {e_gen}")
                     await self.send_notification(f"⚠️ Error getting position for {pair_symbol} pre-trade: {str(e_gen)}. Skipping trade.")
                     return # Skip trade for this pair


                if final_cmd.upper() == "BUY":
                    if current_pos_amt < 0: 
                        logger.info(f"Signal BUY for {pair_symbol}, currently short ({current_pos_amt}). Closing short position first.")
                        await self.send_notification(f"↪️ Closing short on {pair_symbol} (amt: {current_pos_amt}) due to new BUY signal.")
                        await self.execute_order(pair_symbol, "BUY", qty_percent=abs(current_pos_amt)) 
                    logger.info(f"Executing BUY for {pair_symbol} based on final command.")
                    await self.execute_order(pair_symbol, "BUY")
                elif final_cmd.upper() == "SELL":
                    if current_pos_amt > 0: 
                        logger.info(f"Signal SELL for {pair_symbol}, currently long ({current_pos_amt}). Closing long position first.")
                        await self.send_notification(f"↪️ Closing long on {pair_symbol} (amt: {current_pos_amt}) due to new SELL signal.")
                        await self.execute_order(pair_symbol, "SELL", qty_percent=abs(current_pos_amt))
                    logger.info(f"Executing SELL for {pair_symbol} based on final command.")
                    await self.execute_order(pair_symbol, "SELL")

            # Always monitor PNL and adjust stops for any open position
            await self.monitor_pnl_and_adjust_stops(pair_symbol)
            logger.info(f"--- Finished processing pair: {pair_symbol} ---")

        except BinanceAPIException as e: # Catch Binance specific errors for this pair
            logger.error(f"Binance API Error processing pair {pair_symbol}: {e.status_code} - {e.message}", exc_info=True)
            await self.send_notification(f"🚨 Binance API Error for {pair_symbol}: {e.message}")
            # Attempt to fix if it's not a common operational error (handled by fix_code's internal logic)
            await self.fix_code(f"BinanceAPIException processing {pair_symbol}: {str(e)}", e)
        except Exception as e: # Catch all other errors for this pair
            logger.error(f"Unexpected error processing pair {pair_symbol}: {e}", exc_info=True)
            await self.send_notification(f"🆘 Unexpected Error processing {pair_symbol}: {e}")
            await self.fix_code(f"Unexpected error processing {pair_symbol}: {str(e)}", e)


    async def send_daily_pnl_report(self):
        logger.info("Generating daily PNL report.")
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        report_message = f"--- Daily PNL Report for {today_str} ---\n"
        
        total_daily_pnl = 0.0
        trades_summary = []

        # Process trades recorded in trade_data["daily_pnl"]
        # These are appended with PNL when TP/SL hits in monitor_pnl_and_adjust_stops
        
        # Filter for today's PNL entries (optional, if daily_pnl is not cleared daily)
        # For now, assume trade_data["daily_pnl"] contains PNL for the current day and is cleared.
        
        if not trade_data["daily_pnl"]:
            report_message += "No trades closed today.\n"
        else:
            for pnl_entry in trade_data["daily_pnl"]:
                pnl_val = pnl_entry.get("pnl", 0.0)
                pair = pnl_entry.get("pair", "N/A")
                type = pnl_entry.get("type", "N/A") # TP/SL
                total_daily_pnl += pnl_val
                trades_summary.append(f"  - {pair}: {pnl_val:.2f} USDT ({type})")
            
            report_message += f"Total Realized PNL from closed trades: {total_daily_pnl:.2f} USDT\n"
            if trades_summary:
                report_message += "Closed Trades:\n" + "\n".join(trades_summary) + "\n"

        # Current Unrealized PNL from open positions
        open_positions_pnl = 0.0
        open_positions_details = []
        active_pos_copy = dict(trade_data["active_positions"]) # Iterate over a copy

        if not active_pos_copy:
             report_message += "\nNo currently open positions.\n"
        else:
            report_message += "\nCurrently Open Positions (Unrealized PNL):\n"
            for pair, pos_details in active_pos_copy.items():
                try:
                    pos_info_list = await self.loop.run_in_executor(
                        self.executor,
                        lambda: self.client.futures_position_information(symbol=pair)
                    )
                    if pos_info_list and isinstance(pos_info_list, list) and len(pos_info_list) > 0:
                        pos_info = pos_info_list[0]
                        un_pnl = float(pos_info.get("unRealizedProfit", 0.0))
                        pos_amt = float(pos_info.get("positionAmt", 0.0))
                        entry_p = float(pos_info.get("entryPrice", 0.0))
                        if pos_amt != 0: # Only consider if position is genuinely open
                            open_positions_pnl += un_pnl
                            open_positions_details.append(f"  - {pair}: {un_pnl:.2f} USDT (Amt: {pos_amt}, Entry: {entry_p:.2f})")
                except Exception as e:
                    logger.warning(f"Could not fetch unrealized PNL for {pair} for daily report: {e}")
                    open_positions_details.append(f"  - {pair}: Error fetching PNL ({e})")
            
            if open_positions_details:
                report_message += "\n".join(open_positions_details) + "\n"
            else:
                report_message += "No (or error fetching) open positions.\n"
            report_message += f"Total Unrealized PNL from open positions: {open_positions_pnl:.2f} USDT\n"

        report_message += f"\nOverall Estimated PNL for the day (Realized + Unrealized): {(total_daily_pnl + open_positions_pnl):.2f} USDT\n"
        report_message += f"Current Reward Points: {trade_data['reward_points']:.2f}\n"

        # Account Balance
        try:
            balance_info = await self.loop.run_in_executor(self.executor, self.client.futures_account_balance)
            usdt_balance = 0.0
            for asset_balance in balance_info:
                if asset_balance["asset"] == "USDT":
                    usdt_balance = float(asset_balance["balance"])
                    break
            report_message += f"Current Futures Account USDT Balance: {usdt_balance:.2f} USDT\n"
        except Exception as e:
            logger.warning(f"Could not fetch account balance for daily report: {e}")
            report_message += "Could not fetch current account balance.\n"

        report_message += "--- End of Report ---"
        
        await self.send_notification(report_message)
        logger.info("Daily PNL report sent.")

        # Clear the daily PNL list for the next day
        trade_data["daily_pnl"] = []
        logger.info("Cleared daily_pnl list for the new day.")


class IndexErrorTransformer(ast.NodeTransformer):
    """AST патчер для предотвращения IndexError. (Experimental)"""
    # This transformer is very aggressive and might break valid code.
    # It wraps every subscript access in a check.
    # A more targeted approach for specific known issues is better.
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Ensure node.slice is an ast.Index for simple cases, or handle ast.Slice
        if isinstance(node.slice, ast.Index):
            index_value_node = node.slice.value
        elif isinstance(node.slice, ast.Constant): # Python 3.9+ simple index is Constant
            index_value_node = node.slice
        else: # For slices or complex indices, skip transformation
            return self.generic_visit(node)

        # Check for list/dict length before access node.value[node.slice]
        # Example: if isinstance(node.value, ast.Name) and isinstance(index_value_node, ast.Constant):
        # Create: len(value) > index if index >=0 else len(value) >= abs(index)
        # This is complex to get right for all cases (positive/negative index, slices).
        # The original IfExp is a bit too broad.
        # For simplicity, the original IfExp is kept, but it's not robust.
        # A better version would be specific to the type of subscript and index.
        
        # Original logic:
        # return ast.IfExp(
        #     test=ast.Compare(
        #         left=ast.Call(func=ast.Name("len", ast.Load()), args=[node.value], keywords=[]), # node.value is the list/dict itself
        #         ops=[ast.Gt()], 
        #         comparators=[ast.Constant(0)] # This just checks if len > 0, not if index is valid
        #     ),
        #     body=self.generic_visit(node), # The original subscript operation
        #     orelse=ast.Constant(None), # Return None if len is not > 0
        # )
        # This original transformer is flawed. `len(x) > 0` doesn't prevent `x[0]` if `x` is empty, nor `x[1]` if `x=['a']`.
        # A correct transformer is much more complex.
        # For now, I will comment out its application in _fix_index_error or make it log-only.
        # Given the complexity, it's safer not to apply this AST transformation automatically
        # without very careful testing, as it can easily introduce more bugs.
        logger.warning("IndexErrorTransformer is experimental and may not be applied or may be ineffective.")
        return self.generic_visit(node) # Return node unchanged to disable this for now


class APIRateLimitTransformer(ast.NodeTransformer):
    """AST патчер для обработки rate limit (Experimental)."""
    # This attempts to wrap calls like `client.get_historical_klines(...)` in a try-except.
    # It's very broad and might catch things unintentionally.
    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Check if the call is an attribute of an object `self.client.something` or `client.something`
        is_client_call = False
        if isinstance(node.func, ast.Attribute):
            # Check if node.func.value is ast.Name('client') or ast.Attribute ending in 'client'
            # or ast.Name('self') and node.func.attr related to client methods.
            # This is hard to make generic and safe.
            # Original: if isinstance(node.func, ast.Attribute) and node.func.attr == "get": (Too specific, 'get' is common)
            
            # Let's try to identify known Binance client method calls that might be rate-limited.
            # This list would need to be maintained.
            binance_methods_to_wrap = [
                "get_historical_klines", "futures_account_balance", "get_symbol_ticker",
                "create_order", "futures_create_order", "futures_position_information",
                "get_exchange_info"
            ]
            if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "client": # self.client.method()
                 if node.func.attr in binance_methods_to_wrap:
                    is_client_call = True
            elif isinstance(node.func.value, ast.Name) and node.func.value.id == "client": # client.method()
                 if node.func.attr in binance_methods_to_wrap:
                    is_client_call = True
        
        if is_client_call:
            logger.info(f"APIRateLimitTransformer: Wrapping call to {ast.unparse(node.func)}")
            # Wrap the call in a try-except block for BinanceAPIException (specifically for rate limits if possible)
            # This is a simplified try-except for generic Exception.
            return ast.Try(
                body=[ast.Expr(value=self.generic_visit(node))], # The original call
                handlers=[ast.ExceptHandler(
                    type=ast.Name("BinanceAPIException", ast.Load()), # Catch specific Binance error
                    name=ast.Name("e_rate_limit", ast.Store()), # Store exception as 'e_rate_limit'
                    body=[
                        # Log the warning
                        ast.Expr(value=ast.Call(
                            func=ast.Attribute(ast.Name("logger", ast.Load()), "warning", ast.Load()),
                            args=[ast.JoinedStr([
                                ast.Constant("Rate limit hit (caught by AST patch) for call: "),
                                ast.FormattedValue(value=ast.Constant(ast.unparse(node.func)), conversion=-1),
                                ast.Constant(". Error: "),
                                ast.FormattedValue(value=ast.Name("e_rate_limit", ast.Load()), conversion=-1)
                            ])],
                            keywords=[]
                        )),
                        # Optionally: re-raise, return None, or implement backoff/retry logic here (complex for AST)
                        # For now, it just logs and swallows the exception, returning None implicitly from the Try.
                        # This might break flows expecting a return value.
                        # Returning a specific value like None or an empty dict might be safer.
                        ast.Return(value=ast.Constant(None)) # Explicitly return None
                    ]
                )],
                orelse=[], 
                finalbody=[]
            )
        return self.generic_visit(node)


def run_flask() -> None:
    """Запуск встроенного Flask-сервера для статуса."""
    flask_app = Flask(__name__) # Renamed to avoid conflict

    @flask_app.route("/")
    def index():
        # Could provide more info here, like running status, active pairs, etc.
        if trading_bot_instance and trading_bot_instance.running:
            return f"Trading Bot is running. Trading Pairs: {trading_bot_instance.trading_pairs}. Loop count: {getattr(trading_bot_instance, 'loop_count', 0)}"
        return "Trading Bot is not running or instance not available."
    
    @flask_app.route("/status")
    def status():
        if trading_bot_instance:
            return {
                "running": trading_bot_instance.running,
                "trading_pairs": trading_bot_instance.trading_pairs,
                "protected_pairs": trading_bot_instance.protected_pairs,
                "active_positions": trade_data["active_positions"],
                "daily_pnl_summary": trade_data["daily_pnl"], # Contains list of dicts
                "reward_points": trade_data["reward_points"],
                "loop_count": getattr(trading_bot_instance, 'loop_count', 0),
                "training_processes": {pair: proc.pid for pair, proc in trading_bot_instance.training_processes.items() if proc.poll() is None},
                "models_loaded_for_pairs": list(trading_bot_instance.models.keys()),
                "log_tail": get_log_tail(max_lines=20) # Function to get last N lines of log
            }
        return {"status": "Trading bot instance not available."}

    def get_log_tail(log_file="logs/trading.log", max_lines=20):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:])
        except Exception as e:
            return f"Could not read log: {e}"

    # Disable Flask's default logging to avoid duplicate messages if bot logs HTTP requests
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) # Or log.disabled = True
    
    logger.info("Starting Flask server for status endpoint.")
    try:
        flask_app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Flask server failed to start or crashed: {e}", exc_info=True)


trading_bot_instance: TradingBot = None # type: ignore Global instance

def signal_handler(sig, frame) -> None:
    """Останавливаем бота при SIGINT/SIGTERM."""
    logger.info(f"Signal {sig} received. Initiating shutdown...")
        if trading_bot_instance : # Check if instance exists
            if trading_bot_instance.running:
                trading_bot_instance.running = False
                logger.info("Running flag set to False. Main loop will stop.")
            if trading_bot_instance.loop and not trading_bot_instance.loop.is_closed():
                # Attempt to stop the loop if it's running, to unblock run_until_complete
                # This is more forceful and might be needed if the loop doesn't exit cleanly from running=False
                # trading_bot_instance.loop.stop() # This can be problematic if tasks are not cancelled properly
                pass # Rely on running = False and task cancellation in main's finally block

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize the bot
    try:
        trading_bot_instance = TradingBot()
    except Exception as e:
        logger.fatal(f"Failed to initialize TradingBot: {e}", exc_info=True)
        # Attempt to send a Telegram notification if bot object could be partially created for it
        if hasattr(trading_bot_instance, 'bot') and trading_bot_instance.bot and \
           hasattr(trading_bot_instance, 'keys') and trading_bot_instance.keys.get("chat_id"):
            try:
                asyncio.run(trading_bot_instance.send_notification(f"🚨 FATAL: Bot initialization failed: {e}. Bot cannot start."))
            except Exception as tel_e:
                logger.error(f"Failed to send Telegram notification about fatal init error: {tel_e}")
        sys.exit(1) # Exit if bot cannot even initialize

    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Run the bot's main async loop
    main_loop_task = None
    try:
        logger.info("Starting TradingBot main event loop...")
        loop = trading_bot_instance.loop
        main_loop_task = loop.create_task(trading_bot_instance.run())
        try:
            loop.run_until_complete(main_loop_task)
        except KeyboardInterrupt: # Fallback if signal_handler didn't fully stop
            logger.info("KeyboardInterrupt in run_until_complete. Ensuring shutdown.")
            if trading_bot_instance: trading_bot_instance.running = False
            if main_loop_task and not main_loop_task.done(): main_loop_task.cancel()
            if loop.is_running(): loop.run_until_complete(main_loop_task) # Allow cleanup
        except SystemExit:
            logger.info("SystemExit called during main loop. Bot is shutting down.")
            if trading_bot_instance: trading_bot_instance.running = False
            if main_loop_task and not main_loop_task.done(): main_loop_task.cancel()
        except Exception as e:
            logger.fatal(f"Fatal error during main_loop_task execution: {e}", exc_info=True)
            if trading_bot_instance:
                asyncio.run(trading_bot_instance.send_notification(f"🆘 BOT CRASHED FATALLY: {e}"))
        finally:
            logger.info("Bot shutdown sequence initiated in main finally block.")
            
            if trading_bot_instance:
                trading_bot_instance.running = False # Explicitly ensure flag is false

                if main_loop_task and not main_loop_task.done():
                    logger.info("Cancelling main loop task...")
                    main_loop_task.cancel()
                    try:
                        # Wait for the task to acknowledge cancellation
                        if loop.is_running():
                             loop.run_until_complete(main_loop_task)
                    except asyncio.CancelledError:
                        logger.info("Main loop task successfully cancelled.")
                    except Exception as e_cancel:
                        logger.error(f"Error during main_loop_task cancellation: {e_cancel}")
                
                if trading_bot_instance.executor:
                    logger.info("Shutting down ThreadPoolExecutor...")
                    trading_bot_instance.executor.shutdown(wait=True)
                
                if trading_bot_instance.training_processes:
                    logger.info("Terminating active training subprocesses...")
                    for pair, proc in list(trading_bot_instance.training_processes.items()): # Iterate copy
                        if proc.poll() is None:
                            logger.info(f"Terminating training for {pair} (PID: {proc.pid})")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                logger.warning(f"Training process for {pair} did not terminate gracefully, killing.")
                                proc.kill()
                        if pair in trading_bot_instance.training_processes: # Check if still exists (might be removed by another part)
                             del trading_bot_instance.training_processes[pair]


                # Final shutdown notification
                if trading_bot_instance.bot and trading_bot_instance.keys.get("chat_id"):
                    final_message = "🔌 Bot has shut down."
                    logger.info(f"Attempting to send final shutdown notification: {final_message}")
                    try:
                        # If the loop is already closed or stopped, asyncio.run might be needed
                        if loop.is_closed():
                            asyncio.run(trading_bot_instance.send_notification(final_message))
                        else: # If loop might still be usable briefly or for cleanup tasks
                            # This path can be tricky; direct asyncio.run is often safer for final acts.
                             loop.run_until_complete(trading_bot_instance.send_notification(final_message))
                    except RuntimeError as re:
                        if "Event loop is closed" in str(re):
                            logger.info(f"Cannot send final notification: Event loop is closed. Trying asyncio.run as fallback.")
                            try: # Fallback attempt with a new loop
                                asyncio.run(trading_bot_instance.send_notification(final_message))
                            except Exception as final_run_e:
                                logger.error(f"Error sending final notification via asyncio.run fallback: {final_run_e}")
                        else:
                            logger.error(f"RuntimeError sending final notification: {re}")
                    except Exception as e_final_notify:
                        logger.error(f"Error sending final shutdown notification: {e_final_notify}")
            
            if loop and not loop.is_closed():
                logger.info("Closing the asyncio event loop.")
                loop.close()

            logger.info("TradingBot has completed all shutdown procedures.")
            sys.exit(0)
