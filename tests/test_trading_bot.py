import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os

# Adjust path to import from main.py, assuming tests/ is a subdirectory of the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import TradingBot, KeyManager, TradingEnv, trade_data, BinanceAPIException # Assuming these can be imported

class TestTradingBot(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Mock KeyManager instance and its load_keys method
        self.mock_key_manager_instance = MagicMock(spec=KeyManager)
        self.test_keys = {
            "api_key": "test_key", "api_secret": "test_secret",
            "telegram_token": "test_token", "chat_id": "123",
            "trading_pairs": ["BTCUSDT", "ETHUSDT"], # Start with some pairs
            "liquidity_threshold": 300, "volatility_threshold": 0.01,
            "consecutive_loss_threshold": 3, "retraining_cooldown_period_hours": 24,
            "cumulative_pnl_check_trades": 10, "cumulative_pnl_threshold": -50.0,
            "sentiment_strong_positive_threshold": 0.6,
            "sentiment_strong_negative_threshold": -0.6,
            "sentiment_neutral_upper_threshold": 0.2,
            "sentiment_neutral_lower_threshold": -0.2,
        }
        self.mock_key_manager_instance.load_keys.return_value = self.test_keys.copy()

        # Patch external dependencies and TradingBot's internal initializers
        with patch('main.Client', new_callable=MagicMock) as self.mock_binance_client_constructor, \
             patch('main.Bot', new_callable=MagicMock) as self.mock_telegram_bot_constructor, \
             patch('main.PPO.load', new_callable=MagicMock) as self.mock_ppo_load, \
             patch('main.PPO', new_callable=MagicMock) as self.mock_ppo_constructor, \
             patch('main.KeyManager', return_value=self.mock_key_manager_instance), \
             patch.object(TradingBot, '_initialize_models', MagicMock(return_value=None)) as self.mock_initialize_models:

            # _load_keys_and_models is a bit complex with its own async calls if loop is running
            # We can mock its behavior or parts of it.
            # For simplicity, let's assume it correctly sets up self.keys, self.client, self.bot, self.trading_pairs
            # by patching it, but then manually setting these attributes.
            with patch.object(TradingBot, '_load_keys_and_models', MagicMock(return_value=None)) as self.mock_load_keys_and_models_method:
                self.bot = TradingBot() # TradingBot.__init__ calls _load_keys_and_models

                # Manually set attributes that _load_keys_and_models would set
                self.bot.keys = self.test_keys.copy()
                self.bot.client = self.mock_binance_client_constructor.return_value
                self.bot.bot = self.mock_telegram_bot_constructor.return_value
                self.bot.trading_pairs = self.bot.keys["trading_pairs"]
                
                # Re-apply thresholds from keys after manual setup
                self.bot.retraining_cooldown_period_hours = self.bot.keys.get("retraining_cooldown_period_hours", 24)
                self.bot.consecutive_loss_threshold = self.bot.keys.get("consecutive_loss_threshold", 3)
                self.bot.cumulative_pnl_check_trades = self.bot.keys.get("cumulative_pnl_check_trades", 10)
                self.bot.cumulative_pnl_threshold = self.bot.keys.get("cumulative_pnl_threshold", -50.0)
                self.bot.liquidity_threshold = self.bot.keys.get("liquidity_threshold", 300)
                self.bot.volatility_threshold = self.bot.keys.get("volatility_threshold", 0.01)
                self.bot.sentiment_strong_positive_threshold = self.bot.keys.get("sentiment_strong_positive_threshold", 0.6)
                self.bot.sentiment_strong_negative_threshold = self.bot.keys.get("sentiment_strong_negative_threshold", -0.6)
                self.bot.sentiment_neutral_upper_threshold = self.bot.keys.get("sentiment_neutral_upper_threshold", 0.2)
                self.bot.sentiment_neutral_lower_threshold = self.bot.keys.get("sentiment_neutral_lower_threshold", -0.2)


        self.bot.loop = self.loop # Assign the test loop
        self.bot.ai_optimizer = self.mock_ppo_constructor.return_value # General PPO for parameter optimization
        
        # Ensure models dict is initialized for pairs
        self.bot.models = {pair: {} for pair in self.bot.trading_pairs}
        for pair in self.bot.trading_pairs: # Mock an RL model for each pair
            self.bot.models[pair]["rl"] = MagicMock() # This is the PPO model for signals/param opt for a pair
            self.bot.models[pair]["rl"].predict = MagicMock(return_value=(np.array([0.0,0.0,0.0]), None))


        # Reset global trade_data for each test if it's modified
        global trade_data
        trade_data = {
            "reward_points": 0.0, "daily_pnl": [],
            "strategy_success": {}, "active_positions": {},
        }
        self.bot.trade_history = {} # Clear bot's internal trade history too
        self.bot.retraining_cooldown = {}

    def tearDown(self):
        self.loop.close()

    async def mock_get_trading_state(self, pair_symbol, sentiment_score=0.0):
        """Helper to create a mock trading state with a specific sentiment score."""
        # Returns a 15-feature array
        state = np.zeros(15, dtype=np.float32)
        state[10] = sentiment_score # sentiment_score is at index 10
        return state

    @patch('main.TradingBot.send_notification', new_callable=AsyncMock)
    async def test_sentiment_override_buy_strong_negative(self, mock_send_notification):
        pair = "BTCUSDT"
        self.bot.models[pair]["rl"].predict.return_value = (np.array([0.5, 0, 0]), None) # Strong buy signal from PPO
        
        with patch.object(self.bot, '_get_trading_state', new_callable=AsyncMock) as mock_state:
            mock_state.return_value = await self.mock_get_trading_state(pair, sentiment_score=-0.7)
            
            with patch.object(self.bot, 'execute_order', new_callable=AsyncMock) as mock_execute_order:
                await self.bot._process_single_pair(pair)
                
                # Verify it holds (no order executed) and notification sent
                mock_execute_order.assert_not_called()
                mock_send_notification.assert_called_with(f"ℹ️ BUY signal for {pair} overridden by strong negative sentiment (-0.70). Holding.")

    @patch('main.TradingBot.send_notification', new_callable=AsyncMock)
    async def test_sentiment_no_override_buy_neutral(self, mock_send_notification):
        pair = "BTCUSDT"
        self.bot.models[pair]["rl"].predict.return_value = (np.array([0.5, 0, 0]), None) # Strong buy
        
        with patch.object(self.bot, '_get_trading_state', new_callable=AsyncMock) as mock_state:
            mock_state.return_value = await self.mock_get_trading_state(pair, sentiment_score=0.1) # Neutral sentiment
            
            with patch.object(self.bot, 'execute_order', new_callable=AsyncMock) as mock_execute_order:
                await self.bot._process_single_pair(pair)
                mock_execute_order.assert_any_call(pair, "BUY") # Should proceed to buy

    @patch('main.TradingBot.send_notification', new_callable=AsyncMock)
    async def test_sentiment_override_sell_strong_positive(self, mock_send_notification):
        pair = "ETHUSDT"
        self.bot.models[pair]["rl"].predict.return_value = (np.array([-0.5, 0, 0]), None) # Strong sell
        
        with patch.object(self.bot, '_get_trading_state', new_callable=AsyncMock) as mock_state:
            mock_state.return_value = await self.mock_get_trading_state(pair, sentiment_score=0.8) # Strong positive
            
            with patch.object(self.bot, 'execute_order', new_callable=AsyncMock) as mock_execute_order:
                await self.bot._process_single_pair(pair)
                mock_execute_order.assert_not_called()
                mock_send_notification.assert_called_with(f"ℹ️ SELL signal for {pair} overridden by strong positive sentiment (0.80). Holding.")
    
    async def test_parameter_scaling(self):
        test_cases = [
            ([-1.0, -1.0, -1.0], [1.5, 1.0, 0.5]), # Min values
            ([1.0, 1.0, 1.0],   [3.5, 5.0, 2.0]), # Max values
            ([0.0, 0.0, 0.0],   [2.5, 3.0, 1.25]),# Mid values
            ([-2.0, -2.0, -2.0], [1.5, 1.0, 0.5]), # Below min (clamped)
            ([2.0, 2.0, 2.0],   [3.5, 5.0, 2.0]), # Above max (clamped)
        ]
        # Original clamping ranges: risk_factor [1.0, 5.0], tp_vol_mult [0.5, 7.0], sl_vol_mult [0.3, 3.0]
        # Parameter mapping:
        # risk_factor: [-1,1] -> [1.5, 3.5]
        # tp_vol_mult: [-1,1] -> [1.0, 5.0]
        # sl_vol_mult: [-1,1] -> [0.5, 2.0]

        for ppo_actions, expected_params in test_cases:
            with patch.object(self.bot, '_get_trading_state', new_callable=AsyncMock) as mock_state:
                 # Mock PPO to return the test actions
                self.bot.ai_optimizer.predict.return_value = (np.array(ppo_actions), None)
                mock_state.return_value = await self.mock_get_trading_state("BTCUSDT")
                
                await self.bot.optimize_parameters("BTCUSDT")
                
                # Check if parameters are scaled and clamped correctly
                # Note: The PPO for param optimization is self.bot.ai_optimizer
                # The one for signals is self.bot.models[pair]["rl"]
                self.assertAlmostEqual(self.bot.risk_factor, expected_params[0], places=2)
                self.assertAlmostEqual(self.bot.tp_vol_mult, expected_params[1], places=2)
                self.assertAlmostEqual(self.bot.sl_vol_mult, expected_params[2], places=2)

    @patch('main.TradingBot._notify_performance_retraining', new_callable=AsyncMock)
    @patch('main.TradingBot.train_new_pair', new_callable=AsyncMock)
    @patch('main.TradingBot.train_lstm_models', new_callable=AsyncMock)
    async def test_retraining_consecutive_losses(self, mock_train_lstm, mock_train_rl, mock_notify_retrain):
        pair = "BTCUSDT"
        self.bot.consecutive_loss_threshold = 3
        self.bot.trade_history[pair] = [
            {'pnl': -10, 'timestamp': datetime.utcnow() - timedelta(minutes=30)},
            {'pnl': -15, 'timestamp': datetime.utcnow() - timedelta(minutes=20)},
            {'pnl': -5,  'timestamp': datetime.utcnow() - timedelta(minutes=10)},
        ]
        await self.bot._check_and_trigger_retraining(pair)
        mock_notify_retrain.assert_called_once_with(pair, "3 consecutive losses.")
        mock_train_rl.assert_called_once_with(pair)
        mock_train_lstm.assert_called_once_with(pair)
        self.assertIn(pair, self.bot.retraining_cooldown)

    @patch('main.TradingBot._notify_performance_retraining', new_callable=AsyncMock)
    @patch('main.TradingBot.train_new_pair', new_callable=AsyncMock)
    @patch('main.TradingBot.train_lstm_models', new_callable=AsyncMock)
    async def test_retraining_cumulative_pnl(self, mock_train_lstm, mock_train_rl, mock_notify_retrain):
        pair = "ETHUSDT"
        self.bot.cumulative_pnl_check_trades = 3
        self.bot.cumulative_pnl_threshold = -20.0
        self.bot.trade_history[pair] = [
            {'pnl': 5,   'timestamp': datetime.utcnow() - timedelta(minutes=30)},
            {'pnl': -15, 'timestamp': datetime.utcnow() - timedelta(minutes=20)},
            {'pnl': -12, 'timestamp': datetime.utcnow() - timedelta(minutes=10)}, # Cumulative = -22
        ]
        await self.bot._check_and_trigger_retraining(pair)
        expected_reason = "Cumulative PNL of last 3 trades (-22.00) is below threshold (-20.00)."
        mock_notify_retrain.assert_called_once_with(pair, expected_reason)
        mock_train_rl.assert_called_once_with(pair)
        mock_train_lstm.assert_called_once_with(pair)

    @patch('main.TradingBot._notify_performance_retraining', new_callable=AsyncMock)
    async def test_retraining_cooldown(self, mock_notify_retrain):
        pair = "BTCUSDT"
        self.bot.consecutive_loss_threshold = 1
        self.bot.trade_history[pair] = [{'pnl': -10, 'timestamp': datetime.utcnow()}]
        
        # First trigger
        await self.bot._check_and_trigger_retraining(pair)
        mock_notify_retrain.assert_called_once()
        self.assertIn(pair, self.bot.retraining_cooldown)
        
        # Second immediate trigger attempt
        mock_notify_retrain.reset_mock()
        self.bot.trade_history[pair].append({'pnl': -5, 'timestamp': datetime.utcnow()}) # Add another loss
        await self.bot._check_and_trigger_retraining(pair)
        mock_notify_retrain.assert_not_called() # Should be skipped due to cooldown

    @patch('main.TradingBot.send_notification', new_callable=AsyncMock)
    async def test_daily_pnl_report(self, mock_send_notification):
        global trade_data # Ensure we modify the global directly for this test
        trade_data["daily_pnl"] = [
            {"pair": "BTCUSDT", "pnl": 100.0, "type": "TP"},
            {"pair": "ETHUSDT", "pnl": -50.0, "type": "SL"},
        ]
        trade_data["reward_points"] = 5.5

        # Mock client calls for open positions and balance
        mock_btc_pos = [{"symbol": "BTCUSDT", "positionAmt": "0.5", "unRealizedProfit": "25.0", "entryPrice": "50000"}]
        mock_ada_pos = [{"symbol": "ADAUSDT", "positionAmt": "100", "unRealizedProfit": "-10.0", "entryPrice": "1.5"}]
        
        self.bot.client.futures_position_information = AsyncMock()
        self.bot.client.futures_position_information.side_effect = lambda symbol: {
            "BTCUSDT": mock_btc_pos,
            "ADAUSDT": mock_ada_pos,
            "ETHUSDT": [] # ETH was closed
        }.get(symbol, [])
        
        self.bot.client.futures_account_balance = AsyncMock(return_value=[{"asset": "USDT", "balance": "10000.0"}])
        
        # Simulate some active positions in trade_data as well (though report fetches fresh)
        trade_data["active_positions"] = {
            "BTCUSDT": {"side": "BUY", "quantity": 0.5, "entry_price": 50000},
            "ADAUSDT": {"side": "BUY", "quantity": 100, "entry_price": 1.5},
        }
        self.bot.trading_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT"] # Ensure ADA is considered

        await self.bot.send_daily_pnl_report()

        mock_send_notification.assert_called_once()
        report_message = mock_send_notification.call_args[0][0]

        self.assertIn("Total Realized PNL from closed trades: 50.00 USDT", report_message)
        self.assertIn("BTCUSDT: 100.00 USDT (TP)", report_message)
        self.assertIn("ETHUSDT: -50.00 USDT (SL)", report_message)
        self.assertIn("BTCUSDT: 25.00 USDT", report_message) # Unrealized
        self.assertIn("ADAUSDT: -10.00 USDT", report_message) # Unrealized
        self.assertIn("Total Unrealized PNL from open positions: 15.00 USDT", report_message)
        self.assertIn("Overall Estimated PNL for the day (Realized + Unrealized): 65.00 USDT", report_message)
        self.assertIn("Current Reward Points: 5.50", report_message)
        self.assertIn("Current Futures Account USDT Balance: 10000.00 USDT", report_message)
        
        # Check if daily_pnl is cleared
        self.assertEqual(trade_data["daily_pnl"], [])


if __name__ == '__main__':
    unittest.main()
