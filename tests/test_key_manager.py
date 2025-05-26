import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import os

# Ensure the main module can be imported
import sys
# Assuming the script is run from the root of the project or tests/
# Adjust if your project structure is different
try:
    from main import KeyManager # Try direct import if main.py is in PYTHONPATH
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from main import KeyManager


class TestKeyManager(unittest.TestCase):
    def setUp(self):
        self.keys_file_path = "secure/keys.json" # As used in KeyManager
        self.default_keys = {
            "api_key": "DEFAULT_API_KEY",
            "api_secret": "DEFAULT_API_SECRET",
            "telegram_token": "DEFAULT_TELEGRAM_TOKEN",
            "chat_id": "DEFAULT_CHAT_ID",
            "trading_pairs": ["BTCUSDT", "ETHUSDT"],
            "liquidity_threshold": 300,
            "volatility_threshold": 0.01,
            "consecutive_loss_threshold": 3,
            "retraining_cooldown_period_hours": 24,
            "cumulative_pnl_check_trades": 10,
            "cumulative_pnl_threshold": -50.0
        }
        # Ensure secure directory exists for testing save_keys, though open is mocked
        os.makedirs(os.path.dirname(self.keys_file_path), exist_ok=True)


    @patch('builtins.open', new_callable=mock_open)
    def test_load_keys_file_missing(self, mock_file_open):
        """Test load_keys returns defaults when keys.json is missing."""
        mock_file_open.side_effect = FileNotFoundError
        km = KeyManager()
        # Override keys_file to ensure it's not accidentally created or read during test
        km.keys_file = "non_existent_temp_keys.json" 
        keys = km.load_keys()
        self.assertEqual(keys, self.default_keys)

    def test_load_keys_file_missing_no_mock_direct(self):
        """Test load_keys returns defaults when keys.json is missing (direct check)."""
        km = KeyManager()
        original_path = km.keys_file
        km.keys_file = "absolutely_non_existent_keys.json" # A path that won't exist
        keys = km.load_keys()
        self.assertEqual(keys, self.default_keys)
        km.keys_file = original_path # Reset path


    @patch('builtins.open', new_callable=mock_open)
    def test_load_keys_invalid_json(self, mock_file_open):
        """Test load_keys returns defaults when keys.json contains invalid JSON."""
        mock_file_open.return_value.read.return_value = "this is not json"
        
        # Mock json.load to raise JSONDecodeError when called with the bad content
        with patch('json.load', side_effect=json.JSONDecodeError("Error", "doc", 0)):
            km = KeyManager()
            keys = km.load_keys()
            self.assertEqual(keys, self.default_keys)
            mock_file_open.assert_called_once_with(km.keys_file, "r", encoding="utf-8")


    @patch('builtins.open', new_callable=mock_open)
    def test_load_keys_valid_json(self, mock_file_open):
        """Test load_keys loads data correctly from a valid keys.json."""
        valid_data = {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret",
            "telegram_token": "test_telegram_token",
            "chat_id": "test_chat_id",
            "trading_pairs": ["BTCUSDT", "ETHUSDT", "LINKUSDT"],
            "liquidity_threshold": 1000,
            "volatility_threshold": 0.02,
            "consecutive_loss_threshold": 5
            # Not all default keys are present, testing merging logic
        }
        mock_file_open.return_value.read.return_value = json.dumps(valid_data)

        # Patch json.load to return the valid_data
        with patch('json.load', return_value=valid_data):
            km = KeyManager()
            keys = km.load_keys()

        # Check that loaded keys override defaults where specified
        self.assertEqual(keys["api_key"], valid_data["api_key"])
        self.assertEqual(keys["trading_pairs"], valid_data["trading_pairs"])
        self.assertEqual(keys["liquidity_threshold"], valid_data["liquidity_threshold"])
        # Check that default keys are present if not in valid_data
        self.assertEqual(keys["cumulative_pnl_threshold"], self.default_keys["cumulative_pnl_threshold"])
        mock_file_open.assert_called_once_with(km.keys_file, "r", encoding="utf-8")


    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_keys(self, mock_json_dump, mock_file_open):
        """Test save_keys attempts to write correct data to keys.json."""
        km = KeyManager()
        test_data = {"test_key": "test_value"}
        km.save_keys(test_data)

        mock_file_open.assert_called_once_with(km.keys_file, "w", encoding="utf-8")
        mock_json_dump.assert_called_once_with(test_data, mock_file_open.return_value.__enter__.return_value, indent=4)

    @patch.object(KeyManager, 'load_keys')
    @patch.object(KeyManager, 'save_keys')
    def test_update_trading_pairs(self, mock_save_keys, mock_load_keys):
        """Test update_trading_pairs correctly adds new pairs and handles duplicates."""
        initial_keys_data = {
            "trading_pairs": ["BTCUSDT", "ETHUSDT"],
            # other keys...
        }
        mock_load_keys.return_value = initial_keys_data.copy() # Return a copy to avoid modification issues
        
        km = KeyManager()
        
        # Test adding new pairs
        new_pairs_to_add = ["ADAUSDT", "SOLUSDT"]
        km.update_trading_pairs(new_pairs_to_add)
        
        # Verify save_keys was called
        mock_save_keys.assert_called_once()
        # Get the arguments passed to save_keys
        saved_data = mock_save_keys.call_args[0][0]
        expected_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        self.assertCountEqual(saved_data["trading_pairs"], expected_pairs, "Should add new unique pairs")

        # Reset mock for next call
        mock_save_keys.reset_mock()
        # Update initial_keys_data for the next scenario to reflect the last save
        initial_keys_data["trading_pairs"] = expected_pairs 
        mock_load_keys.return_value = initial_keys_data.copy()


        # Test adding duplicate and existing pairs
        duplicate_pairs_to_add = ["ETHUSDT", "XRPUSDT"]
        km.update_trading_pairs(duplicate_pairs_to_add)
        
        mock_save_keys.assert_called_once()
        saved_data_duplicates = mock_save_keys.call_args[0][0]
        expected_pairs_duplicates = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"]
        self.assertCountEqual(saved_data_duplicates["trading_pairs"], expected_pairs_duplicates, "Should handle duplicates and add new unique pairs")

        # Test adding to an empty list of pairs
        mock_save_keys.reset_mock()
        initial_keys_data["trading_pairs"] = []
        mock_load_keys.return_value = initial_keys_data.copy()
        km.update_trading_pairs(["DOTUSDT"])
        mock_save_keys.assert_called_once()
        saved_data_empty = mock_save_keys.call_args[0][0]
        self.assertCountEqual(saved_data_empty["trading_pairs"], ["DOTUSDT"])


if __name__ == '__main__':
    unittest.main()
