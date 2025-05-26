# -*- coding: utf-8 -*-
"""
Sentiment Analyzer class for fetching and analyzing news sentiment for cryptocurrencies.
"""
import logging
import numpy as np
import os
import json
import requests 
import nltk
from datetime import datetime, timedelta, timezone
import time

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except LookupError:
    nltk.download('vader_lexicon', quiet=True) # Added quiet=True
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    from transformers import pipeline
except ImportError:
    logging.warning("transformers library not found. Hugging Face model pipeline will be unavailable.")
    pipeline = None # Ensure pipeline is defined to avoid NameError

# Simple mapping for coin symbols to full names for better news search
COIN_NAME_MAP = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "ADA": "Cardano",
    "SOL": "Solana",
    "XRP": "Ripple XRP",
    "DOT": "Polkadot",
    "DOGE": "Dogecoin",
    "AVAX": "Avalanche",
    "MATIC": "Polygon MATIC",
    # Add more as needed
}

class SentimentAnalyzer:
    """
    Analyzes text sentiment using Hugging Face Transformers and/or NLTK's VADER.
    Includes news fetching from configured sources (e.g., NewsAPI) and caching.
    """
    def __init__(self, news_api_config: dict, cache_dir: str = "sentiment_cache", cache_ttl_seconds: int = 3600, request_timeout: int = 10):
        """
        Initializes the SentimentAnalyzer.

        Args:
            news_api_config (dict): Configuration for news APIs.
            cache_dir (str): Directory to store cached sentiment scores.
            cache_ttl_seconds (int): Time-to-live for cache files in seconds.
            request_timeout (int): Timeout for HTTP requests to news APIs.
        """
        self.news_api_config = news_api_config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger.info("SentimentAnalyzer logger configured with basicConfig.")

        self.sentiment_pipeline = None
        if pipeline:
            try:
                self.logger.info("Attempting to load sentiment pipeline: ProsusAI/finbert")
                self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                self.logger.info("Successfully loaded ProsusAI/finbert model.")
            except Exception as e_finbert:
                self.logger.warning(f"Failed to load ProsusAI/finbert: {e_finbert}. Trying distilbert fallback.")
                try:
                    self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                    self.logger.info("Successfully loaded distilbert-base-uncased-finetuned-sst-2-english model.")
                except Exception as e_distilbert:
                    self.logger.warning(f"Failed to load distilbert fallback model: {e_distilbert}. Hugging Face pipeline will be unavailable.")
                    self.sentiment_pipeline = None
        else:
            self.logger.warning("Transformers 'pipeline' function not available. Hugging Face model support disabled.")

        self.logger.info("Initializing NLTK VADER sentiment analyzer.")
        self.vader_analyzer = SentimentIntensityAnalyzer()

        self.cache_dir = cache_dir
        self.cache_ttl_seconds = cache_ttl_seconds
        self.request_timeout = request_timeout
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"Cache directory '{self.cache_dir}' ensured. TTL: {self.cache_ttl_seconds}s.")
        except OSError as e:
            self.logger.error(f"Could not create cache directory '{self.cache_dir}': {e}")
            self.cache_dir = None 

    def _fetch_from_cache(self, cache_key: str) -> dict | None:
        """
        Fetches data from cache if it exists and is not stale.
        """
        if not self.cache_dir:
            return None
        
        file_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(file_path):
            try:
                file_mod_time = os.path.getmtime(file_path)
                if (time.time() - file_mod_time) < self.cache_ttl_seconds:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.logger.info(f"Cache hit and valid for key: {cache_key}")
                    return data
                else:
                    self.logger.info(f"Cache stale for key: {cache_key} (older than {self.cache_ttl_seconds}s).")
                    return None
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Error reading or processing cache file {file_path}: {e}")
                return None
        self.logger.debug(f"Cache miss for key: {cache_key}")
        return None

    def _save_to_cache(self, cache_key: str, data: dict) -> None:
        """
        Saves data to cache as JSON.
        """
        if not self.cache_dir:
            return

        file_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            # Add a timestamp to the data being saved for potential future use/debugging
            data_to_save = data.copy()
            data_to_save['cached_at_utc'] = datetime.now(timezone.utc).isoformat()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
            self.logger.info(f"Saved data to cache for key: {cache_key}")
        except OSError as e:
            self.logger.warning(f"Error writing cache file {file_path}: {e}")

    def get_news_sentiment(self, coin_symbol: str) -> dict:
        """
        Fetches news from configured sources (NewsAPI example implemented),
        analyzes sentiment, and returns an aggregated score.

        Args:
            coin_symbol (str): The symbol of the coin (e.g., "BTC", "ETH").

        Returns:
            dict: e.g., {"score": final_score, "source": "real_news_api/multiple_sources", "articles_analyzed": count}
        """
        if not self.news_api_config.get("news_fetch_enabled", False):
            self.logger.info("News fetching is disabled globally in news_api_config.")
            return {"score": 0.0, "source": "news_fetch_disabled", "articles_analyzed": 0}

        enabled_sources = [s for s in self.news_api_config.get("sources", []) if s.get("enabled")]
        if not enabled_sources:
            self.logger.info("No news sources are enabled in news_api_config.")
            return {"score": 0.0, "source": "no_sources_enabled", "articles_analyzed": 0}

        overall_cache_key = f"{coin_symbol}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}"
        cached_overall_sentiment = self._fetch_from_cache(overall_cache_key)
        if cached_overall_sentiment:
            self.logger.info(f"Returning overall cached sentiment for {coin_symbol} from key {overall_cache_key}.")
            return cached_overall_sentiment

        all_article_scores = []
        articles_analyzed_count = 0
        sources_used = []
        max_articles_per_source = 5 # Limit articles from each source

        coin_name = COIN_NAME_MAP.get(coin_symbol.upper(), coin_symbol) # Get full name for better search

        for source_config in enabled_sources:
            source_name = source_config.get("name")
            api_key = source_config.get("api_key")
            
            # Check if coin is supported by this source
            supported_coins = source_config.get("coins_supported", [])
            if "general" not in supported_coins and coin_symbol.upper() not in [c.upper() for c in supported_coins]:
                self.logger.debug(f"Skipping source {source_name} as it does not support {coin_symbol}.")
                continue
            
            self.logger.info(f"Processing source: {source_name} for {coin_symbol}")
            sources_used.append(source_name)

            if source_name == "NewsAPI":
                if not api_key or "YOUR_NEWSAPI_KEY" in api_key: # Check for placeholder key
                    self.logger.warning(f"NewsAPI key not configured or is placeholder for source {source_name}. Skipping.")
                    continue
                
                # Construct query: prefer full coin name, fallback to symbol. Include "cryptocurrency".
                query = f'("{coin_name}" OR "{coin_symbol}") AND cryptocurrency'
                url = (f"https://newsapi.org/v2/everything?q={query}"
                       f"&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize={max_articles_per_source * 2}") # Fetch more to filter if needed

                self.logger.debug(f"Fetching NewsAPI URL: {url}")
                try:
                    response = requests.get(url, timeout=self.request_timeout)
                    response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                    news_data = response.json()
                    
                    articles = news_data.get("articles", [])
                    if not articles:
                        self.logger.info(f"No articles found from NewsAPI for {coin_symbol} with query '{query}'.")
                        continue

                    for article in articles[:max_articles_per_source]:
                        title = article.get("title", "")
                        description = article.get("description", "")
                        content = article.get("content", "") # Some articles might have content

                        text_to_analyze = f"{title}. {description if description else ''}. {content if content else ''}"
                        if not text_to_analyze.strip() or text_to_analyze == ". .":
                            self.logger.debug(f"Skipping empty article text for '{title}'")
                            continue
                        
                        article_score = self.analyze_text_sentiment(text_to_analyze)
                        all_article_scores.append(article_score)
                        articles_analyzed_count += 1
                        self.logger.debug(f"Analyzed NewsAPI article '{title[:50]}...': Score {article_score:.4f}")

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error fetching news from NewsAPI for {coin_symbol}: {e}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error decoding JSON from NewsAPI for {coin_symbol}: {e}")
            
            # TODO: Add elif blocks here for other news sources like CryptoPanic
            # elif source_name == "CryptoPanic":
            #     # Implement CryptoPanic fetching logic
            #     pass

        if not all_article_scores:
            self.logger.info(f"No articles found or analyzed for {coin_symbol} from any enabled source.")
            result = {"score": 0.0, "source": "no_articles_found", "articles_analyzed": 0}
        else:
            final_aggregated_score = np.mean(all_article_scores)
            # Ensure score is clipped, though analyze_text_sentiment should also do this.
            final_aggregated_score = np.clip(final_aggregated_score, -1.0, 1.0) 
            self.logger.info(f"Aggregated sentiment for {coin_symbol} from {articles_analyzed_count} articles ({len(sources_used)} sources): {final_aggregated_score:.4f}")
            source_str = "/".join(sorted(list(set(sources_used)))) if sources_used else "multiple_sources"
            result = {"score": float(final_aggregated_score), "source": f"real_news_api/{source_str}", "articles_analyzed": articles_analyzed_count}

        self._save_to_cache(overall_cache_key, result) # Cache the final aggregated result
        return result


    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyzes the sentiment of a single piece of text.
        """
        if not isinstance(text, str) or not text.strip():
            self.logger.warning("Received empty or non-string text for analysis. Returning neutral sentiment.")
            return 0.0

        try:
            if self.sentiment_pipeline:
                self.logger.debug("Using Hugging Face pipeline for sentiment analysis.")
                # Ensure text is not excessively long for the model
                # Most models have a limit like 512 tokens.
                # Simple truncation by string length; token-based truncation is more accurate but complex here.
                max_chars = 2000 # Heuristic, roughly 500 words. Adjust as needed.
                truncated_text = text[:max_chars] if len(text) > max_chars else text

                result = self.sentiment_pipeline(truncated_text, truncation=True)[0] # Some models might need explicit truncation=True
                score = result['score']
                label = result['label'].lower()
                
                # Normalize score based on label (common for FinBERT and similar models)
                if label in ['negative', '1 star', '2 stars', 'label_0'] or ('neg' in label and 'neu' not in label) : # Common negative labels
                    score = -score
                elif label in ['neutral', '3 stars', 'label_1']: # Common neutral labels
                    score = 0.0 
                # Positive labels ('positive', '4 stars', '5 stars', 'label_2') usually have positive scores already.
                # If score is confidence (0 to 1), and label is just 'positive', it's fine.
                
                score = np.clip(score, -1.0, 1.0)
                self.logger.debug(f"HF Pipeline result: Label={label}, RawScore={result['score']:.4f} -> Normalized: {score:.4f} for text: '{text[:50]}...'")
                return score
            else: # Fallback to VADER
                self.logger.debug("Using VADER for sentiment analysis as HF pipeline is unavailable.")
                vs = self.vader_analyzer.polarity_scores(text)
                compound_score = vs['compound']
                self.logger.debug(f"VADER result: Compound={compound_score:.4f} for text: '{text[:50]}...'")
                return compound_score
        except Exception as e:
            self.logger.error(f"Error during sentiment analysis of text: '{text[:50]}...': {e}", exc_info=True)
            if self.sentiment_pipeline is not None: 
                self.logger.warning("Falling back to VADER due to error in HF pipeline during analysis.")
                try:
                    vs = self.vader_analyzer.polarity_scores(text)
                    return vs['compound']
                except Exception as e_vader_fallback:
                    self.logger.error(f"Error during VADER fallback analysis: {e_vader_fallback}")
            return 0.0 

    async def fetch_futures_listings(self, keys_unused: dict = None) -> list[str]:
        self.logger.info("fetch_futures_listings not implemented in new SentimentAnalyzer. Returning placeholder list.")
        return ["BTCUSDT_SA", "ETHUSDT_SA"] 

    async def monitor_news_for_new_coins(self, keys_unused: dict = None) -> list[str]:
        self.logger.info("monitor_news_for_new_coins not implemented in new SentimentAnalyzer. Returning empty list.")
        return []


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger(__name__)
    
    # Example news_api_config (replace with your actual key for NewsAPI to test fetching)
    dummy_config_real_key = {
        "news_fetch_enabled": True,
        "sources": [
            {"name": "NewsAPI", "api_key": os.getenv("NEWSAPI_ORG_KEY", "YOUR_NEWSAPI_KEY_DEFAULT"), "enabled": True, "coins_supported": ["BTC", "ETH", "general"]},
            {"name": "CryptoPanic", "api_key": "YOUR_CRYPTOPANIC_KEY_DEFAULT", "enabled": False, "coins_supported": ["general"]}
        ]
    }
    if "YOUR_NEWSAPI_KEY_DEFAULT" in dummy_config_real_key["sources"][0]["api_key"]:
        test_logger.warning("NewsAPI key is default. Real fetching test will be skipped for NewsAPI.")
        dummy_config_real_key["sources"][0]["enabled"] = False # Disable if key is default

    test_logger.info("Initializing SentimentAnalyzer for testing...")
    analyzer = SentimentAnalyzer(news_api_config=dummy_config_real_key, cache_dir="test_sentiment_cache", cache_ttl_seconds=60) # Short TTL for testing
    
    test_logger.info("\n--- Testing analyze_text_sentiment ---")
    # ... (rest of the __main__ block from previous version can be kept for analyze_text_sentiment tests) ...
    texts_to_analyze = [
        ("This is a great development for Bitcoin!", "Positive"),
        ("The market is crashing, terrible news for Ethereum.", "Negative"),
        ("Everything seems to be stable for Cardano.", "Neutral/Mixed"),
    ]
    for text, desc in texts_to_analyze:
        score = analyzer.analyze_text_sentiment(text)
        test_logger.info(f"Text ({desc}): \"{text}\" -> Score: {score:.4f}")

    test_logger.info("\n--- Testing get_news_sentiment (with potential real fetch if NewsAPI key is valid) ---")
    # Test with a coin supported by NewsAPI (if enabled)
    if dummy_config_real_key["sources"][0]["enabled"]:
        btc_sentiment = analyzer.get_news_sentiment("BTC")
        test_logger.info(f"Sentiment for BTC: {btc_sentiment}")
        
        # Test caching: second call for BTC should be faster if within TTL
        btc_sentiment_cached = analyzer.get_news_sentiment("BTC")
        test_logger.info(f"Sentiment for BTC (cached?): {btc_sentiment_cached}")

        eth_sentiment = analyzer.get_news_sentiment("ETH")
        test_logger.info(f"Sentiment for ETH: {eth_sentiment}")
    else:
        test_logger.info("Skipping NewsAPI specific tests as it's not enabled or key is default.")
        # Test with a generic coin if NewsAPI is disabled, should return no_articles_found or similar
        generic_sentiment = analyzer.get_news_sentiment("SOMECOIN")
        test_logger.info(f"Sentiment for SOMECOIN (NewsAPI disabled): {generic_sentiment}")


    test_logger.info("\n--- Testing Placeholder Async Methods ---")
    import asyncio
    async def run_async_tests():
        listings = await analyzer.fetch_futures_listings()
        test_logger.info(f"Placeholder fetch_futures_listings returned: {listings}")
        new_coins = await analyzer.monitor_news_for_new_coins()
        test_logger.info(f"Placeholder monitor_news_for_new_coins returned: {new_coins}")
    asyncio.run(run_async_tests())

    test_logger.info("\nSentimentAnalyzer testing finished.")
    # import shutil
    # if os.path.exists("test_sentiment_cache"):
    #     shutil.rmtree("test_sentiment_cache")
    #     test_logger.info("Cleaned up test_sentiment_cache directory.")

# Notes:
# - Real news fetching depends on valid API keys in news_api_config.
# - Caching is now time-based.
# - Error handling for network and API issues is included.
# - `requests` library is required: `pip install requests`
# - For Hugging Face models: `pip install transformers torch` (or `tensorflow`)
# - For VADER: `pip install nltk` (and run `nltk.download('vader_lexicon')` once)
