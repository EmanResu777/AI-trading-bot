import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

class DataProcessor:
    """
    Processes raw market data by handling missing values (imputation) and 
    scaling features. It can also add sentiment scores as an additional feature.
    """
    def __init__(self):
        """
        Initializes the DataProcessor with a StandardScaler, SimpleImputer, and a logger.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean') # Initialize imputer
        self.logger = logging.getLogger(__name__)
        # Configure logger if not already configured by the application
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # Prevent duplicate logging if basicConfig was already called by another module
            if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
                 self.logger.addHandler(logging.StreamHandler())
            self.logger.info("DataProcessor logger configured.")


    def prepare_data(self, arr: np.ndarray) -> np.ndarray:
        """
        Prepares the data by imputing missing values and then scaling.

        Args:
            arr: NumPy array of shape (num_samples, num_base_features). 
                 Expected to be 2D. If 1D, it's treated as a single sample.

        Returns:
            NumPy array: The processed (imputed and scaled) data.
        """
        self.logger.info(f"prepare_data called with array of shape {arr.shape}")

        if arr.ndim == 1:
            self.logger.warning(f"Input array is 1D with shape {arr.shape}. Reshaping to (1, {arr.shape[0]}) for processing.")
            arr = arr.reshape(1, -1)
        
        if arr.shape[0] == 0 : # No samples to process
            self.logger.warning("Input array has 0 samples. Returning empty array.")
            return arr


        # NaN Handling with SimpleImputer
        nan_before_imputation = np.isnan(arr).sum()
        if nan_before_imputation > 0:
            self.logger.info(f"NaN values found before imputation: {nan_before_imputation} in array of shape {arr.shape}")
            try:
                arr_imputed = self.imputer.fit_transform(arr)
                self.logger.info(f"Imputation applied. Shape after imputation: {arr_imputed.shape}")
            except Exception as e:
                self.logger.error(f"Error during imputation: {e}. Returning original array for scaling attempt.")
                arr_imputed = arr # Proceed with potentially NaN-containing array
        else:
            self.logger.info(f"No NaN values found in input array of shape {arr.shape}.")
            arr_imputed = arr # No imputation needed

        nan_after_imputation = np.isnan(arr_imputed).sum()
        if nan_after_imputation > 0:
            self.logger.warning(f"NaN values remain after imputation: {nan_after_imputation}. This may cause issues with scaling.")
        else:
            self.logger.info("No NaN values found after imputation step.")
            
        # Scaling
        self.logger.warning(
            "DataProcessor.prepare_data is using scaler.fit_transform(). "
            "For production, it's generally better to fit the scaler on a representative "
            "training dataset once and use scaler.transform() on new/live data to prevent data leakage "
            "and ensure consistent scaling based on the training distribution."
        )
        try:
            scaled_data = self.scaler.fit_transform(arr_imputed)
            self.logger.info(f"Data scaled successfully. Output shape: {scaled_data.shape}")
        except Exception as e:
            self.logger.error(f"Error during scaling: {e}. Returning imputed (or original if imputation failed) array without scaling.")
            return arr_imputed # Return data before scaling error

        return scaled_data

    def add_news_sentiment(self, data: np.ndarray, sentiment_score: float) -> np.ndarray:
        """
        Adds a news sentiment score as an additional feature column to the data.

        Args:
            data: NumPy array of shape (num_samples, num_features_before_sentiment).
                  Can be 1D for a single sample, which will be reshaped.
            sentiment_score: A float representing the news sentiment.

        Returns:
            NumPy array: Data with sentiment score added as a new column,
                         shape (num_samples, num_features_before_sentiment + 1).
        """
        self.logger.info(f"add_news_sentiment called with data shape {data.shape} and sentiment score: {sentiment_score}")

        if data.ndim == 1: # If data is a 1D array (single sample)
            self.logger.info(f"Input data to add_news_sentiment is 1D ({data.shape}). Reshaping to (1, {data.shape[0]}).")
            data = data.reshape(1, -1) # Reshape to (1, num_features)
        
        if data.shape[0] == 0:
            self.logger.warning("Input data to add_news_sentiment has 0 samples. Appending sentiment column to empty array structure.")
            # This will result in an array like np.empty((0, N+1)) if data was np.empty((0,N))
            # Or if data itself was empty like np.array([]), it might lead to issues.
            # Let's ensure data has a second dimension if it's completely empty.
            if data.ndim == 1 and data.size == 0: # e.g. np.array([])
                data = np.empty((0,0)) # make it 2D for hstack to work as expected for column addition
            
            # Create an empty column with the correct number of rows (0) and 1 column for sentiment
            sentiment_column = np.full((data.shape[0], 1), sentiment_score, dtype=float if data.dtype.kind != 'f' else data.dtype)

        else:
             # Match data type if possible, otherwise default to float for sentiment
            sentiment_column = np.full((data.shape[0], 1), sentiment_score, dtype=data.dtype if data.dtype.kind == 'f' else float)

        try:
            data_with_sentiment = np.hstack((data, sentiment_column))
            self.logger.info(f"Sentiment score added. New data shape: {data_with_sentiment.shape}")
            return data_with_sentiment
        except Exception as e:
            self.logger.error(f"Error adding sentiment score: {e}. Returning original data.")
            return data

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    processor = DataProcessor()

    # Test prepare_data
    logger.info("\n--- Testing prepare_data ---")
    sample_data_1 = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]], dtype=float)
    logger.info(f"Original sample_data_1:\n{sample_data_1}")
    processed_data_1 = processor.prepare_data(sample_data_1)
    logger.info(f"Processed sample_data_1:\n{processed_data_1}")

    sample_data_2 = np.array([[10, 20], [15, 25]], dtype=float)
    logger.info(f"\nOriginal sample_data_2:\n{sample_data_2}")
    processed_data_2 = processor.prepare_data(sample_data_2)
    logger.info(f"Processed sample_data_2:\n{processed_data_2}")
    
    sample_data_3_1d = np.array([100, 110, np.nan, 130], dtype=float)
    logger.info(f"\nOriginal sample_data_3_1d: {sample_data_3_1d}")
    processed_data_3_1d = processor.prepare_data(sample_data_3_1d)
    logger.info(f"Processed sample_data_3_1d:\n{processed_data_3_1d}")

    sample_data_empty = np.empty((0,3))
    logger.info(f"\nOriginal sample_data_empty:\n{sample_data_empty}")
    processed_data_empty = processor.prepare_data(sample_data_empty)
    logger.info(f"Processed sample_data_empty:\n{processed_data_empty}")


    # Test add_news_sentiment
    logger.info("\n--- Testing add_news_sentiment ---")
    data_for_sentiment_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    sentiment = 0.75
    logger.info(f"Original data_for_sentiment_1:\n{data_for_sentiment_1}, sentiment: {sentiment}")
    with_sentiment_1 = processor.add_news_sentiment(data_for_sentiment_1, sentiment)
    logger.info(f"With sentiment_1:\n{with_sentiment_1}")

    data_for_sentiment_2_1d = np.array([0.1, 0.2, 0.3])
    sentiment_2 = -0.5
    logger.info(f"\nOriginal data_for_sentiment_2_1d: {data_for_sentiment_2_1d}, sentiment: {sentiment_2}")
    with_sentiment_2 = processor.add_news_sentiment(data_for_sentiment_2_1d, sentiment_2)
    logger.info(f"With sentiment_2:\n{with_sentiment_2}")

    data_for_sentiment_empty = np.empty((0,2))
    sentiment_3 = 0.1
    logger.info(f"\nOriginal data_for_sentiment_empty:\n{data_for_sentiment_empty}, sentiment: {sentiment_3}")
    with_sentiment_3 = processor.add_news_sentiment(data_for_sentiment_empty, sentiment_3)
    logger.info(f"With sentiment_3:\n{with_sentiment_3}")

    # Test with data that has been through prepare_data
    logger.info("\n--- Testing add_news_sentiment on processed data ---")
    if processed_data_1.shape[0] > 0 :
        with_sentiment_on_processed = processor.add_news_sentiment(processed_data_1, 0.99)
        logger.info(f"Processed data_1 with sentiment:\n{with_sentiment_on_processed}")
    else:
        logger.info("Skipping add_news_sentiment on processed_data_1 as it's empty.")
