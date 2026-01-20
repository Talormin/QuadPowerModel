"""
PINN Inference and Prediction Pipeline.

This module provides a standalone service for deploying the trained Physics-Informed 
BiLSTM model to generate power consumption predictions on flight telemetry data.

Key capabilities:
1. Replicates the exact pre-processing steps (Kinematics & Scaling) used in training.
2. Performs batched inference to handle large datasets efficiently.
3. Reconstructs continuous time-series from overlapping sliding window predictions.
4. Exports the augmented dataset with predicted power values.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---

@dataclass
class InferenceConfig:
    """Configuration for the inference process."""
    # Paths
    input_data_path: str = r"E:\Dataset\PINN\Airsim\PINN_Train\final_data_cleaned.csv"
    model_path: str = r"C:\Users\LWJ\Desktop\PINN_CODE\模型训练\读取模型\LSTM模型\模型1\PINN_BiLSTM_v1.keras"
    output_path: str = r"E:\Dataset\PINN\Airsim\PINN_Train\final_data_with_predictions.csv"
    
    # Model Parameters (Must match training config)
    window_size: int = 5
    batch_size: int = 128
    
    # Feature Columns
    col_vx: str = 'Vx (m/s)'
    col_vy: str = 'Vy (m/s)'
    col_vz: str = 'Vz (m/s)'
    col_target: str = 'Power_filtered'


class PINNInferenceService:
    """
    Service class responsible for the full lifecycle of model inference:
    Data Loading -> Preprocessing -> Prediction -> Aggregation -> Saving.
    """

    def __init__(self, config: InferenceConfig):
        self.cfg = config
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.model: Optional[tf.keras.Model] = None

    def load_model(self):
        """Loads the pre-trained Keras model."""
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.cfg.model_path}")
        
        logger.info(f"Loading model from: {self.cfg.model_path}")
        try:
            # compile=False is safer for inference if custom losses/metrics were used
            self.model = tf.keras.models.load_model(self.cfg.model_path, compile=False)
            self.model.summary(print_fn=logger.info)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features, calculates Vh/Vv, and scales data.
        Returns: (Scaled Sequences, Scaled Target for inverse transform later)
        """
        logger.info("Preprocessing data...")
        
        # 1. Feature Engineering
        vx = df[self.cfg.col_vx].values.astype(np.float32)
        vy = df[self.cfg.col_vy].values.astype(np.float32)
        vz = df[self.cfg.col_vz].values.astype(np.float32)
        p_actual = df[self.cfg.col_target].values.astype(np.float32)

        v_h = np.sqrt(vx**2 + vy**2)
        v_v = vz

        # 2. Scaling
        # NOTE: In a strict production environment, you should load the scalers 
        # saved during training (e.g. using joblib) rather than fitting on test data.
        # Here we fit on the provided dataset as requested.
        X_raw = np.stack([v_h, v_v], axis=1)
        y_raw = p_actual.reshape(-1, 1)

        X_scaled = self.scaler_input.fit_transform(X_raw)
        _ = self.scaler_output.fit_transform(y_raw) # Fit output scaler for inverse transform later

        # 3. Sequence Generation
        sequences = []
        # Create sliding windows: [Samples, Window_Size, Features]
        for i in range(len(X_scaled) - self.cfg.window_size + 1):
            sequences.append(X_scaled[i : i + self.cfg.window_size])
        
        return np.array(sequences), X_scaled

    def _batch_predict(self, x_seq: np.ndarray) -> np.ndarray:
        """Runs model prediction in batches to manage memory."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
            
        logger.info(f"Starting batch prediction on {len(x_seq)} sequences...")
        
        predictions = []
        num_samples = len(x_seq)
        
        for i in range(0, num_samples, self.cfg.batch_size):
            end = min(i + self.cfg.batch_size, num_samples)
            batch_x = tf.convert_to_tensor(x_seq[i:end], dtype=tf.float32)
            
            # Model returns [prediction, physical_components]. We only need prediction.
            # Assuming the first output is the total power.
            batch_pred = self.model(batch_x, training=False)
            
            # Handle multi-output models (list) vs single output (tensor)
            if isinstance(batch_pred, list):
                batch_pred = batch_pred[0] # Take the main output
                
            predictions.append(batch_pred.numpy())
            
            if (i // self.cfg.batch_size) % 50 == 0:
                logger.info(f"Processed {end}/{num_samples}...")
                
        return np.concatenate(predictions, axis=0)

    def _aggregate_windowed_predictions(self, 
                                      windowed_preds: np.ndarray, 
                                      total_length: int) -> np.ndarray:
        """
        Reconstructs the time series from overlapping windows by averaging.
        
        Args:
            windowed_preds: Shape [Num_Windows, Window_Size, 1]
            total_length: Target length of the original time series (adjusted for padding).
        """
        logger.info("Aggregating windowed predictions...")
        
        # Flatten the feature dim: [Num_Windows, Window_Size]
        preds_flat = windowed_preds[:, :, 0] 
        
        # Initialize accumulators
        # Note: The output length N = Num_Windows + Window_Size - 1
        reconstructed_len = preds_flat.shape[0] + preds_flat.shape[1] - 1
        
        sum_arr = np.zeros(reconstructed_len, dtype=np.float64)
        count_arr = np.zeros(reconstructed_len, dtype=np.int64)
        
        for i in range(preds_flat.shape[0]):
            window_data = preds_flat[i]
            sum_arr[i : i + self.cfg.window_size] += window_data
            count_arr[i : i + self.cfg.window_size] += 1
            
        # Average and handle division by zero
        avg_preds = sum_arr / np.maximum(count_arr, 1)
        return avg_preds

    def run(self):
        """Executes the full inference pipeline."""
        # 1. Load Data
        if not os.path.exists(self.cfg.input_data_path):
            logger.error(f"Data file missing: {self.cfg.input_data_path}")
            return
        
        df = pd.read_csv(self.cfg.input_data_path)
        logger.info(f"Loaded dataset with {len(df)} rows.")

        # 2. Load Model
        self.load_model()

        # 3. Preprocess
        X_seq, _ = self.preprocess_data(df)
        
        # 4. Predict
        raw_preds_scaled = self._batch_predict(X_seq)
        
        # 5. Aggregate / Reconstruct
        # Note: X_seq generated (N - W + 1) windows.
        # Reconstruction will match the length of data covered by windows.
        aggregated_scaled = self._aggregate_windowed_predictions(
            raw_preds_scaled, total_length=None
        )
        
        # 6. Inverse Transform
        logger.info("Inverse transforming predictions...")
        final_preds = self.scaler_output.inverse_transform(
            aggregated_scaled.reshape(-1, 1)
        ).ravel()

        # 7. Merge & Export
        # Pad the beginning with NaNs because the first (Window_Size-1) points 
        # don't complete a full window in standard valid padding
        pad_width = len(df) - len(final_preds)
        padded_preds = np.pad(final_preds, (pad_width, 0), constant_values=np.nan)
        
        df['P_predicted (W)'] = padded_preds
        
        # Save
        os.makedirs(os.path.dirname(self.cfg.output_path), exist_ok=True)
        df.to_csv(self.cfg.output_path, index=False)
        
        logger.info("=" * 40)
        logger.info("INFERENCE COMPLETE")
        logger.info(f"Output saved to: {self.cfg.output_path}")
        logger.info("Preview:")
        logger.info("\n" + str(df[['Power_filtered', 'P_predicted (W)']].tail()))
        logger.info("=" * 40)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    # Use placeholder paths for safety in the provided code
    CONFIG = InferenceConfig(
        input_data_path=r"E:\Dataset\PINN\Airsim\PINN_Train\final_data_cleaned.csv",
        model_path=r"C:\Users\LWJ\Desktop\PINN_CODE\模型训练\读取模型\LSTM模型\模型1\PINN_BiLSTM_v1.keras",
        output_path=r"E:\Dataset\PINN\Airsim\PINN_Train\final_data_with_predictions.csv",
        window_size=5,
        batch_size=128
    )
    
    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    if not os.path.exists(CONFIG.input_data_path):
        logger.error(f"Input file not found: {CONFIG.input_data_path}")
    else:
        service = PINNInferenceService(CONFIG)
        service.run()
