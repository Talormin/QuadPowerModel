"""
Deep Learning Architecture and Data Pipeline Utilities.

This module contains the definition of the PI-S-LSTM (Physics-Informed Stochastic LSTM)
network and helper functions for sliding window sequence generation.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Generator, Any

class PISLSTMBuilder:
    """
    Builder class for the Physics-Informed Stochastic LSTM architecture.
    Implements a residual Bi-LSTM network with layer normalization.
    """
    
    def __init__(self, 
                 input_timesteps: Optional[int] = None, 
                 feature_dim: int = 2,
                 hidden_units: List[int] = [128, 64],
                 dropout_rate: float = 0.15,
                 l2_reg: float = 0.01):
        """
        Configure the model architecture.
        
        Args:
            input_timesteps: Length of input sequences (None for variable length).
            feature_dim: Dimension of input features (default 2: V_h, V_v).
            hidden_units: List defining LSTM units for each bidirectional layer.
            dropout_rate: Dropout probability for regularization.
            l2_reg: L2 regularization factor for kernel weights.
        """
        self.input_shape = (input_timesteps, feature_dim)
        self.hidden_units = hidden_units
        self.dropout = dropout_rate
        self.reg = tf.keras.regularizers.l2(l2_reg)

    def build(self) -> tf.keras.Model:
        """
        Constructs and returns the compiled Keras model.
        """
        inputs = tf.keras.Input(shape=self.input_shape, name="kinematic_inputs")
        
        # --- Feature Extraction Block (Bi-LSTM) ---
        x = inputs
        
        # First Bidirectional LSTM Layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.hidden_units[0], 
                return_sequences=True,
                name="bilstm_layer_1"
            )
        )(x)
        
        # Second Bidirectional LSTM Layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.hidden_units[1], 
                return_sequences=True,
                name="bilstm_layer_2"
            )
        )(x)
        
        # Normalization for training stability
        x = tf.keras.layers.LayerNormalization(name="layer_norm")(x)
        
        # --- Regression Head (MLP) ---
        # Dense Layer 1
        x = tf.keras.layers.Dense(
            128, 
            activation='relu', 
            kernel_regularizer=self.reg,
            name="dense_1"
        )(x)
        x = tf.keras.layers.Dropout(self.dropout, name="dropout_1")(x)
        
        # Dense Layer 2
        x = tf.keras.layers.Dense(64, activation='relu', name="dense_2")(x)
        x = tf.keras.layers.Dropout(self.dropout, name="dropout_2")(x)
        
        # Dense Layer 3
        x = tf.keras.layers.Dense(64, activation='relu', name="dense_3")(x)
        x = tf.keras.layers.Dropout(self.dropout, name="dropout_3")(x)
        
        # --- Residual Connection ---
        # Project inputs to match dense output dimension if necessary
        residual_projection = tf.keras.layers.Dense(64, name="residual_proj")(inputs)
        x = tf.keras.layers.Add(name="residual_add")([x, residual_projection])
        
        # --- Output Layer ---
        outputs = tf.keras.layers.Dense(1, name="power_prediction")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PI-S-LSTM_V1")
        return model


class SequenceDataProcessor:
    """
    Handles temporal data formatting, including sliding window generation
    and sequence reconstruction.
    """
    
    @staticmethod
    def create_sliding_windows(
        features: np.ndarray, 
        targets: np.ndarray, 
        phy_targets: np.ndarray, 
        alpha_d: np.ndarray, 
        alpha_p: np.ndarray, 
        window_size: int = 10,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates overlapping windows from time-series data.
        
        Args:
            features: Input kinematics (N, D).
            targets: Ground truth power (N, 1).
            phy_targets: Physics model estimates (N, 1).
            alpha_d: Data reliability weights (N, 1).
            alpha_p: Physics reliability weights (N, 1).
            window_size: Size of the look-back window.
            stride: Step size between windows.
            
        Returns:
            Tuple containing windowed arrays for all inputs.
        """
        num_samples = features.shape[0]
        if num_samples < window_size:
            raise ValueError(f"Data length ({num_samples}) is smaller than window size ({window_size}).")
            
        windows_idx = np.arange(0, num_samples - window_size + 1, stride)
        
        X_seq, y_seq, yp_seq, ad_seq, ap_seq = [], [], [], [], []
        
        for i in windows_idx:
            end = i + window_size
            X_seq.append(features[i:end])
            y_seq.append(targets[i:end])
            yp_seq.append(phy_targets[i:end])
            ad_seq.append(alpha_d[i:end])
            ap_seq.append(alpha_p[i:end])
            
        return (np.array(X_seq), np.array(y_seq), np.array(yp_seq), 
                np.array(ad_seq), np.array(ap_seq))

    @staticmethod
    def reconstruct_from_windows(
        windowed_preds: np.ndarray, 
        overlap_strategy: str = 'average'
    ) -> np.ndarray:
        """
        Reconstructs the continuous time-series from overlapping window predictions.
        
        Args:
            windowed_preds: Predictions shape (Num_Windows, Window_Size, Features).
            overlap_strategy: Method to handle overlap ('average' is currently supported).
            
        Returns:
            Flat array of reconstructed predictions.
        """
        if overlap_strategy != 'average':
            raise NotImplementedError(f"Strategy {overlap_strategy} not implemented.")
            
        num_windows, window_size, num_feats = windowed_preds.shape
        # Assuming stride=1
        total_len = num_windows + window_size - 1
        
        sum_arr = np.zeros((total_len, num_feats))
        count_arr = np.zeros((total_len, num_feats))
        
        for w in range(num_windows):
            chunk = windowed_preds[w] # (Window_Size, Feats)
            start_idx = w
            end_idx = w + window_size
            
            sum_arr[start_idx:end_idx] += chunk
            count_arr[start_idx:end_idx] += 1.0
            
        # Avoid division by zero
        reconstructed = sum_arr / np.maximum(count_arr, 1.0)
        return reconstructed
