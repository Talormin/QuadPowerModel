"""
Neural Network Architecture for Data-Driven BiLSTM Baseline.

This module defines a standard Bidirectional LSTM network optimized for time-series 
regression. It maps sequence inputs (Velocity components) directly to the target 
variable (Power) without intermediate physical component decomposition.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import tensorflow as tf

class DataDrivenBiLSTMBuilder:
    """
    Builder for the Baseline BiLSTM Network.
    """
    
    @staticmethod
    def build_model(input_shape=(None, 2)) -> tf.keras.Model:
        """
        Constructs the Keras model.
        
        Args:
            input_shape: Shape of the input sequence (Time steps, Features).
                         Features = 2 (V_h, V_v).
        
        Returns:
            tf.keras.Model: Compiled model with a single output (Total Power).
        """
        inputs = tf.keras.Input(shape=input_shape, name="kinematic_input")

        # --- Recurrent Layers (BiLSTM) ---
        # Layer 1: Feature Extraction
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True),
            name="bilstm_layer_1"
        )(inputs)
        
        # Layer 2: Deep Feature Learning
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True),
            name="bilstm_layer_2"
        )(x)

        # --- Interpretation Layers (MLP) ---
        # Normalization for training stability
        x = tf.keras.layers.LayerNormalization(name="layer_norm")(x)
        
        # Dense Block 1
        x = tf.keras.layers.Dense(
            128, activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name="dense_1"
        )(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)
        
        # Dense Block 2
        x = tf.keras.layers.Dense(64, activation='relu', name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_2")(x)
        
        # Dense Block 3
        x = tf.keras.layers.Dense(64, activation='relu', name="dense_3")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_3")(x)

        # Residual Connection (Project input to match Dense dimension)
        residual = tf.keras.layers.Dense(64, name="residual_projection")(inputs)
        x = tf.keras.layers.Add(name="residual_add")([x, residual])

        # --- Output Layer ---
        # Single output regression: Power (scaled)
        output = tf.keras.layers.Dense(1, name="power_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=output, name="Baseline_BiLSTM")
