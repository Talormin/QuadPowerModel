"""
Neural Network Architecture for Standard PINN.

This module defines the BiLSTM-based architecture that explicitly disentangles
power consumption into four physical components:
1. Vertical Power (Pv)
2. Horizontal Power (Ph)
3. Hover Power (Phov)
4. Additional/Residual Power (Padd)

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import tensorflow as tf

class StandardPINNBuilder:
    """
    Builder for the Physics-Component-Aware BiLSTM Network.
    """
    
    @staticmethod
    def build_model(input_shape=(None, 2)) -> tf.keras.Model:
        """
        Constructs the model.
        
        Returns:
            tf.keras.Model: A model with two outputs:
                1. 'output': Summed total power [Batch, Time, 1]
                2. 'physical_components': Individual components [Batch, Time, 4]
        """
        inputs = tf.keras.Input(shape=input_shape, name="kinematic_input")

        # --- Feature Extraction (BiLSTM) ---
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True),
            name="bilstm_1"
        )(inputs)
        
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True),
            name="bilstm_2"
        )(x)

        # --- Dense Interpretation Layers ---
        x = tf.keras.layers.LayerNormalization()(x)
        
        x = tf.keras.layers.Dense(
            128, activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.15)(x)

        # Residual Skip Connection
        residual = tf.keras.layers.Dense(64)(inputs)
        x = tf.keras.layers.Add()([x, residual])

        # --- Physical Component Branching ---
        # Output dimension 4 corresponds to: [Pv, Ph, Phov, Padd]
        phys_outputs = tf.keras.layers.Dense(4, name="physical_components")(x)

        # --- Physics Aggregation (Summation) ---
        # We use a fixed, non-trainable Dense layer to sum components
        # This is equivalent to tf.reduce_sum but keeps Keras topology clean
        output = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name="output"
        )(phys_outputs)

        return tf.keras.Model(inputs=inputs, outputs=[output, phys_outputs])
