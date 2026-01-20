"""
Explainable AI Service using SHAP GradientExplainer.

This module interprets the trained BiLSTM model by computing Shapley values for 
input features (Horizontal and Vertical Velocity). It handles the complex 
TensorFlow 1.x compatibility session management required by `shap`.

Key Features:
1. TF 1.x Graph Mode Emulation (Required for GradientExplainer).
2. Batched SHAP computation to prevent OOM errors.
3. Feature contribution aggregation over time steps.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import time
import logging
import numpy as np
import shap
import tensorflow as tf
from typing import Tuple, List

# --- GPU & TF Compatibility Configuration ---
# Force CPU to avoid CuDNN LSTM kernel incompatibilities with GradientExplainer
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enable TF 1.x compatibility mode
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

logger = logging.getLogger(__name__)

class SHAPAnalysisEngine:
    """
    Manages the SHAP explanation process for Deep Learning models.
    """

    def __init__(self, model_path: str, background_samples: int = 100):
        self.model_path = model_path
        self.bg_samples = background_samples
        self.session = None
        self.model = None
        self.explainer = None
        self.input_tensor = None
        self.output_tensor = None

    def initialize(self, X_sample_shape: Tuple[int, int]):
        """
        Sets up the TF session, loads the model, and initializes the explainer.
        """
        logger.info("Initializing TF Session for SHAP...")
        self.session = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(self.session)

        logger.info(f"Loading model from {self.model_path}...")
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        except Exception as e:
            logger.critical(f"Model load failed: {e}")
            raise

        # Define symbolic tensors for gradient computation
        # Shape: (None, Window_Size, Features)
        self.input_tensor = tf.compat.v1.placeholder(
            tf.float32, shape=(None, *X_sample_shape), name="shap_input"
        )
        
        # Connect placeholder to model
        model_out = self.model(self.input_tensor)
        
        # Handle multi-output models (Standard PINN returns [Total, Components])
        # We explain the first output (Total Power)
        if isinstance(model_out, list):
            target_out = model_out[0]
        else:
            target_out = model_out
            
        # Squeeze to scalar per time step if necessary
        self.output_tensor = tf.squeeze(target_out, axis=-1)

    def prepare_explainer(self, X_background: np.ndarray):
        """
        Initializes the GradientExplainer with background data.
        """
        logger.info(f"Initializing GradientExplainer with {len(X_background)} background samples...")
        self.explainer = shap.GradientExplainer(
            (self.input_tensor, self.output_tensor),
            X_background,
            session=self.session
        )

    def compute_shap_values(self, X_seq: np.ndarray, batch_size: int = 100) -> List[np.ndarray]:
        """
        Computes SHAP values in batches.
        
        Returns:
            List of arrays, where each array corresponds to a time step in the output sequence.
            Structure: [TimeStep_0_SHAP, TimeStep_1_SHAP, ... TimeStep_T_SHAP]
            Each array shape: (N_samples, Window_Size, Features)
        """
        total_samples = len(X_seq)
        num_batches = int(np.ceil(total_samples / batch_size))
        
        logger.info(f"Starting SHAP computation on {total_samples} samples ({num_batches} batches)...")
        start_time = time.time()
        
        batch_results = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_samples)
            X_batch = X_seq[start:end]
            
            # shap_values returns a list of arrays (one for each output node/timestep)
            batch_shap = self.explainer.shap_values(X_batch)
            batch_results.append(batch_shap)
            
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed batch {i+1}/{num_batches} ({elapsed:.1f}s)")

        logger.info("Aggregating batch results...")
        
        # Re-assemble the structure: List[TimeSteps] -> Array[Total_Samples, ...]
        # Assuming output sequence length T
        T = len(batch_results[0]) 
        final_shap_values = []
        
        for t in range(T):
            # Concatenate the t-th output for all batches
            timestep_data = np.concatenate([b[t] for b in batch_results], axis=0)
            final_shap_values.append(timestep_data)
            
        total_time = time.time() - start_time
        logger.info(f"SHAP computation complete. Total time: {total_time:.2f}s")
        
        return final_shap_values

    def run_prediction(self, X_seq: np.ndarray) -> np.ndarray:
        """Helper to run standard prediction within the same session."""
        return self.session.run(self.model.outputs[0], feed_dict={self.model.inputs[0]: X_seq})

    def close(self):
        if self.session:
            self.session.close()
