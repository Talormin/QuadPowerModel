"""
Configuration and Environment Setup for Data-Driven BiLSTM.

This module manages all hyperparameters, file paths, and environment settings 
(GPU configuration, logging) for the baseline data-driven model training.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import tensorflow as tf
import logging
from dataclasses import dataclass

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GPU Configuration ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPUs detected and configured: {len(gpus)}")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
else:
    logger.info("No GPUs detected. Running on CPU.")

@dataclass
class ModelConfig:
    """Hyperparameters and paths for the Data-Driven BiLSTM model."""
    # Training Parameters
    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-4
    window_size: int = 5
    patience: int = 15
    
    # Physics Constraints (All zero for pure data-driven baseline)
    alpha_weights: list = None 
    alpha_smooth: float = 0.0
    alpha_l2: float = 1e-6 # Regularization still applies to weights
    phy_loss_weight: float = 0.0
    data_loss_weight: float = 1.0 # Pure data loss
    
    # File Paths (Update these relative paths as needed)
    train_data_path: str = "./data/final/final_data_cleaned.csv" 
    save_dir: str = "./saved_models/baseline_data_driven"
    model_name: str = "Data_BiLSTM_v1.keras"

    def __post_init__(self):
        if self.alpha_weights is None:
            self.alpha_weights = [0, 0, 0, 0]
        os.makedirs(self.save_dir, exist_ok=True)

CONFIG = ModelConfig()
