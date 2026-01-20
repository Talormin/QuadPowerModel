"""
Physical Consistency Validation Module.

This module implements the theoretical power consumption model derived from
first-principles aerodynamics. It serves two purposes:
1. Data Augmentation: Injects a 'P_physical' column into the dataset as a physics-informed baseline.
2. Model Validation: Quantifies the deviation between theoretical predictions and filtered telemetry.

The underlying model combines momentum theory for hover/climb with parasitic drag components
for forward flight, augmented by a sparse polynomial residual term.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---

@dataclass
class PhysicsCoefficients:
    """Stores aerodynamic coefficients for the UAV power model."""
    # Horizontal Flight Coefficients
    C1: float = 537.92430435
    C2: float = -11.81444764
    C3: float = -32.51778232
    C4: float = 1851.19680972
    C5: float = 2.00966979
    
    # Vertical Flight Coefficients
    C6: float = 2.77965346e+02
    C7: float = 3.40071433e+01
    C8: float = 2.08030427e-01
    C9: float = 4.00000000e+00
    
    # Base Hover Power
    P_HOVER: float = 337.09

    # Optimized Sparse Regression Coefficients (23 terms)
    SPARSE_COEFFS: np.ndarray = np.array([
        16.2813, -7.1361, -33.7687, 0.0, 16.0607, -0.0, 0.0,
        -0.2804, -0.3995, -0.0025, 178.1034, 99.2792, -1.0379, -284.5549,
        -188.0714, -6.2976, 84.8423, -0.0, 16.1915, -2.3171, 1.0918,
        0.0, -14.524
    ], dtype=float)


class AerodynamicModel:
    """
    Implements the mathematical framework for UAV power estimation.
    """
    
    def __init__(self, coeffs: PhysicsCoefficients = PhysicsCoefficients()):
        self.c = coeffs

    def _compute_horizontal_component(self, v_h: np.ndarray) -> np.ndarray:
        """Calculates power component due to horizontal drag and rotor profile drag."""
        # Numerical stability clamp
        inner_term = np.maximum(1.0 + (v_h**4 / self.c.C4) - (v_h**2 / self.c.C5), 1e-6)
        
        p_h = (self.c.C1 + 
               self.c.C2 * v_h**2 + 
               self.c.C3 * np.sqrt(inner_term) + 
               self.c.C5 * v_h**3)
        
        # Return delta from base hover (C1 is typically close to P_hover but separated in formula)
        return p_h - self.c.C1

    def _compute_vertical_component(self, v_v: np.ndarray) -> np.ndarray:
        """Calculates power component due to climb/descent (Momentum Theory)."""
        # Pre-compute ratios
        r_89 = 4 * self.c.C8 / self.c.C9
        r_79 = 4 * self.c.C7 / self.c.C9
        
        # Ascent (v_v > 0)
        inner_up = np.maximum((1 + r_89) * v_v**2 + r_79, 1e-6)
        val_up = (self.c.C6 + 
                  self.c.C7 * v_v + 
                  self.c.C8 * v_v**3 + 
                  (self.c.C7 + self.c.C8 * v_v**2) * np.sqrt(inner_up))
        
        # Descent (v_v <= 0)
        inner_down = np.maximum((1 - r_89) * v_v**2 + r_79, 1e-6)
        val_down = (self.c.C6 + 
                    self.c.C7 * v_v - 
                    self.c.C8 * v_v**3 + 
                    (self.c.C7 - self.c.C8 * v_v**2) * np.sqrt(inner_down))
        
        p_v = np.where(v_v > 0, val_up, val_down)
        return p_v - self.c.C6

    def _compute_residual_component(self, v_h: np.ndarray, v_v: np.ndarray) -> np.ndarray:
        """Computes the sparse polynomial residual term."""
        # Feature Engineering (Vectorized)
        vh2, vv2 = v_h**2, v_v**2
        vh3, vv3 = v_h**3, v_v**3
        vh4, vv4 = v_h**4, v_v**4
        vh5, vv5 = v_h**5, v_v**5
        
        features = np.stack([
            v_h, v_v,               # 1-2
            vh2, vv2,               # 3-4
            vh3, vv3,               # 5-6
            vh4, vv4,               # 7-8
            vh5, vv5,               # 9-10
            v_h * v_v,              # 11
            vh3 * v_v,              # 12
            v_h * vv3,              # 13
            vh2 * v_v,              # 14
            vv2 * v_h,              # 15
            vh3 * vv2,              # 16
            vh2 * vv2,              # 17
            vh4 * v_v,              # 18
            v_h * vv4,              # 19
            vh5 * v_v,              # 20
            v_h * vv5,              # 21
            np.sin(v_h),            # 22
            np.sin(v_v)             # 23
        ], axis=1)
        
        return features @ self.c.SPARSE_COEFFS

    def predict(self, v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray) -> np.ndarray:
        """
        Predicts total power consumption based on velocity vector.
        
        Args:
            v_x, v_y, v_z: Velocity components in m/s.
            
        Returns:
            Total predicted power in Watts.
        """
        v_h = np.sqrt(v_x**2 + v_y**2)
        v_v = v_z
        
        p_hover = np.full_like(v_h, self.c.P_HOVER)
        p_horiz = self._compute_horizontal_component(v_h)
        p_vert = self._compute_vertical_component(v_v)
        p_resid = self._compute_residual_component(v_h, v_v)
        
        return p_hover + p_horiz + p_vert + p_resid


class ValidationVisualizer:
    """Generates professional plots for model validation."""
    
    @staticmethod
    def plot_time_series(y_true: np.ndarray, y_pred: np.ndarray, sample_limit: int = 5000):
        plt.figure(figsize=(12, 5))
        limit = min(sample_limit, len(y_true))
        
        plt.plot(y_true[:limit], label='Measured (Filtered)', color='gray', alpha=0.7, linewidth=1)
        plt.plot(y_pred[:limit], label='Physics Model Prediction', color='#1f77b4', linewidth=1.5, linestyle='--')
        
        plt.title(f"Model Tracking Performance (First {limit} Samples)")
        plt.xlabel("Sample Index")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_parity(y_true: np.ndarray, y_pred: np.ndarray):
        plt.figure(figsize=(7, 7))
        
        # Scatter plot with density-like alpha
        plt.scatter(y_true, y_pred, s=10, alpha=0.3, color='#1f77b4', edgecolors='none')
        
        # Identity line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (y=x)')
        
        plt.title("Parity Plot: Predicted vs. Actual")
        plt.xlabel("Measured Power (W)")
        plt.ylabel("Predicted Power (W)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


class PhysicsValidator:
    """
    Main controller for the validation pipeline.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.model = AerodynamicModel()
        self.visualizer = ValidationVisualizer()

    def run(self):
        logger.info(f"Loading dataset: {self.file_path}")
        if not os.path.exists(self.file_path):
            logger.error("File not found.")
            return

        df = pd.read_csv(self.file_path)
        required_cols = ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)", "Power_filtered"]
        
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Dataset missing required columns. Needed: {required_cols}")
            return

        # 1. Compute Physics Prediction
        logger.info("Computing physics-based power estimates...")
        v_x = df["Vx (m/s)"].values
        v_y = df["Vy (m/s)"].values
        v_z = df["Vz (m/s)"].values
        
        p_physical = self.model.predict(v_x, v_y, v_z)
        
        # 2. Update DataFrame
        df["P_physical [W]"] = p_physical
        df.to_csv(self.file_path, index=False, float_format="%.6f")
        logger.info(f"Updated dataset saved with 'P_physical [W]' column.")

        # 3. Statistical Evaluation
        p_real = df["Power_filtered"].values
        
        # Metrics
        mape = np.mean(np.abs((p_real - p_physical) / (np.abs(p_real) + 1e-6))) * 100
        r2 = r2_score(p_real, p_physical)
        mae = mean_absolute_error(p_real, p_physical)
        mse = mean_squared_error(p_real, p_physical)
        
        logger.info("=" * 40)
        logger.info("PHYSICS MODEL EVALUATION REPORT")
        logger.info("=" * 40)
        logger.info(f"MAPE : {mape:.2f} %")
        logger.info(f"R²   : {r2:.4f}")
        logger.info(f"MAE  : {mae:.4f} W")
        logger.info(f"MSE  : {mse:.4f} W²")
        logger.info("=" * 40)

        # 4. Visualization
        self.visualizer.plot_time_series(p_real, p_physical)
        self.visualizer.plot_parity(p_real, p_physical)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    # INPUT_CSV = r"E:\Dataset\PINN\Airsim\4种飞行轨迹\水平方向剧烈运动\filtered_output_with_weights.csv"
    INPUT_CSV = "./data/processed/filtered_output_with_weights.csv"
    
    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    if not os.path.exists(os.path.dirname(INPUT_CSV)):
         logger.error(f"Directory not found: {os.path.dirname(INPUT_CSV)}")
    else:
        validator = PhysicsValidator(INPUT_CSV)
        validator.run()
