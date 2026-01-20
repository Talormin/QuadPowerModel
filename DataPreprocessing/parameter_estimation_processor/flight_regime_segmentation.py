"""
Flight Regime Segmentation and Pre-filtering Module.

This module is responsible for separating continuous flight telemetry into 
distinct operating regimes (Horizontal, Ascent, Descent) based on kinematic 
state vectors. It also acts as a coarse filter to remove high-acceleration 
transients that violate quasi-steady-state assumptions required for 
aerodynamic coefficient estimation.

Key Features:
1. High-Acceleration Rejection: Filters out aggressive maneuvers.
2. Vectorized State Classification: Efficiently labels data samples.
3. Dataset Partitioning: Exports regime-specific datasets for downstream analysis.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---

class FlightState(Enum):
    HORIZONTAL = 'horizontal'
    ASCENT = 'ascent'
    DESCENT = 'descent'
    TRANSITION = 'transition/other'

@dataclass
class SegmentationConfig:
    """Hyperparameters for state classification."""
    # File Paths
    input_path: str = "./data/final/all_flight_data.csv"
    output_dir: str = "./data/regimes"
    
    # Pre-filtering Thresholds
    # Max allowed acceleration (m/s^2) to be considered 'quasi-steady'
    max_acceleration_total: float = 2.0  
    
    # State Classification Thresholds
    # Minimum horizontal speed to qualify as 'Horizontal Flight'
    min_horizontal_vel: float = 0.2  # m/s
    # Maximum vertical speed to still be considered 'Horizontal'
    max_vertical_drift: float = 0.3  # m/s
    # Minimum vertical speed to qualify as 'Ascent/Descent'
    min_vertical_vel: float = 0.3    # m/s


class RegimeSegmenter:
    """
    Core processor for flight state segmentation.
    """
    
    def __init__(self, config: SegmentationConfig):
        self.cfg = config
        self.df: Optional[pd.DataFrame] = None
        
        # Ensure output directory exists
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Loads dataset from CSV."""
        path = Path(self.cfg.input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        
        logger.info(f"Loading dataset: {path}")
        self.df = pd.read_csv(path)
        
        # Validate required columns
        required = ['vx', 'vy', 'vz', 'ax', 'ay', 'az']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset missing required kinematic columns: {missing}")

    def filter_transients(self):
        """
        Filters out high-acceleration events.
        Aerodynamic coefficient fitting assumes quasi-steady states.
        """
        if self.df is None: return

        # Calculate total acceleration magnitude
        acc_mag = np.sqrt(
            self.df['ax']**2 + self.df['ay']**2 + self.df['az']**2
        )
        self.df['a_total'] = acc_mag
        
        initial_len = len(self.df)
        
        # Apply filter
        mask = acc_mag <= self.cfg.max_acceleration_total
        self.df = self.df[mask].copy()
        
        removed = initial_len - len(self.df)
        logger.info(f"Transient Filtering: Removed {removed} samples "
                    f"(Acceleration > {self.cfg.max_acceleration_total} m/s^2).")

    def classify_states(self):
        """
        Vectorized classification of flight states.
        """
        if self.df is None: return
        
        logger.info("Classifying flight regimes...")

        # 1. Compute Scalar Velocities
        # V_h = sqrt(vx^2 + vy^2)
        v_h = np.sqrt(self.df['vx']**2 + self.df['vy']**2)
        # V_v = vz (Up is positive assumption, check coordinate system if needed)
        v_v = self.df['vz']
        
        self.df['V_h'] = v_h
        self.df['V_v'] = v_v

        # 2. Vectorized Conditions using numpy
        # Condition: Horizontal Flight
        cond_horiz = (v_h > self.cfg.min_horizontal_vel) & \
                     (np.abs(v_v) < self.cfg.max_vertical_drift)
        
        # Condition: Ascent
        # Note: Must ensure V_h is low to isolate pure vertical drag
        cond_ascent = (v_h < self.cfg.min_horizontal_vel) & \
                      (v_v > self.cfg.min_vertical_vel)
        
        # Condition: Descent
        cond_descent = (v_h < self.cfg.min_horizontal_vel) & \
                       (v_v < -self.cfg.min_vertical_vel)

        # 3. Assign Labels (Priority: Horizontal > Ascent > Descent > Transition)
        # We use np.select for efficient multi-branch logic
        conditions = [cond_horiz, cond_ascent, cond_descent]
        choices = [
            FlightState.HORIZONTAL.value,
            FlightState.ASCENT.value,
            FlightState.DESCENT.value
        ]
        
        self.df['flight_state'] = np.select(
            conditions, 
            choices, 
            default=FlightState.TRANSITION.value
        )

    def export_datasets(self):
        """
        Saves the segmented datasets to disk.
        """
        if self.df is None: return

        # 1. Save Cleaned Master Dataset
        master_path = Path(self.cfg.output_dir) / "filtered_master_dataset.csv"
        self.df.to_csv(master_path, index=False)
        logger.info(f"Saved master filtered dataset: {master_path}")

        # 2. Save Subsets
        groups = self.df.groupby('flight_state')
        
        summary = {}
        
        for state_val, group_df in groups:
            if state_val == FlightState.TRANSITION.value:
                continue # Skip transition data for fitting
                
            filename = f"regime_{state_val}.csv"
            save_path = Path(self.cfg.output_dir) / filename
            group_df.to_csv(save_path, index=False)
            
            summary[state_val] = len(group_df)
            logger.info(f"-> Exported {state_val}: {len(group_df)} samples to {filename}")

        self._print_summary(summary)

    def _print_summary(self, summary: Dict[str, int]):
        total_valid = sum(summary.values())
        logger.info("=" * 40)
        logger.info("SEGMENTATION SUMMARY")
        logger.info("=" * 40)
        for state, count in summary.items():
            logger.info(f"{state.ljust(15)} : {count:6d} samples ({count/total_valid:.1%})")
        logger.info("-" * 40)
        logger.info(f"Total Useful    : {total_valid:6d} samples")
        logger.info("=" * 40)

    def run(self):
        """Execution pipeline."""
        self.load_data()
        self.filter_transients()
        self.classify_states()
        self.export_datasets()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    # INPUT_FILE = r"E:\Dataset\all_flight_data.csv"
    INPUT_FILE = "./data/final/all_flight_data.csv"
    
    CONFIG = SegmentationConfig(
        input_path=INPUT_FILE,
        output_dir="./data/processed_regimes",
        
        # Filtering aggressiveness
        max_acceleration_total=2.0, 
        
        # Classification Logic
        min_horizontal_vel=0.2,   # > 0.2 m/s -> Moving horizontally
        max_vertical_drift=0.3,   # < 0.3 m/s -> Stable altitude
        min_vertical_vel=0.3      # > 0.3 m/s -> Distinct climb/descent
    )

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    if not os.path.exists(os.path.dirname(INPUT_FILE)):
        logger.error(f"Directory not found: {os.path.dirname(INPUT_FILE)}")
    else:
        segmenter = RegimeSegmenter(CONFIG)
        segmenter.run()
