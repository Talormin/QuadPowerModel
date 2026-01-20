"""
Trajectory Signal Post-Processing Module.

This module implements advanced zero-phase filtering (Butterworth) for 
UAV flight data. It includes routines for:
1. Signal conditioning (NaN interpolation, padding).
2. Spectral filtering (SOS-based implementation).
3. Comparative visualization of raw vs. filtered signals.

Designed to suppress high-frequency noise from sensor telemetry while 
preserving kinematic phase information.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration Classes ---

@dataclass
class FilterConfig:
    """
    Configuration object for Butterworth filter parameters.
    """
    sampling_rate: float = 64.45  # Hz
    cutoff_freq: float = 3.0      # Hz
    filter_order: int = 4
    
    def validate(self):
        """Ensure filter parameters satisfy Nyquist theorem."""
        nyquist = 0.5 * self.sampling_rate
        if self.cutoff_freq >= nyquist:
            logger.warning(
                f"Cutoff frequency ({self.cutoff_freq} Hz) exceeds Nyquist limit "
                f"({nyquist} Hz). Clamping to 99% of Nyquist."
            )
            self.cutoff_freq = 0.99 * nyquist
        if self.cutoff_freq <= 0:
            raise ValueError("Cutoff frequency must be positive.")

@dataclass
class ProcessingTask:
    """
    Defines a single column processing task.
    """
    input_col: str
    output_col: str
    config: Optional[FilterConfig] = None  # Uses default if None

# --- Core Logic ---

class SignalProcessor:
    """
    Core engine for digital signal processing algorithms.
    """
    
    @staticmethod
    def apply_zero_phase_filter(data: np.ndarray, config: FilterConfig) -> np.ndarray:
        """
        Applies a zero-phase forward-backward digital filter (filtfilt).
        
        Args:
            data: Raw 1D signal array.
            config: Filter configuration parameters.
            
        Returns:
            Filtered 1D signal array.
        """
        # 1. Parameter Validation
        config.validate()
        
        # 2. Pre-processing: Handle Missing Values
        # Linear interpolation for NaNs to prevent filter explosion
        series = pd.Series(data)
        if series.isna().any():
            series = series.interpolate(method='linear', limit_direction='both')
        clean_data = series.to_numpy()
        
        # 3. Filter Design (Second-Order Sections for stability)
        nyquist = 0.5 * config.sampling_rate
        norm_cutoff = config.cutoff_freq / nyquist
        
        sos = signal.butter(
            N=config.filter_order, 
            Wn=norm_cutoff, 
            btype='low', 
            analog=False, 
            output='sos'
        )
        
        # 4. Apply Filter (Zero-phase)
        # Pad type 'odd' helps reduce edge artifacts
        try:
            filtered_data = signal.sosfiltfilt(sos, clean_data, padtype='odd', padlen=None)
            return filtered_data
        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            return clean_data

class DataVisualizer:
    """
    Handles generation of comparative plots for signal analysis.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            
    def plot_comparison(self, 
                       raw: np.ndarray, 
                       filtered: np.ndarray, 
                       col_name: str, 
                       config: FilterConfig):
        """Generates and saves/shows a comparison plot."""
        plt.figure(figsize=(12, 6))
        
        # Plot styling
        plt.plot(raw, label='Raw Signal', color='gray', alpha=0.5, linewidth=1)
        plt.plot(filtered, label='Filtered (Zero-phase)', color='#1f77b4', linewidth=2)
        
        plt.title(f"Signal Smoothing: {col_name}\n"
                  f"fc={config.cutoff_freq}Hz, fs={config.sampling_rate}Hz, Order={config.filter_order}")
        plt.xlabel("Sample Index")
        plt.ylabel("Magnitude")
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if self.save_dir:
            save_path = os.path.join(self.save_dir, f"filter_result_{col_name}.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

class TrajectoryPostProcessor:
    """
    Main controller for the batch processing pipeline.
    """
    
    def __init__(self, default_config: FilterConfig):
        self.default_config = default_config
        self.processor = SignalProcessor()
        self.visualizer = None  # Initialized in run
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads dataset with error checking."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        return pd.read_csv(file_path)
        
    def run(self, 
            input_path: str, 
            output_path: str, 
            tasks: List[ProcessingTask],
            enable_plotting: bool = True):
        """
        Executes the processing pipeline.
        """
        logger.info(f"Loading data from: {input_path}")
        df = self.load_data(input_path)
        
        # Setup Visualizer
        plot_dir = os.path.dirname(output_path) if enable_plotting else None
        self.visualizer = DataVisualizer(save_dir=plot_dir)
        
        for task in tasks:
            if task.input_col not in df.columns:
                logger.warning(f"Column '{task.input_col}' not found in dataset. Skipping.")
                continue
                
            logger.info(f"Processing column: {task.input_col} -> {task.output_col}")
            
            # Determine configuration (Task specific > Default)
            active_config = task.config if task.config else self.default_config
            
            # Extract and Process
            raw_signal = df[task.input_col].values
            filtered_signal = self.processor.apply_zero_phase_filter(raw_signal, active_config)
            
            # Save to DataFrame
            df[task.output_col] = filtered_signal
            
            # Visualize
            if enable_plotting:
                self.visualizer.plot_comparison(
                    raw_signal, 
                    filtered_signal, 
                    task.input_col, 
                    active_config
                )
                
        # Save Result
        logger.info(f"Saving processed data to: {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Post-processing completed successfully.")

# --- Execution Entry Point ---

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # USER CONFIGURATION SECTION
    # ------------------------------------------------------------------
    
    # 1. Define Paths (Use relative or absolute paths as needed)
    # Placeholder paths used for safety; update with actual environment paths
    INPUT_CSV = "./data/raw_flight_logs/horizontal_maneuver.csv"
    OUTPUT_CSV = "./data/processed/filtered_horizontal_maneuver.csv"
    
    # 2. Global Defaults
    GLOBAL_SETTINGS = FilterConfig(
        sampling_rate=64.45,
        cutoff_freq=3.0,
        filter_order=4
    )
    
    # 3. Define Processing Tasks (Mapping input cols to output cols)
    # You can customize individual filters here if needed
    TASKS = [
        ProcessingTask('Power (W)', 'Power_filtered'),
        ProcessingTask('Vx (m/s)', 'Vx_filtered'),
        ProcessingTask('Vy (m/s)', 'Vy_filtered'),
        ProcessingTask('Vz (m/s)', 'Vz_filtered'),
        # Example of custom config for a specific column:
        # ProcessingTask('Altitude (m)', 'Alt_filtered', FilterConfig(sampling_rate=64.45, cutoff_freq=1.0, filter_order=2))
    ]
    
    # ------------------------------------------------------------------
    # MAIN EXECUTION
    # ------------------------------------------------------------------
    
    # Ensure directory exists for output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Check if input file exists before running
    if not os.path.exists(os.path.dirname(INPUT_CSV)):
         logger.error(f"Input directory does not exist: {os.path.dirname(INPUT_CSV)}")
         logger.info("Please configure the 'INPUT_CSV' path in the script.")
    else:
        pipeline = TrajectoryPostProcessor(default_config=GLOBAL_SETTINGS)
        try:
            pipeline.run(INPUT_CSV, OUTPUT_CSV, TASKS, enable_plotting=True)
        except Exception as main_err:
            logger.critical(f"Pipeline execution failed: {main_err}")
