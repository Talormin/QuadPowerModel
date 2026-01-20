"""
Flight Telemetry Data Cleaning and Quality Control Pipeline.

This module implements a multi-stage filtering process to prepare raw UAV 
power consumption data for machine learning tasks. It addresses two primary 
noise sources:
1. Low-power ground/takeoff states (via Physical Thresholding).
2. Transient sensor drops/spikes (via Rolling Statistical Analysis).

Includes a visualization suite for inspecting global data distribution 
and local signal artifacts.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(class)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---

@dataclass
class CleaningConfig:
    """Hyperparameters for the data cleaning pipeline."""
    # File I/O
    input_path: str
    output_path: str
    target_col: str = 'Power_filtered'
    
    # Phase 1: Physical Thresholds
    hover_power_ref: float = 337.09
    # Threshold for stable flight (e.g., 95% of hover power)
    stable_flight_ratio: float = 0.95
    
    # Phase 2: Statistical Spike Detection
    # Window size for local rolling statistics
    spike_window_size: int = 25
    # Z-score threshold for identifying outliers (higher = less sensitive)
    spike_threshold_sigma: float = 3.0
    
    # Visualization
    zoom_context_points: int = 2000

    @property
    def flight_threshold(self) -> float:
        return self.hover_power_ref * self.stable_flight_ratio


# --- Core Logic: Data Cleaner ---

class FlightTelemetryCleaner:
    """
    Handles the filtering and cleaning logic for flight data.
    """
    
    def __init__(self, config: CleaningConfig):
        self.cfg = config

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from disk."""
        if not os.path.exists(self.cfg.input_path):
            raise FileNotFoundError(f"Input file not found: {self.cfg.input_path}")
        logger.info(f"Loading raw telemetry from: {self.cfg.input_path}")
        return pd.read_csv(self.cfg.input_path)

    def filter_takeoff_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 1: Physical Thresholding.
        Removes data points where power consumption is below the physical 
        minimum required for stable flight (ground effect/idling).
        """
        threshold = self.cfg.flight_threshold
        logger.info(f"Phase 1: Filtering takeoff/ground states (Threshold < {threshold:.2f} W)...")
        
        initial_count = len(df)
        filtered_df = df[df[self.cfg.target_col] > threshold].copy()
        removed_count = initial_count - len(filtered_df)
        
        logger.info(f"-> Removed {removed_count} samples (Ground/Takeoff phase).")
        return filtered_df

    def remove_transient_spikes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
        """
        Phase 2: Statistical Outlier Detection.
        Uses a rolling window to detect sudden negative spikes (sensor dropouts) 
        that deviate significantly from the local median.
        
        Returns:
            Tuple[pd.DataFrame, pd.Index]: Cleaned DataFrame and indices of removed spikes.
        """
        logger.info("Phase 2: Detecting transient sensor spikes...")
        
        series = df[self.cfg.target_col]
        
        # Calculate local statistics
        # Using centered window to look ahead and behind
        rolling_median = series.rolling(
            window=self.cfg.spike_window_size, center=True, min_periods=1
        ).median()
        
        rolling_std = series.rolling(
            window=self.cfg.spike_window_size, center=True, min_periods=1
        ).std()
        
        # Detection logic: Value is lower than median AND deviation > threshold * std
        # We focus on negative spikes (dropouts) as they are common in voltage sensors
        deviation = rolling_median - series
        threshold = self.cfg.spike_threshold_sigma * rolling_std
        
        is_spike = (series < rolling_median) & (deviation > threshold)
        
        spike_count = is_spike.sum()
        spike_indices = df.index[is_spike]
        
        cleaned_df = df[~is_spike].copy()
        
        logger.info(f"-> Detected and removed {spike_count} transient spikes.")
        return cleaned_df, spike_indices

    def save_data(self, df: pd.DataFrame):
        """Saves the cleaned dataset."""
        logger.info(f"Saving cleaned dataset to: {self.cfg.output_path}")
        df.to_csv(self.cfg.output_path, index=False)
        logger.info("Save successful.")


# --- Visualization: Quality Control ---

class QualityControlVisualizer:
    """
    Generates comparative plots (Global vs Local) for quality assurance.
    """
    
    def __init__(self, config: CleaningConfig):
        self.cfg = config
        # Use standard plotting style
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_dashboard(self, 
                       original_df: pd.DataFrame, 
                       final_df: pd.DataFrame, 
                       spike_indices: pd.Index):
        """
        Generates a 2x2 dashboard showing global filtering effects and 
        zoomed-in views of removed artifacts.
        """
        logger.info("Generating Quality Control Dashboard...")
        
        # Identify all discarded points (Phase 1 + Phase 2)
        discarded_indices = sorted(list(set(original_df.index) - set(final_df.index)))
        discarded_df = original_df.loc[discarded_indices]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        
        # --- Top Left: Global View (Clean Data) ---
        ax1 = axes[0, 0]
        ax1.plot(original_df.index, original_df[self.cfg.target_col], 
                 color='lightgray', alpha=0.7, label='Raw Signal')
        ax1.plot(final_df.index, final_df[self.cfg.target_col], 
                 color='#1f77b4', linewidth=0.8, alpha=0.9, label='Cleaned Signal')
        ax1.set_title("Global View: Raw vs. Cleaned", fontsize=14)
        ax1.set_ylabel("Power (W)")
        ax1.legend()

        # --- Top Right: Global View (Discarded Data) ---
        ax2 = axes[0, 1]
        ax2.plot(original_df.index, original_df[self.cfg.target_col], 
                 color='lightgray', alpha=0.5, label='Raw Signal')
        ax2.scatter(discarded_df.index, discarded_df[self.cfg.target_col], 
                    color='red', s=5, alpha=0.6, label='Discarded Points')
        ax2.set_title("Global View: Discarded Artifacts", fontsize=14)
        ax2.legend()

        # --- Bottom Row: Local Zoom on Spikes ---
        if not spike_indices.empty:
            # Select up to 2 representative spikes to zoom in on
            zoom_targets = [spike_indices[0]]
            if len(spike_indices) > 1:
                zoom_targets.append(spike_indices[len(spike_indices)//2])
            
            for i, center_idx in enumerate(zoom_targets):
                ax = axes[1, i]
                
                # Define window
                start = max(0, center_idx - self.cfg.zoom_context_points)
                end = min(len(original_df), center_idx + self.cfg.zoom_context_points)
                
                # Slice data
                local_raw = original_df.loc[start:end]
                local_clean = final_df.loc[start:end]
                local_spikes = original_df.loc[original_df.index.intersection(spike_indices)]
                local_spikes = local_spikes[(local_spikes.index >= start) & (local_spikes.index <= end)]

                ax.plot(local_raw.index, local_raw[self.cfg.target_col], 
                        color='gray', alpha=0.6, label='Raw Context')
                ax.plot(local_clean.index, local_clean[self.cfg.target_col], 
                        color='#1f77b4', label='Retained Data')
                ax.scatter(local_spikes.index, local_spikes[self.cfg.target_col], 
                           color='red', marker='x', s=100, linewidth=2, label='Removed Spike')
                
                ax.set_title(f"Local Detail: Artifact at Index {center_idx}", fontsize=12)
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Power (W)")
                ax.legend()
        else:
            logger.info("No spikes detected. Skipping zoom plots.")
            axes[1, 0].text(0.5, 0.5, "No Spikes Detected", ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, "No Spikes Detected", ha='center', va='center')

        plt.tight_layout()
        plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # USER CONFIGURATION
    # ------------------------------------------------------------------
    # INPUT_FILE = r"E:\Dataset\PINN\Airsim\4种飞行轨迹\水平方向剧烈运动\filtered_output_with_weights.csv"
    # OUTPUT_FILE = r"E:\Dataset\PINN\Airsim\4种飞行轨迹\水平方向剧烈运动\final_data_cleaned.csv"
    
    # Use relative paths for open-source portability
    INPUT_FILE = "./data/processed/filtered_output_with_weights.csv"
    OUTPUT_FILE = "./data/final/final_dataset_cleaned.csv"

    CONFIG = CleaningConfig(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        target_col='Power_filtered',
        hover_power_ref=337.09,
        stable_flight_ratio=0.95,  # 95% of hover power required
        spike_window_size=25,
        spike_threshold_sigma=3.0,
        zoom_context_points=2000
    )

    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------
    
    # Check directory existence
    if not os.path.exists(os.path.dirname(INPUT_FILE)):
        logger.error(f"Directory not found: {os.path.dirname(INPUT_FILE)}")
        logger.info("Please adjust the file paths in the script.")
    else:
        # 1. Initialize
        cleaner = FlightTelemetryCleaner(CONFIG)
        visualizer = QualityControlVisualizer(CONFIG)
        
        # 2. Load
        raw_df = cleaner.load_data()
        
        # 3. Process
        # Phase 1: Filter ground/takeoff
        phase1_df = cleaner.filter_takeoff_phase(raw_df)
        
        # Phase 2: Statistical cleaning
        final_df, spikes = cleaner.remove_transient_spikes(phase1_df)
        
        # 4. Reporting
        removed_total = len(raw_df) - len(final_df)
        logger.info("=" * 40)
        logger.info(f"DATA CLEANING SUMMARY")
        logger.info(f"Original Samples : {len(raw_df)}")
        logger.info(f"Final Samples    : {len(final_df)}")
        logger.info(f"Total Removed    : {removed_total} ({removed_total/len(raw_df):.2%})")
        logger.info("=" * 40)
        
        # 5. Visualize & Save
        visualizer.plot_dashboard(raw_df, final_df, spikes)
        cleaner.save_data(final_df)
