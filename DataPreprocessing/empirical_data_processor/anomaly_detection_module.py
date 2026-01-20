"""
Advanced Anomaly Detection and Data Cleaning Module.

This module implements a multi-stage outlier detection algorithm designed for
UAV power telemetry data. It combines:
1. Local robustness checks (Rolling MAD/Z-Score).
2. Global statistical bounds (Interquartile Range).
3. Cluster-based peak extraction (to remove only the sharpest artifacts).

The goal is to preserve the underlying signal dynamics while surgically removing
sensor glitches and non-physical power spikes.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataclasses import dataclass
from typing import Optional, Tuple, List

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure Plotting Fonts (Fallback strategy for Chinese support)
rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

@dataclass
class AnomalyConfig:
    """Configuration for outlier detection algorithms."""
    # Data Columns
    timestamp_col: str = "Timestamp"
    target_col: str = "Power_filtered"
    
    # Local Outlier Detection (Rolling Z-Score)
    window_size: int = 11          # Size of rolling window
    robust_z_threshold: float = 2.0  # Threshold for local anomalies
    
    # Global Outlier Detection (IQR)
    iqr_multiplier: float = 0.3    # Multiplier for Interquartile Range
    hard_upper_limit: Optional[float] = None # Absolute physical limit
    
    # Cluster Processing
    max_tips_per_cluster: int = 200 # How many peak points to remove per anomaly cluster


class StatisticalOutlierDetector:
    """
    Implements statistical methods for identifying and handling time-series anomalies.
    """
    
    def __init__(self, config: AnomalyConfig):
        self.cfg = config

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the timestamp column is in datetime format."""
        if self.cfg.timestamp_col in df.columns:
            if not np.issubdtype(df[self.cfg.timestamp_col].dtype, np.datetime64):
                try:
                    df[self.cfg.timestamp_col] = pd.to_datetime(df[self.cfg.timestamp_col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Timestamp parsing failed: {e}")
        return df

    def detect_candidates(self, series: pd.Series) -> pd.Series:
        """
        Stage 1: Identify all potential anomaly candidates using Local & Global stats.
        Returns a boolean mask (True = Candidate Anomaly).
        """
        # 1. Local Robust Z-Score (Rolling MAD)
        # rolling_median = series.rolling(window=self.cfg.window_size, center=True, min_periods=3).median()
        # Using a slightly larger min_periods for stability
        min_periods = max(3, self.cfg.window_size // 3)
        
        roll_med = series.rolling(window=self.cfg.window_size, center=True, min_periods=min_periods).median()
        residuals = series - roll_med
        roll_mad = residuals.abs().rolling(window=self.cfg.window_size, center=True, min_periods=min_periods).median()
        
        epsilon = 1e-9 # Prevent division by zero
        robust_z = residuals.abs() / (1.4826 * (roll_mad + epsilon))
        
        # We only care about upward spikes for power consumption
        # Note: residuals > 0 ensures we check positive deviations
        mask_local = (robust_z > self.cfg.robust_z_threshold) & (residuals > 0)
        
        # 2. Global IQR Check
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + self.cfg.iqr_multiplier * iqr
        mask_global = series > upper_bound
        
        # 3. Hard Limit Check
        mask_hard = pd.Series(False, index=series.index)
        if self.cfg.hard_upper_limit is not None:
            mask_hard = series > self.cfg.hard_upper_limit
            
        # Combine masks (Union)
        # NaNs are always treated as anomalies initially
        candidates = mask_local | mask_global | mask_hard | series.isna()
        
        return candidates

    def refine_clusters(self, series: pd.Series, candidate_mask: pd.Series) -> pd.Series:
        """
        Stage 2: Cluster Refinement.
        Instead of removing all candidates, we group consecutive anomalies and 
        only remove the sharpest 'tips' (peaks) from each cluster.
        """
        mask_np = candidate_mask.to_numpy()
        vals_np = series.to_numpy()
        
        # Always remove NaNs
        nan_mask = series.isna().to_numpy()
        
        # Indices of candidates
        candidate_indices = np.where(mask_np)[0]
        final_mask = np.zeros_like(mask_np, dtype=bool)
        
        if candidate_indices.size > 0:
            # Group consecutive indices (difference > 1 implies new group)
            split_points = np.where(np.diff(candidate_indices) > 1)[0] + 1
            clusters = np.split(candidate_indices, split_points)
            
            for cluster in clusters:
                if cluster.size == 0:
                    continue
                
                # Determine how many points to cut from this cluster
                k = min(self.cfg.max_tips_per_cluster, cluster.size)
                
                # Get values in cluster (handle NaNs for sorting)
                cluster_vals = np.nan_to_num(vals_np[cluster], nan=-np.inf)
                
                # Find indices of the top-k highest values
                # argsort returns indices relative to 'cluster' array
                top_k_local_indices = np.argsort(cluster_vals)[-k:]
                top_k_global_indices = cluster[top_k_local_indices]
                
                final_mask[top_k_global_indices] = True
                
        # Combine peaks with NaNs
        return pd.Series(final_mask | nan_mask, index=series.index)

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main execution pipeline.
        Returns: (Cleaned DataFrame, Boolean Mask of Removed Points)
        """
        df = df.copy()
        df = self._parse_timestamps(df)
        
        # Ensure numeric
        df[self.cfg.target_col] = pd.to_numeric(df[self.cfg.target_col], errors='coerce')
        series = df[self.cfg.target_col]
        
        logger.info("Step 1: Detecting candidate anomalies...")
        candidate_mask = self.detect_candidates(series)
        logger.info(f"-> Candidates found: {candidate_mask.sum()}")
        
        logger.info("Step 2: Refining anomaly clusters (Tip Extraction)...")
        final_mask = self.refine_clusters(series, candidate_mask)
        logger.info(f"-> Final points to remove: {final_mask.sum()}")
        
        df_clean = df.loc[~final_mask].copy()
        
        return df_clean, final_mask


class CleaningVisualizer:
    """Helper class for plotting cleaning results."""
    
    @staticmethod
    def plot_full_comparison(raw_series: pd.Series, clean_series: pd.Series, mask: pd.Series, title_suffix: str = ""):
        plt.figure(figsize=(12, 5))
        
        # Use numpy arrays for plotting speed
        idx_all = np.arange(len(raw_series))
        vals_raw = raw_series.to_numpy()
        
        # Plot Raw
        plt.plot(idx_all, vals_raw, label="Raw Signal", linewidth=1, color='lightgray', alpha=0.8)
        
        # Plot Removed Points
        idx_removed = np.where(mask)[0]
        if len(idx_removed) > 0:
            plt.scatter(idx_removed, vals_raw[idx_removed], s=15, c='red', zorder=5, label="Detected Anomalies (Tips)")
        
        # Plot Cleaned (using original indices)
        plt.plot(clean_series.index, clean_series.values, linewidth=1.2, color='#1f77b4', label="Cleaned Signal")
        
        plt.title(f"Signal Cleaning Overview {title_suffix}")
        plt.xlabel("Sample Index")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    @staticmethod
    def plot_segment_zoom(raw_series: pd.Series, clean_series: pd.Series, mask: pd.Series, 
                         start: int, length: int):
        end = min(start + length, len(raw_series))
        
        # Slice data
        raw_seg = raw_series.iloc[start:end]
        mask_seg = mask.iloc[start:end]
        
        # For cleaned data, we need intersection of indices
        clean_seg = clean_series.loc[clean_series.index.intersection(raw_seg.index)]
        
        plt.figure(figsize=(12, 5))
        plt.plot(raw_seg.index, raw_seg.values, linewidth=1.5, color='gray', alpha=0.5, label="Raw Segment")
        
        removed_indices = mask_seg[mask_seg].index
        if len(removed_indices) > 0:
            plt.scatter(removed_indices, raw_seg.loc[removed_indices], s=25, c='red', zorder=5, label="Removed Spikes")
            
        plt.plot(clean_seg.index, clean_seg.values, linewidth=1.5, color='#1f77b4', label="Cleaned Segment")
        
        plt.title(f"Detailed View: Segment [{start}, {end})")
        plt.xlabel("Index")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    # INPUT_FILE = "./data/processed/filtered_horizontal_maneuver.csv"
    # OUTPUT_FILE = "./data/cleaned/cleaned_horizontal_maneuver.csv"
    
    # Placeholder for safety
    INPUT_FILE = r"E:\Dataset\PINN\Airsim\4种飞行轨迹\水平方向剧烈运动\filtered_output.csv"
    OUTPUT_FILE = os.path.splitext(INPUT_FILE)[0] + "_cleaned.csv"
    
    CONFIG = AnomalyConfig(
        target_col="Power_filtered",
        window_size=11,
        robust_z_threshold=2.0,
        iqr_multiplier=0.3,
        max_tips_per_cluster=200
    )
    
    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    if os.path.exists(INPUT_FILE):
        logger.info(f"Loading data from {INPUT_FILE}...")
        df_raw = pd.read_csv(INPUT_FILE)
        
        detector = StatisticalOutlierDetector(CONFIG)
        df_clean, anomaly_mask = detector.process(df_raw)
        
        # Save
        df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"Cleaned dataset saved to {OUTPUT_FILE}")
        logger.info(f"Original Size: {len(df_raw)} | Cleaned Size: {len(df_clean)}")
        
        # Visualization
        viz = CleaningVisualizer()
        
        # 1. Full view
        viz.plot_full_comparison(df_raw[CONFIG.target_col], df_clean[CONFIG.target_col], anomaly_mask)
        
        # 2. Zoomed view (Example: index 1000 to 4000)
        viz.plot_segment_zoom(df_raw[CONFIG.target_col], df_clean[CONFIG.target_col], anomaly_mask, 
                              start=1000, length=3000)
        
    else:
        logger.error(f"File not found: {INPUT_FILE}")
