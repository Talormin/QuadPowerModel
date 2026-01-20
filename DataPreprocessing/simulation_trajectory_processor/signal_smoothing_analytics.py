"""
Trajectory Signal Smoothing and Analytics Module.

This module provides a comprehensive suite for post-processing UAV flight telemetry.
It implements multiple smoothing algorithms (MA, EMA, Savitzky-Golay) to mitigate
sensor noise while preserving kinematic trends.

Key Features:
1. Automated Data Backup: Creates timestamped backups before modification.
2. Multi-Strategy Smoothing: Supports Moving Average, Exponential, and SavGol filters.
3. Intelligent Segmentation: Slices data by time or index for granular analysis.
4. Comparative Visualization: Generates high-resolution plots for full/partial series.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import shutil
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Union

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Enums & Configuration ---

class SmoothingMethod(Enum):
    MOVING_AVERAGE = "ma"
    EXPONENTIAL_MA = "ema"
    SAVITZKY_GOLAY = "savgol"

@dataclass
class SmoothingConfig:
    """Configuration for signal smoothing algorithms."""
    # Target Columns
    target_column: str = "Power_filtered"
    
    # Algorithm Selection
    method: SmoothingMethod = SmoothingMethod.MOVING_AVERAGE
    
    # Hyperparameters
    ma_window: int = 101           # For MA (Odd number recommended)
    ema_alpha: float = 0.05        # For EMA (0 < alpha <= 1)
    savgol_window: int = 101       # For SavGol (Must be odd and > poly_order)
    savgol_poly_order: int = 3     # For SavGol
    
    # I/O Settings
    append_to_csv: bool = True     # Write smoothed data back to source file
    create_backup: bool = True     # Backup original file before writing
    overwrite_col: bool = True     # Overwrite existing column if name conflicts

@dataclass
class SegmentConfig:
    """Configuration for data segmentation/slicing."""
    # Option 1: Manual Index Ranges [(start, end), ...]
    manual_segments_idx: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 5000), (5000, 10000), (10000, 11500)
    ])
    
    # Option 2: Time-based Ranges [(t_start, t_end), ...]
    # If provided, this overrides manual_segments_idx
    manual_segments_time: List[Tuple[float, float]] = field(default_factory=list)
    
    # Option 3: Auto-Segmentation (Fallback)
    use_auto_fallback: bool = True
    auto_num_segments: int = 3
    auto_segment_length: int = 1000

# --- Core Processor ---

class SignalSmoother:
    """
    Core engine for applying smoothing algorithms to time-series data.
    """
    
    def __init__(self, df: pd.DataFrame, config: SmoothingConfig):
        self.df = df
        self.cfg = config
        self._check_dependencies()

    def _check_dependencies(self):
        """Checks for optional dependencies (scipy)."""
        if self.cfg.method == SmoothingMethod.SAVITZKY_GOLAY:
            try:
                import scipy.signal
            except ImportError:
                logger.warning("SciPy not found. Fallback to Moving Average.")
                self.cfg.method = SmoothingMethod.MOVING_AVERAGE

    def apply(self) -> Tuple[np.ndarray, str]:
        """
        Applies the configured smoothing strategy.
        Returns: (Smoothed Array, Generated Column Name)
        """
        raw_data = self.df[self.cfg.target_column].values
        
        if self.cfg.method == SmoothingMethod.MOVING_AVERAGE:
            window = self._ensure_odd(self.cfg.ma_window)
            smoothed = pd.Series(raw_data).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
            col_name = f"{self.cfg.target_column}_ma_w{window}"
            label = f"MA (w={window})"
            
        elif self.cfg.method == SmoothingMethod.EXPONENTIAL_MA:
            smoothed = pd.Series(raw_data).ewm(alpha=self.cfg.ema_alpha, adjust=False).mean().to_numpy()
            col_name = f"{self.cfg.target_column}_ema_a{self.cfg.ema_alpha}"
            label = f"EMA (α={self.cfg.ema_alpha})"
            
        elif self.cfg.method == SmoothingMethod.SAVITZKY_GOLAY:
            from scipy.signal import savgol_filter
            window = self._ensure_odd(max(self.cfg.savgol_window, self.cfg.savgol_poly_order + 2))
            smoothed = savgol_filter(raw_data, window_length=window, polyorder=self.cfg.savgol_poly_order, mode='interp')
            col_name = f"{self.cfg.target_column}_savgol_w{window}_p{self.cfg.savgol_poly_order}"
            label = f"SavGol (w={window}, p={self.cfg.savgol_poly_order})"
            
        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")
            
        return smoothed, col_name, label

    @staticmethod
    def _ensure_odd(n: int) -> int:
        return n if n % 2 != 0 else n + 1

# --- Data Management ---

class DataHandler:
    """
    Handles File I/O, Backup rotation, and Encoding detection.
    """
    
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.encoding_used = None

    def load(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
            
        # Smart encoding detection
        encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
        for enc in encodings:
            try:
                df = pd.read_csv(self.filepath, encoding=enc)
                self.encoding_used = enc
                logger.info(f"Loaded CSV with encoding: {enc}")
                return df
            except UnicodeDecodeError:
                continue
        
        # Fallback
        return pd.read_csv(self.filepath)

    def create_backup(self):
        """Creates a timestamped backup of the original file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.filepath.stem}_backup_{timestamp}{self.filepath.suffix}"
        backup_path = self.filepath.parent / "backups" / backup_name
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.filepath, backup_path)
        logger.info(f"Backup created: {backup_path}")

    def save(self, df: pd.DataFrame):
        """Saves the DataFrame, preserving original encoding."""
        enc = self.encoding_used if self.encoding_used else "utf-8-sig"
        df.to_csv(self.filepath, index=False, encoding=enc)
        logger.info(f"Data saved to: {self.filepath}")

# --- Visualization ---

class TrajectoryVisualizer:
    """
    Manages plotting logic for full series and segments.
    """
    
    def __init__(self, output_dir: Path, dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_series(self, 
                    x: np.ndarray, 
                    y_raw: np.ndarray, 
                    y_smooth: np.ndarray, 
                    title: str, 
                    label_smooth: str, 
                    filename: str):
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_raw, label="Original Signal", color='lightgray', alpha=0.7, linewidth=1)
        plt.plot(x, y_smooth, label=label_smooth, color='#1f77b4', linewidth=1.5)
        
        plt.title(title)
        plt.xlabel("Index / Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()
        logger.info(f"Plot saved: {save_path}")

# --- Main Pipeline ---

class AnalyticsPipeline:
    def __init__(self, csv_path: str, smooth_cfg: SmoothingConfig, seg_cfg: SegmentConfig):
        self.path = Path(csv_path)
        self.smooth_cfg = smooth_cfg
        self.seg_cfg = seg_cfg
        self.handler = DataHandler(self.path)
        self.visualizer = TrajectoryVisualizer(self.path.parent / "analytics_results")

    def _resolve_segments(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Determines analysis segments based on priority: Time > Manual Index > Auto."""
        n = len(df)
        
        # 1. Time-based segmentation
        time_col = next((c for c in df.columns if c.lower() in ['time', 'timestamp', 't']), None)
        if self.seg_cfg.manual_segments_time and time_col:
            t_vals = df[time_col].values
            segments = []
            for t0, t1 in self.seg_cfg.manual_segments_time:
                mask = (t_vals >= t0) & (t_vals < t1)
                indices = np.where(mask)[0]
                if indices.size > 0:
                    segments.append((indices[0], indices[-1] + 1))
            if segments:
                return segments

        # 2. Manual Index segmentation
        if self.seg_cfg.manual_segments_idx:
            return self.seg_cfg.manual_segments_idx

        # 3. Auto segmentation fallback
        if self.seg_cfg.use_auto_fallback:
            win = self.seg_cfg.auto_segment_length
            k = self.seg_cfg.auto_num_segments
            starts = np.linspace(0, max(0, n - win), num=k+2, dtype=int)[1:-1]
            return [(s, s + win) for s in starts]
        
        return []

    def run(self):
        logger.info("Initializing Analytics Pipeline...")
        
        # 1. Load Data
        df = self.handler.load()
        if self.smooth_cfg.target_column not in df.columns:
            logger.error(f"Column '{self.smooth_cfg.target_column}' not found.")
            return

        # 2. Apply Smoothing
        smoother = SignalSmoother(df, self.smooth_cfg)
        y_smooth, col_name, label = smoother.apply()

        # 3. Save Data (Optional)
        if self.smooth_cfg.append_to_csv:
            if self.smooth_cfg.create_backup:
                self.handler.create_backup()
            
            # Handle column name collision
            final_col = col_name
            if not self.smooth_cfg.overwrite_col:
                counter = 1
                while final_col in df.columns:
                    counter += 1
                    final_col = f"{col_name}_{counter}"
            
            df[final_col] = y_smooth
            self.handler.save(df)

        # 4. Visualization (Full Series)
        indices = np.arange(len(df))
        self.visualizer.plot_series(
            indices, 
            df[self.smooth_cfg.target_column].values, 
            y_smooth, 
            f"Full Series Analysis: {label}", 
            label, 
            "full_series_analysis.png"
        )

        # 5. Segment Analysis
        segments = self._resolve_segments(df)
        for i, (start, end) in enumerate(segments):
            # Boundary checks
            s, e = max(0, start), min(len(df), end)
            if s >= e: continue
            
            self.visualizer.plot_series(
                indices[s:e], 
                df[self.smooth_cfg.target_column].values[s:e], 
                y_smooth[s:e], 
                f"Segment {i+1} Analysis [{s}:{e}]", 
                label, 
                f"segment_{i+1}_analysis.png"
            )
            
            # Export Segment Data
            seg_df = df.iloc[s:e]
            seg_csv_path = self.visualizer.output_dir / f"segment_{i+1}_data.csv"
            seg_df.to_csv(seg_csv_path, index=False)
            logger.info(f"Exported segment data: {seg_csv_path}")

        logger.info("Pipeline Execution Completed.")

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    # INPUT_FILE = r"E:\Dataset\PINN\Airsim\cleaned_output.csv"
    INPUT_FILE = "./data/cleaned/cleaned_horizontal_maneuver.csv"
    
    # 1. Configure Smoothing
    SMOOTH_OPTS = SmoothingConfig(
        target_column="Power_filtered",
        method=SmoothingMethod.MOVING_AVERAGE,
        ma_window=101,
        append_to_csv=True,
        create_backup=True
    )
    
    # 2. Configure Segmentation
    # Example: Select specific index ranges for deep-dive
    SEG_OPTS = SegmentConfig(
        manual_segments_idx=[(0, 5000), (6000, 9000)],
        use_auto_fallback=True
    )
    
    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    if not os.path.exists(os.path.dirname(INPUT_FILE)):
         logger.error(f"Directory not found: {os.path.dirname(INPUT_FILE)}")
    else:
        pipeline = AnalyticsPipeline(INPUT_FILE, SMOOTH_OPTS, SEG_OPTS)
        pipeline.run()
