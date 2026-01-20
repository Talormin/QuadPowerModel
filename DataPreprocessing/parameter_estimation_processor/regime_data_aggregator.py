"""
Regime Data Aggregation Module.

This module is responsible for aggregating raw simulation logs from distributed 
directories into consolidated datasets for physical parameter identification.

It performs:
1. Batch ingestion of CSV logs from specific maneuver folders.
2. Kinematic feature extraction (computing V_h from component velocities).
3. Conditional filtering based on vertical velocity (Vz) for ascent/descent separation.
4. Standardization of output columns for the system identification pipeline.

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---

@dataclass
class AggregationConfig:
    """Configuration for data paths and column mappings."""
    base_dir: str = "./data/airsim_raw"
    output_dir: str = "./data/processed_regimes"
    
    # Input Column Names (from AirSim)
    col_vx: str = "Vx (m/s)"
    col_vy: str = "Vy (m/s)"
    col_vz: str = "Vz (m/s)"
    col_power: str = "Power (W)"
    
    # Output Column Names (Standardized)
    out_vh: str = "V_h"
    out_vv: str = "V_v"
    out_power: str = "power"

@dataclass
class AggregationTask:
    """Defines a specific aggregation job."""
    source_folder: str
    output_filename: str
    filter_condition: Optional[Callable[[pd.Series], pd.Series]] = None
    description: str = ""


class RegimeAggregator:
    """
    Processor for merging and transforming simulation logs.
    """
    
    def __init__(self, config: AggregationConfig):
        self.cfg = config
        self.base_path = Path(self.cfg.base_dir)
        self.output_path = Path(self.cfg.output_dir)
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _process_dataframe(self, df: pd.DataFrame, condition: Optional[Callable]) -> pd.DataFrame:
        """Applies filters and computes derived features."""
        # 1. Apply conditional filter (e.g., Vz > 0)
        if condition is not None:
            mask = condition(df[self.cfg.col_vz])
            df = df[mask].copy()
        
        # 2. Compute Horizontal Velocity (V_h)
        v_h = np.sqrt(df[self.cfg.col_vx]**2 + df[self.cfg.col_vy]**2)
        
        # 3. Construct Standardized DataFrame
        df_new = pd.DataFrame({
            self.cfg.out_vh: v_h,
            self.cfg.out_vv: df[self.cfg.col_vz],
            self.cfg.out_power: df[self.cfg.col_power]
        })
        
        return df_new

    def run_task(self, task: AggregationTask):
        """Executes a single aggregation task."""
        folder_path = self.base_path / task.source_folder
        output_file = self.output_path / task.output_filename
        
        logger.info(f"Starting Task: {task.description}")
        logger.info(f"Scanning folder: {folder_path}")
        
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {folder_path}. Skipping.")
            return

        df_buffer = []
        total_rows = 0

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                processed_df = self._process_dataframe(df, task.filter_condition)
                
                if not processed_df.empty:
                    df_buffer.append(processed_df)
                    total_rows += len(processed_df)
            except Exception as e:
                logger.error(f"Error reading {file_path.name}: {e}")

        if df_buffer:
            final_df = pd.concat(df_buffer, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            logger.info(f"-> Saved {total_rows} samples to {output_file}")
        else:
            logger.warning("No valid data remaining after filtering.")

    def run_all(self, tasks: List[AggregationTask]):
        """Runs all defined tasks."""
        logger.info("Initializing Aggregation Pipeline...")
        for task in tasks:
            self.run_task(task)
        logger.info("Pipeline Completed.")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    # Use relative paths
    CONFIG = AggregationConfig(
        base_dir="./data/airsim_raw_split",
        output_dir="./data/training_sets"
    )
    
    # ------------------------------------------------------------------
    # TASK DEFINITIONS
    # ------------------------------------------------------------------
    # Define how each folder should be processed
    TASKS = [
        AggregationTask(
            source_folder="horizontal",
            output_filename="horizontal_motion.csv",
            filter_condition=None,
            description="Aggregating Horizontal Flight Data"
        ),
        AggregationTask(
            source_folder="climb",
            output_filename="vertical_ascent.csv",
            filter_condition=lambda vz: vz > 0.05, # Filter out near-zero noise
            description="Aggregating Vertical Ascent Data (Vz > 0)"
        ),
        AggregationTask(
            source_folder="descent",
            output_filename="vertical_descent.csv",
            filter_condition=lambda vz: vz < -0.05, # Filter out near-zero noise
            description="Aggregating Vertical Descent Data (Vz < 0)"
        )
    ]

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------
    if not os.path.exists(CONFIG.base_dir):
        logger.error(f"Base directory not found: {CONFIG.base_dir}")
        logger.info("Please ensure your simulation logs are organized into subfolders: horizontal, climb, descent.")
    else:
        aggregator = RegimeAggregator(CONFIG)
        aggregator.run_all(TASKS)
