"""Leakage-safe and causal preprocessing shared by PI-S-LSTM experiments.

The fit/transform boundary in this module is intentional:

* raw observations are split before any fitted statistic is computed;
* outlier statistics and Min-Max scalers are fitted on training rows only;
* outlier correction and moving averages use current/past observations only;
* every operation is reset at flight and dataset-split boundaries; and
* sliding windows are created independently inside every flight and split.

The archived numbered preprocessing scripts are retained for provenance, but
the corrected main experiment imports this module directly.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


RAW_FEATURE_COLUMNS = ("Vx (m/s)", "Vy (m/s)", "Vz (m/s)")
RAW_POWER_COLUMN = "Power (W)"
FILTERED_COLUMNS = {
    "Vx (m/s)": "Vx_filtered",
    "Vy (m/s)": "Vy_filtered",
    "Vz (m/s)": "Vz_filtered",
    "Power (W)": "Power_filtered",
}
SPLIT_ORDER = ("train", "validation", "test")
DEFAULT_PLATFORM = "Airsim-horizontal-aggressive"
EPSILON = 1e-12


@dataclass
class WindowedSplit:
    features: np.ndarray
    target: np.ndarray
    physical_target: np.ndarray
    alpha_data: np.ndarray
    alpha_physics: np.ndarray
    original_indices: np.ndarray
    timestamps: np.ndarray
    flight_ids: np.ndarray
    split_names: np.ndarray

    @property
    def number_of_windows(self) -> int:
        return int(self.features.shape[0])


@dataclass
class PreparedExperiment:
    train: WindowedSplit
    validation: WindowedSplit
    test: WindowedSplit
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    artifact: Dict[str, Any]
    processed_rows: pd.DataFrame
    split_manifest: pd.DataFrame
    reports: Dict[str, Dict[str, Any]]


def read_csv_compatible(path: str, **kwargs: Any) -> pd.DataFrame:
    """Read Unicode Windows paths with old pandas versions used by the project."""

    with open(path, "rb") as handle:
        return pd.read_csv(handle, **kwargs)


def write_csv_compatible(frame: pd.DataFrame, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)


def _as_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def synchronize_and_identify_flights(
    frame: pd.DataFrame,
    platform_name: str = DEFAULT_PLATFORM,
    timestamp_column: str = "Timestamp",
    gap_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """Validate the synchronized wide table and identify flight boundaries.

    The current AirSim main file already stores velocity and power in one row at
    a shared timestamp, so no interpolation is necessary here. Missing numeric
    values are left for the causal correction stage; no future sample is used.
    If ``flight_id`` is absent, a new flight starts at a sufficiently large
    timestamp gap. The detected threshold is never learned from validation/test
    target values.
    """

    required = list(RAW_FEATURE_COLUMNS) + [RAW_POWER_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required raw columns: {}".format(missing))

    result = frame.copy()
    result["_original_index"] = np.arange(len(result), dtype=np.int64)
    if "platform" not in result.columns:
        result["platform"] = platform_name
    else:
        result["platform"] = result["platform"].astype(str)

    _as_numeric(result, required)

    if timestamp_column in result.columns:
        numeric_time = pd.to_numeric(result[timestamp_column], errors="coerce")
        if int(numeric_time.notnull().sum()) == len(result):
            result[timestamp_column] = numeric_time.astype(float)
            result = result.sort_values(
                ["platform", timestamp_column, "_original_index"], kind="mergesort"
            ).reset_index(drop=True)
        else:
            # An unreliable timestamp must not be silently used for reordering.
            result[timestamp_column] = np.arange(len(result), dtype=float)
    else:
        result[timestamp_column] = np.arange(len(result), dtype=float)

    if "flight_id" not in result.columns:
        flight_labels = np.empty(len(result), dtype=object)
        for platform, positions in result.groupby("platform", sort=False).groups.items():
            pos = np.asarray(list(positions), dtype=np.int64)
            time_values = np.asarray(result.loc[pos, timestamp_column], dtype=float)
            positive_differences = np.diff(time_values)
            finite_positive = positive_differences[
                np.isfinite(positive_differences) & (positive_differences > 0.0)
            ]
            median_dt = (
                float(np.median(finite_positive))
                if finite_positive.size
                else 1.0
            )
            threshold = (
                float(gap_seconds)
                if gap_seconds is not None
                else max(1.0, 10.0 * median_dt)
            )
            starts = np.r_[True, positive_differences > threshold]
            local_ids = np.cumsum(starts.astype(np.int64))
            flight_labels[pos] = [
                "{}-F{:03d}".format(platform, int(identifier))
                for identifier in local_ids
            ]
        result["flight_id"] = flight_labels
    else:
        result["flight_id"] = result["flight_id"].astype(str)

    return result.reset_index(drop=True)


def _nearest_complete_flight_cut(
    cumulative_rows: np.ndarray, target_rows: float, low: int, high: int
) -> int:
    candidates = np.arange(low, high + 1, dtype=np.int64)
    distances = np.abs(cumulative_rows[candidates - 1] - target_rows)
    return int(candidates[int(np.argmin(distances))])


def split_before_preprocessing(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split rows, preferring complete-flight boundaries."""

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")
    if not (0.0 < validation_fraction < 1.0 - train_fraction):
        raise ValueError("validation_fraction is invalid.")

    blocks: List[pd.DataFrame] = []
    for platform, platform_frame in frame.groupby("platform", sort=False):
        ordered_ids = list(
            platform_frame.groupby("flight_id", sort=False)["_original_index"]
            .min()
            .sort_values()
            .index
        )
        if len(ordered_ids) >= 3:
            counts = np.asarray(
                [
                    int((platform_frame["flight_id"] == flight_id).sum())
                    for flight_id in ordered_ids
                ],
                dtype=np.int64,
            )
            cumulative = np.cumsum(counts)
            total = int(cumulative[-1])
            train_cut = _nearest_complete_flight_cut(
                cumulative, train_fraction * total, 1, len(ordered_ids) - 2
            )
            validation_cut = _nearest_complete_flight_cut(
                cumulative,
                (train_fraction + validation_fraction) * total,
                train_cut + 1,
                len(ordered_ids) - 1,
            )
            assignments: Dict[str, str] = {}
            for index, flight_id in enumerate(ordered_ids):
                if index < train_cut:
                    assignments[flight_id] = "train"
                elif index < validation_cut:
                    assignments[flight_id] = "validation"
                else:
                    assignments[flight_id] = "test"
            part = platform_frame.copy()
            part["_dataset_split"] = [
                assignments[str(flight_id)] for flight_id in part["flight_id"]
            ]
        else:
            # With fewer than three complete logs, preserve the existing
            # chronological 80/10/10 rule inside the long log.
            part = platform_frame.sort_values(
                ["Timestamp", "_original_index"], kind="mergesort"
            ).copy()
            number_of_rows = len(part)
            train_end = max(1, int(np.floor(train_fraction * number_of_rows)))
            validation_end = max(
                train_end + 1,
                int(
                    np.floor(
                        (train_fraction + validation_fraction) * number_of_rows
                    )
                ),
            )
            validation_end = min(validation_end, number_of_rows - 1)
            split_values = np.empty(number_of_rows, dtype=object)
            split_values[:train_end] = "train"
            split_values[train_end:validation_end] = "validation"
            split_values[validation_end:] = "test"
            part["_dataset_split"] = split_values
        blocks.append(part)

    split_frame = (
        pd.concat(blocks, axis=0)
        .sort_values(["platform", "Timestamp", "_original_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    observed = set(split_frame["_dataset_split"])
    if observed != set(SPLIT_ORDER):
        raise AssertionError("Expected train/validation/test, got {}".format(observed))

    original_sets = {
        split: set(
            np.asarray(
                split_frame.loc[
                    split_frame["_dataset_split"] == split, "_original_index"
                ],
                dtype=np.int64,
            )
        )
        for split in SPLIT_ORDER
    }
    if (
        original_sets["train"] & original_sets["validation"]
        or original_sets["train"] & original_sets["test"]
        or original_sets["validation"] & original_sets["test"]
    ):
        raise AssertionError("Raw observation indices overlap across splits.")

    manifest_rows: List[Dict[str, Any]] = []
    grouped = split_frame.groupby(
        ["platform", "flight_id", "_dataset_split"], sort=False
    )
    for (platform, flight_id, split), group in grouped:
        manifest_rows.append(
            {
                "platform": platform,
                "flight_id": flight_id,
                "dataset_split": split,
                "number_of_observations": len(group),
                "first_original_index": int(group["_original_index"].iloc[0]),
                "last_original_index": int(group["_original_index"].iloc[-1]),
                "start_timestamp": float(group["Timestamp"].iloc[0]),
                "end_timestamp": float(group["Timestamp"].iloc[-1]),
            }
        )
    return split_frame, pd.DataFrame(manifest_rows)


def fit_outlier_detector(
    training_frame: pd.DataFrame,
    columns: Sequence[str],
    sigma: float = 3.0,
    history_size: int = 11,
    epsilon: float = EPSILON,
) -> Dict[str, Any]:
    """Fit three-sigma statistics on training rows only."""

    if set(training_frame["_dataset_split"]) != {"train"}:
        raise ValueError("Outlier detector fit accepts training rows only.")
    by_platform: Dict[str, Dict[str, Dict[str, float]]] = {}
    for platform, group in training_frame.groupby("platform", sort=False):
        platform_stats: Dict[str, Dict[str, float]] = {}
        for column in columns:
            values = np.asarray(pd.to_numeric(group[column], errors="coerce"), float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(
                    "No finite training values for {} / {}".format(platform, column)
                )
            mean = float(np.mean(finite))
            standard_deviation = max(float(np.std(finite, ddof=0)), float(epsilon))
            platform_stats[column] = {
                "mean": mean,
                "std": standard_deviation,
                "lower": mean - float(sigma) * standard_deviation,
                "upper": mean + float(sigma) * standard_deviation,
                "fallback_median": float(np.median(finite)),
            }
        by_platform[str(platform)] = platform_stats

    return {
        "method": "training-fitted three-sigma",
        "sigma": float(sigma),
        "history_size": int(history_size),
        "epsilon": float(epsilon),
        "fitted_split": "train",
        "columns": list(columns),
        "by_platform": by_platform,
        "continuous_dynamic_protection": (
            "Only the first finite threshold violation in a consecutive run is "
            "corrected; later consecutive violations are retained as sustained dynamics."
        ),
    }


def _platform_statistics(
    fitted_statistics: Mapping[str, Any], platform: str
) -> Mapping[str, Mapping[str, float]]:
    by_platform = fitted_statistics["by_platform"]
    if platform in by_platform:
        return by_platform[platform]
    if len(by_platform) == 1:
        return next(iter(by_platform.values()))
    raise KeyError("No training outlier statistics for platform '{}'.".format(platform))


def causal_outlier_correction(
    frame: pd.DataFrame,
    fitted_statistics: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Correct isolated anomalies with the median of prior valid observations.

    Filtering is causal: a value at t never reads t+1 or later. Histories reset
    at every flight and dataset-split boundary. Consecutive finite threshold
    violations after the first are retained to protect sustained dynamics.
    """

    result = frame.copy()
    history_size = int(fitted_statistics["history_size"])
    columns = list(fitted_statistics["columns"])
    report: Dict[str, Any] = {
        "detected": {column: 0 for column in columns},
        "corrected": {column: 0 for column in columns},
        "fallbacks": {column: 0 for column in columns},
        "protected_sustained": {column: 0 for column in columns},
    }

    group_columns = ["platform", "flight_id", "_dataset_split"]
    for (platform, _, _), positions in result.groupby(
        group_columns, sort=False
    ).groups.items():
        pos = np.asarray(list(positions), dtype=np.int64)
        platform_stats = _platform_statistics(
            fitted_statistics, str(platform)
        )
        for column in columns:
            statistics = platform_stats[column]
            lower = float(statistics["lower"])
            upper = float(statistics["upper"])
            fallback = float(statistics["fallback_median"])
            values = np.asarray(result.loc[pos, column], dtype=float).copy()
            history: deque = deque(maxlen=history_size)
            previous_was_candidate = False

            for local_index, value in enumerate(values):
                finite = bool(np.isfinite(value))
                candidate = (not finite) or value < lower or value > upper
                if candidate:
                    report["detected"][column] += 1

                protect_sustained = (
                    candidate and finite and previous_was_candidate
                )
                if candidate and not protect_sustained:
                    if history:
                        replacement = float(np.median(np.asarray(history, float)))
                    else:
                        replacement = fallback
                        report["fallbacks"][column] += 1
                    values[local_index] = replacement
                    report["corrected"][column] += 1
                elif protect_sustained:
                    # A consecutive threshold run is evidence of a persistent
                    # regime rather than an isolated sensor spike.
                    history.append(float(value))
                    report["protected_sustained"][column] += 1
                else:
                    history.append(float(value))

                previous_was_candidate = candidate and finite
            result.loc[pos, column] = values

    report["detected_total"] = int(sum(report["detected"].values()))
    report["corrected_total"] = int(sum(report["corrected"].values()))
    report["fallback_total"] = int(sum(report["fallbacks"].values()))
    return result, report


def causal_moving_average(
    frame: pd.DataFrame,
    column_mapping: Mapping[str, str],
    window_size: int = 101,
) -> pd.DataFrame:
    """Apply a trailing mean independently inside every flight and split."""

    if int(window_size) < 1:
        raise ValueError("window_size must be positive.")
    result = frame.copy()
    group_columns = ["platform", "flight_id", "_dataset_split"]
    for input_column, output_column in column_mapping.items():
        result[output_column] = np.nan
        for _, positions in result.groupby(group_columns, sort=False).groups.items():
            pos = np.asarray(list(positions), dtype=np.int64)
            values = pd.Series(
                np.asarray(result.loc[pos, input_column], dtype=float)
            )
            filtered = values.rolling(
                window=int(window_size), min_periods=1, center=False
            ).mean()
            result.loc[pos, output_column] = np.asarray(filtered, dtype=float)
    return result


def _physical_reference_power(v_h: np.ndarray, v_v: np.ndarray) -> np.ndarray:
    """Current project's unchanged physical base plus 23-term SINDy residual."""

    c1, c2, c3, c4, c5 = (
        537.92430435,
        -11.81444764,
        -32.51778232,
        1851.19680972,
        2.00966979,
    )
    c6, c7, c8, c9 = (
        2.77965346e02,
        3.40071433e01,
        2.08030427e-01,
        4.0,
    )
    p_hover = 337.09
    coefficients = np.asarray(
        [
            16.2813,
            -7.1361,
            -33.7687,
            0.0,
            16.0607,
            -0.0,
            0.0,
            -0.2804,
            -0.3995,
            -0.0025,
            178.1034,
            99.2792,
            -1.0379,
            -284.5549,
            -188.0714,
            -6.2976,
            84.8423,
            -0.0,
            16.1915,
            -2.3171,
            1.0918,
            0.0,
            -14.524,
        ],
        dtype=float,
    )

    horizontal_inner = np.maximum(
        1.0 + v_h ** 4 / c4 - v_h ** 2 / c5, 1e-6
    )
    horizontal = c2 * v_h ** 2 + c3 * np.sqrt(horizontal_inner) + c5 * v_h ** 3
    vertical_positive = (
        c7 * v_v
        + c8 * v_v ** 3
        + (c7 + c8 * v_v ** 2)
        * np.sqrt(
            np.maximum(
                (1.0 + 4.0 * c8 / c9) * v_v ** 2 + 4.0 * c7 / c9,
                1e-6,
            )
        )
    )
    vertical_negative = (
        c7 * v_v
        - c8 * v_v ** 3
        + (c7 - c8 * v_v ** 2)
        * np.sqrt(
            np.maximum(
                (1.0 - 4.0 * c8 / c9) * v_v ** 2 + 4.0 * c7 / c9,
                1e-6,
            )
        )
    )
    vertical = np.where(v_v > 0.0, vertical_positive, vertical_negative)
    features = np.stack(
        [
            v_h,
            v_v,
            v_h ** 2,
            v_v ** 2,
            v_h ** 3,
            v_v ** 3,
            v_h ** 4,
            v_v ** 4,
            v_h ** 5,
            v_v ** 5,
            v_h * v_v,
            v_h ** 3 * v_v,
            v_h * v_v ** 3,
            v_h ** 2 * v_v,
            v_v ** 2 * v_h,
            v_h ** 3 * v_v ** 2,
            v_h ** 2 * v_v ** 2,
            v_h ** 4 * v_v,
            v_h * v_v ** 4,
            v_h ** 5 * v_v,
            v_h * v_v ** 5,
            np.sin(v_h),
            np.sin(v_v),
        ],
        axis=1,
    )
    return p_hover + horizontal + vertical + np.dot(features, coefficients)


def _causal_absolute_power_difference(frame: pd.DataFrame) -> np.ndarray:
    differences = np.zeros(len(frame), dtype=float)
    for _, positions in frame.groupby(
        ["platform", "flight_id", "_dataset_split"], sort=False
    ).groups.items():
        pos = np.asarray(list(positions), dtype=np.int64)
        power = np.asarray(frame.loc[pos, "Power_filtered"], dtype=float)
        differences[pos] = np.abs(np.diff(power, prepend=power[0]))
    return differences


def fit_and_apply_derived_columns(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Create unchanged model inputs, physical targets, and adaptive weights."""

    result = frame.copy()
    vx = np.asarray(result["Vx_filtered"], dtype=float)
    vy = np.asarray(result["Vy_filtered"], dtype=float)
    vz = np.asarray(result["Vz_filtered"], dtype=float)
    result["V_h"] = np.sqrt(vx ** 2 + vy ** 2)
    result["V_v"] = vz
    result["P_physical [W]"] = _physical_reference_power(
        np.asarray(result["V_h"], float), np.asarray(result["V_v"], float)
    )

    power_difference = _causal_absolute_power_difference(result)
    training_mask = np.asarray(result["_dataset_split"] == "train")
    weight_scaler = MinMaxScaler()
    weight_scaler.fit(power_difference[training_mask].reshape(-1, 1))
    score = weight_scaler.transform(power_difference.reshape(-1, 1)).reshape(-1)
    score = np.clip(score, 0.0, 1.0)

    epsilon = 1e-16
    r_data = np.power(np.clip(score, epsilon, 1.0), 0.8)
    r_physics = np.power(np.clip(1.0 - score, epsilon, 1.0), 2.0)
    denominator = r_data + 1e-3 * r_physics + epsilon
    alpha_data = np.clip(r_data / denominator, 1e-6, 1.0 - 1e-6)
    result["alpha_data"] = alpha_data
    result["alpha_phy"] = 1.0 - alpha_data
    return result, weight_scaler


def fit_training_scalers(
    processed_frame: pd.DataFrame,
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit both scalers once, using only training observations."""

    training = processed_frame.loc[
        processed_frame["_dataset_split"] == "train"
    ]
    features = np.stack(
        [
            np.asarray(training["V_h"], dtype=float),
            np.asarray(training["V_v"], dtype=float),
        ],
        axis=1,
    )
    target = np.asarray(training["Power_filtered"], dtype=float).reshape(-1, 1)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(features)
    target_scaler.fit(target)
    return feature_scaler, target_scaler


def create_windows_for_split(
    processed_frame: pd.DataFrame,
    split_name: str,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    window_size: int,
    stride: int = 1,
) -> WindowedSplit:
    """Construct aligned sequence-to-sequence windows without crossing bounds.

    The existing PI-S-LSTM outputs one power value per input time step, so the
    target remains the aligned power sequence. The last target in each window is
    the instantaneous endpoint target requested for causal deployment.
    """

    subset = processed_frame.loc[
        processed_frame["_dataset_split"] == split_name
    ].copy()
    feature_blocks: List[np.ndarray] = []
    target_blocks: List[np.ndarray] = []
    physical_blocks: List[np.ndarray] = []
    alpha_data_blocks: List[np.ndarray] = []
    alpha_physics_blocks: List[np.ndarray] = []
    index_blocks: List[np.ndarray] = []
    timestamp_blocks: List[np.ndarray] = []
    flight_labels: List[str] = []
    split_labels: List[str] = []

    for (platform, flight_id), group in subset.groupby(
        ["platform", "flight_id"], sort=False
    ):
        group = group.sort_values(
            ["Timestamp", "_original_index"], kind="mergesort"
        )
        if len(group) < int(window_size):
            continue
        features = np.stack(
            [
                np.asarray(group["V_h"], dtype=float),
                np.asarray(group["V_v"], dtype=float),
            ],
            axis=1,
        )
        target = np.asarray(group["Power_filtered"], dtype=float).reshape(-1, 1)
        physical = np.asarray(group["P_physical [W]"], dtype=float).reshape(-1, 1)
        features_scaled = feature_scaler.transform(features).astype(np.float32)
        target_scaled = target_scaler.transform(target).astype(np.float32)
        physical_scaled = target_scaler.transform(physical).astype(np.float32)
        alpha_data = np.asarray(group["alpha_data"], dtype=np.float32)
        alpha_physics = np.asarray(group["alpha_phy"], dtype=np.float32)
        original_indices = np.asarray(group["_original_index"], dtype=np.int64)
        timestamps = np.asarray(group["Timestamp"], dtype=float)

        starts = range(0, len(group) - int(window_size) + 1, int(stride))
        for start in starts:
            stop = start + int(window_size)
            feature_blocks.append(features_scaled[start:stop])
            target_blocks.append(target_scaled[start:stop])
            physical_blocks.append(physical_scaled[start:stop])
            alpha_data_blocks.append(alpha_data[start:stop])
            alpha_physics_blocks.append(alpha_physics[start:stop])
            index_blocks.append(original_indices[start:stop])
            timestamp_blocks.append(timestamps[start:stop])
            flight_labels.append("{}::{}".format(platform, flight_id))
            split_labels.append(split_name)

    if not feature_blocks:
        raise ValueError(
            "No {} windows could be constructed with window_size={}.".format(
                split_name, window_size
            )
        )
    return WindowedSplit(
        features=np.asarray(feature_blocks, dtype=np.float32),
        target=np.asarray(target_blocks, dtype=np.float32),
        physical_target=np.asarray(physical_blocks, dtype=np.float32),
        alpha_data=np.asarray(alpha_data_blocks, dtype=np.float32),
        alpha_physics=np.asarray(alpha_physics_blocks, dtype=np.float32),
        original_indices=np.asarray(index_blocks, dtype=np.int64),
        timestamps=np.asarray(timestamp_blocks, dtype=float),
        flight_ids=np.asarray(flight_labels, dtype=object),
        split_names=np.asarray(split_labels, dtype=object),
    )


def assert_window_boundaries(windowed: WindowedSplit, split_name: str) -> None:
    if not np.all(windowed.split_names == split_name):
        raise AssertionError("A window crossed a dataset split boundary.")
    for label, indices, timestamps in zip(
        windowed.flight_ids, windowed.original_indices, windowed.timestamps
    ):
        if len(set([label])) != 1:
            raise AssertionError("A window crossed a flight boundary.")
        if np.any(np.diff(timestamps) < 0.0):
            raise AssertionError("A window is not time ordered.")
        if len(indices) != len(timestamps):
            raise AssertionError("Feature/label time indices are misaligned.")


def _json_ready_statistics(statistics: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(statistics, ensure_ascii=False))


def save_preprocessing_artifact(
    artifact: Mapping[str, Any],
    artifact_path: str,
    split_manifest: Optional[pd.DataFrame] = None,
) -> None:
    directory = os.path.dirname(os.path.abspath(artifact_path))
    os.makedirs(directory, exist_ok=True)
    with open(artifact_path, "wb") as handle:
        pickle.dump(dict(artifact), handle, protocol=pickle.HIGHEST_PROTOCOL)

    config_path = os.path.splitext(artifact_path)[0] + ".json"
    public_config = {
        "artifact_version": artifact["artifact_version"],
        "created_utc": artifact["created_utc"],
        "source_data_path": artifact.get("source_data_path"),
        "split_rule": artifact["split_rule"],
        "outlier_statistics": _json_ready_statistics(
            artifact["outlier_statistics"]
        ),
        "causal_filter_window": artifact["causal_filter_window"],
        "window_size": artifact["window_size"],
        "stride": artifact["stride"],
        "feature_columns": artifact["feature_columns"],
        "target_column": artifact["target_column"],
        "scaler_fit_scope": "training observations only",
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(public_config, handle, ensure_ascii=False, indent=2)

    if split_manifest is not None:
        manifest_path = os.path.join(directory, "preprocessing_split_manifest.csv")
        write_csv_compatible(split_manifest, manifest_path)


def load_preprocessing_artifact(artifact_path: str) -> Dict[str, Any]:
    with open(artifact_path, "rb") as handle:
        artifact = pickle.load(handle)
    if int(artifact.get("artifact_version", -1)) != 1:
        raise ValueError("Unsupported preprocessing artifact version.")
    return artifact


def prepare_main_experiment(
    data_path: str,
    artifact_path: Optional[str],
    window_size: int = 5,
    stride: int = 1,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    sigma: float = 3.0,
    history_size: int = 11,
    filter_window: int = 101,
    platform_name: str = DEFAULT_PLATFORM,
    maximum_rows: Optional[int] = None,
    verbose: bool = True,
) -> PreparedExperiment:
    """Run the corrected main preprocessing pipeline."""

    read_kwargs: Dict[str, Any] = {}
    if maximum_rows is not None:
        read_kwargs["nrows"] = int(maximum_rows)
    raw = read_csv_compatible(data_path, **read_kwargs)
    synchronized = synchronize_and_identify_flights(
        raw, platform_name=platform_name
    )
    split_frame, manifest = split_before_preprocessing(
        synchronized,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    training_rows = split_frame.loc[
        split_frame["_dataset_split"] == "train"
    ].copy()
    detector = fit_outlier_detector(
        training_rows,
        columns=list(RAW_FEATURE_COLUMNS) + [RAW_POWER_COLUMN],
        sigma=sigma,
        history_size=history_size,
    )

    processed_parts: List[pd.DataFrame] = []
    reports: Dict[str, Dict[str, Any]] = {}
    for split_name in SPLIT_ORDER:
        part = split_frame.loc[
            split_frame["_dataset_split"] == split_name
        ].copy()
        corrected, report = causal_outlier_correction(part, detector)
        filtered = causal_moving_average(
            corrected, FILTERED_COLUMNS, window_size=filter_window
        )
        processed_parts.append(filtered)
        reports[split_name] = report
    processed = (
        pd.concat(processed_parts, axis=0)
        .sort_values(["platform", "Timestamp", "_original_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    processed, weight_scaler = fit_and_apply_derived_columns(processed)
    feature_scaler, target_scaler = fit_training_scalers(processed)

    windows = {
        split_name: create_windows_for_split(
            processed,
            split_name,
            feature_scaler,
            target_scaler,
            window_size=window_size,
            stride=stride,
        )
        for split_name in SPLIT_ORDER
    }
    for split_name in SPLIT_ORDER:
        assert_window_boundaries(windows[split_name], split_name)

    artifact: Dict[str, Any] = {
        "artifact_version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source_data_path": os.path.abspath(data_path),
        "split_rule": {
            "train_fraction": float(train_fraction),
            "validation_fraction": float(validation_fraction),
            "test_fraction": float(
                1.0 - train_fraction - validation_fraction
            ),
            "ordered": True,
            "prefer_complete_flights": True,
        },
        "outlier_statistics": detector,
        "causal_filter_window": int(filter_window),
        "window_size": int(window_size),
        "stride": int(stride),
        "feature_columns": ["V_h", "V_v"],
        "target_column": "Power_filtered",
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "adaptive_weight_scaler": weight_scaler,
    }
    if artifact_path:
        save_preprocessing_artifact(artifact, artifact_path, manifest)

    if verbose:
        print("\n[Preprocessing] synchronized wide-format input; no interpolation required")
        print(
            "[Preprocessing] split first: train/validation/test = "
            "{}/{}/{} raw observations".format(
                int((processed["_dataset_split"] == "train").sum()),
                int((processed["_dataset_split"] == "validation").sum()),
                int((processed["_dataset_split"] == "test").sum()),
            )
        )
        print(
            "[Preprocessing] detected {} flight logs; boundaries saved in manifest".format(
                int(processed["flight_id"].nunique())
            )
        )
        for split_name in SPLIT_ORDER:
            print(
                "[Preprocessing] {}: outliers detected={}, corrected={}, "
                "fallbacks={}, windows={}".format(
                    split_name,
                    reports[split_name]["detected_total"],
                    reports[split_name]["corrected_total"],
                    reports[split_name]["fallback_total"],
                    windows[split_name].number_of_windows,
                )
            )
        print(
            "[Preprocessing] causal trailing moving average window={}".format(
                filter_window
            )
        )
        print(
            "[Preprocessing] feature/target scalers fitted on training rows only"
        )
        print(
            "[Preprocessing] cross-flight windows=False; cross-split windows=False"
        )
        if artifact_path:
            print(
                "[Preprocessing] reusable online parameters: {}".format(
                    os.path.abspath(artifact_path)
                )
            )

    return PreparedExperiment(
        train=windows["train"],
        validation=windows["validation"],
        test=windows["test"],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        artifact=artifact,
        processed_rows=processed,
        split_manifest=manifest,
        reports=reports,
    )


class OnlineCausalPreprocessor:
    """Stateful one-sample-at-a-time preprocessing for causal deployment."""

    def __init__(self, artifact: Mapping[str, Any], platform: str) -> None:
        self.artifact = artifact
        self.platform = str(platform)
        self.statistics = _platform_statistics(
            artifact["outlier_statistics"], self.platform
        )
        self.history_size = int(
            artifact["outlier_statistics"]["history_size"]
        )
        self.filter_window = int(artifact["causal_filter_window"])
        self.feature_scaler = artifact["feature_scaler"]
        self.valid_history = {
            column: deque(maxlen=self.history_size)
            for column in list(RAW_FEATURE_COLUMNS) + [RAW_POWER_COLUMN]
        }
        self.filter_history = {
            column: deque(maxlen=self.filter_window)
            for column in list(RAW_FEATURE_COLUMNS) + [RAW_POWER_COLUMN]
        }
        self.previous_candidate = {
            column: False
            for column in list(RAW_FEATURE_COLUMNS) + [RAW_POWER_COLUMN]
        }
        self.active_flight_id: Optional[str] = None
        self.feature_window: deque = deque(
            maxlen=int(artifact["window_size"])
        )

    def reset_flight(self, flight_id: str) -> None:
        self.active_flight_id = str(flight_id)
        for history in self.valid_history.values():
            history.clear()
        for history in self.filter_history.values():
            history.clear()
        for column in self.previous_candidate:
            self.previous_candidate[column] = False
        self.feature_window.clear()

    def transform_sample(
        self,
        flight_id: str,
        vx: float,
        vy: float,
        vz: float,
        power: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.active_flight_id != str(flight_id):
            self.reset_flight(str(flight_id))
        raw_values = {
            "Vx (m/s)": float(vx),
            "Vy (m/s)": float(vy),
            "Vz (m/s)": float(vz),
        }
        if power is not None:
            raw_values[RAW_POWER_COLUMN] = float(power)

        filtered_values: Dict[str, float] = {}
        for column, value in raw_values.items():
            stats = self.statistics[column]
            finite = bool(np.isfinite(value))
            candidate = (
                (not finite)
                or value < float(stats["lower"])
                or value > float(stats["upper"])
            )
            protect_sustained = (
                candidate
                and finite
                and self.previous_candidate[column]
            )
            if candidate and not protect_sustained:
                if self.valid_history[column]:
                    corrected = float(
                        np.median(np.asarray(self.valid_history[column], float))
                    )
                else:
                    corrected = float(stats["fallback_median"])
            else:
                corrected = value
                self.valid_history[column].append(float(value))
            self.previous_candidate[column] = candidate and finite
            self.filter_history[column].append(float(corrected))
            filtered_values[column] = float(
                np.mean(np.asarray(self.filter_history[column], float))
            )

        v_h = float(
            np.sqrt(
                filtered_values["Vx (m/s)"] ** 2
                + filtered_values["Vy (m/s)"] ** 2
            )
        )
        v_v = float(filtered_values["Vz (m/s)"])
        scaled = self.feature_scaler.transform(
            np.asarray([[v_h, v_v]], dtype=float)
        )[0].astype(np.float32)
        self.feature_window.append(scaled)
        window = (
            np.asarray(self.feature_window, dtype=np.float32)
            if len(self.feature_window) == self.feature_window.maxlen
            else None
        )
        return {
            "V_h": v_h,
            "V_v": v_v,
            "scaled_features": scaled,
            "input_window": window,
            "window_ready": window is not None,
        }


def quick_check(data_path: str, maximum_rows: int) -> None:
    prepared = prepare_main_experiment(
        data_path=data_path,
        artifact_path=None,
        maximum_rows=maximum_rows,
        verbose=True,
    )
    raw_sets = {
        split_name: set(
            np.asarray(
                prepared.processed_rows.loc[
                    prepared.processed_rows["_dataset_split"] == split_name,
                    "_original_index",
                ],
                dtype=np.int64,
            )
        )
        for split_name in SPLIT_ORDER
    }
    assert not raw_sets["train"] & raw_sets["validation"]
    assert not raw_sets["train"] & raw_sets["test"]
    assert not raw_sets["validation"] & raw_sets["test"]
    print("[Quick check] PASS: fitted parameters, causality, and boundaries validated.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-check", action="store_true")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-rows", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if not arguments.quick_check:
        raise SystemExit("Use --quick-check for preprocessing-only validation.")
    quick_check(os.path.abspath(arguments.data), arguments.max_rows)
