"""Lightweight regression tests for leakage-safe causal preprocessing."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from causal_preprocessing import (
    FILTERED_COLUMNS,
    RAW_FEATURE_COLUMNS,
    RAW_POWER_COLUMN,
    SPLIT_ORDER,
    causal_moving_average,
    causal_outlier_correction,
    fit_outlier_detector,
    prepare_main_experiment,
    split_before_preprocessing,
    synchronize_and_identify_flights,
)


def make_synthetic_log() -> pd.DataFrame:
    rows = []
    timestamp = 1000.0
    original = 0
    for flight in range(9):
        for sample in range(16):
            phase = 0.25 * sample + 0.1 * flight
            rows.append(
                {
                    "Timestamp": timestamp,
                    "Vx (m/s)": 2.0 + np.sin(phase),
                    "Vy (m/s)": 0.5 * np.cos(phase),
                    "Vz (m/s)": 0.2 * np.sin(0.5 * phase),
                    "Power (W)": 430.0
                    + 12.0 * np.sin(phase)
                    + 1.5 * flight,
                    "source_index": original,
                }
            )
            timestamp += 0.02
            original += 1
        timestamp += 2.0
    return pd.DataFrame(rows)


def prepare_in_memory(frame: pd.DataFrame):
    synchronized = synchronize_and_identify_flights(frame)
    split, _ = split_before_preprocessing(synchronized)
    train = split.loc[split["_dataset_split"] == "train"].copy()
    statistics = fit_outlier_detector(
        train, list(RAW_FEATURE_COLUMNS) + [RAW_POWER_COLUMN]
    )
    corrected = {}
    filtered = {}
    for split_name in SPLIT_ORDER:
        part = split.loc[split["_dataset_split"] == split_name].copy()
        corrected[split_name], _ = causal_outlier_correction(part, statistics)
        filtered[split_name] = causal_moving_average(
            corrected[split_name], FILTERED_COLUMNS, window_size=5
        )
    return split, statistics, corrected, filtered


class CausalPreprocessingTests(unittest.TestCase):
    def test_01_test_values_do_not_change_training_outlier_fit(self):
        frame = make_synthetic_log()
        split, statistics_a, _, _ = prepare_in_memory(frame)
        test_indices = np.asarray(
            split.loc[split["_dataset_split"] == "test", "_original_index"],
            dtype=int,
        )
        modified = frame.copy()
        modified.loc[test_indices, "Power (W)"] += 100000.0
        _, statistics_b, _, _ = prepare_in_memory(modified)
        self.assertEqual(statistics_a, statistics_b)

    def test_02_validation_values_do_not_change_training_scaler(self):
        frame = make_synthetic_log()
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.csv")
            second = os.path.join(directory, "second.csv")
            frame.to_csv(first, index=False)
            baseline = prepare_main_experiment(
                first, artifact_path=None, filter_window=5, verbose=False
            )
            validation_indices = set(
                np.asarray(
                    baseline.processed_rows.loc[
                        baseline.processed_rows["_dataset_split"] == "validation",
                        "_original_index",
                    ],
                    dtype=int,
                )
            )
            changed = frame.copy()
            mask = changed.index.to_series().map(lambda value: value in validation_indices)
            changed.loc[mask, "Vx (m/s)"] += 50000.0
            changed.to_csv(second, index=False)
            perturbed = prepare_main_experiment(
                second, artifact_path=None, filter_window=5, verbose=False
            )
            np.testing.assert_allclose(
                baseline.feature_scaler.data_min_,
                perturbed.feature_scaler.data_min_,
            )
            np.testing.assert_allclose(
                baseline.feature_scaler.data_max_,
                perturbed.feature_scaler.data_max_,
            )

    def test_03_test_values_do_not_change_training_preprocessing(self):
        frame = make_synthetic_log()
        split, _, _, filtered_a = prepare_in_memory(frame)
        test_indices = np.asarray(
            split.loc[split["_dataset_split"] == "test", "_original_index"],
            dtype=int,
        )
        changed = frame.copy()
        changed.loc[test_indices, "Vx (m/s)"] = -99999.0
        _, _, _, filtered_b = prepare_in_memory(changed)
        columns = list(FILTERED_COLUMNS.values())
        np.testing.assert_allclose(
            np.asarray(filtered_a["train"][columns], dtype=float),
            np.asarray(filtered_b["train"][columns], dtype=float),
        )

    def test_04_windows_never_cross_flights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            make_synthetic_log().to_csv(path, index=False)
            prepared = prepare_main_experiment(
                path, artifact_path=None, filter_window=5, verbose=False
            )
            for windowed in (
                prepared.train,
                prepared.validation,
                prepared.test,
            ):
                for indices in windowed.original_indices:
                    flight_ids = set(
                        prepared.processed_rows.set_index("_original_index")
                        .loc[indices, "flight_id"]
                        .astype(str)
                    )
                    self.assertEqual(len(flight_ids), 1)

    def test_05_windows_never_cross_dataset_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            make_synthetic_log().to_csv(path, index=False)
            prepared = prepare_main_experiment(
                path, artifact_path=None, filter_window=5, verbose=False
            )
            for expected, windowed in (
                ("train", prepared.train),
                ("validation", prepared.validation),
                ("test", prepared.test),
            ):
                self.assertTrue(np.all(windowed.split_names == expected))

    def test_06_causal_filter_is_invariant_to_future_changes(self):
        split, _, corrected, _ = prepare_in_memory(make_synthetic_log())
        train = corrected["train"].copy()
        one_flight = train.loc[
            train["flight_id"] == train["flight_id"].iloc[0]
        ].copy()
        changed = one_flight.copy()
        changed.iloc[10:, changed.columns.get_loc("Power (W)")] += 1e6
        baseline = causal_moving_average(
            one_flight, {"Power (W)": "filtered"}, window_size=5
        )
        perturbed = causal_moving_average(
            changed, {"Power (W)": "filtered"}, window_size=5
        )
        np.testing.assert_allclose(
            np.asarray(baseline["filtered"].iloc[:10], dtype=float),
            np.asarray(perturbed["filtered"].iloc[:10], dtype=float),
        )

    def test_07_causal_correction_is_invariant_to_future_changes(self):
        split, statistics, _, _ = prepare_in_memory(make_synthetic_log())
        train = split.loc[split["_dataset_split"] == "train"].copy()
        one_flight = train.loc[
            train["flight_id"] == train["flight_id"].iloc[0]
        ].copy()
        changed = one_flight.copy()
        changed.iloc[10:, changed.columns.get_loc("Power (W)")] += 1e6
        baseline, _ = causal_outlier_correction(one_flight, statistics)
        perturbed, _ = causal_outlier_correction(changed, statistics)
        np.testing.assert_allclose(
            np.asarray(baseline["Power (W)"].iloc[:10], dtype=float),
            np.asarray(perturbed["Power (W)"].iloc[:10], dtype=float),
        )

    def test_08_feature_and_label_indices_are_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            make_synthetic_log().to_csv(path, index=False)
            prepared = prepare_main_experiment(
                path, artifact_path=None, filter_window=5, verbose=False
            )
            lookup = prepared.processed_rows.set_index("_original_index")
            for indices, target_scaled in zip(
                prepared.test.original_indices, prepared.test.target
            ):
                target = prepared.target_scaler.inverse_transform(target_scaled)
                expected = np.asarray(
                    lookup.loc[indices, "Power_filtered"], dtype=float
                ).reshape(-1, 1)
                np.testing.assert_allclose(target, expected, rtol=1e-6, atol=1e-5)

    def test_09_time_order_is_preserved_in_every_split(self):
        split, _, _, _ = prepare_in_memory(make_synthetic_log())
        for split_name in SPLIT_ORDER:
            part = split.loc[split["_dataset_split"] == split_name]
            for _, group in part.groupby(["platform", "flight_id"], sort=False):
                times = np.asarray(group["Timestamp"], dtype=float)
                self.assertTrue(np.all(np.diff(times) >= 0.0))

    def test_10_raw_observation_indices_are_disjoint(self):
        split, _, _, _ = prepare_in_memory(make_synthetic_log())
        index_sets = {
            split_name: set(
                np.asarray(
                    split.loc[
                        split["_dataset_split"] == split_name,
                        "_original_index",
                    ],
                    dtype=int,
                )
            )
            for split_name in SPLIT_ORDER
        }
        self.assertFalse(index_sets["train"] & index_sets["validation"])
        self.assertFalse(index_sets["train"] & index_sets["test"])
        self.assertFalse(index_sets["validation"] & index_sets["test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
