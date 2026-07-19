"""Baseline models for the PI-S-LSTM comparative experiment.

Implemented baselines
---------------------
1. MR-PM: analytical rotor-power model used by the archived project.
2. MEC-FM: standardized ridge-regression flight-mechanics surrogate using
   polynomial horizontal/vertical-speed terms.
3. LSTM-PM: data-only BiLSTM sequence regressor.
4. ODP-LSTM: output-domain physics-regularized BiLSTM that predicts power
   directly and penalizes disagreement with the physical baseline.
5. SE-PC: serial error-correction model that adds a learned BiLSTM correction
   to normalized physical power.
6. PINN-SE: tanh multilayer sequence estimator trained with data, physics,
   temporal-smoothness, and L2 penalties.
7. PhysCon-BE: bounded-error physics-constrained BiLSTM whose learned
   correction is limited around the physical baseline.
8. Transformer: lightweight Transformer encoder sequence regressor.
9. TCN: causal temporal convolutional sequence regressor.

Important reproducibility notes
-------------------------------
* Every neural model maps [batch, time, 2] -> [batch, time, 1]. This keeps the
  prediction target identical across baselines.
* The two input channels are expected to be [V_h, V_v].
* Physics-guided models expect a dictionary with keys ``sequence`` and
  ``physical_total``. ``physical_total`` must be normalized with the same
  training-set scaler as measured power.
* The numerical presets below reproduce the model-specific candidate values
  supplied for the comparative experiment. They are intentionally different across
  models and can be overridden through the public builders or ``build_baseline``.
* These presets should be described as Bayesian-search candidate configurations
  unless the corresponding optimization logs are available. Replace them with the
  actual validation-selected values before claiming optimized hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf


TensorLike = Union[tf.Tensor, Mapping[str, tf.Tensor]]


# ---------------------------------------------------------------------------
# Reproducible default configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Training controls shared by neural baselines.

    Values marked ``None`` are intentionally unspecified because they depend on
    the current dataset, sampling frequency, normalization, and validation split.
    Fill them after training-set/validation-set analysis; do not select them using
    the final test set.
    """

    # Dataset dependent: determine from sampling frequency and useful temporal span.
    window_size: Optional[int] = None
    # Interface definition: the present experiment uses [V_h, V_v].
    n_features: int = 2
    # Dataset/optimizer dependent: tune on the validation set.
    learning_rate: Optional[float] = None
    # Dataset size and hardware dependent.
    batch_size: Optional[int] = None
    # Training-curve dependent; use a sufficiently high cap together with early stopping.
    maximum_epochs: Optional[int] = None
    # Validation-curve dependent.
    early_stopping_patience: Optional[int] = None
    # Loss-scale dependent, especially after changing target normalization.
    early_stopping_min_delta: Optional[float] = None
    # Reproducibility control, not a fitted dataset parameter.
    random_seed: int = 42


@dataclass(frozen=True)
class BiLSTMConfig:
    """BiLSTM architecture selected from training/validation experiments."""

    # Dataset size and temporal complexity dependent.
    units: Optional[Tuple[int, ...]] = None
    dense_units: Optional[Tuple[int, ...]] = None
    # Overfitting dependent.
    dropout: Optional[float] = None
    # Feature/target scale and overfitting dependent.
    dense_l2: Optional[float] = None


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Loss weights that must be tuned after fixing all normalization rules."""

    # All four weights depend on target scaling and relative loss magnitudes.
    data_weight: Optional[float] = None
    physics_weight: Optional[float] = None
    smoothness_weight: Optional[float] = None
    global_l2_weight: Optional[float] = None


@dataclass(frozen=True)
class TransformerConfig:
    """Transformer capacity selected from training/validation experiments."""

    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    key_dim: Optional[int] = None
    num_blocks: Optional[int] = None
    ffn_units: Optional[int] = None
    dropout: Optional[float] = None


@dataclass(frozen=True)
class TCNConfig:
    """TCN receptive field and capacity selected for the current dataset."""

    filters: Optional[int] = None
    kernel_size: Optional[int] = None
    dilations: Optional[Tuple[int, ...]] = None
    convolutions_per_block: Optional[int] = None
    dropout: Optional[float] = None


# ---------------------------------------------------------------------------
# Model-specific experiment presets
# ---------------------------------------------------------------------------
# These values reproduce the supplied Bayesian-style candidate configurations.
# They remain overrideable and should be replaced by actual optimization outputs
# when the completed Bayesian-search records are available.

LSTM_PM_TRAINING = TrainingConfig(
    window_size=17,
    learning_rate=3.6e-4,
    batch_size=72,
    maximum_epochs=500,
    early_stopping_patience=28,
    early_stopping_min_delta=8.0e-6,
)
LSTM_PM_ARCHITECTURE = BiLSTMConfig(
    units=(113, 47),
    dense_units=(91, 43, 21),
    dropout=0.176,
    dense_l2=2.8e-4,
)

ODP_LSTM_TRAINING = TrainingConfig(
    window_size=21,
    learning_rate=2.7e-4,
    batch_size=64,
    maximum_epochs=500,
    early_stopping_patience=36,
    early_stopping_min_delta=6.0e-6,
)
ODP_LSTM_ARCHITECTURE = BiLSTMConfig(
    units=(121, 56),
    dense_units=(97, 39, 24),
    dropout=0.143,
    dense_l2=0.0,
)
ODP_LSTM_LOSS = PhysicsLossConfig(
    data_weight=1.0,
    physics_weight=0.137,
    smoothness_weight=6.2e-4,
    global_l2_weight=8.7e-6,
)

SE_PC_TRAINING = TrainingConfig(
    window_size=19,
    learning_rate=4.3e-4,
    batch_size=80,
    maximum_epochs=500,
    early_stopping_patience=31,
    early_stopping_min_delta=9.0e-6,
)
SE_PC_ARCHITECTURE = BiLSTMConfig(
    units=(107, 44),
    dense_units=(86, 35, 18),
    dropout=0.207,
    dense_l2=0.0,
)
SE_PC_LOSS = PhysicsLossConfig(
    data_weight=1.0,
    physics_weight=0.184,
    smoothness_weight=3.9e-4,
    global_l2_weight=1.3e-5,
)
SE_PC_CORRECTION_HEAD_L2 = 2.4e-4

PINN_SE_TRAINING = TrainingConfig(
    window_size=23,
    learning_rate=7.1e-4,
    batch_size=56,
    maximum_epochs=800,
    early_stopping_patience=44,
    early_stopping_min_delta=4.0e-6,
)
PINN_SE_LOSS = PhysicsLossConfig(
    data_weight=1.0,
    physics_weight=0.226,
    smoothness_weight=9.1e-4,
    global_l2_weight=5.8e-6,
)
PINN_SE_HIDDEN_UNITS = (93, 71, 49, 28)
PINN_SE_FIRST_LAYER_L2 = 7.3e-6

PHYSCON_BE_TRAINING = TrainingConfig(
    window_size=18,
    learning_rate=3.1e-4,
    batch_size=88,
    maximum_epochs=500,
    early_stopping_patience=33,
    early_stopping_min_delta=7.0e-6,
)
PHYSCON_BE_ARCHITECTURE = BiLSTMConfig(
    units=(116, 52),
    dense_units=(89, 36, 20),
    dropout=0.162,
    dense_l2=0.0,
)
PHYSCON_BE_LOSS = PhysicsLossConfig(
    data_weight=1.0,
    physics_weight=0.112,
    smoothness_weight=4.8e-4,
    global_l2_weight=9.6e-6,
)
PHYSCON_BE_MAXIMUM_CORRECTION = 0.173

TRANSFORMER_TRAINING = TrainingConfig(
    window_size=16,
    learning_rate=2.4e-4,
    batch_size=48,
    maximum_epochs=400,
    early_stopping_patience=26,
    early_stopping_min_delta=1.1e-5,
)
TRANSFORMER_ARCHITECTURE = TransformerConfig(
    d_model=72,
    num_heads=6,
    key_dim=12,
    num_blocks=3,
    ffn_units=184,
    dropout=0.128,
)

TCN_TRAINING = TrainingConfig(
    window_size=27,
    learning_rate=6.4e-4,
    batch_size=96,
    maximum_epochs=400,
    early_stopping_patience=23,
    early_stopping_min_delta=1.3e-5,
)
TCN_ARCHITECTURE = TCNConfig(
    filters=60,
    kernel_size=5,
    dilations=(1, 2, 4, 8),
    convolutions_per_block=2,
    dropout=0.171,
)

# Backward-compatible generic aliases. Direct model builders use their own
# model-specific constants rather than these aliases.
DEFAULT_TRAINING = LSTM_PM_TRAINING
DEFAULT_BILSTM = LSTM_PM_ARCHITECTURE
DEFAULT_PHYSICS_LOSS = ODP_LSTM_LOSS
DEFAULT_TRANSFORMER = TRANSFORMER_ARCHITECTURE
DEFAULT_TCN = TCN_ARCHITECTURE


def _require_value(name: str, value: Any) -> Any:
    """Return an explicitly configured value or raise a dataset-aware error."""

    if value is None:
        raise ValueError(
            f"{name} is dataset dependent and has intentionally no default. "
            "Set it using only the training/validation data before building or fitting."
        )
    return value


def _require_positive_int(name: str, value: Optional[int]) -> int:
    value = int(_require_value(name, value))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _require_positive_float(name: str, value: Optional[float]) -> float:
    value = float(_require_value(name, value))
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _require_nonnegative_float(name: str, value: Optional[float]) -> float:
    value = float(_require_value(name, value))
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validate_training_shape(config: TrainingConfig) -> Tuple[int, int]:
    window_size = _require_positive_int("TrainingConfig.window_size", config.window_size)
    n_features = _require_positive_int("TrainingConfig.n_features", config.n_features)
    return window_size, n_features


def _validate_bilstm_config(config: BiLSTMConfig) -> Tuple[Tuple[int, ...], Tuple[int, ...], float, float]:
    units = tuple(_require_value("BiLSTMConfig.units", config.units))
    dense_units = tuple(_require_value("BiLSTMConfig.dense_units", config.dense_units))
    if not units or not dense_units or any(int(v) <= 0 for v in units + dense_units):
        raise ValueError("BiLSTM hidden-layer sizes must contain positive integers.")
    dropout = float(_require_value("BiLSTMConfig.dropout", config.dropout))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("BiLSTMConfig.dropout must be in [0, 1).")
    dense_l2 = _require_nonnegative_float("BiLSTMConfig.dense_l2", config.dense_l2)
    return tuple(map(int, units)), tuple(map(int, dense_units)), dropout, dense_l2


def _validate_physics_loss_config(config: PhysicsLossConfig) -> None:
    _require_nonnegative_float("PhysicsLossConfig.data_weight", config.data_weight)
    _require_nonnegative_float("PhysicsLossConfig.physics_weight", config.physics_weight)
    _require_nonnegative_float("PhysicsLossConfig.smoothness_weight", config.smoothness_weight)
    _require_nonnegative_float("PhysicsLossConfig.global_l2_weight", config.global_l2_weight)


def _validate_transformer_config(config: TransformerConfig) -> None:
    d_model = _require_positive_int("TransformerConfig.d_model", config.d_model)
    num_heads = _require_positive_int("TransformerConfig.num_heads", config.num_heads)
    key_dim = _require_positive_int("TransformerConfig.key_dim", config.key_dim)
    _require_positive_int("TransformerConfig.num_blocks", config.num_blocks)
    _require_positive_int("TransformerConfig.ffn_units", config.ffn_units)
    dropout = float(_require_value("TransformerConfig.dropout", config.dropout))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("TransformerConfig.dropout must be in [0, 1).")
    if num_heads * key_dim != d_model:
        raise ValueError(
            "For this implementation, num_heads * key_dim must equal d_model."
        )


def _validate_tcn_config(config: TCNConfig) -> None:
    _require_positive_int("TCNConfig.filters", config.filters)
    _require_positive_int("TCNConfig.kernel_size", config.kernel_size)
    dilations = tuple(_require_value("TCNConfig.dilations", config.dilations))
    if not dilations or any(int(v) <= 0 for v in dilations):
        raise ValueError("TCNConfig.dilations must contain positive integers.")
    _require_positive_int(
        "TCNConfig.convolutions_per_block", config.convolutions_per_block
    )
    dropout = float(_require_value("TCNConfig.dropout", config.dropout))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("TCNConfig.dropout must be in [0, 1).")


def set_reproducible_seed(seed: int = 42) -> None:
    """Set NumPy and TensorFlow seeds without changing deterministic-op policy."""

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_early_stopping(
    config: TrainingConfig = DEFAULT_TRAINING,
) -> tf.keras.callbacks.EarlyStopping:
    """Create the common validation-loss early-stopping callback."""

    patience = _require_positive_int(
        "TrainingConfig.early_stopping_patience", config.early_stopping_patience
    )
    min_delta = _require_nonnegative_float(
        "TrainingConfig.early_stopping_min_delta", config.early_stopping_min_delta
    )
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )


# ---------------------------------------------------------------------------
# Analytical physical baselines
# ---------------------------------------------------------------------------


@dataclass
class MRPMParameters:
    """Aircraft- and dataset-specific MR-PM coefficients.

    The defaults reproduce the calibrated coefficient set supplied for this
    comparative experiment. Override them when changing the aircraft, propulsion
    system, power-measurement chain, or fitted flight dataset.
    """

    c1: Optional[float] = 537.92430435
    c2: Optional[float] = -11.81444764
    c3: Optional[float] = -32.51778232
    c4: Optional[float] = 1851.19680972
    c5: Optional[float] = 2.00966979
    c6: Optional[float] = 277.965346
    c7: Optional[float] = 34.0071433
    c8: Optional[float] = 0.208030427
    c9: Optional[float] = 4.0
    p_hover: Optional[float] = 337.09


class MRPMRegressor:
    """Analytical multirotor power model (MR-PM).

    The constructor uses the supplied experiment preset unless another parameter
    set is provided. ``fit`` refines identifiable shape coefficients and hover power.
    C1 and C6 are held fixed because total power subtracts their offsets, making
    them unidentifiable from total-power observations alone.
    """

    def __init__(self, parameters: Optional[MRPMParameters] = None) -> None:
        self.parameters = parameters or MRPMParameters()
        self.optimization_result_: Optional[Any] = None

    def _require_parameters(self) -> MRPMParameters:
        missing = [
            name
            for name in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "p_hover")
            if getattr(self.parameters, name) is None
        ]
        if missing:
            raise ValueError(
                "MR-PM parameters are aircraft/dataset specific and have no "
                f"default values. Set: {', '.join(missing)}."
            )
        return self.parameters

    @staticmethod
    def _as_vector(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(result)):
            raise ValueError("MR-PM inputs must contain only finite values.")
        return result

    def horizontal_power(self, v_h: np.ndarray) -> np.ndarray:
        p = self._require_parameters()
        v_h = self._as_vector(v_h)
        inner = np.maximum(1.0 + v_h**4 / p.c4 - v_h**2 / p.c5, 1e-8)
        return p.c1 + p.c2 * v_h**2 + p.c3 * np.sqrt(inner) + p.c5 * v_h**3

    def vertical_power(self, v_v: np.ndarray) -> np.ndarray:
        p = self._require_parameters()
        v_v = self._as_vector(v_v)
        inner_up = np.maximum(
            (1.0 + 4.0 * p.c8 / p.c9) * v_v**2 + 4.0 * p.c7 / p.c9,
            1e-8,
        )
        inner_down = np.maximum(
            (1.0 - 4.0 * p.c8 / p.c9) * v_v**2 + 4.0 * p.c7 / p.c9,
            1e-8,
        )
        ascent = (
            p.c6
            + p.c7 * v_v
            + p.c8 * v_v**3
            + (p.c7 + p.c8 * v_v**2) * np.sqrt(inner_up)
        )
        descent = (
            p.c6
            + p.c7 * v_v
            - p.c8 * v_v**3
            + (p.c7 - p.c8 * v_v**2) * np.sqrt(inner_down)
        )
        return np.where(v_v > 0.0, ascent, descent)

    def predict(self, v_h: np.ndarray, v_v: np.ndarray) -> np.ndarray:
        p = self._require_parameters()
        v_h = self._as_vector(v_h)
        v_v = self._as_vector(v_v)
        if v_h.shape != v_v.shape:
            raise ValueError("v_h and v_v must have identical shapes.")
        horizontal_offset = self.horizontal_power(v_h) - p.c1
        vertical_offset = self.vertical_power(v_v) - p.c6
        return p.p_hover + horizontal_offset + vertical_offset

    def fit(
        self,
        v_h: np.ndarray,
        v_v: np.ndarray,
        power: np.ndarray,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
        max_nfev: int = 30000,
    ) -> "MRPMRegressor":
        """Refine MR-PM coefficients by bounded nonlinear least squares.

        ``lower_bounds`` and ``upper_bounds`` are intentionally required because
        physically meaningful ranges depend on the vehicle, units, and dataset.
        Each sequence must follow [c2, c3, c4, c5, c7, c8, c9, p_hover].
        """

        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("SciPy is required to fit MR-PM.") from exc

        v_h = self._as_vector(v_h)
        v_v = self._as_vector(v_v)
        power = self._as_vector(power)
        if not (v_h.shape == v_v.shape == power.shape):
            raise ValueError("v_h, v_v, and power must have identical shapes.")

        p = self._require_parameters()
        initial = np.array(
            [p.c2, p.c3, p.c4, p.c5, p.c7, p.c8, p.c9, p.p_hover],
            dtype=np.float64,
        )
        lower = np.asarray(
            _require_value("MRPMRegressor.fit.lower_bounds", lower_bounds),
            dtype=np.float64,
        ).reshape(-1)
        upper = np.asarray(
            _require_value("MRPMRegressor.fit.upper_bounds", upper_bounds),
            dtype=np.float64,
        ).reshape(-1)
        if lower.size != 8 or upper.size != 8:
            raise ValueError("MR-PM lower_bounds and upper_bounds must each contain 8 values.")
        if np.any(lower >= upper):
            raise ValueError("Every MR-PM lower bound must be smaller than its upper bound.")

        def residual(theta: np.ndarray) -> np.ndarray:
            c2, c3, c4, c5, c7, c8, c9, p_hover = theta
            inner_h = np.maximum(1.0 + v_h**4 / c4 - v_h**2 / c5, 1e-8)
            horizontal_offset = c2 * v_h**2 + c3 * np.sqrt(inner_h) + c5 * v_h**3

            inner_up = np.maximum(
                (1.0 + 4.0 * c8 / c9) * v_v**2 + 4.0 * c7 / c9,
                1e-8,
            )
            inner_down = np.maximum(
                (1.0 - 4.0 * c8 / c9) * v_v**2 + 4.0 * c7 / c9,
                1e-8,
            )
            vertical_up = (
                c7 * v_v
                + c8 * v_v**3
                + (c7 + c8 * v_v**2) * np.sqrt(inner_up)
            )
            vertical_down = (
                c7 * v_v
                - c8 * v_v**3
                + (c7 - c8 * v_v**2) * np.sqrt(inner_down)
            )
            vertical_offset = np.where(v_v > 0.0, vertical_up, vertical_down)
            return p_hover + horizontal_offset + vertical_offset - power

        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=max_nfev,
        )
        (
            p.c2,
            p.c3,
            p.c4,
            p.c5,
            p.c7,
            p.c8,
            p.c9,
            p.p_hover,
        ) = result.x.tolist()
        self.optimization_result_ = result
        return self


@dataclass
class MECFMConfig:
    """Configuration for the polynomial flight-mechanics surrogate.

    ``ridge_alpha`` has no universal value. Select it by cross-validation on the
    current training data after applying the exact feature standardization used
    in deployment.
    """

    ridge_alpha: Optional[float] = 6.8e-5


class MECFMRegressor:
    """Polynomial velocity-power surrogate fitted by standardized ridge regression.

    The feature set includes linear, quadratic, cubic, asymmetric vertical-speed,
    and horizontal/vertical interaction terms. It is intended as a transparent
    low-complexity comparison rather than a first-principles rotor model.
    """

    feature_names: Tuple[str, ...] = (
        "1",
        "Vh",
        "Vv",
        "abs(Vv)",
        "Vh^2",
        "Vv^2",
        "Vh^3",
        "Vh*Vv",
        "Vh*abs(Vv)",
    )

    def __init__(self, config: MECFMConfig = MECFMConfig()) -> None:
        self.config = config
        self.coefficients_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_scale_: Optional[np.ndarray] = None

    @staticmethod
    def _features(v_h: np.ndarray, v_v: np.ndarray) -> np.ndarray:
        v_h = np.asarray(v_h, dtype=np.float64).reshape(-1)
        v_v = np.asarray(v_v, dtype=np.float64).reshape(-1)
        if v_h.shape != v_v.shape:
            raise ValueError("v_h and v_v must have identical shapes.")
        return np.column_stack(
            [
                np.ones_like(v_h),
                v_h,
                v_v,
                np.abs(v_v),
                v_h**2,
                v_v**2,
                v_h**3,
                v_h * v_v,
                v_h * np.abs(v_v),
            ]
        )

    def fit(
        self, v_h: np.ndarray, v_v: np.ndarray, power: np.ndarray
    ) -> "MECFMRegressor":
        features = self._features(v_h, v_v)
        target = np.asarray(power, dtype=np.float64).reshape(-1)
        if features.shape[0] != target.shape[0]:
            raise ValueError("The feature and target sample counts differ.")

        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        mean[0] = 0.0
        scale[0] = 1.0
        scale[scale < 1e-12] = 1.0
        standardized = (features - mean) / scale

        penalty = np.eye(standardized.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        ridge_alpha = _require_nonnegative_float("MECFMConfig.ridge_alpha", self.config.ridge_alpha)
        lhs = standardized.T @ standardized + ridge_alpha * penalty
        rhs = standardized.T @ target
        self.coefficients_ = np.linalg.solve(lhs, rhs)
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        return self

    def predict(self, v_h: np.ndarray, v_v: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("MEC-FM must be fitted before predict is called.")
        features = self._features(v_h, v_v)
        standardized = (features - self.feature_mean_) / self.feature_scale_
        return standardized @ self.coefficients_


# ---------------------------------------------------------------------------
# Shared neural-network utilities
# ---------------------------------------------------------------------------


def _sequence_from_inputs(inputs: TensorLike) -> tf.Tensor:
    if isinstance(inputs, Mapping):
        if "sequence" not in inputs:
            raise KeyError("Physics-guided input must include the 'sequence' key.")
        return tf.convert_to_tensor(inputs["sequence"])
    return tf.convert_to_tensor(inputs)


def _bilstm_backbone(
    sequence: tf.Tensor,
    config: BiLSTMConfig,
    prefix: str,
) -> tf.Tensor:
    units, dense_units, dropout, dense_l2 = _validate_bilstm_config(config)
    x = sequence
    for index, layer_units in enumerate(units):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(layer_units, return_sequences=True),
            name=f"{prefix}_bilstm_{index + 1}",
        )(x)
    x = tf.keras.layers.LayerNormalization(name=f"{prefix}_layer_norm")(x)

    for index, layer_units in enumerate(dense_units):
        regularizer = tf.keras.regularizers.l2(dense_l2) if index == 0 else None
        x = tf.keras.layers.Dense(
            layer_units,
            activation="relu",
            kernel_regularizer=regularizer,
            name=f"{prefix}_dense_{index + 1}",
        )(x)
        x = tf.keras.layers.Dropout(
            dropout, name=f"{prefix}_dropout_{index + 1}"
        )(x)

    residual = tf.keras.layers.Dense(
        dense_units[-1], name=f"{prefix}_residual_projection"
    )(sequence)
    return tf.keras.layers.Add(name=f"{prefix}_residual_add")([x, residual])



class PhysicsGuidedSequenceModel(tf.keras.Model):
    """Keras wrapper that adds output-physics, smoothness, and global L2 losses."""

    def __init__(
        self,
        network: tf.keras.Model,
        loss_config: PhysicsLossConfig,
        name: str,
    ) -> None:
        _validate_physics_loss_config(loss_config)
        super().__init__(name=name)
        self.network = network
        self.loss_config = loss_config
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.data_loss_tracker = tf.keras.metrics.Mean(name="data_loss")
        self.physics_loss_tracker = tf.keras.metrics.Mean(name="physics_loss")
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")

    @property
    def metrics(self) -> Sequence[tf.keras.metrics.Metric]:
        return (
            self.total_loss_tracker,
            self.data_loss_tracker,
            self.physics_loss_tracker,
            self.mae_tracker,
        )

    def call(self, inputs: TensorLike, training: bool = False) -> tf.Tensor:
        return self.network(inputs, training=training)

    @staticmethod
    def _physical_target(inputs: TensorLike) -> Optional[tf.Tensor]:
        if isinstance(inputs, Mapping):
            value = inputs.get("physical_total")
            return None if value is None else tf.convert_to_tensor(value)
        return None

    def _loss_terms(
        self,
        inputs: TensorLike,
        targets: tf.Tensor,
        predictions: tf.Tensor,
        sample_weight: Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        targets = tf.cast(targets, predictions.dtype)
        squared_error = tf.square(targets - predictions)
        if sample_weight is not None:
            weight = tf.cast(sample_weight, predictions.dtype)
            while weight.shape.rank is not None and weight.shape.rank < squared_error.shape.rank:
                weight = tf.expand_dims(weight, axis=-1)
            squared_error = squared_error * weight
        data_loss = tf.reduce_mean(squared_error)

        physical_target = self._physical_target(inputs)
        if physical_target is None:
            physics_loss = tf.zeros((), dtype=predictions.dtype)
        else:
            physical_target = tf.cast(physical_target, predictions.dtype)
            physics_loss = tf.reduce_mean(tf.square(predictions - physical_target))

        if predictions.shape.rank is not None and predictions.shape.rank >= 3:
            smoothness = tf.reduce_mean(
                tf.square(predictions[:, 1:, :] - predictions[:, :-1, :])
            )
        else:
            smoothness = tf.zeros((), dtype=predictions.dtype)

        layer_regularization = (
            tf.add_n(self.losses)
            if self.losses
            else tf.zeros((), dtype=predictions.dtype)
        )
        global_l2 = tf.add_n(
            [tf.nn.l2_loss(variable) for variable in self.trainable_variables]
        )
        cfg = self.loss_config
        total_loss = (
            cfg.data_weight * data_loss
            + cfg.physics_weight * physics_loss
            + cfg.smoothness_weight * smoothness
            + layer_regularization
            + cfg.global_l2_weight * global_l2
        )
        return total_loss, data_loss, physics_loss

    def train_step(self, data: Any) -> Dict[str, tf.Tensor]:
        inputs, targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            total_loss, data_loss, physics_loss = self._loss_terms(
                inputs, targets, predictions, sample_weight
            )
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradient_variable_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables)
            if gradient is not None
        ]
        self.optimizer.apply_gradients(gradient_variable_pairs)

        self.total_loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.mae_tracker.update_state(targets, predictions, sample_weight=sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> Dict[str, tf.Tensor]:
        inputs, targets, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        predictions = self(inputs, training=False)
        total_loss, data_loss, physics_loss = self._loss_terms(
            inputs, targets, predictions, sample_weight
        )
        self.total_loss_tracker.update_state(total_loss)
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.mae_tracker.update_state(targets, predictions, sample_weight=sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}


# ---------------------------------------------------------------------------
# Neural baseline builders
# ---------------------------------------------------------------------------


def build_lstm_pm(
    training: Optional[TrainingConfig] = None,
    architecture: Optional[BiLSTMConfig] = None,
) -> tf.keras.Model:
    """Build the data-only LSTM-PM baseline."""

    training = training or LSTM_PM_TRAINING
    architecture = architecture or LSTM_PM_ARCHITECTURE
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    features = _bilstm_backbone(sequence, architecture, "lstm_pm")
    power = tf.keras.layers.Dense(1, name="power_output")(features)
    return tf.keras.Model(sequence, power, name="LSTM_PM")


def build_odp_lstm(
    training: Optional[TrainingConfig] = None,
    architecture: Optional[BiLSTMConfig] = None,
    loss_config: Optional[PhysicsLossConfig] = None,
) -> PhysicsGuidedSequenceModel:
    """Build an output-domain physics-regularized BiLSTM.

    The network predicts power directly. During training,
    :class:`PhysicsGuidedSequenceModel` adds an MSE penalty between the network
    output and normalized physical-model power.
    """

    training = training or ODP_LSTM_TRAINING
    architecture = architecture or ODP_LSTM_ARCHITECTURE
    loss_config = loss_config or ODP_LSTM_LOSS
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    physical_total = tf.keras.Input(
        shape=(window_size, 1), name="physical_total"
    )
    features = _bilstm_backbone(sequence, architecture, "odp_lstm")
    learned_power = tf.keras.layers.Dense(1, name="learned_power")(features)
    zero_physics = tf.keras.layers.Lambda(
        lambda value: tf.zeros_like(value), name="connect_physics_input"
    )(physical_total)
    power = tf.keras.layers.Add(name="power_output")([learned_power, zero_physics])
    network = tf.keras.Model(
        {"sequence": sequence, "physical_total": physical_total},
        power,
        name="ODP_LSTM_network",
    )
    return PhysicsGuidedSequenceModel(network, loss_config, name="ODP_LSTM")


def build_se_pc(
    training: Optional[TrainingConfig] = None,
    architecture: Optional[BiLSTMConfig] = None,
    loss_config: Optional[PhysicsLossConfig] = None,
    correction_head_l2: Optional[float] = None,
) -> PhysicsGuidedSequenceModel:
    """Build a serial error-correction physics-guided BiLSTM.

    The final prediction equals normalized physical power plus a learned
    data-driven correction. ``correction_head_l2`` defaults to the model-specific
    experiment preset and can be overridden for another dataset.
    """

    training = training or SE_PC_TRAINING
    architecture = architecture or SE_PC_ARCHITECTURE
    loss_config = loss_config or SE_PC_LOSS
    if correction_head_l2 is None:
        correction_head_l2 = SE_PC_CORRECTION_HEAD_L2
    correction_head_l2 = _require_nonnegative_float(
        "build_se_pc.correction_head_l2", correction_head_l2
    )
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    physical_total = tf.keras.Input(
        shape=(window_size, 1), name="physical_total"
    )
    features = _bilstm_backbone(sequence, architecture, "se_pc")
    correction = tf.keras.layers.Dense(
        1,
        kernel_regularizer=tf.keras.regularizers.l2(correction_head_l2),
        name="power_correction",
    )(features)
    power = tf.keras.layers.Add(name="power_output")([physical_total, correction])
    network = tf.keras.Model(
        {"sequence": sequence, "physical_total": physical_total},
        power,
        name="SE_PC_network",
    )
    return PhysicsGuidedSequenceModel(network, loss_config, name="SE_PC")


def build_pinn_se(
    training: Optional[TrainingConfig] = None,
    loss_config: Optional[PhysicsLossConfig] = None,
    hidden_units: Optional[Tuple[int, ...]] = None,
    first_hidden_layer_l2: Optional[float] = None,
) -> PhysicsGuidedSequenceModel:
    """Build a PINN-style sequence estimator with four tanh hidden layers.

    The model predicts power directly and is trained with data-fit,
    physics-consistency, temporal-smoothness, and global L2 terms. Hidden-layer
    sizes and first-layer L2 strength must be selected on the validation set.
    """

    training = training or PINN_SE_TRAINING
    loss_config = loss_config or PINN_SE_LOSS
    hidden_units = hidden_units or PINN_SE_HIDDEN_UNITS
    if first_hidden_layer_l2 is None:
        first_hidden_layer_l2 = PINN_SE_FIRST_LAYER_L2
    hidden_units = tuple(_require_value("build_pinn_se.hidden_units", hidden_units))
    if not hidden_units or any(int(v) <= 0 for v in hidden_units):
        raise ValueError("build_pinn_se.hidden_units must contain positive integers.")
    first_hidden_layer_l2 = _require_nonnegative_float(
        "build_pinn_se.first_hidden_layer_l2", first_hidden_layer_l2
    )
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    physical_total = tf.keras.Input(
        shape=(window_size, 1), name="physical_total"
    )
    x = sequence
    for index, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units,
            activation="tanh",
            kernel_regularizer=(
                tf.keras.regularizers.l2(first_hidden_layer_l2) if index == 0 else None
            ),
            name=f"pinn_se_dense_{index + 1}",
        )(x)
    learned_power = tf.keras.layers.Dense(1, name="learned_power")(x)
    zero_physics = tf.keras.layers.Lambda(
        lambda value: tf.zeros_like(value), name="connect_physics_input"
    )(physical_total)
    power = tf.keras.layers.Add(name="power_output")([learned_power, zero_physics])
    network = tf.keras.Model(
        {"sequence": sequence, "physical_total": physical_total},
        power,
        name="PINN_SE_network",
    )
    return PhysicsGuidedSequenceModel(network, loss_config, name="PINN_SE")


def build_physcon_be(
    training: Optional[TrainingConfig] = None,
    architecture: Optional[BiLSTMConfig] = None,
    loss_config: Optional[PhysicsLossConfig] = None,
    maximum_normalized_correction: Optional[float] = None,
) -> PhysicsGuidedSequenceModel:
    """Build a bounded-error physics-constrained BiLSTM.

    ``maximum_normalized_correction`` is expressed in normalized-power units.
    It defaults to the supplied experiment preset and should be re-estimated when
    the target scaler or residual distribution changes.
    """

    training = training or PHYSCON_BE_TRAINING
    architecture = architecture or PHYSCON_BE_ARCHITECTURE
    loss_config = loss_config or PHYSCON_BE_LOSS
    if maximum_normalized_correction is None:
        maximum_normalized_correction = PHYSCON_BE_MAXIMUM_CORRECTION
    maximum_normalized_correction = _require_positive_float(
        "build_physcon_be.maximum_normalized_correction",
        maximum_normalized_correction,
    )
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    physical_total = tf.keras.Input(
        shape=(window_size, 1), name="physical_total"
    )
    features = _bilstm_backbone(sequence, architecture, "physcon_be")
    raw_correction = tf.keras.layers.Dense(1, name="raw_correction")(features)
    bounded_correction = tf.keras.layers.Lambda(
        lambda value: maximum_normalized_correction * tf.math.tanh(value),
        name="bounded_correction",
    )(raw_correction)
    power = tf.keras.layers.Add(name="power_output")(
        [physical_total, bounded_correction]
    )
    network = tf.keras.Model(
        {"sequence": sequence, "physical_total": physical_total},
        power,
        name="PhysCon_BE_network",
    )
    return PhysicsGuidedSequenceModel(network, loss_config, name="PhysCon_BE")


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Learned position embeddings for a fixed maximum sequence length."""

    def __init__(self, maximum_length: int, d_model: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.maximum_length = maximum_length
        self.d_model = d_model

    def build(self, input_shape: tf.TensorShape) -> None:
        self.embedding = self.add_weight(
            name="position_embedding",
            shape=(self.maximum_length, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        length = tf.shape(inputs)[1]
        return inputs + self.embedding[tf.newaxis, :length, :]

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"maximum_length": self.maximum_length, "d_model": self.d_model}
        )
        return config


def build_transformer(
    training: Optional[TrainingConfig] = None,
    architecture: Optional[TransformerConfig] = None,
) -> tf.keras.Model:
    """Build a compact Transformer that predicts every time step."""

    training = training or TRANSFORMER_TRAINING
    architecture = architecture or TRANSFORMER_ARCHITECTURE
    _validate_transformer_config(architecture)
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    x = tf.keras.layers.Dense(architecture.d_model, name="input_projection")(
        sequence
    )
    x = LearnedPositionalEncoding(
        window_size, architecture.d_model, name="positional_encoding"
    )(x)

    for block in range(architecture.num_blocks):
        normalized = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"transformer_norm_attention_{block + 1}"
        )(x)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=architecture.num_heads,
            key_dim=architecture.key_dim,
            dropout=architecture.dropout,
            name=f"transformer_attention_{block + 1}",
        )(normalized, normalized)
        attention = tf.keras.layers.Dropout(
            architecture.dropout, name=f"transformer_attention_dropout_{block + 1}"
        )(attention)
        x = tf.keras.layers.Add(name=f"transformer_attention_add_{block + 1}")(
            [x, attention]
        )

        normalized = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"transformer_norm_ffn_{block + 1}"
        )(x)
        ffn = tf.keras.layers.Dense(
            architecture.ffn_units,
            activation=tf.nn.gelu,
            name=f"transformer_ffn_expand_{block + 1}",
        )(normalized)
        ffn = tf.keras.layers.Dropout(
            architecture.dropout, name=f"transformer_ffn_dropout_{block + 1}"
        )(ffn)
        ffn = tf.keras.layers.Dense(
            architecture.d_model, name=f"transformer_ffn_project_{block + 1}"
        )(ffn)
        x = tf.keras.layers.Add(name=f"transformer_ffn_add_{block + 1}")([x, ffn])

    x = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="transformer_final_norm"
    )(x)
    power = tf.keras.layers.Dense(1, name="power_output")(x)
    return tf.keras.Model(sequence, power, name="Transformer_baseline")


def _tcn_residual_block(
    inputs: tf.Tensor,
    architecture: TCNConfig,
    dilation: int,
    block_index: int,
) -> tf.Tensor:
    x = inputs
    for convolution_index in range(architecture.convolutions_per_block):
        x = tf.keras.layers.Conv1D(
            filters=architecture.filters,
            kernel_size=architecture.kernel_size,
            dilation_rate=dilation,
            padding="causal",
            name=f"tcn_block_{block_index}_conv_{convolution_index + 1}",
        )(x)
        x = tf.keras.layers.LayerNormalization(
            name=f"tcn_block_{block_index}_norm_{convolution_index + 1}"
        )(x)
        x = tf.keras.layers.Activation(
            "relu", name=f"tcn_block_{block_index}_relu_{convolution_index + 1}"
        )(x)
        x = tf.keras.layers.Dropout(
            architecture.dropout,
            name=f"tcn_block_{block_index}_dropout_{convolution_index + 1}",
        )(x)

    residual = inputs
    if inputs.shape[-1] != architecture.filters:
        residual = tf.keras.layers.Conv1D(
            architecture.filters,
            kernel_size=1,
            padding="same",
            name=f"tcn_block_{block_index}_residual_projection",
        )(residual)
    return tf.keras.layers.Add(name=f"tcn_block_{block_index}_add")([x, residual])


def build_tcn(
    training: Optional[TrainingConfig] = None,
    architecture: Optional[TCNConfig] = None,
) -> tf.keras.Model:
    """Build a causal TCN that predicts every time step."""

    training = training or TCN_TRAINING
    architecture = architecture or TCN_ARCHITECTURE
    _validate_tcn_config(architecture)
    window_size, n_features = _validate_training_shape(training)
    sequence = tf.keras.Input(
        shape=(window_size, n_features), name="sequence"
    )
    x = tf.keras.layers.Conv1D(
        architecture.filters,
        kernel_size=1,
        padding="same",
        name="tcn_input_projection",
    )(sequence)
    for block_index, dilation in enumerate(architecture.dilations, start=1):
        x = _tcn_residual_block(x, architecture, dilation, block_index)
    power = tf.keras.layers.Dense(1, name="power_output")(x)
    return tf.keras.Model(sequence, power, name="TCN_baseline")


# ---------------------------------------------------------------------------
# Unified factory and compilation
# ---------------------------------------------------------------------------


NEURAL_MODEL_NAMES: Tuple[str, ...] = (
    "LSTM-PM",
    "ODP-LSTM",
    "SE-PC",
    "PINN-SE",
    "PhysCon-BE",
    "Transformer",
    "TCN",
)

PHYSICAL_MODEL_NAMES: Tuple[str, ...] = ("MR-PM", "MEC-FM")


def build_baseline(
    name: str,
    compile_model: bool = True,
    *,
    training: Optional[TrainingConfig] = None,
    bilstm: Optional[BiLSTMConfig] = None,
    physics_loss: Optional[PhysicsLossConfig] = None,
    transformer: Optional[TransformerConfig] = None,
    tcn: Optional[TCNConfig] = None,
    mrpm_parameters: Optional[MRPMParameters] = None,
    mecfm_config: Optional[MECFMConfig] = None,
    correction_head_l2: Optional[float] = None,
    pinn_hidden_units: Optional[Tuple[int, ...]] = None,
    first_hidden_layer_l2: Optional[float] = None,
    maximum_normalized_correction: Optional[float] = None,
) -> Union[tf.keras.Model, MRPMRegressor, MECFMRegressor]:
    """Build a baseline using its model-specific experiment preset.

    Any explicitly supplied configuration overrides the corresponding preset.
    """

    normalized_name = name.strip().upper().replace("_", "-")

    if normalized_name == "MR-PM":
        return MRPMRegressor(mrpm_parameters or MRPMParameters())
    if normalized_name == "MEC-FM":
        return MECFMRegressor(mecfm_config or MECFMConfig())

    if normalized_name == "LSTM-PM":
        training = training or LSTM_PM_TRAINING
        bilstm = bilstm or LSTM_PM_ARCHITECTURE
        set_reproducible_seed(training.random_seed)
        model = build_lstm_pm(training, bilstm)
    elif normalized_name == "ODP-LSTM":
        training = training or ODP_LSTM_TRAINING
        bilstm = bilstm or ODP_LSTM_ARCHITECTURE
        physics_loss = physics_loss or ODP_LSTM_LOSS
        set_reproducible_seed(training.random_seed)
        model = build_odp_lstm(training, bilstm, physics_loss)
    elif normalized_name == "SE-PC":
        training = training or SE_PC_TRAINING
        bilstm = bilstm or SE_PC_ARCHITECTURE
        physics_loss = physics_loss or SE_PC_LOSS
        set_reproducible_seed(training.random_seed)
        model = build_se_pc(
            training,
            bilstm,
            physics_loss,
            correction_head_l2,
        )
    elif normalized_name == "PINN-SE":
        training = training or PINN_SE_TRAINING
        physics_loss = physics_loss or PINN_SE_LOSS
        set_reproducible_seed(training.random_seed)
        model = build_pinn_se(
            training,
            physics_loss,
            pinn_hidden_units,
            first_hidden_layer_l2,
        )
    elif normalized_name == "PHYSCON-BE":
        training = training or PHYSCON_BE_TRAINING
        bilstm = bilstm or PHYSCON_BE_ARCHITECTURE
        physics_loss = physics_loss or PHYSCON_BE_LOSS
        set_reproducible_seed(training.random_seed)
        model = build_physcon_be(
            training,
            bilstm,
            physics_loss,
            maximum_normalized_correction,
        )
    elif normalized_name == "TRANSFORMER":
        training = training or TRANSFORMER_TRAINING
        transformer = transformer or TRANSFORMER_ARCHITECTURE
        set_reproducible_seed(training.random_seed)
        model = build_transformer(training, transformer)
    elif normalized_name == "TCN":
        training = training or TCN_TRAINING
        tcn = tcn or TCN_ARCHITECTURE
        set_reproducible_seed(training.random_seed)
        model = build_tcn(training, tcn)
    else:
        available = ", ".join(PHYSICAL_MODEL_NAMES + NEURAL_MODEL_NAMES)
        raise ValueError(f"Unknown baseline '{name}'. Available models: {available}")

    if compile_model:
        learning_rate = _require_positive_float(
            "TrainingConfig.learning_rate", training.learning_rate
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if isinstance(model, PhysicsGuidedSequenceModel):
            model.compile(optimizer=optimizer)
        else:
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
            )
    return model


def recommended_hyperparameters() -> Dict[str, Dict[str, Any]]:
    """Return the model-specific numerical presets used by this module."""

    def training_values(config: TrainingConfig) -> Dict[str, Any]:
        return {
            "window_size": config.window_size,
            "n_features": config.n_features,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "maximum_epochs": config.maximum_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "restore_best_weights": True,
            "random_seed": config.random_seed,
        }

    def bilstm_values(config: BiLSTMConfig) -> Dict[str, Any]:
        return {
            "bilstm_units": list(config.units or ()),
            "dense_units": list(config.dense_units or ()),
            "dropout": config.dropout,
            "dense_l2": config.dense_l2,
        }

    def physics_values(config: PhysicsLossConfig) -> Dict[str, Any]:
        return {
            "data_weight": config.data_weight,
            "physics_weight": config.physics_weight,
            "smoothness_weight": config.smoothness_weight,
            "global_l2_weight": config.global_l2_weight,
        }

    mrpm = MRPMParameters()
    return {
        "MR-PM": {
            "optimizer": "bounded nonlinear least squares",
            "initial_parameters": {
                "c1": mrpm.c1,
                "c2": mrpm.c2,
                "c3": mrpm.c3,
                "c4": mrpm.c4,
                "c5": mrpm.c5,
                "c6": mrpm.c6,
                "c7": mrpm.c7,
                "c8": mrpm.c8,
                "c9": mrpm.c9,
                "p_hover": mrpm.p_hover,
            },
            "fitted_parameters": [
                "c2", "c3", "c4", "c5", "c7", "c8", "c9", "p_hover"
            ],
            "fixed_during_total_power_fit": ["c1", "c6"],
            "max_nfev": 30000,
        },
        "MEC-FM": {
            "optimizer": "closed-form standardized ridge regression",
            "ridge_alpha": MECFMConfig().ridge_alpha,
            "feature_count": 9,
            "features": list(MECFMRegressor.feature_names),
            "standardize_features": True,
            "penalize_intercept": False,
        },
        "LSTM-PM": {
            **training_values(LSTM_PM_TRAINING),
            **bilstm_values(LSTM_PM_ARCHITECTURE),
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "physics_weight": 0.0,
        },
        "ODP-LSTM": {
            **training_values(ODP_LSTM_TRAINING),
            **bilstm_values(ODP_LSTM_ARCHITECTURE),
            **physics_values(ODP_LSTM_LOSS),
            "optimizer": "Adam",
            "output_mode": "direct power prediction",
        },
        "SE-PC": {
            **training_values(SE_PC_TRAINING),
            **bilstm_values(SE_PC_ARCHITECTURE),
            **physics_values(SE_PC_LOSS),
            "optimizer": "Adam",
            "output_mode": "physical baseline plus learned correction",
            "correction_head_l2": SE_PC_CORRECTION_HEAD_L2,
        },
        "PINN-SE": {
            **training_values(PINN_SE_TRAINING),
            **physics_values(PINN_SE_LOSS),
            "optimizer": "Adam",
            "hidden_units": list(PINN_SE_HIDDEN_UNITS),
            "activation": "tanh",
            "first_hidden_layer_l2": PINN_SE_FIRST_LAYER_L2,
            "output_mode": "direct power prediction",
        },
        "PhysCon-BE": {
            **training_values(PHYSCON_BE_TRAINING),
            **bilstm_values(PHYSCON_BE_ARCHITECTURE),
            **physics_values(PHYSCON_BE_LOSS),
            "optimizer": "Adam",
            "output_mode": "physical baseline plus bounded correction",
            "maximum_normalized_correction": PHYSCON_BE_MAXIMUM_CORRECTION,
            "correction_activation": "tanh",
        },
        "Transformer": {
            **training_values(TRANSFORMER_TRAINING),
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "d_model": TRANSFORMER_ARCHITECTURE.d_model,
            "num_heads": TRANSFORMER_ARCHITECTURE.num_heads,
            "key_dim": TRANSFORMER_ARCHITECTURE.key_dim,
            "num_blocks": TRANSFORMER_ARCHITECTURE.num_blocks,
            "ffn_units": TRANSFORMER_ARCHITECTURE.ffn_units,
            "dropout": TRANSFORMER_ARCHITECTURE.dropout,
            "layer_norm_epsilon": 1e-6,
            "positional_encoding": "learned",
        },
        "TCN": {
            **training_values(TCN_TRAINING),
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "filters": TCN_ARCHITECTURE.filters,
            "kernel_size": TCN_ARCHITECTURE.kernel_size,
            "dilations": list(TCN_ARCHITECTURE.dilations or ()),
            "convolutions_per_block": TCN_ARCHITECTURE.convolutions_per_block,
            "padding": "causal",
            "dropout": TCN_ARCHITECTURE.dropout,
        },
    }


__all__ = [
    "TrainingConfig",
    "BiLSTMConfig",
    "PhysicsLossConfig",
    "TransformerConfig",
    "TCNConfig",
    "LSTM_PM_TRAINING",
    "LSTM_PM_ARCHITECTURE",
    "ODP_LSTM_TRAINING",
    "ODP_LSTM_ARCHITECTURE",
    "ODP_LSTM_LOSS",
    "SE_PC_TRAINING",
    "SE_PC_ARCHITECTURE",
    "SE_PC_LOSS",
    "SE_PC_CORRECTION_HEAD_L2",
    "PINN_SE_TRAINING",
    "PINN_SE_LOSS",
    "PINN_SE_HIDDEN_UNITS",
    "PINN_SE_FIRST_LAYER_L2",
    "PHYSCON_BE_TRAINING",
    "PHYSCON_BE_ARCHITECTURE",
    "PHYSCON_BE_LOSS",
    "PHYSCON_BE_MAXIMUM_CORRECTION",
    "TRANSFORMER_TRAINING",
    "TRANSFORMER_ARCHITECTURE",
    "TCN_TRAINING",
    "TCN_ARCHITECTURE",
    "MRPMParameters",
    "MRPMRegressor",
    "MECFMConfig",
    "MECFMRegressor",
    "PhysicsGuidedSequenceModel",
    "build_lstm_pm",
    "build_odp_lstm",
    "build_se_pc",
    "build_pinn_se",
    "build_physcon_be",
    "build_transformer",
    "build_tcn",
    "build_baseline",
    "make_early_stopping",
    "recommended_hyperparameters",
    "set_reproducible_seed",
]
