"""
Governing Equations and Physics Loss Module for Standard PINN.

This module defines the aerodynamic laws, the sparse regression basis functions
(with analytical derivatives), and the specialized loss function that enforces
physical consistency on both values and gradients (Sobolev norms).

Copyright (c) 2026 UAV-Research-Group. All rights reserved.
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class PhysicsCoefficients:
    """Aerodynamic coefficients and optimized sparse regression weights."""
    C1: float = 537.92430435
    C2: float = -11.81444764
    C3: float = -32.51778232
    C4: float = 1851.19680972
    C5: float = 2.00966979
    C6: float = 2.77965346e+02
    C7: float = 3.40071433e+01
    C8: float = 2.08030427e-01
    C9: float = 4.00000000e+00
    P_HOVER: float = 337.09

    # SINDy Optimized Coefficients (23 terms)
    COEFF_OPT: np.ndarray = np.array([
        16.2813, -7.1361, -33.7687, 0.0, 16.0607, -0.0, 0.0,
        -0.2804, -0.3995, -0.0025, 178.1034, 99.2792, -1.0379, -284.5549,
        -188.0714, -6.2976, 84.8423, -0.0, 16.1915, -2.3171, 1.0918,
        0.0, -14.524
    ], dtype=float)

class AnalyticalDerivatives:
    """
    Computes the basis functions matrix and their analytical partial derivatives
    w.r.t V_h and V_v. Used for the 'Derivative Constraint' in the loss function.
    """
    
    @staticmethod
    def compute_basis_and_gradients(V_h: tf.Tensor, V_v: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Constructs the feature matrix and its gradients.
        """
        # Pre-compute powers for efficiency
        Vh_sq, Vv_sq = V_h**2, V_v**2
        Vh_cub, Vv_cub = V_h**3, V_v**3
        Vh_quart, Vv_quart = V_h**4, V_v**4
        Vh_quint, Vv_quint = V_h**5, V_v**5
        
        sin_Vh = tf.sin(V_h)
        sin_Vv = tf.sin(V_v)

        # 1. Basis Function Matrix [Batch, Time, Features]
        features = tf.stack([
            V_h, V_v, Vh_sq, Vv_sq, Vh_cub, Vv_cub, Vh_quart, Vv_quart, Vh_quint, Vv_quint,
            V_h * V_v, Vh_cub * V_v, V_h * Vv_cub, Vh_sq * V_v, Vv_sq * V_h,
            Vh_cub * Vv_sq, Vh_sq * Vv_sq, Vh_quart * V_v, V_h * Vv_quart,
            Vh_quint * V_v, V_h * Vv_quint, sin_Vh, sin_Vv,
        ], axis=-1)

        # 2. Analytical Gradients w.r.t V_h
        dVh = tf.stack([
            tf.ones_like(V_h), tf.zeros_like(V_h), 
            2 * V_h, tf.zeros_like(V_h), 
            3 * Vh_sq, tf.zeros_like(V_h), 
            4 * Vh_cub, tf.zeros_like(V_h), 
            5 * Vh_quart, tf.zeros_like(V_h),
            V_v, 3 * Vh_sq * V_v, Vv_cub, 2 * V_h * V_v, Vv_sq,
            3 * Vh_sq * Vv_sq, 2 * V_h * Vv_sq, 4 * Vh_cub * V_v, Vv_quart,
            5 * Vh_quart * V_v, Vv_quint, tf.cos(V_h), tf.zeros_like(V_h),
        ], axis=-1)

        # 3. Analytical Gradients w.r.t V_v
        dVv = tf.stack([
            tf.zeros_like(V_v), tf.ones_like(V_v), 
            tf.zeros_like(V_v), 2 * V_v, 
            tf.zeros_like(V_v), 3 * Vv_sq, 
            tf.zeros_like(V_v), 4 * Vv_cub, 
            tf.zeros_like(V_v), 5 * Vv_quart,
            V_h, Vh_cub, 3 * V_v**2 * V_h, Vh_sq, 2 * V_v * V_h,
            2 * Vh_cub * V_v, 2 * Vh_sq * V_v, Vh_quart, 4 * V_h * Vv_cub,
            Vh_quint, 5 * V_h * Vv_quart, tf.zeros_like(V_v), tf.cos(V_v),
        ], axis=-1)

        return features, dVh, dVv


class PINNLossCalculator:
    """
    Advanced loss calculator that computes the Sobolev norm (Value + Derivative).
    Uses Auto-Differentiation (Jacobian) for the NN and Analytical Derivatives for Physics.
    """
    
    def __init__(self, 
                 weights: Dict[str, float], 
                 scalers: Tuple[float, float, float]):
        """
        Args:
            weights: Hyperparameters for loss components (alpha_weights).
            scalers: (Vh_range, Vv_range, P_range) for denormalization inside gradient calculation.
        """
        self.w = weights # alpha_weights list
        self.vh_range, self.vv_range, self.p_range = scalers
        self.consts = PhysicsCoefficients()
        
    def _compute_physics_values(self, v_h, v_v):
        """Computes raw physical power values (horizontal, vertical, etc.)."""
        c = self.consts
        # Horizontal
        inner_h = tf.maximum(1 + (v_h**4)/c.C4 - (v_h**2)/c.C5, 1e-6)
        p_h = c.C1 + c.C2 * v_h**2 + c.C3 * tf.sqrt(inner_h) + c.C5 * v_h**3 - c.C1
        
        # Vertical
        term_up = tf.sqrt(tf.maximum((1 + 4*c.C8/c.C9)*v_v**2 + 4*c.C7/c.C9, 1e-6))
        val_up = c.C6 + c.C7*v_v + c.C8*v_v**3 + (c.C7 + c.C8*v_v**2) * term_up
        term_down = tf.sqrt(tf.maximum((1 - 4*c.C8/c.C9)*v_v**2 + 4*c.C7/c.C9, 1e-6))
        val_down = c.C6 + c.C7*v_v - c.C8*v_v**3 + (c.C7 - c.C8*v_v**2) * term_down
        p_v = tf.where(v_v > 0, val_up, val_down) - c.C6
        
        # Hover
        p_hov = tf.fill(tf.shape(v_h), c.P_HOVER)
        
        return p_h, p_v, p_hov

    @tf.function(experimental_relax_shapes=True)
    def compute_loss(self, model, x_seq, p_actual_norm, p_phy_norm, 
                     alpha_d_seq, alpha_p_seq, 
                     lambda_data, lambda_phy, lambda_smooth, lambda_reg,
                     w_deriv, w_num):
        """
        Computes the total PINN loss including Jacobian matching.
        """
        with tf.GradientTape() as tape:
            tape.watch(x_seq)
            # Forward pass: Returns summed output AND individual components
            p_pred_norm, phys_outputs = model(x_seq, training=True)
            
            # phys_outputs shape: [Batch, Time, 4] -> [Pv, Ph, Phov, Padd]

        # --- 1. Jacobian Calculation (Model Gradients) ---
        # Flatten for batch_jacobian: [B, T*4]
        b_dim, t_dim, c_dim = phys_outputs.shape
        phys_flat = tf.reshape(phys_outputs, [b_dim, t_dim * c_dim])
        
        # Compute Jacobian: d(Output)/d(Input) -> [B, T*4, T, 2]
        jac = tape.batch_jacobian(phys_flat, x_seq)
        # Extract diagonal elements (time-aligned gradients) -> [B, T*4, 2]
        diag_jac = tf.linalg.diag_part(jac)
        # Reshape to [B, T, 4, 2] -> [Batch, Time, Component, Input_Dim(Vh, Vv)]
        dp_model_tensor = tf.reshape(diag_jac, [b_dim, t_dim, c_dim, 2])

        # --- 2. Physics Derivatives (Analytical) ---
        # Denormalize inputs for physics calc
        v_h_norm = x_seq[:, :, 0]
        v_v_norm = x_seq[:, :, 1]
        # Assuming scaler min is 0 for simplicity or pre-handled. 
        # Ideally pass scaler.min_, but using range for gradient scaling is correct.
        v_h = v_h_norm * self.vh_range 
        v_v = v_v_norm * self.vv_range

        # Compute Basis and Gradients
        feats, d_basis_dvh, d_basis_dvv = AnalyticalDerivatives.compute_basis_and_gradients(v_h, v_v)
        
        # Compute Residual Component Derivatives (P_add)
        coeff_tensor = tf.cast(self.consts.COEFF_OPT, dtype=tf.float32)
        dp_add_dvh = tf.einsum('btk,k->bt', d_basis_dvh, coeff_tensor)
        dp_add_dvv = tf.einsum('btk,k->bt', d_basis_dvv, coeff_tensor)
        
        # Compute Analytical derivatives for Pv, Ph (Using GradientTape inside helper or analytical formula)
        # For brevity, using a secondary tape or assuming functions return grads.
        # Here we re-use the tape trick for P_h, P_v physics equations locally if analytical form is complex
        with tf.GradientTape(persistent=True) as pt:
            pt.watch([v_h, v_v])
            ph_val, pv_val, _ = self._compute_physics_values(v_h, v_v)
        
        dph_dvh = pt.gradient(ph_val, v_h)
        dpv_dvv = pt.gradient(pv_val, v_v)
        del pt 

        # --- 3. Scale Derivatives to Normalized Space ---
        # d(P_norm)/d(V_norm) = dP/dV * (Range_V / Range_P)
        scale_factor_h = self.vh_range / (self.p_range + 1e-12)
        scale_factor_v = self.vv_range / (self.p_range + 1e-12)
        
        dph_dvh_norm = dph_dvh * scale_factor_h
        dpv_dvv_norm = dpv_dvv * scale_factor_v
        dpadd_dvh_norm = dp_add_dvh * scale_factor_h
        dpadd_dvv_norm = dp_add_dvv * scale_factor_v

        # --- 4. Loss Calculation ---
        
        # A. Data Loss
        loss_data = tf.reduce_mean((p_pred_norm - p_actual_norm) ** 2, axis=1)
        
        # B. Physics Value Loss (Soft Constraint on total power)
        # Note: Or constraint on individual components if ground truth available. 
        # Following original script logic: L2 distance to P_phy
        loss_phy_val = tf.reduce_mean((p_pred_norm - p_phy_norm) ** 2, axis=1)

        # C. Physics Derivative Loss (Sobolev Constraint)
        # Matching Model Jacobian [Component, Input] to Analytical Gradients
        # Components: 0:Pv, 1:Ph, 2:Phov, 3:Padd
        # Inputs: 0:Vh, 1:Vv
        
        term_pv = self.w[0] * (tf.reduce_mean((dp_model_tensor[:,:,0,0] - 0.0)**2, 1) +  # dPv/dVh = 0
                               tf.reduce_mean((dp_model_tensor[:,:,0,1] - dpv_dvv_norm)**2, 1))
        
        term_ph = self.w[1] * (tf.reduce_mean((dp_model_tensor[:,:,1,0] - dph_dvh_norm)**2, 1) + 
                               tf.reduce_mean((dp_model_tensor[:,:,1,1] - 0.0)**2, 1))   # dPh/dVv = 0
                               
        term_phov = self.w[2] * (tf.reduce_mean((dp_model_tensor[:,:,2,0] - 0.0)**2, 1) + # Constant
                                 tf.reduce_mean((dp_model_tensor[:,:,2,1] - 0.0)**2, 1))
                                 
        term_padd = self.w[3] * (tf.reduce_mean((dp_model_tensor[:,:,3,0] - dpadd_dvh_norm)**2, 1) + 
                                 tf.reduce_mean((dp_model_tensor[:,:,3,1] - dpadd_dvv_norm)**2, 1))

        loss_derivs = term_pv + term_ph + term_phov + term_padd

        # Combine Physics Losses
        loss_phys_total = w_deriv * loss_derivs + w_num * loss_phy_val

        # D. Smoothness & Reg
        dp_dt = p_pred_norm[:, 1:, :] - p_pred_norm[:, :-1, :]
        l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

        # E. Weighted Aggregation
        # Handling reliability weights (Alpha)
        alpha_d = tf.reduce_mean(alpha_d_seq, axis=1, keepdims=True) - 0.1
        alpha_p = tf.reduce_mean(alpha_p_seq, axis=1, keepdims=True) + 0.1
        
        if len(loss_phys_total.shape) == 1:
            loss_phys_total = tf.expand_dims(loss_phys_total, axis=-1)

        per_sample_loss = (alpha_d * lambda_data * loss_data + 
                           alpha_p * lambda_phy * loss_phys_total)

        final_loss = (tf.reduce_mean(per_sample_loss) + 
                      lambda_smooth * tf.reduce_mean(dp_dt ** 2) + 
                      lambda_reg * l2_reg)

        return final_loss, loss_data, loss_phys_total
