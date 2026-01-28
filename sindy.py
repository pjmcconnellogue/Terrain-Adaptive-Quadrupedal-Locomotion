"""
Robot Dynamics Model with SINDy Corrections
Handles state prediction, dynamics calculations, and SINDy model training
"""

import numpy as np
import casadi as ca
import pysindy as ps
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import re


# ============================================================================
# STATE PREDICTION
# ============================================================================

def predict_next_state_from_grf_and_state(GRFs, measured_state, dt):
    """
    Predict next robot state given GRFs and current state

    Args:
        GRFs: Ground reaction forces [12] - 4 feet x 3 components
        measured_state: Current state [30] - [pos(3), vel(3), R(9), omega(3), foot_pos(12)]
        dt: Time step

    Returns:
        next_state: Predicted state [18] - [pos(3), vel(3), R(9), omega(3)]
        state_dot: State derivatives [18]
    """
    # Unpack forces (4 feet, each 3D)
    f1 = ca.vertcat(GRFs[0], GRFs[1], GRFs[2])
    f2 = ca.vertcat(GRFs[3], GRFs[4], GRFs[5])
    f3 = ca.vertcat(GRFs[6], GRFs[7], GRFs[8])
    f4 = ca.vertcat(GRFs[9], GRFs[10], GRFs[11])

    # Extract base states
    p = ca.vertcat(*measured_state[0:3])      # Position
    v = ca.vertcat(*measured_state[3:6])      # Velocity
    R_flat = ca.vertcat(*measured_state[6:15]) # Rotation matrix (flattened)
    omega = ca.vertcat(*measured_state[15:18]) # Angular velocity

    # Extract foot positions
    pf1 = ca.vertcat(*measured_state[18:21])
    pf2 = ca.vertcat(*measured_state[21:24])
    pf3 = ca.vertcat(*measured_state[24:27])
    pf4 = ca.vertcat(*measured_state[27:30])

    # Predict next state using dynamics + Euler integration
    next_state, state_dot = predict_next_state(
        f1, f2, f3, f4, pf1, pf2, pf3, pf4,
        p, v, R_flat, omega, dt
    )

    return next_state, state_dot


def predict_next_state(f1, f2, f3, f4, pf1, pf2, pf3, pf4, p, v, R, omega, dt):
    """
    Predict next state using Forward Euler integration

    Args:
        f1-f4: Forces on each foot [3]
        pf1-pf4: Foot positions relative to CoM [3]
        p, v, R, omega: Current state
        dt: Time step

    Returns:
        next_state: Integrated state [18]
        state_dot: State derivatives [18]
    """
    # Compute state derivatives
    state_dot = dynamics(f1, f2, f3, f4, pf1, pf2, pf3, pf4, p, v, R, omega)

    # Forward Euler integration
    p_next = p + dt * state_dot[0:3]
    v_next = v + dt * state_dot[3:6]
    R_next_flat = R + dt * state_dot[6:15]
    omega_next = omega + dt * state_dot[15:18]

    # Combine into next state
    next_state = ca.vertcat(p_next, v_next, R_next_flat, omega_next)

    return next_state, state_dot


# ============================================================================
# DYNAMICS MODEL
# ============================================================================

def dynamics(f1, f2, f3, f4, pf1, pf2, pf3, pf4, p, v, R, omega):
    """
    Robot dynamics with SINDy corrections

    Computes state derivatives: [p_dot, v_dot, R_dot, omega_dot]

    Physics model:
    - Rigid body dynamics
    - Newton-Euler equations
    - SINDy-corrected angular acceleration

    Args:
        f1-f4: Forces on each foot [3]
        pf1-pf4: Foot positions (moment arms) [3]
        p, v: Position and velocity [3]
        R: Rotation matrix (flattened) [9]
        omega: Angular velocity [3]

    Returns:
        rhs: State derivatives [18] - [p_dot(3), v_dot(3), R_dot(9), omega_dot(3)]
    """
    # Robot parameters
    G = np.array([0, 0, 9.81])  # Gravity
    m = 9.63  # Mass (kg)
    I = np.array([  # Inertia tensor (kg·m²)
        [0.012, 0.0, 0.000],
        [0.0, 0.021, 0.0],
        [0.000, 0.0, 0.028]
    ])

    # Total force and torque
    r1, r2, r3, r4 = pf1, pf2, pf3, pf4

    F_total = f1 + f2 + f3 + f4
    tau_total = (ca.cross(r1, f1) + ca.cross(r2, f2) +
                 ca.cross(r3, f3) + ca.cross(r4, f4))

    # Reshape rotation matrix
    RR = ca.reshape(R, 3, 3).T

    # Linear dynamics
    p_dot = v
    v_dot = (1 / m) * F_total - G

    # Rotational dynamics
    R_dot = ca.mtimes(RR, ca.skew(omega))
    omega_dot = ca.mtimes(
        ca.inv(I),
        ca.mtimes(RR, tau_total) - ca.cross(omega, ca.mtimes(I, omega))
    )

    # Currently active SINDy corrections for angular acceleration
    omega_dot[2] = omega_dot[2] - 31.8 * tau_total[2] - 0.68 * tau_total[1] * omega[0]
    omega_dot[1] = omega_dot[1] - 42.5 * tau_total[1]
    omega_dot[0] = omega_dot[0] - 51.2 * tau_total[0] + 3.5 * tau_total[1] + 19.4 * tau_total[0] * omega[1]

    # Reshape R_dot to match state vector format
    R_dot_flat = ca.reshape(R_dot.T, 9, 1)

    # Concatenate state derivatives
    rhs = ca.vertcat(p_dot, v_dot, R_dot_flat, omega_dot)

    return rhs

# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_sindy_features(states, grfs):
    """
    Prepare feature matrix for SINDy training

    Features: Forces (3) + Torques (3) + Omega (3) = 9 total

    Args:
        states: Robot states [N, 30]
        grfs: Ground reaction forces [N, 12]

    Returns:
        X: Feature matrix [N, 9]
        feature_names: List of feature names
    """
    num_samples = grfs.shape[0]
    foot_positions = states[:, 18:]

    # Reshape into per-foot arrays
    grfs_per_foot = grfs.reshape(num_samples, 4, 3)
    fps_per_foot = foot_positions.reshape(num_samples, 4, 3)

    # Calculate features
    F_total = np.sum(grfs_per_foot, axis=1)
    torque_per_foot = np.cross(fps_per_foot, grfs_per_foot, axisc=2)
    tau_total = np.sum(torque_per_foot, axis=1)
    omega_features = states[:, 15:18]

    # Combine features
    X = np.hstack([F_total, tau_total, omega_features])

    feature_names = [
        'F_total_x', 'F_total_y', 'F_total_z',
        'tau_total_x', 'tau_total_y', 'tau_total_z',
        'omega_x', 'omega_y', 'omega_z'
    ]

    return X, feature_names


def calculate_prediction_errors(states, grfs, state_index, dt):
    """
    Calculate prediction errors between actual and model-predicted state derivatives

    Args:
        states: Robot states [N, 30]
        grfs: Ground reaction forces [N, 12]
        state_index: Which state derivative to compute error for
        dt: Time step

    Returns:
        errors: Prediction errors [N-1]
    """
    num_samples = len(states)
    errors = []

    for i in range(min(num_samples - 1, 4000)):
        # Actual change
        actual_change = (states[i + 1, state_index] - states[i, state_index]) / dt

        # Predicted change
        try:
            _, expected_state_dot = predict_next_state_from_grf_and_state(
                grfs[i], states[i], dt
            )
            if hasattr(expected_state_dot, 'full'):
                expected_change = np.array(expected_state_dot.full()).flatten()[state_index]
            else:
                expected_change = np.array(expected_state_dot).flatten()[state_index]
        except Exception as e:
            print(f"Error at step {i}: {e}")
            expected_change = actual_change

        error = actual_change - expected_change
        errors.append(error)

    return np.array(errors)


# ============================================================================
# SINDY MODEL TRAINING
# ============================================================================

def train_sindy_error_with_omega(differences, grfs, states, degree=2, threshold=0.025):
    """
    Train SINDy model for angular acceleration errors
    Uses Forces + Torques + Omega features

    Feature set (9 total):
    - F_total: [x, y, z] - Total force on robot
    - tau_total: [x, y, z] - Total torque on robot
    - omega: [x, y, z] - Angular velocity

    Args:
        differences: State derivative errors [N, 18]
        grfs: Ground reaction forces [N, 12]
        states: Robot states [N, 30]
        degree: Polynomial degree for feature library
        threshold: Sparsity threshold for STLSQ

    Returns:
        model: Trained SINDy model
        x_scaler: Feature scaler
        y_scaler: Target scaler
    """
    print("\n" + "=" * 60)
    print("SINDy Training: Angular Acceleration Error Model")
    print("=" * 60)

    num_samples = grfs.shape[0]
    foot_positions_data = states[:, 18:]

    # Validate input shapes
    if grfs.shape[1] != 12:
        raise ValueError(f"grfs columns error: expected 12, got {grfs.shape[1]}")
    if foot_positions_data.shape[1] != 12:
        raise ValueError(f"foot_positions_data columns error")

    # Prepare features
    X_raw, feature_names = prepare_sindy_features(states, grfs)

    print(f"\nFeature Summary:")
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {num_samples}")
    print(f"  Feature names: {feature_names}")

    # Target: d_omega_z_err (index 17)
    y_original = differences[:, 17].reshape(-1, 1)

    # Scale features and target
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_raw)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_original)

    print(f"\nTarget statistics:")
    print(f"  Mean: {np.mean(y_original):.6f}")
    print(f"  Std: {np.std(y_original):.6f}")

    # Train SINDy model
    print(f"\nModel Configuration:")
    print(f"  Polynomial degree: {degree}")
    print(f"  Sparsity threshold: {threshold}")

    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=degree, include_bias=True),
        optimizer=ps.STLSQ(threshold=threshold)
    )

    model.fit(X_scaled, x_dot=y_scaled, t=1, feature_names=feature_names)

    # Display discovered equations
    print(f"\nDiscovered Model:")
    print("-" * 60)
    model.print()

    # Evaluate performance
    y_pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = y_original.flatten()

    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    if r2 > 0.3:
        print("  Status: Good correlation - model captures meaningful patterns")
    elif r2 > 0.1:
        print("  Status: Moderate correlation - some useful patterns detected")
    else:
        print("  Status: Low correlation - model may not be useful")

    # Generate correction equations
    print(f"\nCorrection Equations:")
    print("-" * 60)

    correction_str = generate_correction_equation(model, feature_names, x_scaler, y_scaler)

    if correction_str:
        print("\nFor training dynamics:")
        print(f"  omega_dot[2] = omega_dot[2] + ({correction_str})")

        print("\nFor MPC dynamics:")
        mpc_correction = correction_str.replace("F_total", "F_total_sym").replace("tau_total", "tau_total_sym")
        print(f"  omega_dot[2] = omega_dot[2] - ({mpc_correction})")
    else:
        print("No significant correction terms found")

    print("=" * 60 + "\n")

    return model, x_scaler, y_scaler


def train_sindy_for_state(measured_state_history, measured_grf_history, robotId, p,
                          state_index, degree=2, threshold=0.025):
    """
    Train SINDy model for any state derivative

    Args:
        measured_state_history: Dictionary of state histories by robot
        measured_grf_history: Dictionary of GRF histories by robot
        robotId: Robot ID to analyze
        p: Parameter dictionary with 'Tmpc' and 'steps'
        state_index: Which state to train for (15, 16, or 17 for omega)
        degree: Polynomial degree
        threshold: Sparsity threshold

    Returns:
        Dictionary with model, scalers, metrics, and predictions
    """
    print(f"\nSINDy Analysis - State {state_index} (Degree {degree})")
    print("=" * 60)

    states = np.array(measured_state_history[robotId])
    num_samples = len(states)

    # Prepare GRF data
    grfs = []
    for i in range(num_samples):
        combined_grf = []
        grf_index = i // 4
        for leg in range(1, 5):
            if grf_index < len(measured_grf_history[robotId][leg]):
                combined_grf.extend(measured_grf_history[robotId][leg][grf_index])
        grfs.append(combined_grf)

    grfs = np.array(grfs)

    # Calculate prediction errors
    dt = p['Tmpc'] / p['steps']
    errors = calculate_prediction_errors(states, grfs, state_index, dt)

    print(f"Computed {len(errors)} error samples")
    print(f"Error statistics: mean={np.mean(errors):.6f}, std={np.std(errors):.6f}")

    # Prepare features
    X, feature_names = prepare_sindy_features(states[:len(errors)], grfs[:len(errors)])
    y = errors.reshape(-1, 1)

    # Scale data
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # Train model
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=degree, include_bias=True),
        optimizer=ps.STLSQ(threshold=threshold)
    )

    model.fit(X_scaled, x_dot=y_scaled, t=1, feature_names=feature_names)

    print(f"\nDiscovered model:")
    print("-" * 60)
    model.print()

    # Evaluate
    y_pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = y.flatten()

    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    print(f"\nPerformance:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    # Generate correction
    print(f"\nCorrection equation:")
    print("-" * 60)

    correction_str = generate_correction_equation(model, feature_names, x_scaler, y_scaler)

    if correction_str:
        print("For training dynamics:")
        print(f"  omega_dot[{state_index - 15}] = omega_dot[{state_index - 15}] + ({correction_str})")

        print("\nFor MPC dynamics:")
        mpc_correction = correction_str.replace("F_total", "F_total_sym").replace("tau_total", "tau_total_sym")
        print(f"  omega_dot[{state_index - 15}] = omega_dot[{state_index - 15}] - ({mpc_correction})")
    else:
        print("No significant correction terms found")

    return {
        'r2': r2,
        'rmse': rmse,
        'model': model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'y_actual': y_actual,
        'y_pred': y_pred,
        'state_index': state_index,
        'degree': degree,
        'dt': dt
    }


# ============================================================================
# CORRECTION EQUATION GENERATION
# ============================================================================

def generate_correction_equation(model, feature_names, x_scaler, y_scaler):
    """
    Generate correction equation string from trained SINDy model

    Args:
        model: Trained SINDy model
        feature_names: List of feature names
        x_scaler: Fitted StandardScaler for features
        y_scaler: Fitted StandardScaler for target

    Returns:
        correction_str: String representing correction equation
    """
    scaled_coeffs = model.coefficients().flatten()
    library_terms = model.feature_library.get_feature_names(feature_names)

    correction_terms = []
    for coeff, term in zip(scaled_coeffs, library_terms):
        if abs(coeff) > 1e-10:
            unscaled_coeff = _unscale_coefficient(coeff, term, feature_names, x_scaler, y_scaler)

            if abs(unscaled_coeff) > 1e-6:
                if term == '1':
                    correction_terms.append(f"{unscaled_coeff:.6f}")
                else:
                    python_term = _convert_to_python_syntax(term)
                    correction_terms.append(f"{unscaled_coeff:.6f}*{python_term}")

    if correction_terms:
        correction_str = " + ".join(correction_terms).replace("+ -", "- ")
        return correction_str
    else:
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _convert_to_python_syntax(term):
    """Convert mathematical notation to Python syntax"""
    python_term = term.replace('^', '**')
    python_term = re.sub(r'(\*\*\d+)\s+([a-zA-Z_])', r'\1 * \2', python_term)
    python_term = re.sub(r'(\])\s+([a-zA-Z_])', r'\1 * \2', python_term)
    python_term = re.sub(r'(\d+(?:\.\d+)?)\s+([a-zA-Z_])', r'\1 * \2', python_term)
    python_term = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'\1 * \2', python_term)
    python_term = re.sub(r'\s+', ' ', python_term)
    return python_term


def _unscale_coefficient(coeff, term, feature_names, x_scaler, y_scaler):
    """
    Unscale coefficient for physical interpretation
    Handles constant, linear, quadratic, and cross terms
    """
    # Constant term
    if term == '1':
        return coeff * y_scaler.scale_[0] + y_scaler.mean_[0]

    # Linear terms
    if term in feature_names:
        var_idx = feature_names.index(term)
        return coeff * y_scaler.scale_[0] / x_scaler.scale_[var_idx]

    # Quadratic terms
    if '**2' in term or '^2' in term:
        base_var = term.replace('**2', '').replace('^2', '')
        if base_var in feature_names:
            var_idx = feature_names.index(base_var)
            return coeff * y_scaler.scale_[0] / (x_scaler.scale_[var_idx] ** 2)

    # Cross terms
    elif ' ' in term:
        vars_in_term = term.split(' ')
        if len(vars_in_term) == 2 and all(var in feature_names for var in vars_in_term):
            var1_idx = feature_names.index(vars_in_term[0])
            var2_idx = feature_names.index(vars_in_term[1])
            return coeff * y_scaler.scale_[0] / (x_scaler.scale_[var1_idx] * x_scaler.scale_[var2_idx])

    # General case - count variable occurrences
    total_scaling = 1.0
    for var in feature_names:
        count = term.count(var)
        # Check for power notation
        power_matches = re.findall(rf'{re.escape(var)}\*\*(\d+)', term)
        for match in power_matches:
            count += int(match) - 1
        if count > 0:
            var_idx = feature_names.index(var)
            total_scaling *= x_scaler.scale_[var_idx] ** count

    return coeff * y_scaler.scale_[0] / total_scaling