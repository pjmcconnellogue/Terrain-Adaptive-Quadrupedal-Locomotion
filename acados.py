"""
Acados MPC Setup and Solver Interface
Handles model creation, OCP formulation, and constraint management
"""

import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import numpy as np
import time
import os


# ============================================================================
# CASADI SYMBOLIC SETUP
# ============================================================================

def setup_casadi_states():
    """Define state variables: position, velocity, rotation matrix, angular velocity"""
    p = ca.SX.sym('p', 3, 1)
    v = ca.SX.sym('v', 3, 1)
    R = ca.SX.sym('R', 9, 1)
    omega = ca.SX.sym('omega', 3, 1)
    states = ca.vertcat(p, v, R, omega)
    n_states = states.size()[0]
    return states, n_states, p, v, R, omega


def setup_casadi_controls():
    """Define control inputs: forces on each foot"""
    f1 = ca.SX.sym('f1', 3, 1)
    f2 = ca.SX.sym('f2', 3, 1)
    f3 = ca.SX.sym('f3', 3, 1)
    f4 = ca.SX.sym('f4', 3, 1)
    controls = ca.vertcat(f1, f2, f3, f4)
    n_controls = controls.size()[0]
    return controls, n_controls, f1, f2, f3, f4


def setup_casadi_params():
    """Define parameters: gravity, mass, inertia, foot positions"""
    G = ca.SX.sym('g', 3, 1)
    m = ca.SX.sym('m')
    I = ca.SX.sym('I', 3, 3)

    # Foot positions relative to center of mass
    pf1 = ca.SX.sym('pf1', 3, 1)
    pf2 = ca.SX.sym('pf2', 3, 1)
    pf3 = ca.SX.sym('pf3', 3, 1)
    pf4 = ca.SX.sym('pf4', 3, 1)

    return G, m, I, pf1, pf2, pf3, pf4


def params():
    """Robot physical parameters"""
    G_value = np.array([0, 0, 9.81])
    m_value = 9.63
    I_value = np.array([
        [0.012, 0.0, 0.000],
        [0.0, 0.021, 0.0],
        [0.000, 0.0, 0.028]
    ])
    return G_value, m_value, I_value


# ============================================================================
# DYNAMICS MODEL
# ============================================================================

def dynamics(f1, f2, f3, f4, G, m, I, pf1, pf2, pf3, pf4, v, R_flat, omega):
    """
    Robot dynamics with SINDy corrections

    Computes: [p_dot, v_dot, R_dot, omega_dot]
    """
    # Total force and torque
    r1, r2, r3, r4 = pf1, pf2, pf3, pf4
    F_total_sym = f1 + f2 + f3 + f4
    tau_total_sym = ca.cross(r1, f1) + ca.cross(r2, f2) + ca.cross(r3, f3) + ca.cross(r4, f4)

    # Rotation matrix handling
    R_sym_3x3 = ca.reshape(R_flat, 3, 3)
    R_world_to_body = R_sym_3x3.T

    # Linear dynamics
    p_dot = v
    v_dot = ((1 / m) * F_total_sym) - G
    R_dot_matrix = ca.mtimes(R_sym_3x3, ca.skew(omega))

    # Angular dynamics (Euler's equation)
    tau_body = ca.mtimes(R_world_to_body, tau_total_sym)
    omega_dot_nominal = ca.mtimes(ca.inv(I), tau_body - ca.cross(omega, ca.mtimes(I, omega)))

    # Extract components for SINDy corrections
    wd_x_nom = omega_dot_nominal[0]
    wd_y_nom = omega_dot_nominal[1]
    wd_z_nom = omega_dot_nominal[2]

    # Apply SINDy corrections
    wd_z_corr = wd_z_nom - 31.8 * tau_total_sym[2] - 0.68 * tau_total_sym[1] * omega[0]
    wd_y_corr = wd_y_nom - 42.5 * tau_total_sym[1]
    wd_x_corr = wd_x_nom - 51.2 * tau_total_sym[0] + 3.5 * tau_total_sym[1] + 19.4 * tau_total_sym[0] * omega[1]

    omega_dot_final = ca.vertcat(wd_x_corr, wd_y_corr, wd_z_corr)

    # Combine into state derivative vector
    R_dot_flat = ca.reshape(R_dot_matrix.T, 9, 1)
    rhs = ca.vertcat(p_dot, v_dot, R_dot_flat, omega_dot_final)

    return rhs


# ============================================================================
# ACADOS MODEL CREATION
# ============================================================================

def create_acados_model():
    """Create Acados model with friction constraints"""
    model = AcadosModel()
    model.name = "srbm_friction"

    # States
    states, n_states, p_sym, v_sym, R_sym, omega_sym = setup_casadi_states()
    model.x = states

    # Controls
    controls, n_controls, f1, f2, f3, f4 = setup_casadi_controls()
    model.u = controls

    # Parameters: foot positions (12) + FSM states (4) = 16
    G_sym, m_sym, I_sym, pf1_sym, pf2_sym, pf3_sym, pf4_sym = setup_casadi_params()
    FSM_k_sym = ca.SX.sym('FSM_k', 4)

    params_p = ca.vertcat(pf1_sym, pf2_sym, pf3_sym, pf4_sym, FSM_k_sym)
    model.p = params_p
    n_p = model.p.size1()

    # Dynamics
    g_value, m_value, I_value = params()
    rhs = dynamics(f1, f2, f3, f4, g_value, m_value, I_value,
                   pf1_sym, pf2_sym, pf3_sym, pf4_sym, v_sym, R_sym, omega_sym)

    model.xdot = ca.SX.sym("xdot", n_states)
    model.f_expl_expr = rhs
    model.f_impl_expr = model.xdot - model.f_expl_expr

    # Friction constraints
    mu = 0.5
    friction_constraints = ca.vertcat(
        model.u[0] - model.u[2] * mu, model.u[1] - model.u[2] * mu,
        model.u[3] - model.u[5] * mu, model.u[4] - model.u[5] * mu,
        model.u[6] - model.u[8] * mu, model.u[7] - model.u[8] * mu,
        model.u[9] - model.u[11] * mu, model.u[10] - model.u[11] * mu
    )

    model.con_h_expr = friction_constraints
    num_path_constraints = model.con_h_expr.size1()
    model.con_h_expr_e = None

    return model, num_path_constraints


# ============================================================================
# OCP FORMULATION
# ============================================================================

def create_ocp(model, num_path_constraints_from_model, x0, Tf, N):
    """Create and configure Acados Optimal Control Problem"""
    ocp = AcadosOcp()

    # Set Acados paths
    acados_install_dir = os.environ.get('ACADOS_INSTALL_DIR')
    if acados_install_dir:
        ocp.acados_lib_path = os.path.join(acados_install_dir, 'lib')
        ocp.acados_include_path = os.path.join(acados_install_dir, 'include')

    ocp.model = model

    # Initialize parameter vector: foot positions (12) + FSM states (4)
    pf_dummy_value = np.zeros(12)
    fsm_dummy_value = np.zeros(4)
    p_val_stage0 = np.concatenate([pf_dummy_value, fsm_dummy_value])

    if p_val_stage0.shape[0] != 16:
        raise ValueError(f"Parameter vector size is {p_val_stage0.shape[0]}, expected 16")

    ocp.parameter_values = p_val_stage0

    # Time horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N

    # Integrator settings
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    # Solver configuration for real-time MPC
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.regularize_method = 'PROJECT'
    ocp.solver_options.levenberg_marquardt = 1e-4
    ocp.solver_options.nlp_solver_max_iter = 50

    # QP solver settings
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.qp_solver_iter_max = 50

    # Tolerances
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq = 1e-4
    ocp.solver_options.nlp_solver_tol_ineq = 1e-4
    ocp.solver_options.nlp_solver_tol_comp = 1e-4

    nx = ocp.model.x.size1()
    nu = ocp.model.u.size1()
    n_p = ocp.model.p.size1()
    nh = ocp.model.con_h_expr.size1() if ocp.model.con_h_expr is not None else 0
    nh_e = ocp.model.con_h_expr_e.size1() if ocp.model.con_h_expr_e is not None else 0

    ocp.dims.nh = nh
    ocp.dims.nh_e = nh_e

    # Initial state constraint
    ocp.constraints.x0 = x0

    # Control bounds
    F_max = 200.0
    ocp.constraints.lbu = -F_max * np.ones(nu)
    ocp.constraints.ubu = F_max * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu)

    # Cost function weights
    Q_diag = [
        100, 100, 100,      # Position
        50, 50, 50,         # Velocity
        50, 250, 250,       # Rotation matrix
        250, 50, 250,
        500, 500, 500,
        1, 1, 1             # Angular velocity
    ]
    Q = np.diag(Q_diag)
    R_diagonal = [1e-2, 1e-2, 1e-3] * 4
    R = np.diag(R_diagonal)

    # Stage cost
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
    ocp.cost.Vx = np.zeros((nx + nu, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((nx + nu, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)
    ocp.cost.yref = np.zeros(nx + nu)

    # Terminal cost
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W_e = Q
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros(nx)

    # Friction constraints: force - mu*Normal <= 0
    if nh > 0:
        lh = np.full(nh, -1e5)
        uh = np.full(nh, 0.0)
        ocp.constraints.lh = lh
        ocp.constraints.uh = uh

    ocp.constraints.constr_type_e = 'BGH'

    # Create solver
    try:
        json_file_name = f'acados_ocp_{model.name}.json'
        ocp_solver = AcadosOcpSolver(ocp, json_file=json_file_name, build=True, generate=True)
        print(f"Acados solver created: {json_file_name}")
        return ocp_solver, N
    except Exception as e:
        print(f"ERROR: Failed to create solver: {e}")
        raise


# ============================================================================
# OCP SOLVER
# ============================================================================

def solve_ocp(ocp_solver, x_current, x_desired_list, fsm_schedule_data, pf_list):
    """
    Solve MPC problem for one timestep

    Args:
        ocp_solver: Acados OCP solver instance
        x_current: Current state [18]
        x_desired_list: Desired states over horizon [N+1, 18]
        fsm_schedule_data: FSM schedule [4, N+1]
        pf_list: Foot positions [12]

    Returns:
        u0: Optimal control for current timestep [12]
        full_u: Full control sequence [N*12]
        x_pred_next: Predicted next state [18]
    """
    N = ocp_solver.acados_ocp.solver_options.N_horizon
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    n_p = ocp_solver.acados_ocp.dims.np

    num_feet = 4
    num_foot_pos_params = 12

    F_max = 200.0
    default_lbu = -F_max * np.ones(nu)
    default_ubu = F_max * np.ones(nu)
    zero_value = 0.0

    # Input validation
    if not isinstance(x_current, np.ndarray) or x_current.shape != (nx,):
        raise ValueError(f"x_current has wrong type/shape {type(x_current)}/{x_current.shape}, expected numpy ({nx},)")
    if len(x_desired_list) != N + 1:
        raise ValueError(f"x_desired_list length {len(x_desired_list)} != {N+1}")
    if not isinstance(fsm_schedule_data, np.ndarray) or fsm_schedule_data.shape != (num_feet, N + 1):
        raise ValueError(f"fsm_schedule_data has wrong type/shape")
    if not isinstance(pf_list, np.ndarray) or pf_list.shape != (num_foot_pos_params,):
        raise ValueError(f"pf_list has wrong type/shape")

    # Set initial state constraint
    ocp_solver.set(0, "lbx", x_current)
    ocp_solver.set(0, "ubx", x_current)

    # Set references and parameters for each stage
    for k in range(N):
        # Reference trajectory
        x_ref_k = np.asarray(x_desired_list[k]).flatten()
        yref_k = np.concatenate([x_ref_k, np.zeros(nu)])
        ocp_solver.set(k, "yref", yref_k)

        # Parameters: foot positions + FSM state
        fsm_k = fsm_schedule_data[:, k+1].flatten()
        p_k = np.concatenate([pf_list, fsm_k])

        if p_k.shape[0] != n_p:
            raise ValueError(f"Parameter vector size mismatch at stage {k}")
        ocp_solver.set(k, "p", p_k)

        # Control bounds (zero for swing legs)
        lbu_k = default_lbu.copy()
        ubu_k = default_ubu.copy()

        for foot_idx in range(num_feet):
            if fsm_k[foot_idx] == 0:  # Swing phase
                control_indices = slice(foot_idx * 3, (foot_idx + 1) * 3)
                lbu_k[control_indices] = zero_value
                ubu_k[control_indices] = zero_value

        ocp_solver.set(k, "lbu", lbu_k)
        ocp_solver.set(k, "ubu", ubu_k)

    # Terminal stage
    yref_e = np.asarray(x_desired_list[N]).flatten()
    ocp_solver.set(N, "yref", yref_e)

    fsm_N = fsm_schedule_data[:, N].flatten()
    p_N = np.concatenate([pf_list, fsm_N])

    if p_N.shape[0] != n_p:
        raise ValueError(f"Parameter vector size mismatch at terminal stage")
    ocp_solver.set(N, "p", p_N)

    # Solve OCP
    status = ocp_solver.solve()

    if status != 0:
        print(f"WARNING: Solver returned status {status}")
        return None, None, None

    # Extract optimal control
    u0 = ocp_solver.get(0, "u")
    if u0 is not None:
        u0 = u0.flatten()
    else:
        u0 = np.zeros(nu)

    # Extract full control sequence
    u_list = []
    for k in range(N):
        u_k = ocp_solver.get(k, "u")
        if u_k is not None:
            u_list.append(u_k.flatten())
        else:
            u_list.append(np.zeros(nu))

    full_u = np.array(u_list).flatten() if u_list else None

    # Get predicted next state
    try:
        x_pred_next = ocp_solver.get(1, 'x')
        if x_pred_next is None:
            x_pred_next = np.zeros(nx)
    except:
        x_pred_next = np.zeros(nx)

    return u0, full_u, x_pred_next