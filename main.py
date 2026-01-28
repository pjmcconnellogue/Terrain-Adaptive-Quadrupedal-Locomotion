"""
Quadruped Robot Simulation with MPC and Impedance Control
Main simulation logic without plotting functions
"""

import numpy as np
from scipy.linalg import expm
import time
from typing import Dict
import pybullet_config
import acados
import sindy

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_ADMITTANCE = True
USE_IMPEDANCE = True

# Admittance parameters: (virtual mass, max adjustment per step)
ADMITTANCE_PARAMS = {
    0: (np.array([20, 20, 50]), 0.7),      # Seeking: Moderate mass, critically damped
    1: (np.array([10, 10, 10]), 0.5),      # Light: Lighter mass
    2: (np.array([2.5, 2.5, 2.5]), 0.1)    # Adjusting: Very light, yields to terrain
}

# Impedance gains: (Kp, Kd) for each contact state
GAIN_TABLE_ADMITTANCE = {
    0: (np.array([200, 200, 100]), np.array([25, 25, 12])),   # Seeking
    1: (np.array([250, 250, 150]), np.array([30, 30, 18])),   # Light
    2: (np.array([300, 300, 200]), np.array([35, 35, 22])),   # Adjusting
    3: (np.array([400, 400, 400]), np.array([45, 45, 35]))    # Solid
}

GAIN_TABLE_NO_ADMITTANCE = {
    0: (np.array([200, 200, 100]), np.array([25, 25, 12])),   # Seeking
    1: (np.array([250, 250, 150]), np.array([30, 30, 18])),   # Light
    2: (np.array([300, 300, 200]), np.array([35, 35, 22])),   # Adjusting
    3: (np.array([400, 400, 400]), np.array([45, 45, 35]))    # Solid
}

CONTACT_STATES = {'seeking': 0, 'light': 1, 'adjusting': 2, 'solid': 3}
STATE_NAMES = ['seeking', 'light', 'adjusting', 'solid']


# ============================================================================
# PARAMETERS AND INITIALIZATION
# ============================================================================

def params() -> Dict:
    """Simulation parameters"""
    return {
        'SimTimeDuration': 13,
        'Tmpc': 0.0125,
        'steps': 8,
        'predHorizon': 8,
        'g': 9.81,
        'Tst': 0.3,
        'Tsw': 0.15,
        'z0': 0.25,
        'yaw_d': 0.,
        'acc_d': 0.1,
        'vel_d': [.2, 0.0],
        'ang_acc_d': [0, 0, 0.05],
    }


def init_variables():
    """Initialize simulation variables"""
    p = params()
    MAX_ITER = int(p['SimTimeDuration'] / p['Tmpc'])
    FSM = np.zeros(4)
    Ta_init = np.zeros(4)
    Tb_init = np.zeros(4)
    return p, MAX_ITER, FSM, Ta_init, Tb_init


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def hatMap(a):
    """Skew-symmetric matrix from vector"""
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def fcn_gen_Xd(t_, Xt, p):
    """Generate desired trajectory"""
    acc_d, vel_d, yaw_d, ang_acc_d = p['acc_d'], p['vel_d'], p['yaw_d'], p['ang_acc_d']
    Xd = np.zeros((18, len(t_)))
    zz = pybullet_config.get_ground_height_for_CoM(robotId)

    for ii, t in enumerate(t_):
        pc_d = np.array([0, 0, p['z0'] + zz])
        dpc_d = np.zeros(3)

        # Linear velocity trajectory
        for jj in range(2):
            if t < vel_d[jj] / (acc_d+0.000001):
                dpc_d[jj] = acc_d * t
                pc_d[jj] = 0.5 * acc_d * t ** 2
            else:
                dpc_d[jj] = vel_d[jj]
                pc_d[jj] = vel_d[jj] * t - 0.5 * vel_d[jj] ** 2 / (acc_d+0.000001)

        ea_d = np.zeros(3)
        wb_d = np.zeros(3)

        if Xt is not None:
            # Sinusoidal yaw trajectory
            yaw_d = np.sin(5 * t_[ii]) * p['yaw_d']
            wb_d[2] = 5 * np.cos(5 * t_[ii]) * p['yaw_d']
            ea_d[2] = yaw_d

        vR_d = expm(hatMap(ea_d)).flatten()
        Xd[:, ii] = np.concatenate((pc_d, dpc_d, vR_d, wb_d))

    return Xd


def fcn_FSM(t_, p, FSM, Ta, Tb):
    """Finite State Machine for gait planning"""
    Tst, Tsw = p['Tst'], p['Tsw']
    t = t_[0]
    s = (t - Ta) / (Tb - Ta + 1E-8)

    for i_leg in range(4):
        if FSM[i_leg] == 0:
            Ta[i_leg] = t if i_leg in [0, 3] else t + 0.5 * (Tst + Tsw)
            Tb[i_leg] = Ta[i_leg] + Tst
            FSM[i_leg] = 1
        elif FSM[i_leg] == 1 and s[i_leg] >= 1 - 1e-7:
            FSM[i_leg] = 2
            Ta[i_leg] = t
            Tb[i_leg] = t + Tsw
        elif FSM[i_leg] == 2 and s[i_leg] >= 1 - 1e-7:
            FSM[i_leg] = 1
            Ta[i_leg] = t
            Tb[i_leg] = t + Tst

    FSM_ = np.tile(FSM, (p['predHorizon'], 1)).T

    for i_leg in range(4):
        for ii in range(1, p['predHorizon']):
            if t_[ii] <= Ta[i_leg]:
                FSM_[i_leg, ii] = 1
            elif Ta[i_leg] < t_[ii] < Tb[i_leg]:
                FSM_[i_leg, ii] = FSM[i_leg]
            elif t_[ii] >= Tb[i_leg] + Tst + Tsw:
                FSM_[i_leg, ii] = FSM[i_leg]
            else:
                FSM_[i_leg, ii] = 2 if FSM[i_leg] == 1 else 1

    return (FSM_ == 1).T.reshape(4 * p['predHorizon']), FSM, Ta, Tb


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PreallocatedArrays:
    """Pre-allocated arrays for hot path operations"""
    def __init__(self):
        self.zeros_3 = np.zeros(3)
        self.temp_vec3_1 = np.zeros(3)
        self.temp_vec3_2 = np.zeros(3)
        self.temp_vec3_3 = np.zeros(3)
        self.temp_grf_slice = np.zeros(3)
        self.temp_force_error = np.zeros(3)
        self.temp_position_error = np.zeros(3)
        self.temp_velocity_error = np.zeros(3)

    def get_zeros_3(self):
        self.zeros_3.fill(0)
        return self.zeros_3


class PositionErrorTracker:
    """Track position errors for analysis"""
    def __init__(self, history_length=1000, dt_simulation=None):
        self.history_length = history_length
        self.position_errors = {leg: [] for leg in range(1, 5)}
        self.dt_simulation = dt_simulation

    def update(self, leg, position_error, iteration=None):
        self.position_errors[leg].append(position_error.copy())


class MemoryPool:
    """Memory pool for frequently allocated arrays"""
    def __init__(self):
        self._vec3_pool = [np.zeros(3) for _ in range(50)]
        self._vec12_pool = [np.zeros(12) for _ in range(20)]
        self._pool_indices = {'vec3': 0, 'vec12': 0}

    def get_vec3(self):
        vec = self._vec3_pool[self._pool_indices['vec3']]
        self._pool_indices['vec3'] = (self._pool_indices['vec3'] + 1) % len(self._vec3_pool)
        vec.fill(0)
        return vec

    def get_vec12(self):
        vec = self._vec12_pool[self._pool_indices['vec12']]
        self._pool_indices['vec12'] = (self._pool_indices['vec12'] + 1) % len(self._vec12_pool)
        vec.fill(0)
        return vec


# Global instances
_prealloc = PreallocatedArrays()
_position_tracker = PositionErrorTracker(dt_simulation=0.00125)
_memory_pool = MemoryPool()
_computation_cache = {'jacobians': {}, 'grfs': {}, 'foot_states': {}}


# ============================================================================
# CACHING FUNCTIONS
# ============================================================================

def get_cached_grfs(robot, plane_id, soft_patch_id, pybullet_iteration):
    """Cache GRF calculations"""
    cache_key = (robot, pybullet_iteration)
    if cache_key not in _computation_cache['grfs']:
        _computation_cache['grfs'][cache_key] = pybullet_config.get_grfs(
            robot, plane_id, soft_patch_id, foot_indices=[2, 6, 10, 14]
        )
    return _computation_cache['grfs'][cache_key]


def get_cached_jacobian(robot, leg_idx, pybullet_iteration, steps_per_mpc):
    """Cache Jacobian calculations"""
    update_frequency = max(1, steps_per_mpc / 2)
    cache_iteration = (pybullet_iteration // update_frequency) * update_frequency
    cache_key = (robot, leg_idx, cache_iteration)

    if cache_key not in _computation_cache['jacobians']:
        J_all = pybullet_config.get_pybullet_jacobians(
            robotId=robot,
            leg_joint_indices=[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
            foot_indices=[2, 6, 10, 14]
        )
        _computation_cache['jacobians'][cache_key] = J_all[leg_idx]
    return _computation_cache['jacobians'][cache_key]


def get_cached_foot_state(robot, foot_link_index, pybullet_iteration):
    """Cache foot state calculations"""
    cache_key = (robot, foot_link_index, pybullet_iteration)
    if cache_key not in _computation_cache['foot_states']:
        _computation_cache['foot_states'][cache_key] = pybullet_config.get_foot_state_world(robot, foot_link_index)
    return _computation_cache['foot_states'][cache_key]


def get_all_motor_angles_batch(robot, motor_ids):
    """Get all motor angles in one batch operation"""
    joint_states = [pybullet_config.p.getJointState(robot, motor_id) for motor_id in motor_ids]
    return {motor_id: joint_states[i][0] for i, motor_id in enumerate(motor_ids)}


# ============================================================================
# CONTROL FUNCTIONS
# ============================================================================

def get_contact_state_int(contact_state_str):
    """Convert string state to integer"""
    return CONTACT_STATES.get(contact_state_str, 0)


def calculate_virtual_damper_shadow(force_error, damping_coefficient, dt):
    """Calculate position adjustment from force error using virtual damper"""
    velocity_command = force_error / damping_coefficient
    position_adjustment = velocity_command * dt
    return velocity_command, position_adjustment


def calculate_impedance_torque(robot, leg, J, anchor_pos, foot_pos, foot_vel, Kp, Kd, stance_state=None):
    """Calculate impedance control torques"""
    if anchor_pos is None:
        return _prealloc.get_zeros_3(), _prealloc.get_zeros_3()

    position_error = _prealloc.temp_position_error
    position_error[:] = anchor_pos - foot_pos

    velocity_error = _prealloc.temp_velocity_error
    velocity_error[:] = -foot_vel

    corrective_force_world = _prealloc.temp_vec3_3
    corrective_force_world[:] = (Kp * position_error) + (Kd * velocity_error)

    corrective_tau = J.T @ corrective_force_world

    return corrective_tau, corrective_force_world.copy()


def control_leg(leg, U, robot, plane_id, soft_patch_id, stance, lift, U_history,
                measured_grf_history, angles, stance_anchor_points, admittance_states,
                ii, i, footanchorpos, newstance, newlift, soft_patch_force_data=None, p=None):
    """Low-level leg control with admittance and impedance"""

    pybullet_iteration = p['steps'] * ii + i
    leg_idx = leg - 1
    joint_base = 4 * leg_idx
    joint_indices = [joint_base, joint_base + 1, joint_base + 2]
    foot_link_index = 3 + joint_base
    grf_slice = slice(3 * leg_idx, 3 * leg_idx + 3)

    # Measure ground reaction forces
    all_GRFs = get_cached_grfs(robot, plane_id, soft_patch_id, pybullet_iteration)
    measured_leg_grf = all_GRFs[grf_slice]
    measured_grf_history.append(measured_leg_grf.copy())

    # Check soft patch contact
    contact_points = pybullet_config.p.getContactPoints(bodyA=robot, bodyB=soft_patch_id, linkIndexA=foot_link_index)
    num_contacts = len(contact_points)

    # Get foot state
    foot_pos, foot_vel = get_cached_foot_state(robot, foot_link_index, pybullet_iteration)

    # Update position tracker
    anchorpoint = stance_anchor_points[leg]
    if anchorpoint is not None:
        err = anchorpoint - foot_pos
    else:
        err = np.zeros_like(foot_pos)

    if stance[leg] == 2:
        _position_tracker.update(leg, err)
    else:
        _position_tracker.update(leg, np.zeros_like(foot_pos))

    # LIFT PHASE
    if lift[leg] == 1:
        newlift[leg] = 1
        joint_angles = {joint_base: 0., joint_base + 1: -1.1, joint_base + 2: 2.55}
        pybullet_config.apply_pd_control(robot, joint_angles)
        pybullet_config.fix_motor(joint_angles, robot)
        lift[leg] = 0

        # Reset admittance state
        admittance_states[leg].update({
            'contact_state': 'adjusting',
            'cumulative_offset': np.zeros(3),
            'was_in_contact': False,
            'prev_state': None,
            'prev_corrective_force': np.zeros(3),
            'prev_total_desired': np.zeros(3)
        })

        U_history.append(np.zeros(3))
        return U_history, measured_grf_history

    # STANCE PHASE 1: Initialize stance
    if stance[leg] == 1:
        newlift[leg] = 0
        if footanchorpos[2] < -0.05:
            footanchorpos[2] = -0.03

        stance_anchor_points[leg] = footanchorpos.copy()
        stance[leg] = 2
        newstance[leg] = 1
        U_history.append(np.zeros(3))
        return U_history, measured_grf_history

    # STANCE PHASE 2: Main control
    if stance[leg] == 2:
        if newstance[leg] == 1:
            pybullet_config.initiate_torque_control(robot, leg_joint_indices=joint_indices)
            newstance[leg] = 0

        foot_pos, foot_vel = get_cached_foot_state(robot, foot_link_index, pybullet_iteration)

        # Calculate desired forces
        DESIRED_CONTACT_FORCE = U[grf_slice]
        prev_corrective_force = admittance_states[leg].get('prev_corrective_force', _prealloc.get_zeros_3())

        DESIRED_TOTAL_FORCE = _prealloc.temp_vec3_1
        DESIRED_TOTAL_FORCE[:] = DESIRED_CONTACT_FORCE + prev_corrective_force

        force_error = _prealloc.temp_force_error
        force_error[:] = DESIRED_TOTAL_FORCE - measured_leg_grf

        # Adaptive thresholds based on desired force
        desired_fz = abs(DESIRED_CONTACT_FORCE[2])
        reference_force = desired_fz
        THRESHOLD_CONTACT_ENTRY = 0.05 * reference_force
        THRESHOLD_CONTACT_EXIT = 0.005 * reference_force
        THRESHOLD_LIGHT_MAX = 0.25 * reference_force
        THRESHOLD_FORCE_ERROR = 0.25 * reference_force

        # Contact state detection
        was_in_contact = admittance_states[leg].get('was_in_contact', False)
        measured_fz = measured_leg_grf[2]
        force_error_z = abs(force_error[2])

        if not was_in_contact:
            if measured_fz > THRESHOLD_CONTACT_ENTRY:
                admittance_states[leg]['was_in_contact'] = True
                current_contact_state = 'light'
                current_contact_state_int = 1
            else:
                current_contact_state = 'seeking'
                current_contact_state_int = 0
        else:
            if measured_fz < THRESHOLD_CONTACT_EXIT:
                admittance_states[leg]['was_in_contact'] = False
                current_contact_state = 'seeking'
                current_contact_state_int = 0
            elif measured_fz < THRESHOLD_LIGHT_MAX:
                current_contact_state = 'light'
                current_contact_state_int = 1
            elif force_error_z > THRESHOLD_FORCE_ERROR:
                current_contact_state = 'adjusting'
                current_contact_state_int = 2
            else:
                current_contact_state = 'solid'
                current_contact_state_int = 3

        admittance_states[leg]['contact_state'] = current_contact_state

        # Update anchor point based on contact state
        if current_contact_state != 'solid':
            # Track actual position with smoothing during non-solid contact
            foot_pos_current, _ = get_cached_foot_state(robot, foot_link_index, pybullet_iteration)
            stance_anchor_points[leg] = 0.95 * stance_anchor_points[leg] + 0.05 * foot_pos_current
        elif admittance_states[leg].get('prev_state') != 'solid':
            # Lock to actual position when entering solid contact
            foot_pos_lock, _ = get_cached_foot_state(robot, foot_link_index, pybullet_iteration)
            stance_anchor_points[leg] = foot_pos_lock.copy()

        # Admittance control
        if current_contact_state_int in ADMITTANCE_PARAMS:
            D_virtual, max_adjustment_per_step = ADMITTANCE_PARAMS[current_contact_state_int]
            dt_simulation = p['Tmpc'] / p['steps']
            velocity_cmd, position_adjustment = calculate_virtual_damper_shadow(
                force_error, D_virtual, dt_simulation
            )
            position_adjustment_limited = np.clip(
                position_adjustment, -max_adjustment_per_step, max_adjustment_per_step
            )

            if USE_ADMITTANCE:
                admittance_states[leg]['cumulative_offset'] += position_adjustment_limited
                max_total_offset = 0.05
                admittance_states[leg]['cumulative_offset'][2] = np.clip(
                    admittance_states[leg]['cumulative_offset'][2],
                    -max_total_offset, max_total_offset
                )

            stance_anchor_points[leg] = footanchorpos - np.array([
                0, 0, admittance_states[leg]['cumulative_offset'][2]
            ])

        elif current_contact_state == 'solid':
            if admittance_states[leg].get('prev_state') != 'solid':
                stance_anchor_points[leg] = foot_pos.copy()
                admittance_states[leg]['cumulative_offset'].fill(0)

        admittance_states[leg]['prev_state'] = current_contact_state

    # Torque calculation
    if stance[leg] == 2 or stance[leg] == 0:
        J = get_cached_jacobian(robot, leg_idx, pybullet_iteration, steps_per_mpc=p['steps'])

        # MPC feedforward
        leg_U = U[grf_slice]
        desired_forces = -leg_U
        R = pybullet_config.get_base_rotation_matrix(robot)
        tau_feedforward = J.T @ R.T @ desired_forces

        # Impedance feedback
        contact_state_int = get_contact_state_int(admittance_states[leg].get('contact_state', 'seeking'))
        gain_table = GAIN_TABLE_ADMITTANCE if USE_ADMITTANCE else GAIN_TABLE_NO_ADMITTANCE
        Kp, Kd = gain_table.get(contact_state_int, gain_table[0])

        corrective_tau, corrective_force = calculate_impedance_torque(
            robot, leg, J, stance_anchor_points[leg], foot_pos, foot_vel,
            Kp=Kp, Kd=Kd, stance_state=stance[leg]
        )

        # Store for next iteration
        current_total_desired = leg_U + corrective_force
        admittance_states[leg].update({
            'prev_total_desired': current_total_desired.copy(),
            'prev_corrective_force': corrective_force.copy()
        })

        # Apply total torque
        tau = tau_feedforward + corrective_tau if USE_IMPEDANCE else tau_feedforward
        pybullet_config.apply_torques(robot, leg_joint_indices=joint_indices, tau=tau)

        # Record desired forces
        if stance[leg] == 2:
            desired_forces_total_for_recording = leg_U + corrective_force
            U_history.append(desired_forces_total_for_recording)
        else:
            U_history.append(np.zeros(3))

        # Soft patch data recording
        if soft_patch_force_data is not None:
            if num_contacts == 0:
                force_error_soft_patch = 0
                desired_force = 0
                measured_force = 0
            else:
                if leg_U[2] == 0:
                    force_error_soft_patch = 0
                    desired_force = 0
                    measured_force = measured_leg_grf[2]
                else:
                    force_error_soft_patch = (leg_U[2] + corrective_force[2] - measured_leg_grf[2])
                    desired_force = leg_U[2] + corrective_force[2]
                    measured_force = measured_leg_grf[2]

            soft_patch_data = soft_patch_force_data
            soft_patch_data['errors'].append(force_error_soft_patch)
            soft_patch_data['desired'].append(desired_force)
            soft_patch_data['measured'].append(measured_force)
            soft_patch_data['leg'].append(leg)
            soft_patch_data['iteration'].append(pybullet_iteration)
    else:
        U_history.append(np.zeros(3))

    return U_history, measured_grf_history


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation(p, MAX_ITER, FSM, Ta_init, Tb_init, robotIds, plane_id, soft_patch_id):
    """Main simulation loop"""

    # Create MPC model
    model, num_fsm_states = acados.create_acados_model()

    n_robots = len(robotIds)
    n_legs = 4

    # Initialize state arrays
    stance = {robot: np.zeros(n_legs + 1, dtype=np.int32) for robot in robotIds}
    newstance = {robot: np.zeros(n_legs + 1, dtype=np.int32) for robot in robotIds}
    newlift = {robot: np.zeros(n_legs + 1, dtype=np.int32) for robot in robotIds}
    lift = {robot: np.zeros(n_legs + 1, dtype=np.int32) for robot in robotIds}

    u0 = {robot: np.zeros(12 * p['predHorizon']) for robot in robotIds}
    bool_prev = {robot: np.ones(4, dtype=bool) for robot in robotIds}
    stance_angles = {robot: np.zeros((4, 3)) for robot in robotIds}
    footanchorpos = {robot: np.zeros((4, 3)) for robot in robotIds}
    stance_anchor_points = {robot: {leg: None for leg in range(1, 5)} for robot in robotIds}

    admittance_states = {
        leg: {
            'contact_state': 'adjusting',
            'cumulative_offset': np.zeros(3),
            'was_in_contact': False,
            'prev_state': None,
            'prev_corrective_force': np.zeros(3),
            'prev_total_desired': np.zeros(3)
        } for leg in range(1, 5)
    }

    # History tracking
    U_history = {robot: {leg: [] for leg in range(1, 5)} for robot in robotIds}
    measured_grf_history = {robot: {leg: [] for leg in range(1, 5)} for robot in robotIds}
    robot_state_history = {robot: [] for robot in robotIds}
    measured_state_history = {robot: [] for robot in robotIds}
    desired_state_history = {robot: [] for robot in robotIds}
    mpc_time_history = []
    mpc_solve_times = []

    motor_ids = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    motor_angle_history = {robot: {motor_id: [] for motor_id in motor_ids} for robot in robotIds}
    simulation_time_history = []

    soft_patch_force_data = {robot: {
        'errors': [],
        'desired': [],
        'measured': [],
        'leg': [],
        'iteration': []
    } for robot in robotIds}

    # Set initial robot pose
    initial_joint_angles = {
        0: 0., 1: -1.1, 2: 2.1, 4: 0., 5: -1.1, 6: 2.1,
        8: 0., 9: -1.1, 10: 2.1, 12: 0., 13: -1.1, 14: 2.1
    }

    for robot in robotIds:
        pybullet_config.set_joint_angles(robot, initial_joint_angles)
        for _ in range(1250):
            pybullet_config.fix_motor(initial_joint_angles, robot)
            pybullet_config.simulation_step()

    # Initialize torque control
    for robot in robotIds:
        for leg_idx in range(4):
            joint_indices = [4 * leg_idx, 4 * leg_idx + 1, 4 * leg_idx + 2]
            pybullet_config.initiate_torque_control(robot, leg_joint_indices=joint_indices)

    # Create MPC solver
    robotId_main = robotIds[0] if robotIds else None
    if robotId_main is None:
        return

    Xt = pybullet_config.get_robot_state(robotId_main)
    ocp_solver, _ = acados.create_ocp(
        model, num_fsm_states, np.array(Xt[0:18]),
        Tf=p['Tmpc'] * p['predHorizon'], N=p['predHorizon']
    )

    real_start_time = time.time()
    xd0 = np.zeros(18)
    print("Initialization complete. Starting simulation...")

    # ========================================================================
    # MAIN SIMULATION LOOP
    # ========================================================================
    for mpc_iteration in range(MAX_ITER):
        # Clear cache periodically
        if mpc_iteration % 100 == 0:
            _computation_cache['grfs'].clear()
            _computation_cache['jacobians'].clear()
            _computation_cache['foot_states'].clear()

        # Generate time horizon and FSM schedule
        t_horizon_array = p['Tmpc'] * mpc_iteration + p['Tmpc'] * np.arange(p['predHorizon'])
        t_horizon = t_horizon_array

        mpc_time_history.append(t_horizon[0])
        bool_inStance, FSM, Ta, Tb = fcn_FSM(t_horizon, p, FSM, Ta_init, Tb_init)

        U_robot = {}

        # Solve MPC for each robot
        for robot in robotIds:
            Xt = pybullet_config.get_robot_state(robot)
            robot_state_history[robot].append(Xt[0:18].copy())

            # Generate desired trajectory
            Xd_plan = fcn_gen_Xd(t_horizon, Xt, p)
            desired_state_history[robot].append(Xd_plan[:, 0].copy())

            # Prepare MPC inputs
            x_desired_list = [Xt[0:18] if mpc_iteration == 0 else xd0[0:18]] + \
                             [Xd_plan[:, i] for i in range(p['predHorizon'])]
            xd0 = np.array(Xd_plan[:, 1])
            FSM_full = np.concatenate([bool_prev[robot], bool_inStance])
            pf_list = np.array(Xt[18:30])

            # Solve MPC
            start_time = time.perf_counter()
            uopt, u_full, x_pred_next = acados.solve_ocp(
                ocp_solver, x_current=np.array(Xt[0:18]),
                x_desired_list=x_desired_list,
                fsm_schedule_data=FSM_full.reshape((p['predHorizon'] + 1, 4)).T,
                pf_list=pf_list
            )
            solve_time = time.perf_counter() - start_time
            mpc_solve_times.append(solve_time)

            u_full = u_full.flatten()
            u0[robot] = np.hstack([u_full[12:], u_full[-12:]])
            U_robot[robot] = uopt

            # Update stance angles and anchor positions
            for i in range(4):
                stance_angles[robot][i], footanchorpos[robot][i] = \
                    pybullet_config.compute_joint_angles_to_anchor_point(robot, Xt[6:15], end_effector=3 + 4 * i)

            # Update leg state flags
            bool_inStance_current = bool_inStance[0:4]
            state_changes = (bool_inStance_current != bool_prev[robot])
            for i in range(4):
                if state_changes[i]:
                    stance[robot][i + 1] = int(bool_inStance_current[i])
                    lift[robot][i + 1] = int(not bool_inStance_current[i])
            bool_prev[robot] = bool_inStance_current.copy()

        # Execute sub-steps at higher frequency
        SIM_STEPS_PER_MPC = p['steps']
        counter = 0

        for i in range(SIM_STEPS_PER_MPC):
            current_sim_time = time.time() - real_start_time
            simulation_time_history.append(current_sim_time)

            for robot in robotIds:
                motor_angles = get_all_motor_angles_batch(robot, motor_ids)

                # Control each leg
                for leg in range(1, 5):
                    U_history[robot][leg], measured_grf_history[robot][leg] = control_leg(
                        leg=leg, U=U_robot[robot], robot=robot, plane_id=plane_id,
                        soft_patch_id=soft_patch_id, stance=stance[robot], lift=lift[robot],
                        U_history=U_history[robot][leg],
                        measured_grf_history=measured_grf_history[robot][leg],
                        angles=stance_angles[robot], stance_anchor_points=stance_anchor_points[robot],
                        admittance_states=admittance_states, ii=mpc_iteration, i=counter,
                        footanchorpos=footanchorpos[robot][leg - 1], newstance=newstance[robot],
                        newlift=newlift[robot], soft_patch_force_data=soft_patch_force_data[robot],
                        p=p
                    )

                measured_state = pybullet_config.get_robot_state(robot)
                measured_state_history[robot].append(measured_state)

                for motor_id in motor_ids:
                    motor_angle_history[robot][motor_id].append(motor_angles[motor_id])

            pybullet_config.simulation_step()
            counter += 1

        # Real-time synchronization
        sim_target_time = t_horizon[0]
        real_elapsed_time = time.time() - real_start_time
        if sim_target_time > real_elapsed_time:
            time.sleep(sim_target_time - real_elapsed_time)

        # Progress indicator
        if mpc_iteration % 20 == 0:
            print(f"Progress: {mpc_iteration}/{MAX_ITER} iterations ({100*mpc_iteration/MAX_ITER:.1f}%)")

    pybullet_config.disconnect()
    print("Simulation complete.")

    # Return collected data
    return {
        'params': p,
        'U_history': U_history,
        'measured_grf_history': measured_grf_history,
        'robot_state_history': robot_state_history,
        'measured_state_history': measured_state_history,
        'desired_state_history': desired_state_history,
        'mpc_time_history': mpc_time_history,
        'motor_angle_history': motor_angle_history,
        'simulation_time_history': simulation_time_history,
        'soft_patch_force_data': soft_patch_force_data,
        'position_tracker': _position_tracker,
        'mpc_solve_times': mpc_solve_times,
        'robotIds': robotIds
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    p, MAX_ITER, FSM, Ta_init, Tb_init = init_variables()
    robotId, plane_id, soft_patch_id = pybullet_config.setup_simulation(p['Tmpc'], p['steps'])

    # Make robotId global for trajectory generation
    globals()['robotId'] = robotId

    results = run_simulation(p, MAX_ITER, FSM, Ta_init, Tb_init, [robotId], plane_id, soft_patch_id)

    # Save results
    import pickle
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\nResults saved to 'simulation_results.pkl'")
    print("Run the plotting script to visualize results.")