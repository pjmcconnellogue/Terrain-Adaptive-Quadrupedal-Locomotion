"""
Quadruped Robot Simulation - Plotting and Analysis
All visualization and analysis functions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import sindy

from main import PositionErrorTracker


# ============================================================================
# BASIC PLOTTING FUNCTIONS
# ============================================================================

def plot_grf_history(robot_id, history_u, history_grf, ylim_xy=(-7, 7), ylim_z=(-50, 50)):
    """Plot ground reaction force tracking"""
    for leg in range(1, 5):
        U_array = np.array(history_u[leg])
        measured_GRF_array = np.array(history_grf[leg])

        if U_array.size == 0 or measured_GRF_array.size == 0:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Leg {leg} Force Tracking (Robot {robot_id})")

        for i, axis in enumerate(['X', 'Y', 'Z']):
            axes[i].plot(U_array[:, i], '-', label=f"U_{axis} (Desired)", linewidth=2)
            axes[i].plot(measured_GRF_array[:, i], '--', label=f"Measured GRF {axis}", linewidth=1.5)
            axes[i].set_ylabel(f"Force {axis} (N)")

            if i == 2:
                axes[i].set_ylim(ylim_z)
            else:
                axes[i].set_ylim(ylim_xy)

            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout()
        plt.show()


def plot_motor_angles(robot_id, time_history, angle_history):
    """Plot motor angles over time"""
    plt.figure(figsize=(12, 6))
    motor_labels = {0: "Motor 0 (Hip Ab)", 1: "Motor 1 (Hip Pitch)", 2: "Motor 2 (Knee)"}

    for motor_id, history in angle_history.items():
        if motor_id in motor_labels:
            plt.plot(time_history, history, label=motor_labels[motor_id])

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title(f"Robot {robot_id} Motor Angles Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_state_tracking(robot_id, time_history, actual_states_hist, desired_states_hist):
    """Plot state tracking (actual vs desired)"""
    state_indices_to_plot = {
        0: 'X Position (m)', 1: 'Y Position (m)', 2: 'Z Position (m)',
        15: 'Vx Linear Velocity (m/s)', 16: 'Vy Linear Velocity (m/s)',
        17: 'Vz Linear Velocity (m/s)',
    }

    actual_states = np.array(actual_states_hist)
    desired_states = np.array(desired_states_hist)

    num_plots = len(state_indices_to_plot)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, num_plots * 3), sharex=True)
    fig.suptitle(f"Robot {robot_id} State Tracking (Actual vs Desired)", fontsize=16)
    if num_plots == 1:
        axes = [axes]

    for i, (idx, label) in enumerate(state_indices_to_plot.items()):
        axes[i].plot(time_history, actual_states[:, idx], label='Actual')
        axes[i].plot(time_history, desired_states[:, idx], '--', label='Desired')
        axes[i].set_ylabel(label)
        axes[i].grid(True)
        if i == 0:
            axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_position_errors(position_tracker):
    """Plot position errors for all legs"""
    if any(len(position_tracker.position_errors[leg]) < 10 for leg in range(1, 5)):
        print("Insufficient data for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, leg in enumerate(range(1, 5)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        if len(position_tracker.position_errors[leg]) > 0:
            errors = np.array(position_tracker.position_errors[leg])
            iteration_array = np.arange(len(errors))

            if position_tracker.dt_simulation is not None:
                time_array = iteration_array * position_tracker.dt_simulation
                x_label = 'Time (s)'
            else:
                time_array = iteration_array
                x_label = 'Iteration'

            ax.plot(time_array, errors[:, 0], 'r-', label='X Error', linewidth=1.5)
            ax.plot(time_array, errors[:, 1], 'g-', label='Y Error', linewidth=1.5)
            ax.set_xlabel(x_label, fontsize=18)
            ax.set_ylabel('Position Error (m)', fontsize=18)
            ax.set_title(f'Foot {leg} Position Error', fontsize=18)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_ylim(-0.025, 0.05)

    plt.tight_layout()
    plt.show()


# ============================================================================
# FORCE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_force_performance(robot_id, U_history, measured_grf_history):
    """Analyze force tracking performance"""
    print(f"\nForce Analysis - Robot {robot_id}")
    print("=" * 50)

    for leg in range(1, 5):
        if leg not in U_history or leg not in measured_grf_history:
            continue

        U_array = np.array(U_history[leg])
        measured_array = np.array(measured_grf_history[leg])

        if U_array.size == 0 or measured_array.size == 0:
            continue

        force_errors = U_array - measured_array
        force_error_z = force_errors[:, 2]

        mean_error = np.mean(force_error_z)
        rms_error = np.sqrt(np.mean(force_error_z ** 2))
        max_error = np.max(np.abs(force_error_z))

        contact_threshold = 5.0
        contact_forces = measured_array[:, 2]
        contact_percentage = (np.sum(contact_forces > contact_threshold) / len(contact_forces)) * 100
        avg_contact_force = np.mean(contact_forces[contact_forces > contact_threshold])

        print(f"\nLeg {leg}:")
        print(f"  Mean Z Force Error: {mean_error:.2f} N")
        print(f"  RMS Z Force Error:  {rms_error:.2f} N")
        print(f"  Max Z Force Error:  {max_error:.2f} N")
        print(f"  Contact Percentage: {contact_percentage:.1f}%")
        print(f"  Avg Contact Force:  {avg_contact_force:.1f} N")


def plot_force_analysis(robot_id, U_history, measured_grf_history):
    """Create comprehensive force analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Robot {robot_id} - Force Analysis", fontsize=16)

    # Force tracking errors
    ax1 = axes[0, 0]
    for leg in range(1, 5):
        if leg in U_history and leg in measured_grf_history:
            U_array = np.array(U_history[leg])
            measured_array = np.array(measured_grf_history[leg])
            if U_array.size > 0 and measured_array.size > 0:
                errors = U_array[:, 2] - measured_array[:, 2]
                ax1.plot(errors, label=f'Leg {leg}', alpha=0.8)

    ax1.set_title('Force Tracking Errors (Z-axis)')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Force Error (N)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Desired vs Actual scatter
    ax2 = axes[0, 1]
    for leg in range(1, 5):
        if leg in U_history and leg in measured_grf_history:
            U_array = np.array(U_history[leg])
            measured_array = np.array(measured_grf_history[leg])
            if U_array.size > 0 and measured_array.size > 0:
                desired_z = U_array[:, 2]
                actual_z = measured_array[:, 2]
                ax2.scatter(desired_z, actual_z, alpha=0.5, label=f'Leg {leg}', s=5)

    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Tracking')
    ax2.set_title('Desired vs Actual Force (Z-axis)')
    ax2.set_xlabel('Desired Force (N)')
    ax2.set_ylabel('Actual Force (N)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # RMS errors
    ax3 = axes[1, 0]
    legs = []
    rms_errors = []

    for leg in range(1, 5):
        if leg in U_history and leg in measured_grf_history:
            U_array = np.array(U_history[leg])
            measured_array = np.array(measured_grf_history[leg])
            if U_array.size > 0 and measured_array.size > 0:
                errors = U_array[:, 2] - measured_array[:, 2]
                rms_error = np.sqrt(np.mean(errors ** 2))
                legs.append(leg)
                rms_errors.append(rms_error)

    if legs:
        x_pos = np.arange(len(legs))
        ax3.bar(x_pos, rms_errors, alpha=0.7, color='red')
        ax3.set_title('RMS Force Tracking Error')
        ax3.set_xlabel('Leg')
        ax3.set_ylabel('RMS Error (N)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Leg {leg}' for leg in legs])
        ax3.grid(True, alpha=0.3)

    # Force magnitudes
    ax4 = axes[1, 1]
    for leg in range(1, 5):
        if leg in measured_grf_history:
            measured_array = np.array(measured_grf_history[leg])
            if measured_array.size > 0:
                force_z = measured_array[:, 2]
                ax4.plot(force_z, label=f'Leg {leg}', alpha=0.8)

    ax4.set_title('Measured Z Forces Over Time')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Force Z (N)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# SOFT PATCH ANALYSIS
# ============================================================================

def analyze_soft_patch_forces(robot_id, soft_patch_data):
    """Analyze force performance on soft patch"""
    if len(soft_patch_data['errors']) == 0:
        print("No soft patch contact detected")
        return

    errors = np.array(soft_patch_data['errors'])
    legs = np.array(soft_patch_data['leg'])

    print(f"\nSoft Patch Analysis - Robot {robot_id}")
    print("=" * 50)

    for leg in range(1, 5):
        leg_mask = legs == leg
        if np.sum(leg_mask) > 0:
            leg_errors = errors[leg_mask]
            print(f"Leg {leg}: {len(leg_errors)} contacts, "
                  f"RMS error: {np.sqrt(np.mean(leg_errors ** 2)):.2f} N")


def plot_soft_patch_analysis(robot_id, soft_patch_data):
    """Plot soft patch specific force analysis"""
    if len(soft_patch_data['errors']) == 0:
        print("No soft patch data to plot")
        return

    errors = np.array(soft_patch_data['errors'])
    desired = np.array(soft_patch_data['desired'])
    measured = np.array(soft_patch_data['measured'])
    legs = np.array(soft_patch_data['leg'])
    iterations = np.array(soft_patch_data['iteration'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Robot {robot_id} - Soft Patch Force Analysis", fontsize=16)

    colors = ['red', 'blue', 'green', 'orange']

    # Errors over time
    ax1 = axes[0, 0]
    for leg in range(1, 5):
        leg_mask = legs == leg
        if np.sum(leg_mask) > 0:
            ax1.scatter(iterations[leg_mask], errors[leg_mask],
                        c=colors[leg - 1], label=f'Leg {leg}', alpha=0.7, s=20)
    ax1.set_title('Soft Patch Force Errors Over Time')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Force Error (N)')
    ax1.set_ylim(-100, 50)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error distribution
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=100, alpha=0.7, color='purple')
    ax2.axvline(np.mean(errors), color='red', linestyle='--',
                label=f'Mean: {np.mean(errors):.1f}N')
    ax2.set_title('Soft Patch Force Error Distribution')
    ax2.set_xlabel('Force Error (N)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.set_xlim(-100, 100)
    ax2.grid(True, alpha=0.3)

    # Desired vs Measured
    ax3 = axes[1, 0]
    for leg in range(1, 5):
        leg_mask = legs == leg
        if np.sum(leg_mask) > 0:
            ax3.scatter(desired[leg_mask], measured[leg_mask],
                        c=colors[leg - 1], label=f'Leg {leg}', alpha=0.7, s=20)

    lims = [min(np.min(desired), np.min(measured)), max(np.max(desired), np.max(measured))]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Tracking')
    ax3.set_title('Soft Patch: Desired vs Measured Forces')
    ax3.set_xlabel('Desired Force (N)')
    ax3.set_ylabel('Measured Force (N)')
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, 50)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # RMS errors per leg
    ax4 = axes[1, 1]
    leg_rms_errors = []
    leg_labels = []

    for leg in range(1, 5):
        leg_mask = legs == leg
        if np.sum(leg_mask) > 0:
            leg_errors = errors[leg_mask]
            rms_error = np.sqrt(np.mean(leg_errors ** 2))
            leg_rms_errors.append(rms_error)
            leg_labels.append(f'Leg {leg}')

    if leg_rms_errors:
        x_pos = np.arange(len(leg_labels))
        bars = ax4.bar(x_pos, leg_rms_errors, alpha=0.7, color=colors[:len(leg_labels)])
        ax4.set_title('Soft Patch RMS Force Error by Leg')
        ax4.set_xlabel('Leg')
        ax4.set_ylabel('RMS Error (N)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(leg_labels)
        ax4.grid(True, alpha=0.3)

        for bar, value in zip(bars, leg_rms_errors):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{value:.1f}N', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_soft_patch_error_magnitudes(robot_id, soft_patch_data, p):
    """Plot magnitude of soft patch force errors"""
    errors = np.array(soft_patch_data['errors'])
    legs = np.array(soft_patch_data['leg'])
    iterations = np.array(soft_patch_data['iteration'])

    dt_simulation = p['Tmpc'] / p['steps']
    time_array = iterations * dt_simulation
    error_magnitudes = np.abs(errors)

    colors = ['red', 'blue', 'green', 'orange']

    # Individual leg plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, leg in enumerate(range(1, 5)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        leg_mask = legs == leg

        if np.sum(leg_mask) > 0:
            ax.plot(time_array[leg_mask], error_magnitudes[leg_mask],
                    c=colors[i], alpha=0.8, linewidth=1.5)
            ax.set_title(f'Leg {leg} - Force Error Magnitude', fontsize=16)
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_ylabel('|Force Error| (N)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 60)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_xlim(7, 11)

    plt.tight_layout()
    plt.show()

    # Combined plot
    fig2, ax = plt.subplots(figsize=(12, 6))

    for leg in range(1, 5):
        leg_mask = legs == leg
        if np.sum(leg_mask) > 0:
            ax.plot(time_array[leg_mask], error_magnitudes[leg_mask],
                    c=colors[leg - 1], label=f'Leg {leg}', alpha=0.8, linewidth=1.5)

    ax.set_title('Force Error Magnitude - All Legs')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Force Error| (N)')
    ax.set_ylim(0, 60)
    ax.set_xlim(6, 11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# STATE VISUALIZATION
# ============================================================================

def plot_simplified_states(measured_state_history, desired_state_history, robotId, p):
    """Plot position (x,y) and yaw angle vs time"""
    actual_states = np.array(measured_state_history[robotId])
    desired_states = np.array(desired_state_history[robotId])

    actual_x = actual_states[:, 0]
    actual_y = actual_states[:, 1]
    desired_x = desired_states[:, 0]
    desired_y = desired_states[:, 1]

    # Extract yaw from rotation matrices
    actual_yaw_angles = []
    desired_yaw_angles = []

    for i in range(len(actual_states)):
        R = actual_states[i, 6:15].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        actual_yaw_angles.append(np.degrees(yaw))

    for i in range(len(desired_states)):
        R = desired_states[i, 6:15].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        desired_yaw_angles.append(np.degrees(yaw))

    dt_actual = p['Tmpc'] / p['steps']
    dt_desired = p['Tmpc']

    actual_time = np.arange(len(actual_states)) * dt_actual
    desired_time = np.arange(len(desired_states)) * dt_desired

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10))

    # X Position
    axes[0].plot(actual_time, actual_x, 'b-', linewidth=1, label='Actual X', alpha=0.8)
    axes[0].plot(desired_time, desired_x, 'r-', linewidth=2, label='Desired X')
    axes[0].set_xlabel('Time (s)', fontsize=14)
    axes[0].set_ylabel('X Position (m)', fontsize=14)
    axes[0].set_title('Robot X Position', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=12)

    # Y Position
    axes[1].plot(actual_time, actual_y, 'b-', linewidth=1, label='Actual Y', alpha=0.8)
    axes[1].plot(desired_time, desired_y, 'r-', linewidth=2, label='Desired Y')
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_ylabel('Y Position (m)', fontsize=14)
    axes[1].set_title('Robot Y Position', fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=12)

    # Yaw
    axes[2].plot(actual_time, actual_yaw_angles, 'b-', linewidth=1, label='Actual Yaw', alpha=0.8)
    axes[2].plot(desired_time, desired_yaw_angles, 'r-', linewidth=2, label='Desired Yaw')
    axes[2].set_xlabel('Time (s)', fontsize=14)
    axes[2].set_ylabel('Yaw (degrees)', fontsize=14)
    axes[2].set_title('Robot Yaw Angle', fontsize=16)
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal State Values:")
    print(f"X Position: Actual = {actual_x[-1]:.3f} m, Desired = {desired_x[-1]:.3f} m")
    print(f"Y Position: Actual = {actual_y[-1]:.3f} m, Desired = {desired_y[-1]:.3f} m")
    print(f"Yaw Angle:  Actual = {actual_yaw_angles[-1]:.1f}°, "
          f"Desired = {desired_yaw_angles[-1]:.1f}°")


def quick_yaw_plot(measured_state_history, desired_state_history, robotId, p):
    """Plot yaw angle and rate tracking"""
    actual_states = np.array(measured_state_history[robotId])
    desired_states = np.array(desired_state_history[robotId])

    # Extract yaw angles
    actual_yaw_angles = []
    for i in range(len(actual_states)):
        R = actual_states[i, 6:15].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        actual_yaw_angles.append(yaw)

    desired_yaw_angles = []
    for i in range(len(desired_states)):
        R = desired_states[i, 6:15].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        desired_yaw_angles.append(yaw)

    dt_actual = p['Tmpc'] / p['steps']
    dt_desired = p['Tmpc']

    actual_time = np.arange(len(actual_yaw_angles)) * dt_actual
    desired_time = np.arange(len(desired_yaw_angles)) * dt_desired

    plt.figure(figsize=(15, 6))

    # Yaw angle
    plt.subplot(1, 2, 1)
    plt.plot(actual_time, actual_yaw_angles, 'b-', linewidth=1, label='Actual Yaw')
    plt.plot(desired_time, desired_yaw_angles, 'r-', linewidth=2, label='Desired Yaw')
    plt.xlabel('Time (seconds)', fontsize=18)
    plt.ylabel('Yaw (radians)', fontsize=18)
    plt.title('Robot Yaw Angle: Actual vs Desired', fontsize=18)
    plt.legend(fontsize=14)
    plt.ylim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Angular velocity
    plt.subplot(1, 2, 2)
    actual_omega_z = actual_states[:, 17]
    desired_omega_z = desired_states[:, 17]
    plt.plot(actual_time, actual_omega_z, 'b-', linewidth=1, label='Actual ω_z')
    plt.plot(desired_time, desired_omega_z, 'r-', linewidth=2, label='Desired ω_z')
    plt.xlabel('Time (seconds)', fontsize=18)
    plt.ylabel('Angular Velocity (radians/s)', fontsize=18)
    plt.title('Robot Yaw Rate: Actual vs Desired', fontsize=18)
    plt.legend(fontsize=14)
    plt.ylim(-1.75, 1.75)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()

    return actual_yaw_angles, desired_yaw_angles


# ============================================================================
# MPC PERFORMANCE
# ============================================================================

def plot_mpc_solve_times(mpc_solve_times):
    """Plot MPC optimization times"""
    solve_times_ms = np.array(mpc_solve_times) * 1000

    plt.figure(figsize=(10, 6))
    plt.plot(solve_times_ms, linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--', label='10ms control cycle limit')
    plt.xlabel('Control Iteration')
    plt.ylabel('MPC Solve Time (ms)')
    plt.title('MPC Optimization Time per Control Cycle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 12)
    plt.show()

    print(f"\nMPC Solve Time Statistics:")
    print(f"Average: {np.mean(solve_times_ms):.2f} ms")
    print(f"Max: {np.max(solve_times_ms):.2f} ms")
    print(f"Min: {np.min(solve_times_ms):.2f} ms")
    print(f"% cycles under 10ms: {np.sum(solve_times_ms < 10) / len(solve_times_ms) * 100:.1f}%")

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(solve_times_ms, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=12.5, color='r', linestyle='--', label='12.5ms limit')
    plt.axvline(x=np.mean(solve_times_ms), color='g', linestyle='-',
                label=f'Mean: {np.mean(solve_times_ms):.2f}ms')
    plt.xlabel('MPC Solve Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of MPC Solve Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# SINDY VISUALIZATION
# ============================================================================

def plot_sindy_results(sindy_result):
    """Plot SINDy model predictions vs actual errors"""
    y_actual = sindy_result['y_actual']
    y_pred = sindy_result['y_pred']
    state_index = sindy_result['state_index']
    r2 = sindy_result['r2']
    dt = sindy_result['dt']

    plt.figure(figsize=(12, 4))

    # Time series
    plt.subplot(1, 2, 1)
    time_actual = np.arange(len(y_actual)) * dt
    subsample = slice(None, None, max(1, len(time_actual) // 1250))
    plt.plot(time_actual[subsample], y_actual[subsample], label='Actual Error', alpha=0.7)
    plt.plot(time_actual[subsample], y_pred[subsample], label='SINDy Prediction', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel(r'$\dot{\omega}_z$ Error (rad/s²)', fontsize=18)
    plt.title(f'SINDy Error Prediction (R²={r2:.3f})', fontsize=18)
    plt.ylim(-100, 100)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_actual[::10], y_pred[::10], alpha=0.5, s=1, color='blue', label='Data Points')
    min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Match')
    plt.xlabel(r'Actual $\dot{\omega}_z$ Error (rad/s²)', fontsize=18)
    plt.ylabel(r'Predicted $\dot{\omega}_z$ Error (rad/s²)', fontsize=18)
    plt.title(f'Predicted vs Actual Errors', fontsize=18)
    plt.xlim(-50, 100)
    plt.ylim(-50, 100)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_polynomial_degrees(measured_state_history, measured_grf_history, robotId, p,
                               state_index, degrees=[1, 2, 3]):
    """Compare SINDy performance across different polynomial degrees"""
    print(f"\nComparing Polynomial Degrees for State {state_index}")
    print("=" * 60)

    results = {}

    for degree in degrees:
        print(f"\nTesting degree {degree}...")
        result = sindy.train_sindy_for_state(
            measured_state_history, measured_grf_history,
            robotId, p, state_index, degree=degree
        )
        results[degree] = result

        # Plot each result
        plot_sindy_results(result)

    print(f"\nComparison Summary:")
    print("-" * 60)
    for degree in degrees:
        r2 = results[degree]['r2']
        rmse = results[degree]['rmse']
        print(f"Degree {degree}: R² = {r2:.6f}, RMSE = {rmse:.6f}")

    best_degree = max(degrees, key=lambda d: results[d]['r2'])
    print(f"\nBest degree: {best_degree} (R² = {results[best_degree]['r2']:.6f})")

    return results


# ============================================================================
# MAIN PLOTTING SCRIPT
# ============================================================================

def main():
    """Load simulation results and generate all plots"""
    print("Loading simulation results...")
    with open('simulation_results.pkl', 'rb') as f:
        results = pickle.load(f)

    p = results['params']
    robotId = results['robotIds'][0]

    print("\n" + "="*60)
    print("Generating Analysis Plots")
    print("="*60)

    # Basic tracking plots
    plot_grf_history(robotId, results['U_history'][robotId],
                     results['measured_grf_history'][robotId])

    plot_motor_angles(robotId, results['simulation_time_history'],
                      results['motor_angle_history'][robotId])

    plot_state_tracking(robotId, results['mpc_time_history'],
                        results['robot_state_history'][robotId],
                        results['desired_state_history'][robotId])

    # Force analysis
    analyze_force_performance(robotId, results['U_history'][robotId],
                              results['measured_grf_history'][robotId])
    plot_force_analysis(robotId, results['U_history'][robotId],
                        results['measured_grf_history'][robotId])

    # Soft patch analysis
    analyze_soft_patch_forces(robotId, results['soft_patch_force_data'][robotId])
    plot_soft_patch_analysis(robotId, results['soft_patch_force_data'][robotId])
    plot_soft_patch_error_magnitudes(robotId, results['soft_patch_force_data'][robotId], p)

    # Position errors
    plot_position_errors(results['position_tracker'])

    # State visualization
    plot_simplified_states(results['measured_state_history'],
                           results['desired_state_history'], robotId, p)
    quick_yaw_plot(results['measured_state_history'],
                   results['desired_state_history'], robotId, p)

    # MPC performance
    plot_mpc_solve_times(results['mpc_solve_times'])

    # SINDy analysis
    sindy_results = compare_polynomial_degrees(
        results['measured_state_history'],
        results['measured_grf_history'],
        robotId, p, state_index=17, degrees=[1, 2, 3]
    )

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


if __name__ == '__main__':
    main()