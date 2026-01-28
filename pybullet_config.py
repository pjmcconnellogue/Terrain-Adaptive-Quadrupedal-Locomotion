"""
PyBullet Interface for Quadruped Robot Simulation
Handles robot physics, terrain, control, and state management
"""

import pybullet as p
import pybullet_data
import numpy as np
import os


# ============================================================================
# SIMULATION SETUP
# ============================================================================

def setup_simulation(tMPC, steps):
    """
    Initialize PyBullet simulation with terrain, robot, and soft patch
    
    Args:
        tMPC: MPC timestep
        steps: Number of simulation sub-steps per MPC cycle
        
    Returns:
        robotId: Robot body ID
        terrain_id: Terrain body ID
        soft_patch_id: Soft patch body ID
    """
    physicsClient = p.connect(p.GUI, options="--opengl --width=800 --height=600")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set physics parameters
    p.setTimeStep(tMPC / steps)
    p.setGravity(0, 0, -9.81)

    # Create terrain
    terrain_id = _create_terrain()

    # Create soft patch
    soft_patch_id = _create_soft_patch()

    # Load robot
    robotId = _load_robot()

    return robotId, terrain_id, soft_patch_id


def _create_terrain():
    """Create heightfield terrain with slope and ruggedness"""
    num_rows = 128
    num_cols = 128
    terrain_width = 40.0
    terrain_length = 20.0
    slope_steepness = 0.15
    base_ruggedness_scale = 0.075

    # Generate height data
    y_coords = np.linspace(0, terrain_length, num_cols)
    slope_grid = np.tile(y_coords, (num_rows, 1)) * slope_steepness
    ruggedness_grid = (np.random.rand(num_rows, num_cols) - .1) * base_ruggedness_scale
    heightfield_data = (slope_grid + ruggedness_grid).flatten().tolist()

    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[terrain_width / num_cols, terrain_length / num_rows, 1.0],
        heightfieldData=heightfield_data,
        numHeightfieldRows=num_rows,
        numHeightfieldColumns=num_cols,
    )

    terrain_id = p.createMultiBody(0, terrain_shape)
    p.resetBasePositionAndOrientation(terrain_id, [0, 0, 0], [0, 0, 0, 1])
    p.changeDynamics(terrain_id, -1, lateralFriction=1)

    return terrain_id


def _create_soft_patch():
    """Create compliant soft patch on terrain"""
    soft_patch_half_extents = [10, 4, 0.0]
    patch_x_center = 110
    patch_y_center = 4.05
    soft_patch_base_z = 0.0025

    soft_patch_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=soft_patch_half_extents
    )

    soft_patch_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=soft_patch_shape,
        basePosition=[patch_x_center, patch_y_center, soft_patch_base_z],
        baseOrientation=[0, 0, 0, 1]
    )

    # Set compliant contact properties
    p.changeDynamics(
        soft_patch_id, -1,
        contactStiffness=1000,
        contactDamping=100,
        lateralFriction=1,
        restitution=0.01
    )

    return soft_patch_id


def _load_robot():
    """Load quadruped robot from URDF"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "urdf")
    urdf_file = os.path.join(urdf_path, "paws.urdf")
    robotId = p.loadURDF(urdf_file, [0, 0, 0.375], useFixedBase=False)
    return robotId


def disconnect():
    """Disconnect from PyBullet simulation"""
    p.disconnect()


# ============================================================================
# ROBOT STATE QUERIES
# ============================================================================

def get_robot_state(robotId):
    """
    Get complete robot state vector

    Returns:
        state_vector: [pos(3), vel(3), R_flat(9), angvel(3), foot_pos(12)]
                     Total: 30 elements
    """
    # Base state
    pos, rot_quat = p.getBasePositionAndOrientation(robotId)
    vel, angvel = p.getBaseVelocity(robotId)

    # Rotation matrix (flattened)
    rot_matrix = np.array(p.getMatrixFromQuaternion(rot_quat)).reshape(3, 3)
    rot_flat = rot_matrix.flatten().tolist()

    # Foot positions relative to CoM
    foot_link_indices = [3, 7, 11, 15]
    foot_positions = get_foot_positions_relative_to_com(robotId, foot_link_indices)

    # Construct state vector
    state_vector = list(pos) + list(vel) + rot_flat + list(angvel)

    for key in sorted(foot_positions.keys()):
        state_vector.extend(foot_positions[key])

    return state_vector


def get_base_rotation_matrix(robotId):
    """Get 3x3 rotation matrix from base orientation"""
    _, base_orientation = p.getBasePositionAndOrientation(robotId)
    rotation_matrix = np.array(p.getMatrixFromQuaternion(base_orientation)).reshape(3, 3)
    return rotation_matrix


def get_foot_positions_relative_to_com(robotId, foot_link_indices):
    """
    Get foot positions relative to robot CoM

    Returns:
        dict: {'foot_0': [x,y,z], 'foot_1': [x,y,z], ...}
    """
    com_position, _ = p.getBasePositionAndOrientation(robotId)
    foot_positions = {}

    for i, foot_link_id in enumerate(foot_link_indices):
        foot_world_position = p.getLinkState(robotId, foot_link_id)[0]
        relative_position = np.array(foot_world_position) - np.array(com_position)
        foot_positions[f"foot_{i}"] = relative_position

    return foot_positions


def get_foot_state_world(robotId, foot_link_index):
    """
    Get foot position and velocity in world frame

    Returns:
        position_world: [x, y, z]
        velocity_world: [vx, vy, vz]
    """
    link_state = p.getLinkState(
        bodyUniqueId=robotId,
        linkIndex=foot_link_index,
        computeLinkVelocity=1
    )

    position_world = np.array(link_state[0])
    velocity_world = np.array(link_state[6])

    return position_world, velocity_world


# ============================================================================
# GROUND CONTACT AND TERRAIN
# ============================================================================

def get_ground_height_for_CoM(robotId):
    """Get ground height at robot CoM using raycast for trajectory planning"""
    base_pos, _ = p.getBasePositionAndOrientation(robotId)
    zpos = base_pos[2]
    xpos = base_pos[0]
    ypos = base_pos[1] - 4

    ray_start = [xpos, ypos, zpos]
    ray_end = [xpos, ypos, -0.2]
    result = p.rayTest(ray_start, ray_end)[0]

    if result[0] != -1:
        if result[0] == 1:  # Hit robot itself
            return 0.0
        return result[3][2]
    else:
        return 0.0


def get_ground_height(x, y, z, robot_body_id):
    """Get ground height at specific XY location using raycast for foot placement"""
    ray_start = [x, y, z + 0.1]
    ray_end = [x, y, -0.5]
    result = p.rayTest(ray_start, ray_end)[0]

    if result[0] != -1:
        if result[0] == robot_body_id:  # Hit robot itself
            return 0.0
        return result[3][2]
    else:
        return 0.0


def get_grfs(robotId, plane_id, soft_patch_id, foot_indices):
    """
    Calculate ground reaction forces for each foot

    Returns:
        grf_world_list: Flattened array [12] of forces in world frame
                       [leg1_fx, leg1_fy, leg1_fz, leg2_fx, ...]
    """
    grf_per_leg_world = {idx: np.zeros(3) for idx in foot_indices}

    # Gather all contact points
    terrain_contacts = p.getContactPoints(bodyA=robotId, bodyB=plane_id)
    soft_patch_contacts = p.getContactPoints(bodyA=robotId, bodyB=soft_patch_id) if soft_patch_id is not None else []

    # Sum forces for each foot
    for point in terrain_contacts + soft_patch_contacts:
        foot_index = point[3]
        if foot_index in foot_indices:
            # Assemble total force vector
            normal_force_vec = np.array(point[7]) * point[9]
            friction_force_1_vec = np.array(point[11]) * point[10]
            friction_force_2_vec = np.array(point[13]) * point[12]
            grf_world = normal_force_vec + friction_force_1_vec + friction_force_2_vec

            grf_per_leg_world[foot_index] += grf_world

    # Return as flattened array
    grf_world_list = np.concatenate([grf_per_leg_world[idx] for idx in sorted(foot_indices)])

    return grf_world_list


# ============================================================================
# KINEMATICS AND JACOBIANS
# ============================================================================

def get_pybullet_jacobians(robotId, leg_joint_indices, foot_indices):
    """
    Calculate Jacobians for all legs

    Returns:
        np.array: Array of 3x3 Jacobians for each leg [4, 3, 3]
    """
    pb_jacobians = []

    # Get all joint states efficiently
    all_joint_states = p.getJointStates(robotId, leg_joint_indices)
    joint_positions = [state[0] for state in all_joint_states]
    joint_velocities = [state[1] for state in all_joint_states]
    zero_vec = [0] * len(joint_positions)

    for i, foot_idx in enumerate(foot_indices):
        foot_idx = foot_idx + 1

        pb_jacobian_lin, pb_jacobian_ang = p.calculateJacobian(
            robotId, foot_idx, [0, 0, 0], joint_positions, joint_velocities, zero_vec
        )

        pb_jacobian = np.vstack((pb_jacobian_lin, pb_jacobian_ang))

        # Extract 3x3 Jacobian for this leg
        start_col = 6 + 3 * i
        end_col = start_col + 3

        pb_jacobian_leg = pb_jacobian[:3, start_col:end_col]
        pb_jacobians.append(pb_jacobian_leg)

    return np.array(pb_jacobians)


def compute_joint_angles_to_anchor_point(robotId, R, end_effector):
    """
    Calculate joint angles to place foot at target position using IK
    Accounts for stance geometry and terrain height

    Args:
        robotId: Robot body ID
        R: 3x3 rotation matrix (flattened or 2D)
        end_effector: End effector link index

    Returns:
        joints: Tuple of 3 joint angles
        target_position: [x, y, z] target position in world frame
    """
    # Ensure R is proper 3x3 matrix
    R = np.array(R).reshape(3, 3)

    hip_index = end_effector - 1
    leg_index = int(((end_effector + 1) / 4) - 1)

    # Get hip position in world frame
    hip_position = np.array(p.getLinkState(robotId, hip_index)[0])

    # Stance geometry offsets in robot frame
    # Leg numbering: 0=FL, 1=FR, 2=RL, 3=RR
    if leg_index in [0, 1]:  # Front legs
        x_offset = 0.0
    else:  # Rear legs
        x_offset = -0.02

    if leg_index in [0, 2]:  # Left legs
        y_offset = -0.03
    else:  # Right legs
        y_offset = 0.03

    local_offset = np.array([x_offset, y_offset, 0])

    # Transform offset to world frame
    world_offset = R @ local_offset
    target_position_world = hip_position + world_offset

    # Get ground height at foot target location
    base_pos, _ = p.getBasePositionAndOrientation(robotId)
    zpos = base_pos[2]

    final_ground_height = get_ground_height(
        x=target_position_world[0],
        y=target_position_world[1],
        z=zpos - 0.075,
        robot_body_id=robotId
    )

    # Final target position with ground clearance
    target_position = np.array([
        target_position_world[0],
        target_position_world[1],
        final_ground_height + 0.025
    ])

    # Solve inverse kinematics
    joint_indices = [leg_index * 3, (leg_index * 3) + 1, (leg_index * 3) + 2]

    joint_angles = p.calculateInverseKinematics(
        bodyUniqueId=robotId,
        endEffectorLinkIndex=end_effector,
        targetPosition=target_position,
    )

    joints = tuple(joint_angles[i] for i in joint_indices)

    return joints, target_position


# ============================================================================
# MOTOR CONTROL
# ============================================================================

def set_joint_angles(robotId, joint_angles):
    """Set joint angles directly (for initialization)"""
    for joint_idx, angle in joint_angles.items():
        p.resetJointState(robotId, joint_idx, targetValue=angle)


def apply_pd_control(robotId, joint_angles_dict, kp=1, kd=0.5):
    """
    Apply PD position control to joints

    Args:
        robotId: Robot body ID
        joint_angles_dict: {joint_idx: target_angle, ...}
        kp: Proportional gain
        kd: Derivative gain
    """
    joint_indices = list(joint_angles_dict.keys())
    target_positions = list(joint_angles_dict.values())
    num_joints = len(joint_indices)

    p.setJointMotorControlArray(
        bodyUniqueId=robotId,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_positions,
        positionGains=[kp] * num_joints,
        velocityGains=[kd] * num_joints
    )


def fix_motor(joint_angles, robotId):
    """Lock motors at specific angles using high-stiffness position control"""
    for joint_index, target_angle in joint_angles.items():
        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=100
        )


def initiate_torque_control(robotId, leg_joint_indices):
    """Initialize joints for torque control by disabling internal motors"""
    for joint_idx in leg_joint_indices:
        p.setJointMotorControl2(robotId, joint_idx, p.VELOCITY_CONTROL, force=0)


def apply_torques(robotId, leg_joint_indices, tau):
    """
    Apply torque control to leg joints

    Args:
        robotId: Robot body ID
        leg_joint_indices: List of joint indices
        tau: Array of torques (same length as joint_indices)
    """
    for idx, joint_idx in enumerate(leg_joint_indices):
        p.setJointMotorControl2(robotId, joint_idx, p.TORQUE_CONTROL, force=tau[idx])


# ============================================================================
# SIMULATION STEPPING
# ============================================================================

def simulation_step():
    """Advance simulation by one timestep"""
    p.stepSimulation()