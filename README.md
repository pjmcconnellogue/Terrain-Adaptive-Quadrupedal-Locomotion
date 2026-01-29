# Terrain-Adaptive Quadrupedal Locomotion

Terrain-Adaptive Quadrupedal Locomotion Using Model Predictive Control with Impedance Control and SINDy Model Correction

## Overview

This implementation features:
- **Hierarchical control**: MPC layer + Impedance control
- **Terrain-adaptive compliance**: Admittance control with contact state detection
- **SINDy dynamics correction**: Data-driven model refinement for angular dynamics

## Demo

![Robot Navigating Rugged Slope](Rugged_Slope_Trotting.gif)

*Quadruped robot climbing rugged slope with terrain-adaptive control*

## Requirements

**System:**
- Python 3.8+
- Ubuntu 20.04+ (or equivalent Linux distribution)

**Python Packages:**
- numpy, scipy, matplotlib
- casadi
- pybullet
- pysindy, scikit-learn
- acados_template

**Acados:**
- Requires compilation from source
- Installation guide: https://docs.acados.org/installation/
- Set `ACADOS_SOURCE_DIR` environment variable after installation

## Installation
```bash
# Install Python packages
pip install numpy scipy matplotlib casadi pybullet pysindy scikit-learn

# Install Acados (see https://docs.acados.org/installation/)
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
export ACADOS_SOURCE_DIR="/path/to/acados"

# Install Acados Python interface
cd $ACADOS_SOURCE_DIR/interfaces/acados_template
pip install -e .
```

## Usage
```bash
# Run simulation
python3 main.py

# Generate analysis plots (requires simulation_results.pkl from main.py)
python3 plot.py
```

## File Structure

- `main.py` - Main simulation loop with MPC and impedance control
- `pybullet_config.py` - PyBullet physics interface and terrain setup
- `acados.py` - MPC formulation and Acados solver configuration
- `sindy.py` - SINDy model training and dynamics correction identification
- `plot.py` - Visualization and performance analysis
- `urdf/` - Robot URDF and mesh files

## Key Parameters

### MPC Configuration (main.py)
```python
Tmpc = 0.0125          # MPC timestep (12.5ms)
predHorizon = 8        # Prediction horizon steps
steps = 8              # Integration sub-steps per MPC cycle
```

### Contact State Machine
- **Seeking** (0): Searching for contact - high compliance
- **Light** (1): Initial contact detected - moderate compliance
- **Adjusting** (2): Stabilizing contact - low compliance
- **Solid** (3): Firm contact established - stiff control

Impedance gains and admittance parameters vary by contact state.

## Algorithm Overview

**Control Architecture:**
- **MPC Layer**: Computes optimal ground reaction forces over prediction horizon
- **Impedance/Admittance Control**: Tracks foot positions with adaptive compliance and force-error feedback
- **Force Mapping**: Jacobian-based conversion of desired forces to joint torques

**SINDy Dynamics Correction:**
- Data-driven corrections applied to angular acceleration dynamics
- Trained offline from simulation data

## Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'acados_template'`**
- Ensure Acados Python interface is installed: `cd $ACADOS_SOURCE_DIR/interfaces/acados_template && pip install -e .`

**Issue: `libacados.so: cannot open shared object file`**
- Add to `~/.bashrc`: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${ACADOS_SOURCE_DIR}/lib"`
