#!/usr/bin/env python3
"""
RobustMPC-Pharma V2 - Main Application Entrypoint

This script is the heart of the deployed application. It loads the configuration,
initializes all components from the robust_mpc library, and runs the main
continuous control loop against the pharmaceutical granulation simulator.

Usage:
    python run_controller.py [--config config.yaml] [--steps 1000] [--no-realtime]

Author: PharmaControl-Pro Development Team
Version: 2.0.0
"""

import yaml
import time
import joblib
import torch
import numpy as np
import os
import sys
import argparse
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# --- Add project paths ---
# Updated paths to work with new V1/V2 directory structure
sys.path.insert(0, os.path.abspath('./robust_mpc'))
sys.path.insert(0, os.path.abspath('../V1/src/'))

# --- V2 Imports ---
try:
    from estimators import KalmanStateEstimator
    from models import ProbabilisticTransformer
    from optimizers import GeneticOptimizer
    from core import RobustMPCController
    print("V2 RobustMPC components loaded successfully")
except ImportError as e:
    print(f"Failed to import V2 components: {e}")
    print("Note: Full deployment requires trained models and all dependencies")
    sys.exit(1)

# --- V1 Imports (for the simulator) ---
try:
    from plant_simulator import AdvancedPlantSimulator
    print("V1 Plant simulator loaded successfully")
except ImportError as e:
    print(f"V1 simulator not available: {e}")
    print("Using mock simulator for demonstration")
    AdvancedPlantSimulator = None

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load the main configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict containing all configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)

def get_default_config() -> Dict[str, Any]:
    """Return default configuration if config.yaml is not available."""
    return {
        'files': {
            'model_path': './models/probabilistic_model.pth',
            'scalers_path': './data/scalers.joblib'
        },
        'process': {
            'cma_names': ['d50', 'lod'],
            'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
            'cpp_full_names': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
        },
        'model': {
            'hyperparameters': {
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.15
            }
        },
        'kalman_filter': {
            'process_noise_std': 1.0,
            'measurement_noise_std': 15.0
        },
        'mpc': {
            'lookback': 36,
            'horizon': 50,
            'integral_gain': 0.05,
            'risk_beta': 1.5,
            'mc_samples': 25,
            'cma_names': ['d50', 'lod'],
            'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
            'cpp_full_names': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
            'cpp_constraints': {
                'spray_rate': {'min_val': 80.0, 'max_val': 180.0},
                'air_flow': {'min_val': 400.0, 'max_val': 700.0},
                'carousel_speed': {'min_val': 20.0, 'max_val': 40.0}
            },
            'population_size': 40,
            'generations': 15,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7
        },
        'simulation': {
            'total_steps': 1000,
            'step_interval_seconds': 1.0,
            'initial_state': {'d50': 450.0, 'lod': 1.2},
            'initial_cpps': {'spray_rate': 100.0, 'air_flow': 450.0, 'carousel_speed': 25.0},
            'target_setpoint': {'d50': 380.0, 'lod': 1.8}
        }
    }

class MockPlantSimulator:
    """Mock simulator for demonstration when V1 simulator is not available."""
    
    def __init__(self, initial_state):
        self.state = initial_state.copy()
        self.time = 0
        
    def step(self, cpps):
        """Mock plant dynamics."""
        self.time += 1
        
        # Simple dynamics
        spray_effect = (cpps['spray_rate'] - 130) * 0.5
        air_effect = (cpps['air_flow'] - 550) * -0.02
        
        self.state['d50'] = max(300, min(500, 
            self.state['d50'] * 0.98 + spray_effect + np.random.normal(0, 2)))
        self.state['lod'] = max(0.5, min(3.0,
            self.state['lod'] * 0.95 + air_effect * 0.1 + np.random.normal(0, 0.05)))
            
        return self.state.copy()

def load_components(config: Dict[str, Any]):
    """
    Initialize and return all necessary components for the controller.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized RobustMPCController instance
    """
    print("Loading V2 RobustMPC Components...")
    
    # Device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load scalers (if available)
    scalers = None
    scalers_path = config['files']['scalers_path']
    if os.path.exists(scalers_path):
        try:
            scalers = joblib.load(scalers_path)
            print("Loaded data scalers")
        except Exception as e:
            print(f"Warning: Could not load scalers: {e}")
    else:
        print("Scalers file not found, using default preprocessing")

    # Initialize Predictive Model
    model_config = config['model']
    model = ProbabilisticTransformer(
        cma_features=len(config['process']['cma_names']),
        cpp_features=len(config['process']['cpp_full_names']),
        **model_config['hyperparameters']
    )
    
    # Load pre-trained model (if available)
    model_path = config['files']['model_path']
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ Loaded pre-trained probabilistic model")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            print("📋 Using randomly initialized model for demonstration")
    else:
        print("📋 Pre-trained model not found, using random initialization")
    
    model.to(device)
    model.eval()

    # Initialize Kalman Filter State Estimator
    # Note: In production, A and B matrices would be system-identified
    # For demonstration, we use simple identity/zero matrices
    n_states = len(config['process']['cma_names'])
    n_inputs = len(config['process']['cpp_names'])
    
    A = np.eye(n_states)  # Simple identity dynamics
    B = np.zeros((n_states, n_inputs))  # Zero direct input effect for demo
    initial_state = np.array(list(config['simulation']['initial_state'].values()))
    
    estimator = KalmanStateEstimator(
        transition_matrix=A,
        control_matrix=B,
        initial_state_mean=initial_state,
        **config['kalman_filter']
    )
    print("✅ Initialized Kalman Filter state estimator")

    # Initialize MPC Controller
    controller = RobustMPCController(
        model=model,
        estimator=estimator,
        optimizer_class=GeneticOptimizer,
        config=config['mpc'],
        scalers=scalers
    )
    print("✅ Initialized Robust MPC Controller")
    print("🎯 V2 system ready for deployment!")

    return controller

def run_control_loop(controller, config: Dict[str, Any], args):
    """
    Run the main control loop.
    
    Args:
        controller: Initialized RobustMPCController
        config: Configuration dictionary
        args: Command line arguments
    """
    # Initialize Plant Simulator
    if AdvancedPlantSimulator is not None:
        plant = AdvancedPlantSimulator(initial_state=config['simulation']['initial_state'])
        print("✅ Using V1 AdvancedPlantSimulator")
    else:
        plant = MockPlantSimulator(initial_state=config['simulation']['initial_state'])
        print("📋 Using mock plant simulator")
    
    # Initialize simulation state
    current_cpps = config['simulation']['initial_cpps'].copy()
    setpoint = np.array(list(config['simulation']['target_setpoint'].values()))
    
    print(f"Target Setpoint: d50={setpoint[0]:.1f}μm, LOD={setpoint[1]:.2f}%")
    print(f"Running for {args.steps} steps")
    print("STARTING REAL-TIME CONTROL LOOP")
    
    try:
        performance_log = []
        
        for t in range(args.steps):
            # 1. Get true state and create noisy measurement
            true_state_dict = plant.step(current_cpps)
            true_state_vec = np.array(list(true_state_dict.values()))
            
            # Add realistic measurement noise
            noise_std = np.array([8.0, 0.15])  # d50, LOD noise levels
            noisy_measurement = true_state_vec + np.random.normal(0, noise_std)
            
            # 2. Prepare control input vector
            control_input_vec = np.array(list(current_cpps.values()))

            # 3. Get new control action from MPC (after sufficient history)
            if t >= config['mpc']['lookback']:
                try:
                    suggested_action = controller.suggest_action(
                        measurement=noisy_measurement,
                        control_input=control_input_vec,
                        setpoint=setpoint
                    )
                    # Update control actions
                    current_cpps = dict(zip(config['process']['cpp_names'], suggested_action))
                except Exception as e:
                    print(f"⚠️  Controller error at step {t}: {e}")
                    # Continue with previous control action

            # 4. Calculate performance metrics
            tracking_error = np.abs(true_state_vec - setpoint)
            total_error = np.sum(tracking_error)
            
            # Log performance
            performance_log.append({
                'step': t,
                'true_state': true_state_vec.copy(),
                'noisy_measurement': noisy_measurement.copy(),
                'control_action': control_input_vec.copy(),
                'tracking_error': tracking_error.copy(),
                'total_error': total_error
            })

            # 5. Console output
            if t % 10 == 0 or t < 10:  # Print every 10 steps, plus first 10
                status = "LEARNING" if t < config['mpc']['lookback'] else "CONTROLLING"
                print(f"Step {t:04d} [{status:>12}] | "
                      f"State: d50={true_state_dict['d50']:6.1f}μm, LOD={true_state_dict['lod']:5.2f}% | "
                      f"Action: S={current_cpps['spray_rate']:5.1f}, A={current_cpps['air_flow']:5.1f}, C={current_cpps['carousel_speed']:4.1f} | "
                      f"Error: {total_error:6.2f}")

            # 6. Real-time delay (if enabled)
            if not args.no_realtime:
                time.sleep(config['simulation']['step_interval_seconds'])
                
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⏹️  Control loop terminated by user")
    
    # Performance summary
    if performance_log:
        final_errors = [entry['total_error'] for entry in performance_log[-50:]]
        avg_final_error = np.mean(final_errors)
        print("PERFORMANCE SUMMARY")
        print(f"Final Average Error (last 50 steps): {avg_final_error:.3f}")
        print(f"Total Steps Completed: {len(performance_log)}")
        
        if hasattr(controller, 'get_performance_metrics'):
            metrics = controller.get_performance_metrics()
            print(f"Total Control Actions: {metrics.get('total_control_actions', 'N/A')}")
            print(f"Average Optimization Time: {metrics.get('mean_optimization_time', 'N/A')}")
        
        print("V2 RobustMPC demonstration completed successfully")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RobustMPC-Pharma V2 Controller')
    parser.add_argument('--config', default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of control steps to run (overrides config)')
    parser.add_argument('--no-realtime', action='store_true',
                       help='Run simulation as fast as possible (no time delays)')
    return parser.parse_args()

def main():
    """Main application entry point."""
    print("RobustMPC-Pharma V2 - Industrial Control System")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override steps if provided via command line
    if args.steps is not None:
        config['simulation']['total_steps'] = args.steps
    
    # Load and initialize all components
    try:
        controller = load_components(config)
    except Exception as e:
        print(f"❌ Failed to initialize controller: {e}")
        print("📋 Check that all required files and dependencies are available")
        sys.exit(1)
    
    # Run the main control loop
    run_control_loop(controller, config, args)

if __name__ == '__main__':
    main()