import torch
import os
import yaml
import numpy as np
from datetime import datetime
import random
from pathlib import Path


def get_short_names_ml_signals(use_operational_switching: bool = True, use_abnormal_event: bool = True,
                               use_emergency_event: bool = True) -> list:
    """
    This function returns a set of short names ML signals for (without i_bus).

    Args:
        use_operational_switching (bool): Include operational switching signals.
        use_abnormal_event (bool): Include abnormal event signals.
        use_emergency_event (bool): Include emergency event signals.

    Returns:
        list: A list of ML signals for the given bus.
    """
    # FIXME: rewrite so that it is recorded at the very beginning and counted 1 time, and not at every request

    ml_operational_switching = [
        # --- Working switching ---
        'ML_1',  # Working switching, without specification
        'ML_1_1',  # Operational activation, without specification
        'ML_1_1_1',  # Operating start-up, engine start-up
        'ML_1_2'  # Operational shutdown, without specification
    ]
    ml_abnormal_event = [
        # --- Abnormal events
        'ML_2',      # Anomaly, without clarification
        'ML_2_1',    # Single phase-to-ground fault, without specification
        'ML_2_1_1',  # Sustainable single phase-to-ground fault
        'ML_2_1_2',  # Steady attenuating single phase-to-ground fault, with rare breakouts
        'ML_2_1_3',  # Arc intermittent single phase-to-ground fault
        'ML_2_2',    # Damping fluctuations from emergency processes
        'ML_2_3',    # Voltage drawdown
        'ML_2_3_1',  # Voltage drawdown when starting the engine
        'ML_2_4',    # Current fluctuations, without specification
        'ML_2_4_1',  # Current fluctuations when starting the engine
        'ML_2_4_2',  # Current fluctuations from frequency-driven motors
        'ML_2_5_1',  # Voltage fluctuations
        'ML_2_6',    # Tests
        'ML_2_7_1',  # Reversed voltage phases
        'ML_2_7_2',  # Reversed current phases
    ]

    ml_emergency_event = [
        # --- Emergency events ----
        'ML_3',    # Emergency events, without clarification
        'ML_3_1',  # An accident due to incorrect operation of the device, without clarification
        'ML_3_2',  # Terminal malfunction
        'ML_3_3',  # Two-phase earth fault
        'ML_3_4',  # Phase break in voltage circuits
        'ML_3_5',  # Signal noise resulting in failure to operate
    ]

    ml_signals = []
    if use_operational_switching:
        ml_signals.extend(ml_operational_switching)
    if use_abnormal_event:
        ml_signals.extend(ml_abnormal_event)
    if use_emergency_event:
        ml_signals.extend(ml_emergency_event)

    return ml_signals, ml_operational_switching, ml_abnormal_event, ml_emergency_event

def get_short_names_ml_analog_signals() -> list:
    """
    This function returns a set of short names ML analog signals for (without i_bus).

    Args:

    Returns:
        list: A set of ML signals for the given bus.
    """

    ml_current = [
        'IA', 'IB', 'IC', 'IN'
    ]
    ml_votage_BB = [
        'UA BB', 'UB BB', 'UC BB', 'UN BB', 'UAB BB', 'UBC BB', 'UCA BB',
    ]
    ml_votage_CL = [
        'UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL',
    ]
    # TODO: signals I_raw, U_raw, I|dif-1, I | braking-1 are not taken into account

    ml_signals = []
    ml_signals.extend(ml_current)
    ml_signals.extend(ml_votage_BB)
    ml_signals.extend(ml_votage_CL)

    return ml_signals


def get_ml_signals(i_bus, use_operational_switching=True, use_abnormal_event=True, use_emergency_event=True):
    """
    This function returns a set of ML signals for a given bus.

    Args:
        i_bus (str): The bus number.
        use_operational_switching (bool): Include operational switching signals.
        use_abnormal_event (bool): Include abnormal event signals.
        use_emergency_event (bool): Include emergency event signals.

    Returns:
        set: A set of ML signals for the given bus.
    """
    # FIXME: rewrite so that it is recorded at the very beginning and counted 1 time, and not at every request
    ml_operational_switching = {
        # --- Working switching ---
        f'MLsignal_{i_bus}_1',  # Working switching, without specification
        f'MLsignal_{i_bus}_1_1',  # Operational activation, without specification
        f'MLsignal_{i_bus}_1_1_1',  # Operating start-up, engine start-up
        f'MLsignal_{i_bus}_1_2',  # Operational shutdown, without specification
    }

    ml_abnormal_event = {
        # --- Abnormal events
        f'MLsignal_{i_bus}_2',      # Anomaly, without clarification
        f'MLsignal_{i_bus}_2_1',    # Single phase-to-ground fault, without specification
        f'MLsignal_{i_bus}_2_1_1',  # Sustainable single phase-to-ground fault
        f'MLsignal_{i_bus}_2_1_2',  # Steady attenuating single phase-to-ground fault, with rare breakouts
        f'MLsignal_{i_bus}_2_1_3',  # Arc intermittent single phase-to-ground fault
        f'MLsignal_{i_bus}_2_2',    # Damping fluctuations from emergency processes
        f'MLsignal_{i_bus}_2_3',    # Voltage drawdown
        f'MLsignal_{i_bus}_2_3_1',  # Voltage drawdown when starting the engine
        f'MLsignal_{i_bus}_2_4',    # Current fluctuations, without specification
        f'MLsignal_{i_bus}_2_4_1',  # Current fluctuations when starting the engine
        f'MLsignal_{i_bus}_2_4_2',  # Current fluctuations from frequency-driven motors
        f'MLsignal_{i_bus}_2_5_1',  # Voltage fluctuations
        f'MLsignal_{i_bus}_2_6',    # Tests
        f'MLsignal_{i_bus}_2_7_1',  # Reversed voltage phases
        f'MLsignal_{i_bus}_2_7_2',  # Reversed current phases
    }

    ml_emergency_event = {
        # --- Emergency events ----
        f'MLsignal_{i_bus}_3',    # Emergency events, without clarification
        f'MLsignal_{i_bus}_3_1',  # An accident due to incorrect operation of the device, without clarification
        f'MLsignal_{i_bus}_3_2',  # Terminal malfunction
        f'MLsignal_{i_bus}_3_3',  # Two-phase earth fault
        f'MLsignal_{i_bus}_3_4',  # Phase break in voltage circuits
        f'MLsignal_{i_bus}_3_5'   # Signal noise resulting in failure to operate
    }

    ml_signals = set()
    if use_operational_switching:
        ml_signals.update(ml_operational_switching)
    if use_abnormal_event:
        ml_signals.update(ml_abnormal_event)
    if use_emergency_event:
        ml_signals.update(ml_emergency_event)

    return ml_signals

def get_available_device():
    """Get available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_experiment_dir(base_dir, experiment_name=None):
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{experiment_name}_{timestamp}" if experiment_name else f"experiment_{timestamp}"
    exp_dir = Path(base_dir) / exp_name

    # Create subdirectories
    os.makedirs(exp_dir / "checkpoints", exist_ok=True)
    os.makedirs(exp_dir / "metrics", exist_ok=True)

    return exp_dir

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)