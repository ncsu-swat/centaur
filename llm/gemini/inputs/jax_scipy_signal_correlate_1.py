
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_correlate_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, mode='full', method='auto', precision='default'
    in1 = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    in2 = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    })

    # Input 2: 1D arrays with negative values, mode='same', method='direct', precision='high'
    in1 = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    in2 = np.array([1.0, -2.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    })

    # Input 3: 1D arrays, float64, mode='valid', method='fft', precision='highest'
    in1 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    in2 = np.array([0.1, 0.2], dtype=np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": "highest"
    })

    # Input 4: 2D arrays, mode='full', method='auto', precision='default'
    in1 = np.random.randn(4, 4).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    })

    # Input 5: 2D arrays, mode='same', method='direct', precision='high'
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    })

    # Input 6: 2D arrays, float64, mode='valid', method='fft', precision='highest'
    in1 = np.random.randn(6, 6).astype(np.float64)
    in2 = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": "highest"
    })

    # Input 7: 3D arrays, mode='full', method='auto', precision='default'
    in1 = np.random.randn(3, 3, 3).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    })

    # Input 8: 3D arrays, mode='same', method='direct', precision='high'
    in1 = np.random.randn(4, 4, 4).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    })

    # Input 9: 3D arrays, float64, mode='valid', method='fft', precision='highest'
    in1 = np.random.randn(5, 5, 5).astype(np.float64)
    in2 = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": "highest"
    })

    # Input 10: 4D arrays, mode='full', method='auto', precision='default'
    in1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    in2 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    })

    return list_of_inputs

generated_inputs["jax.scipy.signal.correlate_1"] = jax_scipy_signal_correlate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.correlate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.correlate_1'.")


check_valid('jax.scipy.signal.correlate', generated_inputs['jax.scipy.signal.correlate_1'], lib="jax", suffix=1)
