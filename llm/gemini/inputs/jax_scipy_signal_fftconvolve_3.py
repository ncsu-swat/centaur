
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_fftconvolve_inputs():
    list_of_inputs = []

    # 1. 1D, full, float32
    in1 = np.random.randn(10).astype(np.float32)
    in2 = np.random.randn(3).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "axes": [0]
    })

    # 2. 1D, same, float64
    in1 = np.random.randn(12).astype(np.float64)
    in2 = np.random.randn(5).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "axes": [0]
    })

    # 3. 1D, valid, float32 with negative values
    in1 = np.random.uniform(-10.0, 10.0, size=(15,)).astype(np.float32)
    in2 = np.random.uniform(-5.0, 5.0, size=(4,)).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "axes": [0]
    })

    # 4. 2D, full, float32, convolve along all axes
    in1 = np.random.randn(8, 8).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "axes": [0, 1]
    })

    # 5. 2D, same, float32, convolve along single axis (axis 0)
    in1 = np.random.randn(10, 10).astype(np.float32)
    in2 = np.random.randn(4, 10).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "axes": [0]
    })

    # 6. 2D, valid, float64, convolve along single axis (axis 1)
    in1 = np.random.randn(6, 6).astype(np.float64)
    in2 = np.random.randn(6, 3).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "axes": [1]
    })

    # 7. 3D, full, float32, convolve along all axes
    in1 = np.random.randn(5, 5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3, 3).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "axes": [0, 1, 2]
    })

    # 8. 3D, same, float32, convolve along subset of axes
    in1 = np.random.randn(6, 6, 6).astype(np.float32)
    in2 = np.random.randn(6, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "axes": [1, 2]
    })

    # 9. 3D, valid, float64, convolve along subset of axes
    in1 = np.random.randn(8, 8, 8).astype(np.float64)
    in2 = np.random.randn(3, 8, 3).astype(np.float64)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "axes": [0, 2]
    })

    # 10. 4D, full, float32, convolve along subset of axes
    in1 = np.random.randn(4, 4, 4, 4).astype(np.float32)
    in2 = np.random.randn(2, 4, 2, 4).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "axes": [0, 2]
    })

    # 11. 2D, same, integer-like float32 values
    in1 = np.random.randint(-10, 10, size=(12, 12)).astype(np.float32)
    in2 = np.random.randint(-5, 5, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "axes": [0, 1]
    })

    return list_of_inputs

generated_inputs["jax.scipy.signal.fftconvolve_3"] = jax_scipy_signal_fftconvolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.fftconvolve_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.fftconvolve_3'.")


check_valid('jax.scipy.signal.fftconvolve', generated_inputs['jax.scipy.signal.fftconvolve_3'], lib="jax", suffix=3)
