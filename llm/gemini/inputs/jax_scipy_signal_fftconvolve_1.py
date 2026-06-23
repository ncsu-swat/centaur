
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftconvolve_inputs():
    list_of_inputs = []

    # All inputs use in1 of shape (5,) and in2 of shape (3,) to minimize JIT compilation overhead.
    
    # Mode: 'full'
    # Input 1: positive float32
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "full", "axes": [0]})

    # Input 2: negative float32
    in1 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    in2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "full", "axes": [0]})

    # Input 3: mixed float32
    in1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    in2 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "full", "axes": [0]})

    # Input 4: float64
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    in2 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "full", "axes": [0]})

    # Mode: 'same'
    # Input 5: positive float32
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "same", "axes": [0]})

    # Input 6: negative float32
    in1 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    in2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "same", "axes": [0]})

    # Input 7: mixed float32
    in1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    in2 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "same", "axes": [0]})

    # Mode: 'valid'
    # Input 8: positive float32
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "valid", "axes": [0]})

    # Input 9: negative float32
    in1 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    in2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "valid", "axes": [0]})

    # Input 10: mixed float32
    in1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    in2 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    list_of_inputs.append({"in1": in1, "in2": in2, "mode": "valid", "axes": [0]})

    return list_of_inputs

generated_inputs["jax.scipy.signal.fftconvolve_1"] = fftconvolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.fftconvolve_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.fftconvolve_1'.")


check_valid('jax.scipy.signal.fftconvolve', generated_inputs['jax.scipy.signal.fftconvolve_1'], lib="jax", suffix=1)
