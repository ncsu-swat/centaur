
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_convolve_inputs():
    list_of_inputs = []

    # Input 1: 1D, full, auto, small size
    in1 = np.array([1., 2., 3.], dtype=np.float32)
    in2 = np.array([0.5, 1.0], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": (1.0, 1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D, same, direct, small size
    in1 = np.array([-1., 2., -3., 4.], dtype=np.float32)
    in2 = np.array([1., -1.], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": (1.0, 1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D, valid, fft, small size
    in1 = np.array([1., 2., 3., 4.], dtype=np.float32)
    in2 = np.array([1., 1.], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "valid",
        "method": "fft",
        "precision": (1.0, 1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D, full, direct, small size
    in1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    in2 = np.array([[0.5, 0.5]], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "direct",
        "precision": (1.0, 1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D, same, fft, small size
    in1 = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    in2 = np.array([[1., 0.], [0., -1.]], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "fft",
        "precision": (1.0, 1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.convolve_2"] = jax_scipy_signal_convolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve_2'.")


check_valid('jax.scipy.signal.convolve', generated_inputs['jax.scipy.signal.convolve_2'], lib="jax", suffix=2)
