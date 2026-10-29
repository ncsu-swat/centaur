
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftconvolve_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, full mode, float32
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, same mode, float64 with negative values
    in1 = np.array([-2.5, 1.0, -0.5, 3.2, -1.8, 4.0], dtype=np.float64)
    in2 = np.array([0.2, -0.1, 0.2], dtype=np.float64)
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays, valid mode
    in1 = np.arange(10, dtype=np.float32)
    in2 = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "valid", "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, full mode, both axes
    np.random.seed(42)
    in1 = np.random.randn(5, 5).astype(np.float32)
    in2 = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D arrays, same mode, single axis convolved
    in1 = np.random.randn(6, 8).astype(np.float32)
    in2 = np.random.randn(6, 3).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D arrays, valid mode
    in1 = np.random.randn(8, 8).astype(np.float64)
    in2 = np.random.randn(3, 4).astype(np.float64)
    input_dict = {"in1": in1, "in2": in2, "mode": "valid", "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, full mode, all axes
    in1 = np.random.randn(4, 4, 4).astype(np.float32)
    in2 = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": (0, 1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays, same mode, subset of axes
    in1 = np.random.randn(3, 6, 6).astype(np.float32)
    in2 = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": (1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays, full mode, last two axes
    in1 = np.random.randn(2, 2, 8, 8).astype(np.float32)
    in2 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": (2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex arrays, full mode
    in1 = (
        np.random.randn(4, 4).astype(np.float32)
        + 1j * np.random.randn(4, 4).astype(np.float32)
    ).astype(np.complex64)
    in2 = (
        np.random.randn(2, 2).astype(np.float32)
        + 1j * np.random.randn(2, 2).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.signal.fftconvolve_2"] = fftconvolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.fftconvolve_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.fftconvolve_2'.")


check_valid('jax.scipy.signal.fftconvolve', generated_inputs['jax.scipy.signal.fftconvolve_2'], lib="jax", suffix=2)
