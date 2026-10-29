
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def fftconvolve_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, full mode, axes=0
    in1 = np.array([1.0, 2.0, -1.0, 3.0], dtype=np.float32)
    in2 = np.array([0.5, -0.5], dtype=np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, same mode, axes=0
    in1 = np.array([-2.5, 1.0, 3.2, -0.8, 4.0], dtype=np.float64)
    in2 = np.array([1.0, -1.0, 2.0], dtype=np.float64)
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array, valid mode, axes=-1
    in1 = np.array([1, 4, 9, 16, 25, 36], dtype=np.int32)
    in2 = np.array([1, 2, 1], dtype=np.int32)
    input_dict = {"in1": in1, "in2": in2, "mode": "valid", "axes": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, full mode, axes=0
    np.random.seed(42)
    in1 = np.random.randn(8, 6).astype(np.float32)
    in2 = np.random.randn(3, 6).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, same mode, axes=1
    in1 = np.random.randn(5, 10).astype(np.float32)
    in2 = np.random.randn(5, 4).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array, valid mode, axes=-1
    in1 = np.random.randn(6, 12).astype(np.float64)
    in2 = np.random.randn(6, 5).astype(np.float64)
    input_dict = {"in1": in1, "in2": in2, "mode": "valid", "axes": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, full mode, axes=1
    in1 = np.random.randn(4, 8, 5).astype(np.float32)
    in2 = np.random.randn(4, 3, 5).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex64 array, same mode, axes=2
    in1 = (np.random.randn(2, 3, 6) + 1j * np.random.randn(2, 3, 6)).astype(
        np.complex64
    )
    in2 = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(
        np.complex64
    )
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, valid mode, negative axis axes=-2
    in1 = np.random.randn(7, 5).astype(np.float32)
    in2 = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "valid", "axes": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, linspace, full mode, axes=-1
    in1 = np.linspace(-5.0, 5.0, 20, dtype=np.float32)
    in2 = np.array([0.2, 0.5, 0.2], dtype=np.float32)
    input_dict = {"in1": in1, "in2": in2, "mode": "full", "axes": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float64 array, same mode, axes=0
    in1 = np.random.randn(10, 4, 3).astype(np.float64)
    in2 = np.random.randn(4, 4, 3).astype(np.float64)
    input_dict = {"in1": in1, "in2": in2, "mode": "same", "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
