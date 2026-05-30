
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_arrays_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    a = np.array([1, 2, -3], dtype=np.int32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array
    a = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array
    a = np.random.randn(3, 1, 4).astype(np.float64)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean array
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar
    a = np.array(5, dtype=np.int32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex array
    a = np.array([1 + 2j, -3 - 4j], dtype=np.complex64)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty/zero-sized 2D array
    a = np.empty((0, 5), dtype=np.float32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-dimensional float32 array
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D integer array with negative values
    a = np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array
    a = np.array([-1.5, 2.5, -3.5], dtype=np.float32)
    input_dict = {"args": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.broadcast_arrays"] = broadcast_arrays_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.broadcast_arrays' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.broadcast_arrays'.")


check_valid('jax.numpy.broadcast_arrays', generated_inputs['jax.numpy.broadcast_arrays'], lib="jax", suffix=0)
