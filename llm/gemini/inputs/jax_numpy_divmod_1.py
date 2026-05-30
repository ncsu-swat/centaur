
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 arrays
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = np.array([3, 4, 7], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 with negative dividend values
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    x2 = np.array([3] * 11, dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 arrays
    x1 = np.array([6.0, 6.0, 6.0], dtype=np.float32)
    x2 = np.array([1.9, 2.5, 3.1], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, int64
    x1 = np.array([[12, 15], [18, 21]], dtype=np.int64)
    x2 = np.array([[5, 4], [3, 2]], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting, 2D array and 1D array
    x1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    x2 = np.array([3, 4, 5], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 arrays with random positive values
    x1 = np.random.uniform(10.0, 100.0, (2, 3, 4)).astype(np.float64)
    x2 = np.random.uniform(1.0, 10.0, (2, 3, 4)).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D arrays (scalars represented as tensors)
    x1 = np.array(45, dtype=np.int32)
    x2 = np.array(7, dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D arrays with negative divisor values
    x1 = np.array([10, -10, 10, -10], dtype=np.int32)
    x2 = np.array([3, 3, -3, -3], dtype=np.int32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer values (int64)
    x1 = np.array([10**12, 10**15], dtype=np.int64)
    x2 = np.array([3 * 10**10, 7 * 10**12], dtype=np.int64)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensors of float32
    x1 = np.random.uniform(5.0, 15.0, (2, 2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(1.0, 3.0, (2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.divmod_1"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_1'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_1'], lib="jax", suffix=1)
