
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = np.array([3, 4, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D int32 arrays with negative values
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    x2 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float32 arrays
    x1 = np.array([[6.0, 6.0], [6.0, 6.0]], dtype=np.float32)
    x2 = np.array([[2.0, 2.5], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting (2D and 1D)
    x1 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    x2 = np.array([3, 7], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D arrays, float64
    x1 = np.random.uniform(10, 100, (2, 3, 4)).astype(np.float64)
    x2 = np.random.uniform(1, 10, (2, 3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D arrays (scalars as numpy arrays)
    x1 = np.array(15, dtype=np.int32)
    x2 = np.array(4, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Larger dimensions (4D)
    x1 = np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int32)
    x2 = np.random.randint(1, 10, size=(2, 2, 2, 2)).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Broadcasting with one dimension being 1
    x1 = np.random.randint(10, 50, size=(1, 5)).astype(np.int32)
    x2 = np.random.randint(1, 5, size=(5, 1)).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: float32 with negative values
    x1 = np.array([-10.5, -5.5, 5.5, 10.5], dtype=np.float32)
    x2 = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Large integer division (int64)
    x1 = np.array([100000000000, -200000000000], dtype=np.int64)
    x2 = np.array([3000000, 3000000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs

generated_inputs["jax.numpy.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_1'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_1'], lib="jax", suffix=1)
