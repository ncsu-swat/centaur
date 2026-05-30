
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def less_inputs():
    list_of_inputs = []

    # Input 1: Scalar int32
    x = np.int32(5)
    y = np.int32(10)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 2: Scalar int64 with negative values
    x = np.int64(-3)
    y = np.int64(-1)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: 1D int32 arrays
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([3, 2, 1], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: 1D int64 arrays
    x = np.array([-5, 0, 5], dtype=np.int64)
    y = np.array([-6, 0, 6], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: 2D int32 arrays with broadcasting
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: 2D int64 arrays
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[4, 3], [2, 1]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: 3D int8 arrays
    x = np.arange(8, dtype=np.int8).reshape(2, 2, 2)
    y = (np.ones((2, 2, 2)) * 4).astype(np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: 1D uint32 arrays
    x = np.array([0, 10, 20], dtype=np.uint32)
    y = np.array([5, 10, 15], dtype=np.uint32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: 0D arrays
    x = np.array(5, dtype=np.int32)
    y = np.array(10, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: 1D int16 arrays with negative and positive limits
    x = np.array([-32768, 32767], dtype=np.int16)
    y = np.array([0, 0], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.less_7"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_7'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_7'], lib="jax", suffix=7)
