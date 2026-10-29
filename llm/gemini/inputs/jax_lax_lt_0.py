
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_lax_lt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([-2.5, 0.0, 3.2, 5.0], dtype=np.float32)
    y = np.array([-1.0, 0.0, 2.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays with broadcasting (same number of dimensions)
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 2, 2]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Scalar (0-D array) and 2D float64 array
    x = np.array(0.0, dtype=np.float64)
    y = np.array([[-1.5, 0.5], [2.0, -3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D array and scalar (0-D array) int64
    x = np.array([[10, -5], [0, 8]], dtype=np.int64)
    y = np.array(5, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D float32 arrays with broadcast
    x = np.ones((2, 1, 4), dtype=np.float32)
    y = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D scalars comparison
    x = np.array(-10, dtype=np.int8)
    y = np.array(-5, dtype=np.int8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Boolean arrays
    x = np.array([False, True, False, True], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D float32 arrays
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Unsigned integer arrays uint8
    x = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    y = np.array([[1, 254], [128, 100]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Float16 with inf and nan values
    x = np.array([-np.inf, 0.0, np.nan, 1.0], dtype=np.float16)
    y = np.array([0.0, np.inf, 1.0, np.nan], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 1D int16 arrays with identical values
    x = np.array([100, -200, 300], dtype=np.int16)
    y = np.array([100, -200, 300], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.lt"] = jax_lax_lt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lt'.")


check_valid('jax.lax.lt', generated_inputs['jax.lax.lt'], lib="jax", suffix=0)
