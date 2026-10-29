
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isposinf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with inf, -inf, nan, and finite values
    x = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 2: 0D scalar float64 array with positive infinity
    x = np.array(np.inf, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 3: 0D scalar float32 array with finite value
    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 4: 2D float64 array with mixed infinite and finite values
    x = np.array(
        [[np.inf, -np.inf, 3.14], [0.0, np.nan, np.inf]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 5: 1D float16 array
    x = np.array([-np.inf, 0.0, np.inf, -5.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 6: 2D int32 array
    x = np.array([[-10, 0, 5], [100, -200, 300]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 7: 3D float32 array of shape (2, 3, 4)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    x[0, 1, 2] = np.inf
    x[1, 2, 0] = -np.inf
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 8: 4D float64 array
    x = np.full((2, 2, 2, 2), np.inf, dtype=np.float64)
    x[0, 0, 0, 0] = -np.inf
    x[1, 1, 1, 1] = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 9: 1D empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 10: 1D int64 array
    x = np.array([0, -1, 1, 9223372036854775807], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 11: 2D float64 array with NaNs
    x = np.full((3, 3), np.nan, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 12: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32)
    x[0, 0, 0, 1, 0] = np.inf
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    return list_of_inputs


generated_inputs["jax.numpy.isposinf_1"] = isposinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isposinf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isposinf_1'.")


check_valid('jax.numpy.isposinf', generated_inputs['jax.numpy.isposinf_1'], lib="jax", suffix=1)
