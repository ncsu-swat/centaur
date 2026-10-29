
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isinf_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive/negative infinity, nan, and finite values (float32)
    x = np.array([np.inf, -np.inf, np.nan, 0.0, 1.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar array with infinity (float64)
    x = np.array(np.inf, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float64 containing inf and finite values
    x = np.array(
        [[1.0, np.inf], [-np.inf, 2.5], [np.nan, -0.0]], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D integer array (int32)
    x = np.array([-10, 0, 5, 100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of float32 with inf and finite values
    x = np.array(
        [
            [[np.inf, -1.0], [0.0, -np.inf]],
            [[np.nan, 3.14], [-np.inf, np.inf]],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D complex numbers array (complex64)
    x = np.array(
        [2 + 3j, complex(np.inf, 1), complex(0, -np.inf), 0 + 0j],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex numbers array (complex128)
    x = np.array(
        [
            [complex(np.inf, np.nan), complex(1.0, 2.0)],
            [complex(-np.inf, 0.0), complex(0.0, np.inf)],
        ],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty array of float32
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D float32 array
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x[0, 1, 2, 3] = np.inf
    x[1, 2, 0, 4] = -np.inf
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Boolean array (bool dtype)
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D int64 array
    x = np.array([-1000, 0, 1000, 2147483648], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.isinf_1"] = isinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isinf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isinf_1'.")


check_valid('jax.numpy.isinf', generated_inputs['jax.numpy.isinf_1'], lib="jax", suffix=1)
