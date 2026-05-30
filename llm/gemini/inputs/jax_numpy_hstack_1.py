
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hstack_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, dtype float32
    tup = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float32})

    # Input 2: 2D array, int32, dtype int32
    tup = np.arange(8).reshape(4, 2).astype(np.int32)
    list_of_inputs.append({"tup": tup, "dtype": np.int32})

    # Input 3: 3D array, float64, dtype float64
    tup = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"tup": tup, "dtype": np.float64})

    # Input 4: 3D array, int64, dtype int64
    tup = np.arange(12).reshape(3, 2, 2).astype(np.int64)
    list_of_inputs.append({"tup": tup, "dtype": np.int64})

    # Input 5: 4D array, float32, dtype float32
    tup = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float32})

    # Input 6: Large 2D array, float32, dtype float32
    tup = np.ones((10, 20), dtype=np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float32})

    # Input 7: 2D array with negative values, float32, dtype float64
    tup = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float64})

    # Input 8: 2D array, int32, dtype int32
    tup = np.array([[10, 20, 30]], dtype=np.int32)
    list_of_inputs.append({"tup": tup, "dtype": np.int32})

    # Input 9: 3D array, float32, dtype float32
    tup = np.ones((4, 1, 3), dtype=np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.float32})

    # Input 10: 2D boolean array, bool, dtype bool
    tup = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append({"tup": tup, "dtype": np.bool_})

    # Input 11: 2D complex array, complex128, dtype complex128
    tup = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
    list_of_inputs.append({"tup": tup, "dtype": np.complex128})

    return list_of_inputs

generated_inputs["jax.numpy.hstack_1"] = hstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hstack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hstack_1'.")


check_valid('jax.numpy.hstack', generated_inputs['jax.numpy.hstack_1'], lib="jax", suffix=1)
