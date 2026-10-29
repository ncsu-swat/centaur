
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hstack_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    tup = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.float32)})

    # Input 2: 2D int32 array with negative values
    tup = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.int32)})

    # Input 3: 1D float64 array
    tup = np.array([0.5, -1.5, 2.5, -3.5], dtype=np.float64)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.float64)})

    # Input 4: 3D float32 array
    tup = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.float32)})

    # Input 5: 2D int16 array
    tup = np.array([[-10, 20], [30, -40], [-50, 60]], dtype=np.int16)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.int16)})

    # Input 6: 4D float32 array
    tup = np.ones((2, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.float32)})

    # Input 7: 2D bool array
    tup = np.array([[True, False, True], [False, True, False]], dtype=bool)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(bool)})

    # Input 8: 3D int64 array with negative numbers
    tup = np.array(
        [[[1, -2], [3, 4]], [[-5, 6], [7, -8]], [[9, 10], [-11, 12]]],
        dtype=np.int64,
    )
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.int64)})

    # Input 9: 2D complex64 array
    tup = np.array([[1.0 + 2.0j, -3.0j], [4.0, 5.0 - 1.0j]], dtype=np.complex64)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.complex64)})

    # Input 10: 2D uint8 array with type promotion to uint32
    tup = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.uint32)})

    # Input 11: 1D int32 array with type conversion to int64
    tup = np.array([-100, 200, -300, 400, -500], dtype=np.int32)
    list_of_inputs.append({"tup": tup, "dtype": np.dtype(np.int64)})

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
