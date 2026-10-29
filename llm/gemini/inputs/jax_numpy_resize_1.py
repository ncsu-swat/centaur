
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def resize_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, expanding
    a = np.array([1.0, 2.5, -3.2, 4.8], dtype=np.float32)
    new_shape = 8
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 2: 2D float32 array, truncating
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    new_shape = 4
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 3: 1D int32 array with negative values
    a = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    new_shape = 12
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 4: 3D float64 array
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    new_shape = 5
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 5: 2D boolean array
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    new_shape = 9
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 6: 1D complex64 array
    a = np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64)
    new_shape = 6
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 7: 0D scalar array (int64)
    a = np.array(42, dtype=np.int64)
    new_shape = 7
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 8: 4D uint8 array
    a = np.arange(16, dtype=np.uint8).reshape((2, 2, 2, 2))
    new_shape = 20
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 9: 1D float16 array, new_shape = 1
    a = np.array([0.5, -1.5, 2.5], dtype=np.float16)
    new_shape = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 10: 2D int8 array with negative and positive values
    a = np.array([[-128, -64], [0, 64], [127, -1]], dtype=np.int8)
    new_shape = 15
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    # Input 11: 3D float32 array, expanding to larger 1D shape
    a = np.ones((2, 2, 3), dtype=np.float32)
    new_shape = 30
    list_of_inputs.append(copy.deepcopy({"a": a, "new_shape": new_shape}))

    return list_of_inputs


generated_inputs["jax.numpy.resize_1"] = resize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.resize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.resize_1'.")


check_valid('jax.numpy.resize', generated_inputs['jax.numpy.resize_1'], lib="jax", suffix=1)
