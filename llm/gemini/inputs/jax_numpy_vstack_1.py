
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vstack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array (unstacked into scalars and vertically stacked)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {"tup": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative values
    a = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    input_dict = {"tup": a, "dtype": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    a = np.random.randn(4, 5).astype(np.float64)
    input_dict = {"tup": a, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D boolean array
    a = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    input_dict = {"tup": a, "dtype": np.bool_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    a = np.random.randn(3, 4, 2).astype(np.float32)
    input_dict = {"tup": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array
    a = np.array([-10, 0, 10, 20], dtype=np.int64)
    input_dict = {"tup": a, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float16 array cast to float32
    a = np.array([[1.5, -2.5], [0.0, 3.5]], dtype=np.float16)
    input_dict = {"tup": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array
    a = np.ones((2, 3, 2, 2), dtype=np.float32)
    input_dict = {"tup": a, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64 array
    a = np.array([[1.0 + 2.0j, -3.0 + 4.0j], [5.0 - 1.0j, 0.0 + 2.0j]], dtype=np.complex64)
    input_dict = {"tup": a, "dtype": np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D uint8 array
    a = np.full((3, 3), 255, dtype=np.uint8)
    input_dict = {"tup": a, "dtype": np.uint8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.vstack_1"] = vstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vstack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vstack_1'.")


check_valid('jax.numpy.vstack', generated_inputs['jax.numpy.vstack_1'], lib="jax", suffix=1)
