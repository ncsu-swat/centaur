
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def result_type_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    args = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 2: 2D int32 array with negative values
    args = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 3: 0D scalar float64 array
    args = np.array(-3.14159, dtype=np.float64)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 4: 1D int64 array with large values
    args = np.array([-(2**33), 0, 2**33], dtype=np.int64)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 5: 3D float32 array
    args = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 6: 2D float64 array
    args = np.array([[0.5, -1.5], [2.5, -3.5]], dtype=np.float64)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 7: 4D float32 array
    args = np.ones((1, 2, 2, 1), dtype=np.float32)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 8: 0D scalar int32 array
    args = np.array(42, dtype=np.int32)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 9: 1D float64 array
    args = np.linspace(-10.0, 10.0, num=15, dtype=np.float64)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 10: 2D int64 array
    args = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int64)
    list_of_inputs.append({"args": copy.deepcopy(args)})

    # Input 11: 3D int32 array
    args = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append({"args": copy.deepcopy(args)})

    return list_of_inputs


generated_inputs["jax.numpy.result_type_3"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_3'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_3'], lib="jax", suffix=3)
