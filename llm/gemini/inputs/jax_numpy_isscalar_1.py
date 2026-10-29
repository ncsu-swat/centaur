
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isscalar_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 array
    input_dict = {"element": np.array(3.14, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D negative int32 array
    input_dict = {"element": np.array(-42, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D boolean array
    input_dict = {"element": np.array(True, dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D complex64 array
    input_dict = {"element": np.array(1.0 + 2.0j, dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D float64 array with negative value
    input_dict = {"element": np.array(-0.005, dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with single element
    input_dict = {"element": np.array([10], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with multiple elements
    input_dict = {"element": np.array([-1.0, 0.0, 1.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty 1D array
    input_dict = {"element": np.array([], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array
    input_dict = {"element": np.zeros((3, 3), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array
    input_dict = {"element": np.ones((2, 4, 3), dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D random float32 array
    input_dict = {
        "element": np.random.randn(2, 3, 2, 2).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isscalar_1"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_1'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_1'], lib="jax", suffix=1)
