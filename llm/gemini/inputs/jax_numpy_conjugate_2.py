
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conjugate_inputs():
    list_of_inputs = []

    # Input 1: Python int
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python negative int
    input_dict = {"x": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: np.int32 scalar
    input_dict = {"x": np.int32(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.int64 scalar
    input_dict = {"x": np.int64(-999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of int32
    input_dict = {"x": np.array([1, -2, 3, -4], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of int64
    input_dict = {"x": np.array([[1, 2], [-3, -4]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array of int16
    input_dict = {"x": np.ones((2, 3, 4), dtype=np.int16) * -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of uint8
    input_dict = {"x": np.array([0, 128, 255], dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array of int8
    input_dict = {"x": np.zeros((2, 2, 2, 2), dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 0D array of int32
    input_dict = {"x": np.array(-15, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty array of int64
    input_dict = {"x": np.array([], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.conjugate_2"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_2'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_2'], lib="jax", suffix=2)
