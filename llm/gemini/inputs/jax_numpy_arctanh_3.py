
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctanh_inputs():
    list_of_inputs = []

    # Input 1: Scalar int32 (value: 0)
    input_dict = {"x": np.int32(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar int64 (value: 1)
    input_dict = {"x": np.int64(1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar int16 (value: -1)
    input_dict = {"x": np.int16(-1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of int32 containing values inside [-1, 1]
    input_dict = {"x": np.array([-1, 0, 1], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of int64 containing values inside and outside [-1, 1]
    input_dict = {"x": np.array([-2, -1, 0, 1, 2], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array of int16
    input_dict = {"x": np.array([[0, 1], [-1, 0]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of int8 with larger integers
    input_dict = {"x": np.array([[5, -3], [12, -1]], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of uint32
    input_dict = {"x": np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array of int64
    input_dict = {"x": np.arange(-4, 4, dtype=np.int64).reshape(2, 2, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of int32
    input_dict = {"x": np.random.randint(-10, 10, size=(2, 2, 2, 2), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.arctanh_3"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctanh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctanh_3'.")


check_valid('jax.numpy.arctanh', generated_inputs['jax.numpy.arctanh_3'], lib="jax", suffix=3)
