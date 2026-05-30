
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cbrt_inputs():
    list_of_inputs = []

    # Input 1: Scalar integer
    input_dict = {"x": np.int32(8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of positive integers (perfect cubes)
    input_dict = {"x": np.array([1, 8, 27, 64, 125], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of negative and positive integers (perfect cubes)
    input_dict = {"x": np.array([-1, -8, -27, 0, 1, 8, 27], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of integers (int32)
    input_dict = {"x": np.array([[216, 125, 64], [-27, -8, -1]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of integers (int16)
    input_dict = {"x": np.array([[[1, 8], [27, 64]], [[-1, -8], [-27, -64]]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with larger integers (int64)
    input_dict = {"x": np.array([1000000, -1000000, 8000000, -8000000], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of small integers (int8)
    input_dict = {"x": np.array([[-125, -64, -27], [27, 64, 125]], dtype=np.int8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array of unsigned integers (uint32)
    input_dict = {"x": np.array([[[[1, 8], [27, 64]]]], dtype=np.uint32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D array (scalar array) of int64
    input_dict = {"x": np.array(-343, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of random integers (int32)
    input_dict = {"x": np.random.randint(-1000, 1000, size=(100,)).astype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cbrt_3"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_3'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_3'], lib="jax", suffix=3)
