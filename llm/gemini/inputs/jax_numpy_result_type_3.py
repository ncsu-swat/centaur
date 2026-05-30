
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    input_dict = {"args": np.random.randn(5).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array
    input_dict = {"args": np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array
    input_dict = {"args": np.random.randn(2, 2, 2).astype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D int64 array
    input_dict = {"args": np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D (scalar) float32 array
    input_dict = {"args": np.array(1.5, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array with negative values
    input_dict = {"args": np.array([-10, -20, -30], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float64 array with large values
    input_dict = {"args": np.array([[1e10, -1e10], [2e10, -2e10]], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D int32 array
    input_dict = {"args": np.zeros((1, 2, 1, 3, 1), dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array with zeros
    input_dict = {"args": np.zeros((10,), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int64 array
    input_dict = {"args": np.ones((2, 3, 4), dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
