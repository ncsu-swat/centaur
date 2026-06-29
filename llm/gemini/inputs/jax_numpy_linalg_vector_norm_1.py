
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard L2 norm
    x = np.random.randn(5).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 0,
        "keepdims": False,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, L1 norm along axis 1, keeping dimensions
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 1,
        "keepdims": True,
        "ord": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, negative values, L-inf norm (represented as ord=-1 is min absolute value, wait ord=np.inf is usually inf but signature says integer, so let's use ord=-1)
    x = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "x": x,
        "axis": -1,
        "keepdims": False,
        "ord": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array with negative and positive values, ord=-2
    x = (np.random.randn(4, 5) * 10).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 0,
        "keepdims": True,
        "ord": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, L2 norm along axis 2
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 2,
        "keepdims": False,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 array, ord=3 norm along axis 1
    x = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "axis": 1,
        "keepdims": True,
        "ord": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array (converted to float inside norm, but input as integer array)
    x = np.array([-3, 4, -12, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "axis": -1,
        "keepdims": False,
        "ord": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float64 array, L1 norm along axis -2
    x = np.random.randn(5, 2).astype(np.float64)
    input_dict = {
        "x": x,
        "axis": -2,
        "keepdims": True,
        "ord": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, large scale values
    x = (np.random.randn(4, 4, 4) * 100).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 0,
        "keepdims": False,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, L1 norm along axis 3, keeping dimensions
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 3,
        "keepdims": True,
        "ord": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.vector_norm_1"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vector_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vector_norm_1'.")


check_valid('jax.numpy.linalg.vector_norm', generated_inputs['jax.numpy.linalg.vector_norm_1'], lib="jax", suffix=1)
