
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_linalg_vector_norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, ord=2, keepdims=False
    input_dict = {
        "x": np.array([1.0, -2.0, 3.0], dtype=np.float32),
        "axis": (0,),
        "keepdims": False,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, ord=1, keepdims=True
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "axis": (1,),
        "keepdims": True,
        "ord": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, ord=2
    input_dict = {
        "x": np.random.randn(2, 3, 4).astype(np.float64),
        "axis": (0, 2),
        "keepdims": False,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, ord=3, keepdims=True
    input_dict = {
        "x": np.random.randn(5, 5).astype(np.float32),
        "axis": (0,),
        "keepdims": True,
        "ord": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, ord=1
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "axis": (1, 3),
        "keepdims": False,
        "ord": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 array, ord=0
    input_dict = {
        "x": np.array([-10.0, 0.0, 10.0], dtype=np.float64),
        "axis": (0,),
        "keepdims": True,
        "ord": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, ord=2, keepdims=True
    input_dict = {
        "x": np.random.randn(3, 4, 5).astype(np.float32),
        "axis": (1,),
        "keepdims": True,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, ord=1, negative values
    input_dict = {
        "x": np.random.randn(10, 2).astype(np.float32),
        "axis": (1,),
        "keepdims": False,
        "ord": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, ord=3, keepdims=True
    input_dict = {
        "x": np.random.randn(3, 3, 3).astype(np.float32),
        "axis": (0, 1),
        "keepdims": True,
        "ord": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, ord=2, keepdims=False
    input_dict = {
        "x": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "axis": (2,),
        "keepdims": False,
        "ord": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.vector_norm_4"] = jax_numpy_linalg_vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vector_norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vector_norm_4'.")


check_valid('jax.numpy.linalg.vector_norm', generated_inputs['jax.numpy.linalg.vector_norm_4'], lib="jax", suffix=4)
