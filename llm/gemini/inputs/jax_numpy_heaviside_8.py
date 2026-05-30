
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: Scalar int and scalar float
    input_dict = {
        "x1": np.int32(0),
        "x2": np.float32(0.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int array and scalar float
    input_dict = {
        "x1": np.array([-2, -1, 0, 1, 2], dtype=np.int32),
        "x2": np.float32(0.5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int array and 1D float array (same shape)
    input_dict = {
        "x1": np.array([-5, 0, 5], dtype=np.int64),
        "x2": np.array([0.1, 0.2, 0.3], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int array and scalar float
    input_dict = {
        "x1": np.array([[-1, 0], [1, 2]], dtype=np.int32),
        "x2": np.float32(0.75)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int array and 1D float array (broadcasting)
    input_dict = {
        "x1": np.array([[-3, 0, 3], [4, -1, 0]], dtype=np.int32),
        "x2": np.array([0.2, 0.5, 0.8], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int array and 3D float array
    input_dict = {
        "x1": np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32),
        "x2": np.random.uniform(0.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large arrays with int64 and float64
    input_dict = {
        "x1": np.random.randint(-100, 100, size=(10, 10)).astype(np.int64),
        "x2": np.random.random((10, 10)).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int array (zeros only) and 1D float array
    input_dict = {
        "x1": np.zeros(5, dtype=np.int32),
        "x2": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int array and 2D float array (broadcasting with size 1)
    input_dict = {
        "x1": np.array([[-1, 0, 1]], dtype=np.int32),
        "x2": np.array([[0.5], [0.6]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int array and scalar float
    input_dict = {
        "x1": np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int8),
        "x2": np.float32(1.0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_8"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_8'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_8'], lib="jax", suffix=8)
