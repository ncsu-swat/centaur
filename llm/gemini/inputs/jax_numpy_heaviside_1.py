
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x1 = np.array([-1.5, 0.0, 2.0], dtype=np.float32)
    x2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 2D float64 arrays with negative, positive and zero values
    x1 = np.array([[-2.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    x2 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: Broadcasting 0D array (scalar array) for x2
    x1 = np.array([[-3.0, 0.0, 5.0]], dtype=np.float32)
    x2 = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Broadcasting 1D array for x2 and 2D array for x1
    x1 = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, -2.0]], dtype=np.float32)
    x2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: Integer arrays (int32)
    x1 = np.array([-5, 0, 10], dtype=np.int32)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: 3D float32 arrays
    x1 = np.random.uniform(-5, 5, (2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(0, 1, (2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: Float16 arrays
    x1 = np.array([-10.0, 0.0, 10.0], dtype=np.float16)
    x2 = np.array([0.75, 0.75, 0.75], dtype=np.float16)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: Broadcasting (1, 3) and (3, 1) shapes
    x1 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    x2 = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Int64 arrays
    x1 = np.array([[-10, 0], [0, 10]], dtype=np.int64)
    x2 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: 4D arrays
    x1 = np.random.uniform(-10, 10, (1, 2, 2, 1)).astype(np.float32)
    x2 = np.random.uniform(0, 1, (1, 2, 2, 1)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_1"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_1'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_1'], lib="jax", suffix=1)
