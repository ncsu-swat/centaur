
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def heaviside_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard case
    x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x2 = 0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array
    x1 = np.array([[-1.5, 0.0], [0.0, 2.5]], dtype=np.float64)
    x2 = 1
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array
    x1 = np.array([[[-1, 0], [1, -2]], [[2, 0], [-3, 4]]], dtype=np.int32)
    x2 = 5
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int64 array with negative x2
    x1 = np.array([0, 0, 0, 1, -1], dtype=np.int64)
    x2 = -1
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 2D float32 array with random numbers and zeros
    x1 = np.random.uniform(-10, 10, size=(10, 10)).astype(np.float32)
    x1[x1 < -8] = 0.0  # Introduce exact zeros
    x2 = 2
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array
    x1 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x2 = 0
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with all zeros
    x1 = np.zeros((10,), dtype=np.float32)
    x2 = 1
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D array (scalar tensor)
    x1 = np.array(0.0, dtype=np.float32)
    x2 = 42
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, negative values
    x1 = np.random.uniform(-5.0, -0.1, size=(5, 5)).astype(np.float64)
    x2 = -3
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, all positive values
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    x2 = 10
    input_dict = {"x1": x1, "x2": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.heaviside_3"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_3'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_3'], lib="jax", suffix=3)
