
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive float y
    x = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
    y = 2.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, zero float y
    x = np.array([[0, 1], [2, 0]], dtype=np.int32)
    y = 0.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, negative float y
    x = np.array([[[ -1.5, 2.3], [1.1, -1.5]], [[0.0, -1.5], [-1.5, 4.5]]], dtype=np.float64)
    y = -1.5
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D float32 array, positive float y
    x = np.array(4.2, dtype=np.float32)
    y = 4.2
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float16 array, positive float y
    x = np.array([10.0, 20.0, 10.0], dtype=np.float16)
    y = 10.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, negative zero float y
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    y = -0.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int64 array, positive float y
    x = np.array([[5, 6, 7], [8, 5, 9]], dtype=np.int64)
    y = 5.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int8 array, negative float y
    x = np.array([-3, -2, -1, 0, 1], dtype=np.int8)
    y = -3.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, positive float y
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = 1.0
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 array, large float y
    x = np.array([[123.45, 0.0], [0.0, 123.45]], dtype=np.float64)
    y = 123.45
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.equal_2"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_2'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_2'], lib="jax", suffix=2)
