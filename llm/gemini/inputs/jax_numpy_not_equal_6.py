
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, y is True
    x = np.array([True, False, True, False], dtype=bool)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D integer array (binary), y is False
    x = np.array([[1, 0], [0, 1]], dtype=np.int32)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float array, y is True
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D array (scalar), y is True
    x = np.array(0, dtype=np.int32)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float array with negative values, y is False
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float64)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D boolean array, y is True
    x = np.array([[True, True], [False, False]], dtype=bool)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float array, y is False
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D integer array, y is True
    x = np.random.randint(-10, 10, size=(2, 2, 3)).astype(np.int64)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D boolean array, y is False
    x = np.random.choice([True, False], size=(10, 10))
    y = False
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int16 array, y is True
    x = np.ones((2, 1, 2, 1, 2), dtype=np.int16)
    y = True
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_6"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_6'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_6'], lib="jax", suffix=6)
