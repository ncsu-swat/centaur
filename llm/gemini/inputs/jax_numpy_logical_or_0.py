
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logical_or_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean arrays of the same shape
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([False, False, True, True], dtype=bool)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean arrays of the same shape
    x = np.random.choice([True, False], size=(3, 3))
    y = np.random.choice([True, False], size=(3, 3))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasted 1D and 2D arrays
    x = np.random.choice([True, False], size=(1, 4))
    y = np.random.choice([True, False], size=(3, 4))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer arrays (0 and positive values)
    x = np.array([0, 1, 2, 0], dtype=np.int32)
    y = np.array([1, 0, 0, 0], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float arrays (0.0 and positive/negative values)
    x = np.array([0.0, 1.5, -2.3, 0.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar-like arrays
    x = np.array(True, dtype=bool)
    y = np.array(False, dtype=bool)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 2))
    y = np.random.choice([True, False], size=(2, 2, 2))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with dimensions of size 1 (1x3 and 3x1)
    x = np.array([[True, False, True]], dtype=bool)
    y = np.array([[False], [True], [False]], dtype=bool)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Mixed types (bool and float64)
    x = np.array([True, False, True], dtype=bool)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D boolean arrays with compatible broadcasting sizes
    x = np.random.choice([True, False], size=(2, 1, 3, 3))
    y = np.random.choice([True, False], size=(1, 2, 3, 3))
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Negative integer arrays
    x = np.array([-1, 0, -5, 3], dtype=np.int16)
    y = np.array([0, 0, 1, -2], dtype=np.int16)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.logical_or"] = logical_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logical_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logical_or'.")


check_valid('jax.numpy.logical_or', generated_inputs['jax.numpy.logical_or'], lib="jax", suffix=0)
