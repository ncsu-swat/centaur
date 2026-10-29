
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def degrees_inputs():
    list_of_inputs = []

    # Input 1: 1D array of standard radian angles (float32)
    x = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative radian angles (float64)
    x = np.array([-2 * np.pi, -np.pi, -np.pi / 2, -np.pi / 3], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array (float32)
    x = np.array(
        [[0.0, np.pi / 6], [np.pi / 3, np.pi / 2], [3 * np.pi / 4, np.pi]],
        dtype=np.float32,
    )
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with mixed signs
    x = np.array([[-np.pi, np.pi / 4], [np.pi / 2, -3 * np.pi / 2]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array
    x = np.linspace(-np.pi, np.pi, 24, dtype=np.float32).reshape(2, 3, 4)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D single-element array
    x = np.array([np.pi], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D integer array
    x = np.array([-3, -1, 0, 1, 3], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty 1D array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with special values (inf, -inf, nan)
    x = np.array([0.0, np.inf, -np.inf, np.nan], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of small values
    x = np.random.uniform(-np.pi, np.pi, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array with large values
    x = np.array([1e4, -1e4, 1e-4, -1e-4], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.degrees_5"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_5'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_5'], lib="jax", suffix=5)
