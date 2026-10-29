
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def degrees_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([0.0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float64 array with negative values
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float32 array (2x3)
    x = np.array(
        [[0.0, np.pi / 6, np.pi / 4], [np.pi / 3, np.pi / 2, np.pi]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D float64 array (3x2) with mixed signs
    x = np.array(
        [[-np.pi, np.pi], [-np.pi / 2, np.pi / 2], [-np.pi / 4, np.pi / 4]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array (2x2x2)
    x = np.array(
        [
            [[0.0, np.pi / 2], [np.pi, 2 * np.pi]],
            [[-np.pi / 2, -np.pi], [3 * np.pi / 2, -2 * np.pi]],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D float32 array with special values
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D float64 array with small and large values
    x = np.array([1e-7, -1e-7, 1e5, -1e5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D single-element array
    x = np.array([np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D float32 array with various values
    x = np.array([-1.0, 1.0, -2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 4D float32 array (1x2x1x2)
    x = np.array([[[[0.0, np.pi]], [[-np.pi, np.pi / 2]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.degrees_4"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_4'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_4'], lib="jax", suffix=4)
