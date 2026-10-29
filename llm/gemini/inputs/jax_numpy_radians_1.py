
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def radians_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(180.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of common angles (float32)
    x = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D array with negative angles (float64)
    x = np.array([-360.0, -180.0, -90.0, -45.0, 0.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array of integers (int32)
    x = np.array([[0, 90], [180, 270]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D array with float16
    x = np.array([[-45.0, 45.0], [-135.0, 135.0]], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D array of random angles (float32)
    x = np.random.uniform(-360.0, 360.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array (float64)
    x = np.random.uniform(0.0, 720.0, size=(2, 2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with large degree values (int64)
    x = np.array([720, 1080, 1440, 1800, -720], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array with special float values (nan, inf, -inf)
    x = np.array([np.nan, np.inf, -np.inf, 0.0, 180.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Multi-dimensional empty array
    x = np.zeros((3, 0, 2), dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.radians_1"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_1'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_1'], lib="jax", suffix=1)
