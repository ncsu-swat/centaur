
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def deg2rad_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with standard positive angles
    x = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float64 array with negative angles
    x = np.array([-30.0, -45.0, -90.0, -180.0, -360.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar-like 0D array
    x = np.array(45.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array of floats
    x = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of floats using random values
    x = np.random.uniform(-360.0, 360.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D integer array
    x = np.array([0, 90, 180, 270, 360], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D integer array (int64) with negative values
    x = np.array([[-45, -90], [45, 90]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with extremely large and small angles
    x = np.array([1e6, -1e6, 1e-5, -1e-5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array of floats
    x = np.random.uniform(-180.0, 180.0, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array containing special values (NaN and Inf)
    x = np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.deg2rad_1"] = deg2rad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.deg2rad_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.deg2rad_1'.")


check_valid('jax.numpy.deg2rad', generated_inputs['jax.numpy.deg2rad_1'], lib="jax", suffix=1)
