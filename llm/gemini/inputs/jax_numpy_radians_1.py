
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_radians_inputs():
    list_of_inputs = []

    # Input 1: 1D array with standard angles in float32
    x = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with negative angles in float64
    x = np.array([-45.0, -90.0, -180.0, -270.0, -360.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of angles (float32)
    x = np.array([[0.0, 90.0], [180.0, 270.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Scalar (0D array)
    x = np.array(180.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of angles (float32)
    x = np.random.uniform(-360.0, 360.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Integer degrees (int32)
    x = np.array([0, 90, 180, 270, 360], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of integer degrees (int64)
    x = np.array([[-45, -90], [45, 90]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array with random values
    x = np.random.uniform(-720.0, 720.0, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with extremely large degree values
    x = np.array([1000.0, 5000.0, 10000.0, -10000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array with small fractional degrees
    x = np.array([0.1, 0.01, 0.001, -0.05], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Empty array (0 elements) but with structure
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.radians_1"] = jax_numpy_radians_inputs()

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
