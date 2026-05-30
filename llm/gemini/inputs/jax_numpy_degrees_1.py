
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def degrees_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with standard radian values
    x = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float64 array with negative and positive values
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float32 array (matrix)
    x = np.array([[0.0, np.pi / 6], [np.pi / 3, np.pi / 2]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar equivalent)
    x = np.array(np.pi, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float64 array
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D int32 array (integer radians)
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D int64 array
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array with special values (nan, inf, -inf)
    x = np.array([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float16 array
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.degrees_1"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_1'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_1'], lib="jax", suffix=1)
