
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array with common angles
    x = np.array([0.0, np.pi/4, np.pi/2, np.pi, 2*np.pi], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with negative angles
    x = np.array([[-np.pi, -np.pi/2], [np.pi/3, -np.pi/4]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0D array) float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array with random values
    x = np.random.uniform(-2*np.pi, 2*np.pi, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D float16 array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D int32 array
    x = np.array([-3, -1, 0, 1, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D int64 array
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float64 array with small random values
    x = np.random.randn(2, 2, 2, 2).astype(np.float64) * 0.01
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with NaNs and Infs
    x = np.array([np.nan, np.inf, -np.inf, 0.0, np.pi], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array with a sequence of values
    x = np.linspace(-10.0, 10.0, 100, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.rad2deg_1"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rad2deg_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rad2deg_1'.")


check_valid('jax.numpy.rad2deg', generated_inputs['jax.numpy.rad2deg_1'], lib="jax", suffix=1)
