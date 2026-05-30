
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: positive float and 1D float32 array
    x1 = 10.0
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 2: negative float and 1D float32 array with negative/positive values
    x1 = -12.5
    x2 = np.array([-2.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 3: positive float and 2D float64 array
    x1 = 5.0
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 4: zero float and 1D int32 array (no zeros in divisor)
    x1 = 0.0
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 5: large float and 3D float32 array
    x1 = 100.0
    x2 = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 6: negative float and 1D float32 array with small decimal values
    x1 = -1.0
    x2 = np.array([0.5, -0.5, 2.5], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 7: float and 1D float32 array of ones multiplied by 2
    x1 = 20.5
    x2 = np.ones((5,), dtype=np.float32) * 2.0
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 8: very large float and 1D float64 array of powers of 10
    x1 = 1e5
    x2 = np.array([1e2, 1e3, 1e4], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 9: negative float and 2D int64 array
    x1 = -50.2
    x2 = np.array([[-10, 10], [-5, 5]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 10: float (pi representation) and 1D float32 array
    x1 = 3.14159
    x2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.floor_divide_5"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_5'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_5'], lib="jax", suffix=5)
