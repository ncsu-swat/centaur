
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive float divisor
    x1 = np.array([3.0, -1.0, 4.0, 7.5], dtype=np.float32)
    x2 = 2.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 array, negative float divisor
    x1 = np.array([[-5.5, 2.3], [7.1, -8.9]], dtype=np.float64)
    x2 = -1.2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 1D int32 array, integer-valued float divisor
    x1 = np.array([10, -15, 20, -25], dtype=np.int32)
    x2 = 3.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D float32 array (scalar tensor), positive float divisor
    x1 = np.array(9.75, dtype=np.float32)
    x2 = 2.25
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 array with various ranges
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 2D int64 array with zeros and negative values
    x1 = np.array([[0, -8, 12], [15, -2, 7]], dtype=np.int64)
    x2 = -4.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float64 array
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float64)
    x2 = 2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D float16 array
    x1 = np.array([-3.5, -1.0, 0.0, 2.5, 6.0], dtype=np.float16)
    x2 = 1.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float32 array with larger magnitude values
    x1 = np.array([[100.5, -250.75], [333.3, -412.1]], dtype=np.float32)
    x2 = 7.3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 3D int32 array
    x1 = np.arange(-12, 12, dtype=np.int32).reshape(2, 3, 4)
    x2 = 5.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 1D float64 array with special constant divisor
    x1 = np.array([-10.0, -np.pi, 0.0, np.pi, 10.0], dtype=np.float64)
    x2 = 3.141592653589793
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmod_2"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_2'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_2'], lib="jax", suffix=2)
