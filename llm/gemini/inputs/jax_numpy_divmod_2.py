
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def divmod_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive integer divisor
    x1 = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D int64 array with negative values, negative integer divisor
    x1 = np.array([-15, -7, 0, 8, 23], dtype=np.int64)
    x2 = -4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float32 array, positive integer divisor
    x1 = np.array([[1.5, 2.7, 3.2], [4.9, 5.1, 6.8]], dtype=np.float32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D int32 array, positive integer divisor
    x1 = np.arange(-12, 12, dtype=np.int32).reshape(4, 6)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float64 array, negative integer divisor
    x1 = np.linspace(-10.0, 10.0, 24, dtype=np.float64).reshape(2, 3, 4)
    x2 = -3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array (int32), positive integer divisor
    x1 = np.array(42, dtype=np.int32)
    x2 = 7
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 1D int32 array, positive integer divisor
    x1 = np.array([0, 15, 100, 255, 1024], dtype=np.int32)
    x2 = 4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 4D int32 array, positive integer divisor
    x1 = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int32)
    x2 = 10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float32 array, negative integer divisor
    x1 = np.array([[-5.5, -2.1], [3.4, 7.9]], dtype=np.float32)
    x2 = -2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D int64 array with large values, large integer divisor
    x1 = np.array([1000000, 2500000, -7894561], dtype=np.int64)
    x2 = 100
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D int32 array containing zeros, divisor 1
    x1 = np.zeros((2, 2, 2), dtype=np.int32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.divmod_2"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_2'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_2'], lib="jax", suffix=2)
