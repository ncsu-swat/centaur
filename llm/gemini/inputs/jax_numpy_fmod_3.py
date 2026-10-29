
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmod_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with positive integer divisor
    x1 = np.array([3.5, 4.2, 5.0, 6.7], dtype=np.float32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D int array with negative integer divisor
    x1 = np.array([-7, -4, 0, 5, 9], dtype=np.int32)
    x2 = -4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float64 array with positive integer divisor
    x1 = np.array([[10.5, -2.3], [-5.1, 7.8]], dtype=np.float64)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D int64 array with positive integer divisor
    x1 = np.array([[12, 15, 18], [22, 25, 28]], dtype=np.int64)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 array with negative integer divisor
    x1 = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    x2 = -3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array with positive integer divisor
    x1 = np.array(7.5, dtype=np.float32)
    x2 = 4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array
    x1 = np.ones((2, 2, 2, 2), dtype=np.float32) * 9.5
    x2 = 6
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D float array containing zeros and negatives
    x1 = np.array([0.0, -0.0, -15.5, 15.5], dtype=np.float32)
    x2 = 7
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Large integer values
    x1 = np.array([100000, -250000, 345678], dtype=np.int64)
    x2 = 1000
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D float16 array
    x1 = np.array([[-3.5, 2.5], [1.5, -4.5]], dtype=np.float16)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Divisor is 1
    x1 = np.array([1.25, 2.5, 3.75, 5.0], dtype=np.float32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmod_3"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_3'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_3'], lib="jax", suffix=3)
