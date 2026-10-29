
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def power_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats with positive float exponent
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D array of positive floats with fractional exponent
    x1 = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 3D array with negative float exponent
    x1 = np.ones((2, 3, 4), dtype=np.float64) * 2.0
    x2 = -1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 1D integer array with positive float exponent
    x1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    x2 = 3.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 4D array of floats with zero exponent
    x1 = np.random.uniform(1.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = 0.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 2D array with negative and positive values, even integer float exponent
    x1 = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32)
    x2 = 4.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 0D scalar array with float exponent
    x1 = np.array(5.0, dtype=np.float32)
    x2 = 1.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 2D array float64 with negative fractional exponent
    x1 = np.array([[4.0, 16.0], [25.0, 64.0]], dtype=np.float64)
    x2 = -0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 1D array float32 with exponent 1.0
    x1 = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    x2 = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 3D array float16 with positive fractional exponent
    x1 = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float16)
    x2 = 2.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 2D array with integer values, fractional float exponent
    x1 = np.array([[1, 8], [27, 64]], dtype=np.int64)
    x2 = 1.0 / 3.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.power_3"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_3'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_3'], lib="jax", suffix=3)
