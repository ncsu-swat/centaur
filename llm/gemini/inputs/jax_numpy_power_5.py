
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def power_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor exponent, positive base
    x1 = 2.0
    x2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D integer tensor exponent
    x1 = 3.5
    x2 = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Negative base with integer-valued float array
    x1 = -2.0
    x2 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D tensor (scalar array)
    x1 = 5.0
    x2 = np.array(3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D tensor exponent with float64
    x1 = 1.5
    x2 = np.ones((2, 3, 4), dtype=np.float64) * 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Fraction exponent, base < 1
    x1 = 0.25
    x2 = np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D tensor with int64 exponent
    x1 = 10.0
    x2 = np.arange(16, dtype=np.int64).reshape((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Base 0.0 with positive exponents
    x1 = 0.0
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Negative exponents with float base
    x1 = 4.0
    x2 = np.array([-1.0, -2.0, -0.5, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D int32 tensor exponent
    x1 = 2.5
    x2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 1D float64 tensor exponent
    x1 = 1.2
    x2 = np.array([0.0, 1.5, 2.5, 3.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.power_5"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_5'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_5'], lib="jax", suffix=5)
