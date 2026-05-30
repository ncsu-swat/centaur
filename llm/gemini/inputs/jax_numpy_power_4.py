
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_power_inputs():
    list_of_inputs = []

    # Input 1: positive base, 1D integer exponent array
    x1 = 2
    x2 = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 2: positive base, 2D integer exponent array
    x1 = 3
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 3: positive base, 1D float32 exponent array
    x1 = 5
    x2 = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 4: negative base, 1D integer exponent array
    x1 = -2
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 5: large base, 3D float32 random exponent array
    x1 = 10
    x2 = np.random.uniform(0, 3, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 6: positive base, 1D int64 random exponent array
    x1 = 4
    x2 = np.random.randint(0, 6, size=(5,)).astype(np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 7: zero base, 1D float32 exponent array
    x1 = 0
    x2 = np.array([1, 2, 3], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 8: base 1, 2D float64 random exponent array
    x1 = 1
    x2 = np.random.randn(2, 2).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 9: negative base, 2D int64 exponent array
    x1 = -1
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 10: positive base, float fractional exponents
    x1 = 8
    x2 = np.array([0.333, 0.5], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.power_4"] = jax_numpy_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_4'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_4'], lib="jax", suffix=4)
