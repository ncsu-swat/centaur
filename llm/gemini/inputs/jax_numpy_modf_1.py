
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def modf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative floats
    x = np.array([-3.4, -5.7, 0.6, 1.5, 2.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 2: 2D float64 array
    x = np.array([[1.25, -2.5], [3.75, -4.125]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 3: 0D scalar float32 array
    x = np.array(4.8, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 4: 3D float32 array with uniform distribution
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 5: 1D float16 array
    x = np.array([0.1, -0.9, 1.5, -2.3], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 6: 4D float32 array
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 7: 2D float64 array with whole numbers and zeros
    x = np.array([[0.0, -0.0], [5.0, -5.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 8: 1D float32 array with small fractional parts
    x = np.array([1.0001, -2.0005, 3.00001], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 9: 2D float32 array with large values
    x = np.array(
        [[1000.5, -2000.25], [3000.75, -4000.125]], dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 10: 1D float64 array generated via linspace
    x = np.linspace(-10.5, 10.5, 9, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    # Input 11: 3D float64 array with standard normal distribution
    x = np.random.randn(3, 2, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "out": None}))

    return list_of_inputs


generated_inputs["jax.numpy.modf_1"] = modf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.modf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.modf_1'.")


check_valid('jax.numpy.modf', generated_inputs['jax.numpy.modf_1'], lib="jax", suffix=1)
