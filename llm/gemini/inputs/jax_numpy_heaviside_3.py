
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative, zero, and positive values
    x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D float64 array with non-zero integer x2
    x1 = np.array([-5.5, 0.0, 3.2, -0.1, 0.0, 4.4], dtype=np.float64)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D integer array
    x1 = np.array([[-2, 0, 3], [5, -1, 0]], dtype=np.int32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D float32 array with negative integer x2
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = -1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 array
    x1 = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
    x2 = 5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 4D float64 array
    x1 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    x2 = 10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 1D int64 array with all zeros
    x1 = np.array([0, 0, 0, 0], dtype=np.int64)
    x2 = -3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D float16 array
    x1 = np.array([-1.5, 0.0, 1.5], dtype=np.float16)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float32 array with single element
    x1 = np.array([[0.0]], dtype=np.float32)
    x2 = 42
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 5D float32 array
    x1 = np.random.uniform(-10.0, 10.0, size=(1, 2, 1, 3, 2)).astype(
        np.float32
    )
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_3"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_3'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_3'], lib="jax", suffix=3)
