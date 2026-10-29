
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyval_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D p and 0D scalar x
    p = np.array([2.0, -1.0, 3.0], dtype=np.float32)
    x = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 16})
    )

    # Input 2: Integer polynomial and 1D integer x
    p = np.array([1, 2, 3, 4], dtype=np.int32)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 1})
    )

    # Input 3: Float64 1D p and 2D float64 x
    p = np.array([0.5, -2.5, 1.0], dtype=np.float64)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 4})
    )

    # Input 4: Complex numbers
    p = np.array([1.0 + 2.0j, -0.5 + 1.0j, 3.0 - 1.0j], dtype=np.complex64)
    x = np.array([0.5 - 0.5j, 1.0 + 0.0j, -1.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 8})
    )

    # Input 5: Single coefficient (constant polynomial)
    p = np.array([42.0], dtype=np.float32)
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 2})
    )

    # Input 6: 3D input x with negative and positive values
    p = np.array([-1.0, 0.0, 2.0, -5.0], dtype=np.float32)
    x = np.ones((2, 3, 4), dtype=np.float32) * -0.5
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 16})
    )

    # Input 7: High degree polynomial
    p = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 32})
    )

    # Input 8: 4D input x with int64
    p = np.array([3, -2, 1], dtype=np.int64)
    x = np.zeros((2, 2, 2, 2), dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 2})
    )

    # Input 9: Complex128 polynomial and 2D complex128 x
    p = np.array([2.0 - 3.0j, 1.5 + 0.5j], dtype=np.complex128)
    x = np.array([[1.0j, -2.0], [3.0 - 1.0j, 0.0]], dtype=np.complex128)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 16})
    )

    # Input 10: Linear polynomial with large unroll value
    p = np.array([2.5, -4.0], dtype=np.float32)
    x = np.linspace(-10.0, 10.0, 50, dtype=np.float32).reshape(5, 10)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "x": x, "unroll": 64})
    )

    return list_of_inputs


generated_inputs["jax.numpy.polyval_1"] = polyval_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyval_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyval_1'.")


check_valid('jax.numpy.polyval', generated_inputs['jax.numpy.polyval_1'], lib="jax", suffix=1)
