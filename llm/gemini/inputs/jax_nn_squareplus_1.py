
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squareplus_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with standard b=4
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    b = 4
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 2: 2D float32 matrix with b=1
    x = np.random.randn(3, 4).astype(np.float32)
    b = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 3: 3D float64 tensor with b=2
    x = np.random.randn(2, 3, 5).astype(np.float64)
    b = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 4: 0D scalar float32 with b=5
    x = np.array(2.5, dtype=np.float32)
    b = 5
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 5: 4D float32 tensor with b=10
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    b = 10
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 6: 1D float16 with b=4
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float16)
    b = 4
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 7: 2D float64 with b=0
    x = np.array([[-2.0, 0.0], [1.0, 3.0]], dtype=np.float64)
    b = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 8: 5D float32 tensor with b=3
    x = np.random.randn(1, 2, 2, 2, 2).astype(np.float32)
    b = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 9: Large magnitude values float32 with b=100
    x = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
    b = 100
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 10: All zeros 1D float32 array with b=16
    x = np.zeros((8,), dtype=np.float32)
    b = 16
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    # Input 11: Negative values only with b=8
    x = -np.abs(np.random.randn(4, 4).astype(np.float32))
    b = 8
    list_of_inputs.append(copy.deepcopy({"x": x, "b": b}))

    return list_of_inputs


generated_inputs["jax.nn.squareplus_1"] = squareplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.squareplus_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.squareplus_1'.")


check_valid('jax.nn.squareplus', generated_inputs['jax.nn.squareplus_1'], lib="jax", suffix=1)
