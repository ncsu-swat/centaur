
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
from jax import lax


def cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float64 array with positive values
    x = np.array([[1.0, 8.0], [27.0, 64.0]], dtype=np.float64)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0D float32 scalar tensor
    x = np.array(-125.0, dtype=np.float32)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float32 array with random normal values
    x = np.random.randn(2, 3, 4).astype(np.float32)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 4D float32 array with positive values
    x = np.random.uniform(0.1, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 1D float16 array with negative and positive values
    x = np.array([-64.0, -8.0, 0.0, 8.0, 64.0], dtype=np.float16)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 2D float64 array with fractional values
    x = np.array([[0.125, 0.001], [0.027, 0.008]], dtype=np.float64)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * 8.0
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 1D float64 array with special float values
    x = np.array([0.0, -0.0, np.inf, -np.inf], dtype=np.float64)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2D float32 array with strictly negative values
    x = -np.random.exponential(scale=5.0, size=(4, 5)).astype(np.float32)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 3D float64 array with uniform random values
    x = np.random.uniform(-100.0, 100.0, size=(3, 2, 4)).astype(np.float64)
    accuracy = lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.cbrt_1"] = cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cbrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cbrt_1'.")


check_valid('jax.lax.cbrt', generated_inputs['jax.lax.cbrt_1'], lib="jax", suffix=1)
