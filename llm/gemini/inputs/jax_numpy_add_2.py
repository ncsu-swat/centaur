
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor + positive float
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = 10.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D float32 tensor + negative float
    x = np.random.randn(2, 3).astype(np.float32)
    y = -5.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D float64 tensor + zero float
    x = np.random.randn(2, 2, 2).astype(np.float64)
    y = 0.0
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D int32 tensor + float
    x = np.array([-1, 0, 1, 2], dtype=np.int32)
    y = 1.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 4D float32 tensor + small float
    x = np.random.randn(1, 2, 2, 1).astype(np.float32)
    y = 1e-4
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 0D (scalar) float32 tensor + float
    x = np.array(5.5, dtype=np.float32)
    y = 2.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 2D float64 tensor with large values + float
    x = np.array([[1e5, -1e5], [2e5, -2e5]], dtype=np.float64)
    y = 3.14159
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 1D float32 tensor + very large float
    x = np.array([0.1, 0.2], dtype=np.float32)
    y = 1e10
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 5D float32 tensor + float
    x = np.ones((1, 2, 1, 2, 1), dtype=np.float32)
    y = -0.5
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 2D int16 tensor + negative float
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = -2.718
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.add_2"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_2'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_2'], lib="jax", suffix=2)
