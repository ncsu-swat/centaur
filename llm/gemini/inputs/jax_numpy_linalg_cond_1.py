
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cond_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, p=2
    x = np.random.randn(3, 3).astype(np.float32)
    p = 2
    list_of_inputs.append({"x": x, "p": p})

    # Input 2: 2D square float32 matrix, p=1
    x = np.random.randn(4, 4).astype(np.float32)
    p = 1
    list_of_inputs.append({"x": x, "p": p})

    # Input 3: 2D square float64 matrix, p=-1
    x = np.random.randn(5, 5).astype(np.float64)
    p = -1
    list_of_inputs.append({"x": x, "p": p})

    # Input 4: 2D rectangular float32 matrix, p=2
    x = np.random.randn(3, 4).astype(np.float32)
    p = 2
    list_of_inputs.append({"x": x, "p": p})

    # Input 5: 3D batched square float32 matrix, p=1
    x = np.random.randn(2, 3, 3).astype(np.float32)
    p = 1
    list_of_inputs.append({"x": x, "p": p})

    # Input 6: 3D batched rectangular float64 matrix, p=-2
    x = np.random.randn(2, 3, 2).astype(np.float64)
    p = -2
    list_of_inputs.append({"x": x, "p": p})

    # Input 7: 2D square float32 matrix, p=-2
    x = np.random.randn(6, 6).astype(np.float32)
    p = -2
    list_of_inputs.append({"x": x, "p": p})

    # Input 8: 3D batched square float32 matrix, p=2
    x = np.random.randn(2, 4, 4).astype(np.float32)
    p = 2
    list_of_inputs.append({"x": x, "p": p})

    # Input 9: 3D batched square float64 matrix, p=-1
    x = np.random.randn(1, 5, 5).astype(np.float64)
    p = -1
    list_of_inputs.append({"x": x, "p": p})

    # Input 10: 2D rectangular float32 matrix, p=-2
    x = np.random.randn(4, 2).astype(np.float32)
    p = -2
    list_of_inputs.append({"x": x, "p": p})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.cond_1"] = cond_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cond_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cond_1'.")


check_valid('jax.numpy.linalg.cond', generated_inputs['jax.numpy.linalg.cond_1'], lib="jax", suffix=1)
