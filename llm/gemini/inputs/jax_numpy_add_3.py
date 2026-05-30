
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def add_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive integer
    x = np.random.randn(5).astype(np.float32)
    y = 5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D int32 array, negative integer
    x = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    y = -3
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D float64 array, zero
    x = np.random.randn(2, 2, 2).astype(np.float64)
    y = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: Scalar-like 0D int64 array, positive integer
    x = np.array(42).astype(np.int64)
    y = 10
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D float32 array with negative and positive values, negative integer
    x = np.random.uniform(-5.0, 5.0, (2, 2, 3, 3)).astype(np.float32)
    y = -100
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 1D uint8 array, positive integer
    x = np.random.randint(0, 100, size=(10,)).astype(np.uint8)
    y = 7
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 5D float32 array, positive integer
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    y = 50
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: 2D float16 array, negative integer
    x = np.random.randn(4, 4).astype(np.float16)
    y = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 1D int16 array, large positive integer
    x = np.random.randint(-50, 50, size=(8,)).astype(np.int16)
    y = 1000
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 3D int8 array, positive integer
    x = np.random.randint(-5, 5, size=(2, 3, 4)).astype(np.int8)
    y = 12
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.add_3"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_3'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_3'], lib="jax", suffix=3)
