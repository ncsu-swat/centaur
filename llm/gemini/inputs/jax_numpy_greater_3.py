
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array and positive integer
    x = np.array([1.5, 2.3, 0.5, -1.2], dtype=np.float32)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D float64 array with negative values and negative integer
    x = np.array([[-1.5, -2.0, -0.5], [-3.1, 4.2, -5.0]], dtype=np.float64)
    y = -2
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D int32 array and zero
    x = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int32)
    y = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 1D int64 array with large values and large integer
    x = np.array([1000000, 2000000, 3000000], dtype=np.int64)
    y = 1500000
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D float32 array and positive integer
    x = np.random.randn(2, 3, 2, 2).astype(np.float32) * 10
    y = 5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 0D float32 array (scalar array) and integer
    x = np.array(5.5, dtype=np.float32)
    y = 5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 2D int16 array and negative integer
    x = np.array([[-5, -4], [-3, -2]], dtype=np.int16)
    y = -3
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: 3D float16 array and integer
    x = np.random.uniform(-100, 100, size=(2, 2, 3)).astype(np.float16)
    y = 50
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 1D uint8 array and integer
    x = np.array([0, 127, 255], dtype=np.uint8)
    y = 127
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 5D int8 array and small integer
    x = np.random.randint(-5, 5, size=(1, 2, 2, 1, 3)).astype(np.int8)
    y = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.greater_3"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_3'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_3'], lib="jax", suffix=3)
