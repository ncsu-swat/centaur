
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_less_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, y = 0
    x = np.array([-5, 0, 5], dtype=np.int32)
    y = int(0)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D float32 array, y = 5
    x = np.random.uniform(-10.0, 10.0, (3, 3)).astype(np.float32)
    y = int(5)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D int64 array, y = -3
    x = np.random.randint(-10, 10, size=(2, 2, 2)).astype(np.int64)
    y = int(-3)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 1D float64 array, y = 100
    x = np.array([99.9, 100.0, 100.1], dtype=np.float64)
    y = int(100)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D int16 array, y = 1
    x = np.random.randint(-5, 5, size=(2, 2, 2, 2)).astype(np.int16)
    y = int(1)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 2D uint8 array, y = 128
    x = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
    y = int(128)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 1D float32 array with negative values, y = -10
    x = np.array([-15.5, -10.0, -5.5], dtype=np.float32)
    y = int(-10)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: 0D array (scalar tensor), y = 42
    x = np.array(10, dtype=np.int32)
    y = int(42)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 3D float32 array, y = 0
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = int(0)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 5D int32 array, y = 10
    x = np.random.randint(0, 20, size=(2, 2, 2, 2, 2)).astype(np.int32)
    y = int(10)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.less_3"] = jax_numpy_less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_3'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_3'], lib="jax", suffix=3)
