
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def negative_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([0., -3., 7.], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.randn(2, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D float64 array
    x = np.array([-1.5, 0.0, 3.14, -99.9], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D float64 array
    x = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D int32 array
    x = np.array([-10, 0, 10, 20], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D int32 array
    x = np.random.randint(-100, 100, size=(4, 3)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D int64 array
    x = np.array([-123456789, 0, 987654321], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D int64 array
    x = np.random.randint(-1000, 1000, size=(3, 5)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.negative_1"] = negative_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.negative_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.negative_1'.")


check_valid('jax.numpy.negative', generated_inputs['jax.numpy.negative_1'], lib="jax", suffix=1)
