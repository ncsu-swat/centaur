
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trunc_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-1.5, -0.7, 0.0, 0.7, 1.5, 2.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array
    x = np.random.uniform(-10, 10, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float16 array
    x = np.random.uniform(-5, 5, size=(2, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D float32 scalar array
    x = np.array(-3.14, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D int32 array
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D float32 array
    x = np.random.uniform(-100, 100, size=(2, 3, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D float32 with large and small values
    x = np.array([[1e5 + 0.5, -1e5 - 0.5], [0.12345, -0.12345]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 containing signed zeros
    x = np.array([-0.0, 0.0, -0.9, 0.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D int64 array
    x = np.random.randint(-100, 100, size=(4, 4)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D float64 array
    x = np.random.uniform(-1, 1, size=(2, 2, 1, 3, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.trunc_1"] = trunc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trunc_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trunc_1'.")


check_valid('jax.numpy.trunc', generated_inputs['jax.numpy.trunc_1'], lib="jax", suffix=1)
