
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asin_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor (scalar array), float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D tensor, float32, positive values
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D tensor, float64, negative values
    x = np.array([-0.1, -0.5, -0.9], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D tensor, float32, mix of values including 0 and boundaries
    x = np.array([[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D tensor, float64, random values in [-1, 1]
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D tensor, int32 (containing -1, 0, 1)
    x = np.array([-1, 0, 1], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D tensor, complex64, testing complex support
    x = (np.random.uniform(-1, 1, size=(2, 2)) + 1j * np.random.uniform(-1, 1, size=(2, 2))).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D tensor, float32, small values
    x = np.random.uniform(-0.1, 0.1, size=(1, 2, 2, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D tensor, float16
    x = np.array([-0.75, 0.25, 0.75], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D tensor, complex128
    x = (np.random.uniform(-2, 2, size=(3, 3)) + 1j * np.random.uniform(-2, 2, size=(3, 3))).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D tensor with dense grid
    x = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asin'.")


check_valid('jax.numpy.asin', generated_inputs['jax.numpy.asin'], lib="jax", suffix=0)
