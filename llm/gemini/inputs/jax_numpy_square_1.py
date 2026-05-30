
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_square_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([3.0, -2.0, 5.3, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D int32 array
    x = np.array([[2, 4], [5, 6]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D complex64 array
    x = np.array([1 - 3j, -1j, 2], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float64 array
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D float32 array
    x = np.array(4.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D int16 array
    x = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D float16 array
    x = np.random.randn(3, 3).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D int64 array
    x = np.array([0, -1, 2, -3, 4], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D float32 array
    x = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D complex128 array
    x = np.array([[1 + 1j, 2 - 2j], [3 + 3j, 4 - 4j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D float32 array containing special float values
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.square_1"] = jax_numpy_square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_1'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_1'], lib="jax", suffix=1)
