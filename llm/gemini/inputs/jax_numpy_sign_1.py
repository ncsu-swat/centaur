
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_sign_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-10.0, -0.5, 0.0, 0.5, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array
    x = np.array([[-1.5, 2.3, 0.0], [4.5, -5.6, -0.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array
    x = np.array([[[1, -2], [0, 3]], [[-4, 5], [6, -7]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D complex64 array
    x = np.array([1.0 + 1.0j, -2.0 - 2.0j, 0.0 + 0.0j, 3.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D complex128 array
    x = np.array([[1j, -1j], [2 + 3j, 0]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D float32 array (scalar tensor)
    x = np.array(-5.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D int64 array
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float16 array
    x = np.array([-0.1, 0.0, 0.2, -0.3], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large 2D float32 array with random values
    x = np.random.uniform(-10.0, 10.0, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D int8 array
    x = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.sign_1"] = generate_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sign_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sign_1'.")


check_valid('jax.numpy.sign', generated_inputs['jax.numpy.sign_1'], lib="jax", suffix=1)
