
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ravel_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, C-order
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, F-order
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"a": a, "order": "F"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array, C-order
    a = np.array([-10, 0, 10, 20, 30], dtype=np.int32)
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, C-order
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 array, F-order
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "order": "F"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int64 array, C-order
    a = np.random.randint(-100, 100, size=(2, 2, 3, 3)).astype(np.int64)
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D (scalar) array, C-order
    a = np.array(42.0, dtype=np.float32)
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array, C-order
    a = np.random.choice([True, False], size=(5, 5))
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D int16 array, F-order
    a = np.random.randint(-10, 10, size=(2, 1, 3, 1, 2)).astype(np.int16)
    input_dict = {"a": a, "order": "F"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array, C-order
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"a": a, "order": "C"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D uint8 array, F-order
    a = np.random.randint(0, 255, size=(4, 4, 2)).astype(np.uint8)
    input_dict = {"a": a, "order": "F"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ravel"] = ravel_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ravel'.")


check_valid('jax.numpy.ravel', generated_inputs['jax.numpy.ravel'], lib="jax", suffix=0)
