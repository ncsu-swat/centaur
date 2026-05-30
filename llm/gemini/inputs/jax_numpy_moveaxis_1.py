
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def moveaxis_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array, move axis 0 to 2
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "source": 0, "destination": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float64 array, move axis 2 to 0
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "source": 2, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D int32 array, move axis 1 to -1 (negative destination)
    a = np.random.randint(0, 10, size=(2, 3, 4, 5)).astype(np.int32)
    input_dict = {"a": a, "source": 1, "destination": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D float32 array, move axis -1 to 1 (negative source)
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "source": -1, "destination": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array, move axis 0 to 1
    a = np.random.randint(0, 10, size=(5, 10)).astype(np.int64)
    input_dict = {"a": a, "source": 0, "destination": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, move axis 1 to 0
    a = np.random.randn(10, 5).astype(np.float32)
    input_dict = {"a": a, "source": 1, "destination": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D float32 array, move axis 2 to 4
    a = np.random.randn(2, 2, 3, 3, 4).astype(np.float32)
    input_dict = {"a": a, "source": 2, "destination": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D float32 array, move axis -3 to -1
    a = np.random.randn(2, 2, 3, 3, 4).astype(np.float32)
    input_dict = {"a": a, "source": -3, "destination": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D complex64 array, move axis 0 to 1
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    input_dict = {"a": a, "source": 0, "destination": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6D int16 array, move axis 3 to 1
    a = np.random.randint(-100, 100, size=(2, 2, 2, 3, 3, 4)).astype(np.int16)
    input_dict = {"a": a, "source": 3, "destination": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.moveaxis_1"] = moveaxis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.moveaxis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.moveaxis_1'.")


check_valid('jax.numpy.moveaxis', generated_inputs['jax.numpy.moveaxis_1'], lib="jax", suffix=1)
