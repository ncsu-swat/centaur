
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rot90_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, positive rotation, standard axes
    m = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"m": m, "k": 1, "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, int32, 180 degree rotation, reversed axes
    m = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    input_dict = {"m": m, "k": 2, "axes": (1, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float64, negative rotation
    m = np.random.randn(6, 4).astype(np.float64)
    input_dict = {"m": m, "k": -1, "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, int64, rotation in the last two dimensions
    m = np.random.randint(0, 100, size=(2, 3, 4)).astype(np.int64)
    input_dict = {"m": m, "k": 1, "axes": (1, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float32, rotation with -2 times 90 degrees on non-adjacent axes
    m = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {"m": m, "k": -2, "axes": (0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, boolean, 270 degree rotation
    m = np.random.choice([True, False], size=(2, 2, 4, 4))
    input_dict = {"m": m, "k": 3, "axes": (2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float32, k=0 (no rotation but output copy)
    m = np.random.randn(2, 3, 2, 4).astype(np.float32)
    input_dict = {"m": m, "k": 0, "axes": (1, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D array, float64, large negative k
    m = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {"m": m, "k": -5, "axes": (0, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, float32, k=5 (equivalent to k=1)
    m = np.random.randn(2, 2).astype(np.float32)
    input_dict = {"m": m, "k": 5, "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, complex64, rotating back and forth (k=2)
    m = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(np.complex64)
    input_dict = {"m": m, "k": 2, "axes": (2, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.rot90"] = rot90_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rot90'.")


check_valid('jax.numpy.rot90', generated_inputs['jax.numpy.rot90'], lib="jax", suffix=0)
