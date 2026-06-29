
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D array, even length, float32, axes=0
    x = np.arange(10, dtype=np.float32)
    input_dict = {"x": x, "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, odd length, int32, axes=0
    x = np.arange(9, dtype=np.int32)
    input_dict = {"x": x, "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32, axes=0
    x = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"x": x, "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, float64, axes=1
    x = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"x": x, "axes": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, complex64, axes=2
    x = (np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)).astype(np.complex64)
    input_dict = {"x": x, "axes": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, float32, axes=-1 (negative axis)
    x = np.random.randn(2, 4, 6).astype(np.float32)
    input_dict = {"x": x, "axes": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, int16, axes=2
    x = np.random.randint(-10, 10, size=(2, 3, 4, 5)).astype(np.int16)
    input_dict = {"x": x, "axes": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, boolean, axes=0
    x = np.random.choice([True, False], size=(6, 6))
    input_dict = {"x": x, "axes": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, complex128, axes=-1
    x = (np.random.randn(11) + 1j * np.random.randn(11)).astype(np.complex128)
    input_dict = {"x": x, "axes": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, float32, axes=3
    x = np.random.randn(2, 2, 3, 3, 4).astype(np.float32)
    input_dict = {"x": x, "axes": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifftshift_1"] = ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftshift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftshift_1'.")


check_valid('jax.numpy.fft.ifftshift', generated_inputs['jax.numpy.fft.ifftshift_1'], lib="jax", suffix=1)
