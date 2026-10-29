
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D odd-length array, float32, axis 0
    x = np.array([0.0, 0.2, 0.4, -0.4, -0.2], dtype=np.float32)
    input_dict = {"x": x, "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D even-length array, float64, axis 0
    x = np.linspace(-10.0, 10.0, 8, dtype=np.float64)
    input_dict = {"x": x, "axes": (0,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, complex64, shifting both axes
    x = (
        np.random.randn(4, 4).astype(np.float32)
        + 1j * np.random.randn(4, 4).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {"x": x, "axes": (0, 1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, mixed dimensions (odd, even), complex128, shift axis 1
    x = (
        np.random.randn(5, 6).astype(np.float64)
        + 1j * np.random.randn(5, 6).astype(np.float64)
    ).astype(np.complex128)
    input_dict = {"x": x, "axes": (1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D integer array, negative axis index
    x = np.arange(49, dtype=np.int32).reshape(7, 7)
    input_dict = {"x": x, "axes": (-1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, shift axes 0 and 2
    x = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"x": x, "axes": (0, 2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D complex64 array, negative axes indices
    x = (
        np.random.randn(6, 6, 6).astype(np.float32)
        + 1j * np.random.randn(6, 6, 6).astype(np.float32)
    ).astype(np.complex64)
    input_dict = {"x": x, "axes": (-3, -1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float64 array, shifting multiple axes
    x = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict = {"x": x, "axes": (1, 2, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D complex128 array, shifting first and last axes
    x = (
        np.random.randn(2, 2, 2, 2).astype(np.float64)
        + 1j * np.random.randn(2, 2, 2, 2).astype(np.float64)
    ).astype(np.complex128)
    input_dict = {"x": x, "axes": (0, -1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array with negative values
    x = np.array([-5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 13], dtype=np.int64)
    input_dict = {"x": x, "axes": (-1,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D float32 array
    x = np.random.randn(2, 2, 3, 3, 2).astype(np.float32)
    input_dict = {"x": x, "axes": (1, 3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.ifftshift_3"] = ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftshift_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftshift_3'.")


check_valid('jax.numpy.fft.ifftshift', generated_inputs['jax.numpy.fft.ifftshift_3'], lib="jax", suffix=3)
