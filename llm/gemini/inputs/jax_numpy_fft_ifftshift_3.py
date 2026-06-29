
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, axes 1-tuple
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axes = (0,)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int array with negative elements, axes 1-tuple
    x = np.array([-5, -3, -1, 1, 3, 5], dtype=np.int32)
    axes = (0,)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float array, shift along axis 0
    x = np.random.randn(4, 5).astype(np.float32)
    axes = (0,)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array, shift along axis 1
    x = np.random.randn(4, 5).astype(np.float32)
    axes = (1,)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float array, shift along both axes (0, 1)
    x = np.random.randn(6, 6).astype(np.float64)
    axes = (0, 1)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float array, shift along axis 2
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = (2,)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float array, shift along axes (0, 2)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = (0, 2)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float array, shift along all axes (0, 1, 2)
    x = np.random.randn(2, 2, 2).astype(np.float32)
    axes = (0, 1, 2)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int array, shift along (1, 3)
    x = np.random.randint(-10, 10, size=(2, 3, 2, 3)).astype(np.int64)
    axes = (1, 3)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex array, axes 1-tuple
    x = (np.random.randn(6) + 1j * np.random.randn(6)).astype(np.complex64)
    axes = (0,)
    input_dict = {"x": x, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D complex array, axes (0, 1)
    x = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex128)
    axes = (0, 1)
    input_dict = {"x": x, "axes": axes}
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
