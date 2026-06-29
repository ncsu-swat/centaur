
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, single axis list
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 2: 1D int32, negative elements
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 3: 2D float32, shift along first axis
    x = np.random.randn(4, 4).astype(np.float32)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 4: 2D float64, shift along second axis
    x = np.random.randn(5, 5).astype(np.float64)
    axes = [1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 5: 2D complex64, shift along both axes
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    axes = [0, 1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 6: 3D float32, shift along specific axes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 7: 3D int16
    x = np.random.randint(-100, 100, size=(3, 3, 3)).astype(np.int16)
    axes = [1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 8: 4D float32, shift negative axes
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axes = [-4, -1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 9: 1D bool array
    x = np.array([True, False, True, False, True], dtype=np.bool_)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 10: 2D float32 with negative axis
    x = np.random.randn(6, 6).astype(np.float32)
    axes = [-1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 11: 3D float64, all axes listed
    x = np.random.randn(3, 4, 5).astype(np.float64)
    axes = [0, 1, 2]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifftshift_2"] = ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifftshift_2'.")


check_valid('jax.numpy.fft.ifftshift', generated_inputs['jax.numpy.fft.ifftshift_2'], lib="jax", suffix=2)
