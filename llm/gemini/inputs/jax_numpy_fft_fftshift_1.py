
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, even length, axis 0
    x = np.array([0., 1., 2., 3., 4., 5.], dtype=np.float32)
    axes = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 2: 1D float array, odd length, axis 0
    x = np.array([0., 1., 2., 3., 4.], dtype=np.float32)
    axes = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 3: 2D float array, axis 0
    x = np.random.randn(4, 4).astype(np.float32)
    axes = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 4: 2D float array, axis 1
    x = np.random.randn(4, 4).astype(np.float32)
    axes = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 5: 2D float array, negative axis
    x = np.random.randn(4, 4).astype(np.float32)
    axes = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 6: 3D float array, axis 0
    x = np.random.randn(3, 3, 3).astype(np.float32)
    axes = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 7: 3D float array, axis 2
    x = np.random.randn(3, 3, 3).astype(np.float32)
    axes = 2
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 8: 3D complex array, axis 1
    x = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(np.complex64)
    axes = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 9: 4D integer array, axis 3
    x = np.arange(16).reshape(2, 2, 2, 2).astype(np.int32)
    axes = 3
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 10: 1D double array, negative axis
    x = np.arange(10).astype(np.float64)
    axes = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 11: 2D float array, axis -2
    x = np.random.randn(5, 6).astype(np.float32)
    axes = -2
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftshift_1"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftshift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftshift_1'.")


check_valid('jax.numpy.fft.fftshift', generated_inputs['jax.numpy.fft.fftshift_1'], lib="jax", suffix=1)
