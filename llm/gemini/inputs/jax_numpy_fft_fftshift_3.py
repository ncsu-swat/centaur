
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, shifting axis 0
    x = np.arange(10, dtype=np.float32)
    axes = (0,)
    list_of_inputs.append({"x": x, "axes": axes})

    # Input 2: 2D float32 array, shifting axis 0
    x = np.random.randn(5, 5).astype(np.float32)
    axes = (0,)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 3: 2D float64 array, shifting axis 1
    x = np.random.randn(4, 8).astype(np.float64)
    axes = (1,)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 4: 2D float32 array, shifting both axes
    x = np.random.randn(6, 6).astype(np.float32)
    axes = (0, 1)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 5: 3D float32 array, shifting all axes
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = (0, 1, 2)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 6: 1D complex64 array, shifting axis 0
    x = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    axes = (0,)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 7: 4D float32 array, shifting axes (1, 2)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = (1, 2)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 8: 1D int32 array with negative and positive values, shifting axis 0
    x = np.array([-5, -3, -1, 1, 3, 5], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 9: 3D float64 array, shifting subset of axes (0, 2)
    x = np.random.randn(10, 10, 10).astype(np.float64)
    axes = (0, 2)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 10: 2D int32 array, shifting axis 0
    x = np.arange(12, dtype=np.int32).reshape(3, 4)
    axes = (0,)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    # Input 11: 1D float32 array, shifting negative axis (-1)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axes = (-1,)
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": axes})

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftshift_3"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftshift_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftshift_3'.")


check_valid('jax.numpy.fft.fftshift', generated_inputs['jax.numpy.fft.fftshift_3'], lib="jax", suffix=3)
