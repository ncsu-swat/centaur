
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, shifting the only axis
    x = np.linspace(-10, 10, 10).astype(np.float32)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 2: 1D float64 array with odd number of elements
    x = np.linspace(-5, 5, 11).astype(np.float64)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 3: 2D complex64 array, shifting both axes
    x = (np.random.randn(8, 8) + 1j * np.random.randn(8, 8)).astype(np.complex64)
    axes = [0, 1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 4: 2D int32 array, shifting only the second axis (axis 1)
    x = np.arange(16).reshape(4, 4).astype(np.int32)
    axes = [1]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 5: 3D float32 array, shifting axis 0 and 2
    x = np.random.randn(4, 4, 4).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 6: 2D float64 array with negative and positive values, shifting axis 0
    x = np.random.uniform(-100, 100, (6, 6)).astype(np.float64)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 7: 3D complex128 array, shifting axis 1 and 2
    x = (np.random.randn(3, 5, 5) + 1j * np.random.randn(3, 5, 5)).astype(np.complex128)
    axes = [1, 2]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 8: 1D int16 array with negative values
    x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.int16)
    axes = [0]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 9: 4D float32 array, shifting all axes
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = [0, 1, 2, 3]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 10: 3D float32 array, shifting only one of the axes (axis 2)
    x = np.random.randn(3, 3, 3).astype(np.float32)
    axes = [2]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    # Input 11: 2D float32 array, shifting with negative axis indices
    x = np.random.randn(5, 5).astype(np.float32)
    axes = [-1, -2]
    list_of_inputs.append({"x": copy.deepcopy(x), "axes": copy.deepcopy(axes)})

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftshift_2"] = fftshift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftshift_2'.")


check_valid('jax.numpy.fft.fftshift', generated_inputs['jax.numpy.fft.fftshift_2'], lib="jax", suffix=2)
