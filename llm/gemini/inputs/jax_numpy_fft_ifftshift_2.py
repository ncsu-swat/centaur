
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ifftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, shifting axis 0
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 2: 1D complex64 array with even length, negative axis
    x = np.array([1 + 2j, 3 - 1j, -2 + 0j, 0 - 4j], dtype=np.complex64)
    axes = [-1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 3: 2D float64 array, shifting both axes
    x = np.arange(12, dtype=np.float64).reshape((3, 4))
    axes = [0, 1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 4: 2D int32 array, shifting only axis 1
    x = np.arange(9, dtype=np.int32).reshape((3, 3))
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 5: 2D complex128 array, shifting only axis 0
    x = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(
        np.complex128
    )
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 6: 3D float32 array, shifting non-adjacent axes
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 7: 3D float64 array, using negative axes
    x = np.random.randn(5, 5, 5).astype(np.float64)
    axes = [-2, -1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 8: 3D complex64 array, shifting middle axis
    x = (np.random.randn(3, 4, 2) + 1j * np.random.randn(3, 4, 2)).astype(
        np.complex64
    )
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 9: 4D float32 array, shifting multiple axes
    x = np.random.randn(2, 4, 3, 5).astype(np.float32)
    axes = [1, 3]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 10: 4D complex128 array, shifting all axes explicitly
    x = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(
        np.complex128
    )
    axes = [0, 1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 11: 5D int64 array, shifting negative axes
    x = np.random.randint(-10, 10, size=(2, 3, 2, 3, 2), dtype=np.int64)
    axes = [-3, -1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 12: 1D array with an empty axes list (no-op shift)
    x = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
    axes = []
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

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
