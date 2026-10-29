
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, shift axis 0
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 2: 1D complex64 array, negative axis index
    x = np.array(
        [1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j, 0.0 + 0.0j], dtype=np.complex64
    )
    axes = [-1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 3: 2D float64 array, shift both axes [0, 1]
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    axes = [0, 1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 4: 2D int32 array, shift only axis 1
    x = np.array(
        [[-5, -4, -3], [-2, -1, 0], [1, 2, 3], [4, 5, 6]], dtype=np.int32
    )
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 5: 2D complex128 array, shift only axis 0
    x = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(
        np.complex128
    )
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 6: 3D float32 array, shift axes [0, 2]
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 7: 3D float64 array, empty axes list (no-op shift)
    x = np.random.randn(4, 4, 4).astype(np.float64)
    axes = []
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 8: 3D int64 array, negative axes indices
    x = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    axes = [-3, -1]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 9: 4D float32 array, shift all axes [0, 1, 2, 3]
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 2, 3)).astype(np.float32)
    axes = [0, 1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 10: 4D complex64 array, shift subset of axes
    x = (
        np.random.randn(2, 4, 3, 5) + 1j * np.random.randn(2, 4, 3, 5)
    ).astype(np.complex64)
    axes = [1, 3]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 11: 5D float32 array, shift axis 2
    x = np.ones((2, 2, 3, 2, 2), dtype=np.float32)
    axes = [2]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

    # Input 12: 1D bool array, shift axis 0
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"x": x, "axes": axes}))

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
