
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hfft_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64, backward norm
    a = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {"a": a, "n": 6, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, ortho norm, odd n
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"a": a, "n": 5, "axis": 1, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D complex128, forward norm, axis 0
    a = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex128)
    input_dict = {"a": a, "n": 8, "axis": 0, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex64, backward norm, negative axis
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    input_dict = {"a": a, "n": 4, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64, ortho norm, axis 1
    a = np.random.randn(2, 6, 3).astype(np.float64)
    input_dict = {"a": a, "n": 7, "axis": 1, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32, forward norm
    a = np.random.randn(8).astype(np.float32)
    input_dict = {"a": a, "n": 10, "axis": 0, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex128, backward norm
    a = (np.random.randn(2, 3, 5, 2) + 1j * np.random.randn(2, 3, 5, 2)).astype(np.complex128)
    input_dict = {"a": a, "n": 6, "axis": 2, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex64, ortho norm, negative axis -2
    a = (np.random.randn(5, 4) + 1j * np.random.randn(5, 4)).astype(np.complex64)
    input_dict = {"a": a, "n": 3, "axis": -2, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D complex64, forward norm, axis -3
    a = (np.random.randn(4, 3, 2) + 1j * np.random.randn(4, 3, 2)).astype(np.complex64)
    input_dict = {"a": a, "n": 12, "axis": -3, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex128, ortho norm, minimal n
    a = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex128)
    input_dict = {"a": a, "n": 1, "axis": 0, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float32, backward norm, small n
    a = np.random.randn(4, 2).astype(np.float32)
    input_dict = {"a": a, "n": 2, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.hfft"] = hfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.hfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.hfft'.")


check_valid('jax.numpy.fft.hfft', generated_inputs['jax.numpy.fft.hfft'], lib="jax", suffix=0)
