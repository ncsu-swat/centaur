
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_fft_inputs():
    list_of_inputs = []

    # Input 1: 1D FFT, complex64
    x = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    input_dict = {"x": x, "fft_type": "FFT", "fft_lengths": (8,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D IFFT, complex64
    x = (np.random.randn(16) + 1j * np.random.randn(16)).astype(np.complex64)
    input_dict = {"x": x, "fft_type": "IFFT", "fft_lengths": (16,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D FFT, complex64
    x = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
    input_dict = {"x": x, "fft_type": "FFT", "fft_lengths": (4, 8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched 2D IFFT, complex128
    x = (np.random.randn(2, 8, 16) + 1j * np.random.randn(2, 8, 16)).astype(
        np.complex128
    )
    input_dict = {"x": x, "fft_type": "IFFT", "fft_lengths": (8, 16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D RFFT, float32
    x = np.random.randn(16).astype(np.float32)
    input_dict = {"x": x, "fft_type": "RFFT", "fft_lengths": (16,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D IRFFT, complex64 (fft length 16 -> input size 16 // 2 + 1 = 9)
    x = (np.random.randn(9) + 1j * np.random.randn(9)).astype(np.complex64)
    input_dict = {"x": x, "fft_type": "IRFFT", "fft_lengths": (16,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D RFFT, float32
    x = np.random.randn(4, 10).astype(np.float32)
    input_dict = {"x": x, "fft_type": "RFFT", "fft_lengths": (4, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D IRFFT, complex64 (fft lengths (4, 10) -> trailing dim 10 // 2 + 1 = 6)
    x = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(np.complex64)
    input_dict = {"x": x, "fft_type": "IRFFT", "fft_lengths": (4, 10)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D FFT, complex64
    x = (np.random.randn(2, 4, 4, 4) + 1j * np.random.randn(2, 4, 4, 4)).astype(
        np.complex64
    )
    input_dict = {"x": x, "fft_type": "FFT", "fft_lengths": (4, 4, 4)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched 1D RFFT, float64
    x = np.random.randn(3, 5, 32).astype(np.float64)
    input_dict = {"x": x, "fft_type": "RFFT", "fft_lengths": (32,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Batched 1D IRFFT, complex128 (fft length 32 -> input trailing size 17)
    x = (np.random.randn(3, 5, 17) + 1j * np.random.randn(3, 5, 17)).astype(
        np.complex128
    )
    input_dict = {"x": x, "fft_type": "IRFFT", "fft_lengths": (32,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.fft_2"] = lax_fft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.fft_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.fft_2'.")


check_valid('jax.lax.fft', generated_inputs['jax.lax.fft_2'], lib="jax", suffix=2)
