
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fft_inputs():
    list_of_inputs = []

    # Case 1: 1D FFT (complex64)
    x = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    list_of_inputs.append({"x": x, "fft_type": "FFT", "fft_lengths": (8,)})

    # Case 2: 1D IFFT (complex128)
    x = (np.random.randn(16) + 1j * np.random.randn(16)).astype(np.complex128)
    list_of_inputs.append({"x": x, "fft_type": "IFFT", "fft_lengths": (16,)})

    # Case 3: 2D FFT (complex64)
    x = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
    list_of_inputs.append({"x": x, "fft_type": "FFT", "fft_lengths": (4, 8)})

    # Case 4: 2D IFFT (complex64)
    x = (np.random.randn(8, 16) + 1j * np.random.randn(8, 16)).astype(np.complex64)
    list_of_inputs.append({"x": x, "fft_type": "IFFT", "fft_lengths": (8, 16)})

    # Case 5: 1D RFFT (float32)
    x = np.random.randn(16).astype(np.float32)
    list_of_inputs.append({"x": x, "fft_type": "RFFT", "fft_lengths": (16,)})

    # Case 6: 1D IRFFT (complex64)
    x = (np.random.randn(9) + 1j * np.random.randn(9)).astype(np.complex64)
    list_of_inputs.append({"x": x, "fft_type": "IRFFT", "fft_lengths": (16,)})

    # Case 7: 2D RFFT (float64)
    x = np.random.randn(4, 8).astype(np.float64)
    list_of_inputs.append({"x": x, "fft_type": "RFFT", "fft_lengths": (4, 8)})

    # Case 8: 2D IRFFT (complex64)
    x = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(np.complex64)
    list_of_inputs.append({"x": x, "fft_type": "IRFFT", "fft_lengths": (4, 8)})

    # Case 9: 3D FFT (complex64) with batch dimension
    x = (np.random.randn(2, 4, 4, 4) + 1j * np.random.randn(2, 4, 4, 4)).astype(np.complex64)
    list_of_inputs.append({"x": x, "fft_type": "FFT", "fft_lengths": (4, 4, 4)})

    # Case 10: 1D RFFT with batch dimension
    x = np.random.randn(5, 32).astype(np.float32)
    list_of_inputs.append({"x": x, "fft_type": "RFFT", "fft_lengths": (32,)})

    return list_of_inputs

generated_inputs["jax.lax.fft_1"] = fft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.fft_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.fft_1'.")


check_valid('jax.lax.fft', generated_inputs['jax.lax.fft_1'], lib="jax", suffix=1)
