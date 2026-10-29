
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fft_inputs():
    list_of_inputs = []

    # Input 1: 1D FFT (complex64)
    x = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "FFT", "fft_lengths": (8,)})
    )

    # Input 2: 1D IFFT (complex64)
    x = (np.random.randn(16) + 1j * np.random.randn(16)).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "IFFT", "fft_lengths": (16,)})
    )

    # Input 3: 2D FFT (complex64)
    x = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "FFT", "fft_lengths": (4, 8)})
    )

    # Input 4: Batched 1D FFT with negative values
    x = (np.random.uniform(-5.0, 5.0, (2, 3, 32)) + 1j * np.random.uniform(
        -5.0, 5.0, (2, 3, 32)
    )).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "FFT", "fft_lengths": (32,)})
    )

    # Input 5: Batched 2D IFFT (complex64)
    x = (np.random.randn(3, 8, 8) + 1j * np.random.randn(3, 8, 8)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "IFFT", "fft_lengths": (8, 8)})
    )

    # Input 6: 3D FFT (complex64)
    x = (np.random.randn(2, 4, 8) + 1j * np.random.randn(2, 4, 8)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "FFT", "fft_lengths": (2, 4, 8)})
    )

    # Input 7: 1D RFFT (float32)
    x = np.random.randn(16).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "RFFT", "fft_lengths": (16,)})
    )

    # Input 8: 2D RFFT (float32)
    x = np.random.randn(6, 12).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "RFFT", "fft_lengths": (6, 12)})
    )

    # Input 9: Batched 2D RFFT with negative values (float32)
    x = np.random.uniform(-10.0, 10.0, (2, 4, 16)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "RFFT", "fft_lengths": (4, 16)})
    )

    # Input 10: 1D IRFFT even length (complex64)
    x = (np.random.randn(9) + 1j * np.random.randn(9)).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "IRFFT", "fft_lengths": (16,)})
    )

    # Input 11: 2D IRFFT (complex64)
    x = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "IRFFT", "fft_lengths": (4, 10)})
    )

    # Input 12: Batched 1D IRFFT odd length (complex64)
    x = (np.random.randn(5, 8) + 1j * np.random.randn(5, 8)).astype(
        np.complex64
    )
    list_of_inputs.append(
        copy.deepcopy({"x": x, "fft_type": "IRFFT", "fft_lengths": (15,)})
    )

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
