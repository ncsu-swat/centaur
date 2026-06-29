
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def istft_inputs():
    list_of_inputs = []

    # Input 1, standard single-channel STFT
    Zxx = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, high sampling frequency and float64 dtypes
    Zxx = (np.random.randn(5, 10) + 1j * np.random.randn(5, 10)).astype(np.complex128)
    fs = np.array(16000.0, dtype=np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hamming',
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, two-sided spectrum (input_onesided=False)
    Zxx = (np.random.randn(8, 6) + 1j * np.random.randn(8, 6)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hann',
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': False,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, multi-dimensional signal with batch dimension
    Zxx = (np.random.randn(2, 3, 5) + 1j * np.random.randn(2, 3, 5)).astype(np.complex64)
    fs = np.array(2.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, transposed time and frequency axes
    Zxx = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': True,
        'boundary': True,
        'time_axis': 0,
        'freq_axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, boxcar window and zero overlap
    Zxx = (np.random.randn(3, 8) + 1j * np.random.randn(3, 8)).astype(np.complex64)
    fs = np.array(44100.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'boxcar',
        'nperseg': 4,
        'noverlap': 0,
        'nfft': 4,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, larger segments with bartlett window
    Zxx = (np.random.randn(9, 12) + 1j * np.random.randn(9, 12)).astype(np.complex128)
    fs = np.array(8000.0, dtype=np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'bartlett',
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 16,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, two-sided larger segments with blackman window
    Zxx = (np.random.randn(16, 12) + 1j * np.random.randn(16, 12)).astype(np.complex128)
    fs = np.array(8000.0, dtype=np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'blackman',
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 16,
        'input_onesided': False,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, custom larger overlap (75%)
    Zxx = (np.random.randn(2, 5, 20) + 1j * np.random.randn(2, 5, 20)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hann',
        'nperseg': 8,
        'noverlap': 6,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, batch dimension at another position
    Zxx = (np.random.randn(5, 2, 20) + 1j * np.random.randn(5, 2, 20)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': 'hann',
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': False,
        'time_axis': 2,
        'freq_axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_4"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_4'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_4'], lib="jax", suffix=4)
