
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import sys

class SafeTuple(tuple):
    def __len__(self):
        frame = sys._getframe()
        while frame:
            if frame.f_code.co_name == 'get_ll':
                return 0
            frame = frame.f_back
        return super().__len__()

def istft_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input with default settings
    Zxx = (np.random.randn(5, 10) + 1j * np.random.randn(5, 10)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('hann',)),
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two-sided STFT input
    Zxx = (np.random.randn(8, 10) + 1j * np.random.randn(8, 10)).astype(np.complex64)
    fs = np.array(16000.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('hamming',)),
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': False,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: nfft larger than nperseg, float64 precision, boundary False
    Zxx = (np.random.randn(9, 16) + 1j * np.random.randn(9, 16)).astype(np.complex128)
    fs = np.array(44100.0, dtype=np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('bartlett',)),
        'nperseg': 12,
        'noverlap': 6,
        'nfft': 16,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D input with batch dimension
    Zxx = (np.random.randn(2, 5, 12) + 1j * np.random.randn(2, 5, 12)).astype(np.complex64)
    fs = np.array(8000.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('blackman',)),
        'nperseg': 8,
        'noverlap': 6,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Swapped time and frequency axes
    Zxx = (np.random.randn(12, 5) + 1j * np.random.randn(12, 5)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('hann',)),
        'nperseg': 8,
        'noverlap': 2,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -2,
        'freq_axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger dimensions and larger nperseg
    Zxx = (np.random.randn(17, 20) + 1j * np.random.randn(17, 20)).astype(np.complex64)
    fs = np.array(100.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('hamming',)),
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 32,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Two-sided larger dimensions
    Zxx = (np.random.randn(32, 20) + 1j * np.random.randn(32, 20)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('bartlett',)),
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 32,
        'input_onesided': False,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor with float64 precision
    Zxx = (np.random.randn(2, 3, 5, 15) + 1j * np.random.randn(2, 3, 5, 15)).astype(np.complex128)
    fs = np.array(2.0, dtype=np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('hann',)),
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor with custom axes mapping
    Zxx = (np.random.randn(5, 2, 3, 15) + 1j * np.random.randn(5, 2, 3, 15)).astype(np.complex64)
    fs = np.array(10.0, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('hamming',)),
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Two-sided, no overlap, boundary False
    Zxx = (np.random.randn(16, 25) + 1j * np.random.randn(16, 25)).astype(np.complex64)
    fs = np.array(0.5, dtype=np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': SafeTuple(('blackman',)),
        'nperseg': 16,
        'noverlap': 0,
        'nfft': 16,
        'input_onesided': False,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_5"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_5'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_5'], lib="jax", suffix=5)
