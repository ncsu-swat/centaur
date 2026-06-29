
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array([0.0])

def istft_inputs():
    list_of_inputs = []

    # 1. Standard one-sided ISTFT
    shape = (9, 10)  # nfft // 2 + 1 = 9 for nfft = 16
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
        'window': SafeTuple(('hann',)),
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 16,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. Two-sided ISTFT with custom fs
    shape = (32, 15)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex128)
    input_dict = {
        'Zxx': Zxx,
        'fs': 16000.0,
        'window': SafeTuple(('hamming',)),
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 32,
        'input_onesided': False,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. One-sided with custom window
    shape = (33, 8)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 44100.0,
        'window': SafeTuple(('bartlett',)),
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 64,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Multi-dimensional Zxx
    shape = (2, 5, 12)  # batch=2, freq=5, time=12
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 8000.0,
        'window': SafeTuple(('blackman',)),
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Multi-dimensional Zxx with custom axes
    shape = (16, 20, 3)  # freq=16, time=20, batch=3
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex128)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
        'window': SafeTuple(('flattop',)),
        'nperseg': 16,
        'noverlap': 12,
        'nfft': 16,
        'input_onesided': False,
        'boundary': True,
        'time_axis': 1,
        'freq_axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. Large frequency bin
    shape = (65, 20)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 8000.0,
        'window': SafeTuple(('bohman',)),
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 128,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Low fs value
    shape = (5, 6)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 0.5,
        'window': SafeTuple(('parzen',)),
        'nperseg': 8,
        'noverlap': 2,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Medium size complex128
    shape = (7, 10)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex128)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
        'window': SafeTuple(('nuttall',)),
        'nperseg': 12,
        'noverlap': 6,
        'nfft': 12,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Two-sided with large overlap
    shape = (24, 14)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 2.0,
        'window': SafeTuple(('triang',)),
        'nperseg': 24,
        'noverlap': 18,
        'nfft': 24,
        'input_onesided': False,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. Small test case with boundary False
    shape = (6, 12)
    Zxx = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 100.0,
        'window': SafeTuple(('boxcar',)),
        'nperseg': 10,
        'noverlap': 5,
        'nfft': 10,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_2"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_2'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_2'], lib="jax", suffix=2)
