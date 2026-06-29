
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_istft_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    Zxx = (np.random.randn(3, 5) + 1j * np.random.randn(3, 5)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
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

    # Input 2: Two-sided STFT input
    Zxx = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 16000.0,
        'window': 'hamming',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': False,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger segment size and complex128
    Zxx = (np.random.randn(5, 10) + 1j * np.random.randn(5, 10)).astype(np.complex128)
    input_dict = {
        'Zxx': Zxx,
        'fs': 44100.0,
        'window': 'hann',
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boundary False
    Zxx = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 100.0,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: nfft larger than nperseg
    Zxx = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 8000.0,
        'window': 'bartlett',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 8,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional input with batch dimension
    Zxx = (np.random.randn(2, 3, 5) + 1j * np.random.randn(2, 3, 5)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
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

    # Input 7: Swapped time and frequency axes
    Zxx = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -2,
        'freq_axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different window function
    Zxx = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 2.0,
        'window': 'boxcar',
        'nperseg': 6,
        'noverlap': 3,
        'nfft': 6,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Zero overlap
    Zxx = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 0,
        'nfft': 4,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Custom axes mapping in a 3D array
    Zxx = (np.random.randn(3, 2, 5) + 1j * np.random.randn(3, 2, 5)).astype(np.complex64)
    input_dict = {
        'Zxx': Zxx,
        'fs': 0.5,
        'window': 'hann',
        'nperseg': 4,
        'noverlap': 2,
        'nfft': 4,
        'input_onesided': True,
        'boundary': True,
        'time_axis': 2,
        'freq_axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_1"] = jax_scipy_signal_istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_1'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_1'], lib="jax", suffix=1)
