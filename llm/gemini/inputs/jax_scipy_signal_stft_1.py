
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_stft_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D real signal, default settings
    x = np.random.randn(1000).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 256,
        'noverlap': 128,
        'nfft': 256,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D signal, custom fs, hamming window, axis=0
    x = np.random.randn(500, 10).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 100.0,
        'window': 'hamming',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'even',
        'padded': True,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex 1D signal, two-sided spectrum
    x = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
    input_dict = {
        'x': x,
        'fs': 1000.0,
        'window': 'boxcar',
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 64,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': False,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D signal, custom axis, larger nfft for interpolation
    x = np.random.randn(8, 256, 4).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 50.0,
        'window': 'hann',
        'nperseg': 100,
        'noverlap': 50,
        'nfft': 256,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'odd',
        'padded': True,
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Double precision 1D signal, no overlap
    x = np.random.randn(2000).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 44100.0,
        'window': 'blackman',
        'nperseg': 512,
        'noverlap': 0,
        'nfft': 512,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'constant',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small signal, small segment size, no padding
    x = np.random.randn(100).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 8000.0,
        'window': 'bartlett',
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 32,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': False,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D signal with negative axis
    x = np.random.randn(32, 512).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 16000.0,
        'window': 'hamming',
        'nperseg': 256,
        'noverlap': 192,
        'nfft': 512,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D signal, large segment, odd overlap
    x = np.random.randn(4096).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 2.0,
        'window': 'hann',
        'nperseg': 1024,
        'noverlap': 256,
        'nfft': 1024,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'even',
        'padded': True,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D signal, processing along axis 2
    x = np.random.randn(2, 3, 128).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 0.5,
        'window': 'boxcar',
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 64,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'constant',
        'padded': False,
        'axis': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex signal with double precision
    x = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex128)
    input_dict = {
        'x': x,
        'fs': 22050.0,
        'window': 'hann',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.stft_1"] = jax_scipy_signal_stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_1'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_1'], lib="jax", suffix=1)
