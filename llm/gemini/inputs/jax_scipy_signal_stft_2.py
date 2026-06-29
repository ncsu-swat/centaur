
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_stft_inputs():
    list_of_inputs = []

    # Input 1
    x = np.random.randn(1000).astype(np.float32)
    window = np.hanning(256).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 16000.0,
        'window': window,
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

    # Input 2
    x = np.random.randn(500).astype(np.float32)
    window = np.ones(128).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 1.0,
        'window': window,
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 256,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'even',
        'padded': False,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.random.randn(10, 1000).astype(np.float32)
    window = np.hamming(512).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 8000.0,
        'window': window,
        'nperseg': 512,
        'noverlap': 256,
        'nfft': 512,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'odd',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.random.randn(1000, 5).astype(np.float32)
    window = np.blackman(200).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 44100.0,
        'window': window,
        'nperseg': 200,
        'noverlap': 100,
        'nfft': 256,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'constant',
        'padded': True,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.random.randn(100).astype(np.float64)
    window = np.hamming(16).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 100.0,
        'window': window,
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 32,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': False,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.random.randn(2, 3, 400).astype(np.float32)
    window = np.hanning(64).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 2.5,
        'window': window,
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 64,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.random.randn(2, 400, 3).astype(np.float32)
    window = np.ones(100).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 1000.0,
        'window': window,
        'nperseg': 100,
        'noverlap': 50,
        'nfft': 128,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'even',
        'padded': True,
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
    window = np.hanning(128).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 120.0,
        'window': window,
        'nperseg': 128,
        'noverlap': 0,
        'nfft': 128,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.randn(2000).astype(np.float32)
    window = np.hamming(500).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 50.0,
        'window': window,
        'nperseg': 500,
        'noverlap': 400,
        'nfft': 512,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.random.randn(300).astype(np.float32)
    window = np.hanning(30).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 8.0,
        'window': window,
        'nperseg': 30,
        'noverlap': 15,
        'nfft': 64,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'odd',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.stft_2"] = jax_scipy_signal_stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_2'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_2'], lib="jax", suffix=2)
