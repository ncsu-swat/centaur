
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_stft_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    input_dict = {
        'x': np.random.randn(128).astype(np.float32),
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 32,
        'detrend': 'constant',
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, different window and no overlap
    input_dict = {
        'x': np.random.randn(4, 256).astype(np.float32),
        'fs': 100.0,
        'window': 'hamming',
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 64,
        'detrend': 'linear',
        'return_onesided': False,
        'boundary': 'even',
        'padded': False,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Axis = 0, bartlett window
    input_dict = {
        'x': np.random.randn(256, 2).astype(np.float32),
        'fs': 44100.0,
        'window': 'bartlett',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': 'constant',
        'return_onesided': True,
        'boundary': 'odd',
        'padded': True,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float64
    input_dict = {
        'x': np.random.randn(2, 3, 512).astype(np.float64),
        'fs': 0.5,
        'window': 'blackman',
        'nperseg': 256,
        'noverlap': 128,
        'nfft': 512,
        'detrend': 'linear',
        'return_onesided': True,
        'boundary': 'constant',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small nperseg, no overlap
    input_dict = {
        'x': np.random.randn(64).astype(np.float32),
        'fs': 8000.0,
        'window': 'boxcar',
        'nperseg': 8,
        'noverlap': 0,
        'nfft': 16,
        'detrend': 'constant',
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': False,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger nfft than nperseg
    input_dict = {
        'x': np.random.randn(1000).astype(np.float32),
        'fs': 16000.0,
        'window': 'hann',
        'nperseg': 100,
        'noverlap': 25,
        'nfft': 256,
        'detrend': 'linear',
        'return_onesided': True,
        'boundary': 'even',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis = 1
    input_dict = {
        'x': np.random.randn(4, 128, 2).astype(np.float32),
        'fs': 2.0,
        'window': 'hamming',
        'nperseg': 32,
        'noverlap': 8,
        'nfft': 32,
        'detrend': 'constant',
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': True,
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small overlap
    input_dict = {
        'x': np.random.randn(300).astype(np.float32),
        'fs': 1.5,
        'window': 'hann',
        'nperseg': 50,
        'noverlap': 10,
        'nfft': 100,
        'detrend': 'constant',
        'return_onesided': True,
        'boundary': 'odd',
        'padded': False,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High overlap
    input_dict = {
        'x': np.random.randn(500).astype(np.float32),
        'fs': 10.0,
        'window': 'bartlett',
        'nperseg': 200,
        'noverlap': 190,
        'nfft': 200,
        'detrend': 'linear',
        'return_onesided': False,
        'boundary': 'constant',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, axis = 0 with minimal size
    input_dict = {
        'x': np.random.randn(10, 10).astype(np.float32),
        'fs': 0.1,
        'window': 'boxcar',
        'nperseg': 10,
        'noverlap': 5,
        'nfft': 10,
        'detrend': 'constant',
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.stft_4"] = generate_stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_4'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_4'], lib="jax", suffix=4)
