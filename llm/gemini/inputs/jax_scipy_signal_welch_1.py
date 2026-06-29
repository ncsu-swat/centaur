
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def welch_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D signal with default-like parameters
    x1 = np.random.randn(1024).astype(np.float32)
    input_dict1 = {
        'x': x1,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 256,
        'noverlap': 128,
        'nfft': 256,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different sampling frequency and window type
    x2 = np.random.randn(500).astype(np.float32)
    input_dict2 = {
        'x': x2,
        'fs': 100.0,
        'window': 'boxcar',
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 64,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D array, computing along the last axis (axis=-1)
    x3 = np.random.randn(10, 1000).astype(np.float32)
    input_dict3 = {
        'x': x3,
        'fs': 1000.0,
        'window': 'hamming',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D array, computing along axis 0
    x4 = np.random.randn(200, 5).astype(np.float32)
    input_dict4 = {
        'x': x4,
        'fs': 250.0,
        'window': 'hann',
        'nperseg': 50,
        'noverlap': 25,
        'nfft': 100,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'density',
        'axis': 0,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D array with custom parameters
    x5 = np.random.randn(4, 300, 8).astype(np.float32)
    input_dict5 = {
        'x': x5,
        'fs': 50.0,
        'window': 'bartlett',
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 64,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': 1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float64 1D array with larger segment size
    x6 = np.random.randn(2048).astype(np.float64)
    input_dict6 = {
        'x': x6,
        'fs': 2000.0,
        'window': 'hann',
        'nperseg': 512,
        'noverlap': 256,
        'nfft': 512,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Small segment length and no overlap
    x7 = np.random.randn(100).astype(np.float32)
    input_dict7 = {
        'x': x7,
        'fs': 10.0,
        'window': 'boxcar',
        'nperseg': 10,
        'noverlap': 0,
        'nfft': 10,
        'detrend': 'linear',
        'return_onesided': False,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Larger FFT size for zero padding
    x8 = np.random.randn(512).astype(np.float32)
    input_dict8 = {
        'x': x8,
        'fs': 100.0,
        'window': 'hamming',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 512,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: 2D array computing along negative axis (-2)
    x9 = np.random.randn(64, 500).astype(np.float32)
    input_dict9 = {
        'x': x9,
        'fs': 8000.0,
        'window': 'hann',
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 32,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -2,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: High-dimensional (4D) input
    x10 = np.random.randn(2, 3, 4, 256).astype(np.float32)
    input_dict10 = {
        'x': x10,
        'fs': 44100.0,
        'window': 'hann',
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 128,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["jax.scipy.signal.welch_1"] = welch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.welch_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.welch_1'.")


check_valid('jax.scipy.signal.welch', generated_inputs['jax.scipy.signal.welch_1'], lib="jax", suffix=1)
