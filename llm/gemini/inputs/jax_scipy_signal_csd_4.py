
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def csd_inputs():
    list_of_inputs = []

    # Input 1, valid - 1D arrays with standard settings
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
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
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid - 2D arrays, median average, spectrum scaling
    x = np.random.randn(500, 10).astype(np.float32)
    y = np.random.randn(500, 10).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 100.0,
        'window': 'hamming',
        'nperseg': 100,
        'noverlap': 50,
        'nfft': 128,
        'detrend': 'linear',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': 0,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid - float64 data types and boxcar window
    x = np.random.randn(1024).astype(np.float64)
    y = np.random.randn(1024).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 44100.0,
        'window': 'boxcar',
        'nperseg': 512,
        'noverlap': 256,
        'nfft': 512,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid - 3D arrays with custom axis
    x = np.random.randn(5, 500, 3).astype(np.float32)
    y = np.random.randn(5, 500, 3).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 2.5,
        'window': 'hann',
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 64,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'density',
        'axis': 1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - small segment size
    x = np.random.randn(16).astype(np.float32)
    y = np.random.randn(16).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 8.0,
        'window': 'hann',
        'nperseg': 8,
        'noverlap': 4,
        'nfft': 8,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - large dimensions and FFT zero padding
    x = np.random.randn(5000).astype(np.float32)
    y = np.random.randn(5000).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 1000.0,
        'window': 'hann',
        'nperseg': 1024,
        'noverlap': 512,
        'nfft': 2048,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid - 3D arrays with negative axis index
    x = np.random.randn(4, 1000, 2).astype(np.float32)
    y = np.random.randn(4, 1000, 2).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 50.0,
        'window': 'hamming',
        'nperseg': 200,
        'noverlap': 100,
        'nfft': 256,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': -2,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - zero overlap
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 100,
        'noverlap': 0,
        'nfft': 100,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid - non-power-of-two FFT length
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 12.34,
        'window': 'hann',
        'nperseg': 150,
        'noverlap': 50,
        'nfft': 300,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid - 4D input array
    x = np.random.randn(2, 3, 500, 4).astype(np.float32)
    y = np.random.randn(2, 3, 500, 4).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 250.0,
        'window': 'hann',
        'nperseg': 256,
        'noverlap': 128,
        'nfft': 256,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': 2,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.csd_4"] = csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_4'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_4'], lib="jax", suffix=4)
