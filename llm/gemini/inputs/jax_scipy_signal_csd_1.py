
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_csd_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32
    x = np.random.randn(128).astype(np.float32)
    y = np.random.randn(128).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 32,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64, different fs, nfft, detrend, scaling, average
    x = np.random.randn(256).astype(np.float64)
    y = np.random.randn(256).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 100.0,
        'window': 'hann',
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 128,
        'detrend': 'linear',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': 0,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, axis=-1
    x = np.random.randn(10, 512).astype(np.float32)
    y = np.random.randn(10, 512).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 50.0,
        'window': 'hann',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, axis=0, no overlap
    x = np.random.randn(512, 5).astype(np.float32)
    y = np.random.randn(512, 5).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 250.0,
        'window': 'hann',
        'nperseg': 64,
        'noverlap': 0,
        'nfft': 64,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': 0,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, float32, median average
    x = np.random.randn(2, 3, 200).astype(np.float32)
    y = np.random.randn(2, 3, 200).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 10.0,
        'window': 'hann',
        'nperseg': 50,
        'noverlap': 25,
        'nfft': 100,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Short signal, return_onesided=False
    x = np.random.randn(64).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 1.0,
        'window': 'hann',
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 16,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 with high fs, linear detrend
    x = np.random.randn(1000).astype(np.float64)
    y = np.random.randn(1000).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 1000.0,
        'window': 'hann',
        'nperseg': 100,
        'noverlap': 0,
        'nfft': 100,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32, larger zero-padded FFT length (nfft > nperseg)
    x = np.random.randn(500).astype(np.float32)
    y = np.random.randn(500).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 44100.0,
        'window': 'hann',
        'nperseg': 100,
        'noverlap': 50,
        'nfft': 512,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64, non-default axis, return_onesided=False
    x = np.random.randn(4, 128).astype(np.float64)
    y = np.random.randn(4, 128).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 2.5,
        'window': 'hann',
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 64,
        'detrend': 'linear',
        'return_onesided': False,
        'scaling': 'density',
        'axis': 1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32, axis in the middle (axis=1)
    x = np.random.randn(4, 256, 3).astype(np.float32)
    y = np.random.randn(4, 256, 3).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'fs': 8000.0,
        'window': 'hann',
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': 1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.csd_1"] = jax_scipy_signal_csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_1'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_1'], lib="jax", suffix=1)
