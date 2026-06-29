
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def welch_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D real signal
    nperseg_1 = 32
    input_dict_1 = {
        'x': np.random.randn(128).astype(np.float32),
        'fs': 1.0,
        'window': np.hanning(nperseg_1).astype(np.float32),
        'nperseg': nperseg_1,
        'noverlap': 16,
        'nfft': 32,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Complex 1D signal with different sampling frequency and median average
    nperseg_2 = 64
    input_dict_2 = {
        'x': (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64),
        'fs': 100.0,
        'window': np.hamming(nperseg_2).astype(np.float32),
        'nperseg': nperseg_2,
        'noverlap': 32,
        'nfft': 64,
        'detrend': 'linear',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Multi-dimensional array (2D) along axis 0, double precision
    nperseg_3 = 50
    input_dict_3 = {
        'x': np.random.randn(100, 10).astype(np.float64),
        'fs': 2000.0,
        'window': np.blackman(nperseg_3).astype(np.float64),
        'nperseg': nperseg_3,
        'noverlap': 25,
        'nfft': 128,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': 0,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D array along axis 1 with false return_onesided
    nperseg_4 = 100
    input_dict_4 = {
        'x': np.random.randn(5, 200, 3).astype(np.float32),
        'fs': 50.0,
        'window': np.hanning(nperseg_4).astype(np.float32),
        'nperseg': nperseg_4,
        'noverlap': 50,
        'nfft': 100,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': 1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Minimal overlap, rectangular window (all ones)
    nperseg_5 = 100
    input_dict_5 = {
        'x': np.random.randn(500).astype(np.float32),
        'fs': 10.0,
        'window': np.ones(nperseg_5).astype(np.float32),
        'nperseg': nperseg_5,
        'noverlap': 0,
        'nfft': 100,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Maximum possible overlap
    nperseg_6 = 256
    input_dict_6 = {
        'x': np.random.randn(1000).astype(np.float32),
        'fs': 44100.0,
        'window': np.hanning(nperseg_6).astype(np.float32),
        'nperseg': nperseg_6,
        'noverlap': 255,
        'nfft': 256,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Large FFT zero-padding
    nperseg_7 = 16
    input_dict_7 = {
        'x': np.random.randn(100).astype(np.float32),
        'fs': 1.0,
        'window': np.hanning(nperseg_7).astype(np.float32),
        'nperseg': nperseg_7,
        'noverlap': 8,
        'nfft': 1024,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: 2D array with float32 Hamming window
    nperseg_8 = 128
    input_dict_8 = {
        'x': np.random.randn(4, 1024).astype(np.float32),
        'fs': 1000.0,
        'window': np.hamming(nperseg_8).astype(np.float32),
        'nperseg': nperseg_8,
        'noverlap': 64,
        'nfft': 128,
        'detrend': 'constant',
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Real input returning two-sided spectrum with median average
    nperseg_9 = 32
    input_dict_9 = {
        'x': np.random.randn(128).astype(np.float32),
        'fs': 1.0,
        'window': np.hanning(nperseg_9).astype(np.float32),
        'nperseg': nperseg_9,
        'noverlap': 16,
        'nfft': 32,
        'detrend': 'constant',
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: 4D tensor, processing along axis 2
    nperseg_10 = 50
    input_dict_10 = {
        'x': np.random.randn(2, 2, 200, 2).astype(np.float32),
        'fs': 12.5,
        'window': np.blackman(nperseg_10).astype(np.float32),
        'nperseg': nperseg_10,
        'noverlap': 10,
        'nfft': 64,
        'detrend': 'linear',
        'return_onesided': True,
        'scaling': 'density',
        'axis': 2,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["jax.scipy.signal.welch_3"] = welch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.welch_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.welch_3'.")


check_valid('jax.scipy.signal.welch', generated_inputs['jax.scipy.signal.welch_3'], lib="jax", suffix=3)
