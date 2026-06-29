
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __len__(self):
        return 0

def jax_scipy_signal_welch_inputs():
    list_of_inputs = []

    # Input 1: 1D array with 'hann' window
    x = np.random.randn(1024).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 100.0,
        'window': SafeTuple(('hann',)),
        'nperseg': 256,
        'noverlap': 128,
        'nfft': 256,
        'detrend': False,
        'return_onesided': True,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Small size, 'hamming' window, alternative scaling and average
    x = np.random.randn(128).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 1.0,
        'window': SafeTuple(('hamming',)),
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 64,
        'detrend': False,
        'return_onesided': False,
        'scaling': 'spectrum',
        'axis': 0,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, computing along axis 0 with 'boxcar' window
    x = np.random.randn(512, 10).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 1000.0,
        'window': SafeTuple(('boxcar',)),
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 128,
        'detrend': False,
        'return_onesided': True,
        'scaling': 'density',
        'axis': 0,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, computing along axis 1 with 'blackman' window
    x = np.random.randn(4, 1000).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 50.0,
        'window': SafeTuple(('blackman',)),
        'nperseg': 100,
        'noverlap': 50,
        'nfft': 100,
        'detrend': False,
        'return_onesided': False,
        'scaling': 'density',
        'axis': 1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, computing along axis -1 with 'bartlett' window
    x = np.random.randn(2, 3, 500).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 256.0,
        'window': SafeTuple(('bartlett',)),
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 128,
        'detrend': False,
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, computing along axis 1 with 'flattop' window
    x = np.random.randn(5, 300, 2).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 0.5,
        'window': SafeTuple(('flattop',)),
        'nperseg': 120,
        'noverlap': 0,
        'nfft': 256,
        'detrend': False,
        'return_onesided': True,
        'scaling': 'density',
        'axis': 1,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 1D array with 'parzen' window
    x = np.random.randn(2048).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 10.5,
        'window': SafeTuple(('parzen',)),
        'nperseg': 512,
        'noverlap': 256,
        'nfft': 512,
        'detrend': False,
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': 0,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High sampling frequency, short segments, 'bohman' window
    x = np.random.randn(100).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 44100.0,
        'window': SafeTuple(('bohman',)),
        'nperseg': 16,
        'noverlap': 8,
        'nfft': 16,
        'detrend': False,
        'return_onesided': False,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Median average, 'blackmanharris' window
    x = np.random.randn(256).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 20.0,
        'window': SafeTuple(('blackmanharris',)),
        'nperseg': 64,
        'noverlap': 16,
        'nfft': 64,
        'detrend': False,
        'return_onesided': True,
        'scaling': 'spectrum',
        'axis': 0,
        'average': 'median'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Specific overlap and FFT parameters, 'nuttall' window
    x = np.random.randn(1000).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 123.45,
        'window': SafeTuple(('nuttall',)),
        'nperseg': 200,
        'noverlap': 150,
        'nfft': 200,
        'detrend': False,
        'return_onesided': False,
        'scaling': 'density',
        'axis': -1,
        'average': 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.welch_2"] = jax_scipy_signal_welch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.welch_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.welch_2'.")


check_valid('jax.scipy.signal.welch', generated_inputs['jax.scipy.signal.welch_2'], lib="jax", suffix=2)
