
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class SafeTuple(tuple):
    def __array__(self, dtype=None, copy=None):
        return np.array(list(self), dtype=object)

def jax_scipy_signal_stft_inputs():
    list_of_inputs = []

    # Case 1: 1D input, standard sampling frequency, Hann window
    x = np.random.randn(1000).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 16000.0,
        'window': SafeTuple(('hann',)),
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

    # Case 2: 1D input, Hamming window, two-sided spectrum, even boundary
    x = np.random.randn(500).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 1.0,
        'window': SafeTuple(('hamming',)),
        'nperseg': 128,
        'noverlap': 64,
        'nfft': 256,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'even',
        'padded': False,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D input, Blackman window, high frequency, odd boundary
    x = np.random.randn(10, 1000).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 44100.0,
        'window': SafeTuple(('blackman',)),
        'nperseg': 512,
        'noverlap': 256,
        'nfft': 512,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'odd',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 2D input, Bartlett window, constant boundary
    x = np.random.randn(200, 50).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 8000.0,
        'window': SafeTuple(('bartlett',)),
        'nperseg': 64,
        'noverlap': 32,
        'nfft': 128,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'constant',
        'padded': False,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 3D input, Flattop window, two-sided
    x = np.random.randn(5, 10, 2000).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 100.0,
        'window': SafeTuple(('flattop',)),
        'nperseg': 200,
        'noverlap': 100,
        'nfft': 200,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D input with float64 precision, Bohman window
    x = np.random.randn(128).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 2.0,
        'window': SafeTuple(('bohman',)),
        'nperseg': 32,
        'noverlap': 16,
        'nfft': 64,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 2D float64 input, Blackmanharris window, even boundary
    x = np.random.randn(50, 500).astype(np.float64)
    input_dict = {
        'x': x,
        'fs': 50.0,
        'window': SafeTuple(('blackmanharris',)),
        'nperseg': 100,
        'noverlap': 50,
        'nfft': 100,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'even',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: 3D input, Nuttall window
    x = np.random.randn(2, 300, 4).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 10.0,
        'window': SafeTuple(('nuttall',)),
        'nperseg': 50,
        'noverlap': 25,
        'nfft': 50,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'odd',
        'padded': False,
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: Large 1D input, Parzen window
    x = np.random.randn(4000).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 22050.0,
        'window': SafeTuple(('parzen',)),
        'nperseg': 1024,
        'noverlap': 512,
        'nfft': 1024,
        'detrend': False,
        'return_onesided': True,
        'boundary': 'zeros',
        'padded': True,
        'axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 2D input with Triang window
    x = np.random.randn(100, 250).astype(np.float32)
    input_dict = {
        'x': x,
        'fs': 400.0,
        'window': SafeTuple(('triang',)),
        'nperseg': 80,
        'noverlap': 40,
        'nfft': 128,
        'detrend': False,
        'return_onesided': False,
        'boundary': 'constant',
        'padded': True,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.stft_3"] = jax_scipy_signal_stft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.stft_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.stft_3'.")


check_valid('jax.scipy.signal.stft', generated_inputs['jax.scipy.signal.stft_3'], lib="jax", suffix=3)
