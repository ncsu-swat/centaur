
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_signal_istft_inputs():
    list_of_inputs = []

    # Input 1: Standard one-sided, default axes, Hanning window with 50% overlap
    nperseg = 256
    nfft = 256
    Zxx = (np.random.randn(129, 10) + 1j * np.random.randn(129, 10)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 128,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two-sided, 50% overlap, 16kHz sampling rate
    nperseg = 128
    nfft = 128
    Zxx = (np.random.randn(128, 15) + 1j * np.random.randn(128, 15)).astype(np.complex64)
    fs = np.array(16000.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 64,
        'nfft': nfft,
        'input_onesided': False,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Transposed axes (freq_axis=0, time_axis=1), no boundary
    nperseg = 64
    nfft = 64
    Zxx = (np.random.randn(33, 8) + 1j * np.random.randn(33, 8)).astype(np.complex64)
    fs = np.array(44100.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 32,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': False,
        'time_axis': 1,
        'freq_axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched Zxx with float64/complex128 precision
    nperseg = 256
    nfft = 256
    Zxx = (np.random.randn(2, 129, 10) + 1j * np.random.randn(2, 129, 10)).astype(np.complex128)
    fs = np.array(8000.0, dtype=np.float64)
    window = np.hanning(nperseg).astype(np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 128,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: nfft > nperseg, Hanning window
    nperseg = 100
    nfft = 128
    Zxx = (np.random.randn(65, 12) + 1j * np.random.randn(65, 12)).astype(np.complex64)
    fs = np.array(1000.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 50,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small window size, no boundary
    nperseg = 32
    nfft = 32
    Zxx = (np.random.randn(17, 5) + 1j * np.random.randn(17, 5)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 16,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 75% overlap, swapped negative axes
    nperseg = 64
    nfft = 64
    Zxx = (np.random.randn(10, 64) + 1j * np.random.randn(10, 64)).astype(np.complex64)
    fs = np.array(2.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 48,
        'nfft': nfft,
        'input_onesided': False,
        'boundary': True,
        'time_axis': -2,
        'freq_axis': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multi-dimensional batch
    nperseg = 256
    nfft = 256
    Zxx = (np.random.randn(3, 2, 129, 15) + 1j * np.random.randn(3, 2, 129, 15)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 128,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small sizes
    nperseg = 8
    nfft = 8
    Zxx = (np.random.randn(5, 4) + 1j * np.random.randn(5, 4)).astype(np.complex64)
    fs = np.array(8.0, dtype=np.float32)
    window = np.hanning(nperseg).astype(np.float32)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 4,
        'nfft': nfft,
        'input_onesided': True,
        'boundary': True,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Two-sided, float64 precision, no boundary
    nperseg = 16
    nfft = 16
    Zxx = (np.random.randn(16, 6) + 1j * np.random.randn(16, 6)).astype(np.complex128)
    fs = np.array(1.0, dtype=np.float64)
    window = np.hanning(nperseg).astype(np.float64)
    input_dict = {
        'Zxx': Zxx,
        'fs': fs,
        'window': window,
        'nperseg': nperseg,
        'noverlap': 8,
        'nfft': nfft,
        'input_onesided': False,
        'boundary': False,
        'time_axis': -1,
        'freq_axis': -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_6"] = jax_scipy_signal_istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_6'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_6'], lib="jax", suffix=6)
