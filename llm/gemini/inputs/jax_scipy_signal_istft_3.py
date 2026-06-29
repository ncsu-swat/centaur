
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def istft_inputs():
    list_of_inputs = []

    # Case 1
    Zxx = (np.random.randn(129, 10) + 1j * np.random.randn(129, 10)).astype(np.complex64)
    fs = 16000.0
    window = np.hanning(256).astype(np.float32)
    nperseg = 256
    noverlap = 128
    nfft = 256
    input_onesided = True
    boundary = True
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 2
    Zxx = (np.random.randn(256, 10) + 1j * np.random.randn(256, 10)).astype(np.complex64)
    fs = 8000.0
    window = np.hanning(256).astype(np.float32)
    nperseg = 256
    noverlap = 128
    nfft = 256
    input_onesided = False
    boundary = False
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 3
    Zxx = (np.random.randn(10, 129) + 1j * np.random.randn(10, 129)).astype(np.complex64)
    fs = 44100.0
    window = np.hamming(256).astype(np.float32)
    nperseg = 256
    noverlap = 64
    nfft = 256
    input_onesided = True
    boundary = True
    time_axis = 0
    freq_axis = 1
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 4
    Zxx = (np.random.randn(2, 65, 20) + 1j * np.random.randn(2, 65, 20)).astype(np.complex64)
    fs = 1.0
    window = np.hanning(128).astype(np.float32)
    nperseg = 128
    noverlap = 64
    nfft = 128
    input_onesided = True
    boundary = True
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 5
    Zxx = (np.random.randn(129, 15) + 1j * np.random.randn(129, 15)).astype(np.complex64)
    fs = 2.0
    window = np.hanning(100).astype(np.float32)
    nperseg = 100
    noverlap = 50
    nfft = 256
    input_onesided = True
    boundary = True
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 6
    Zxx = (np.random.randn(65, 8) + 1j * np.random.randn(65, 8)).astype(np.complex64)
    fs = 100.0
    window = np.hanning(128).astype(np.float32)
    nperseg = 128
    noverlap = 0
    nfft = 128
    input_onesided = True
    boundary = False
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 7
    Zxx = (np.random.randn(257, 12) + 1j * np.random.randn(257, 12)).astype(np.complex128)
    fs = 1.0
    window = np.ones(512, dtype=np.float64)
    nperseg = 512
    noverlap = 256
    nfft = 512
    input_onesided = True
    boundary = True
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 8
    Zxx = (np.random.randn(33, 30) + 1j * np.random.randn(33, 30)).astype(np.complex64)
    fs = 1000.0
    window = np.hamming(64).astype(np.float32)
    nperseg = 64
    noverlap = 48
    nfft = 64
    input_onesided = True
    boundary = True
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 9
    Zxx = (np.random.randn(33, 2, 15) + 1j * np.random.randn(33, 2, 15)).astype(np.complex64)
    fs = 1.5
    window = np.hanning(64).astype(np.float32)
    nperseg = 64
    noverlap = 32
    nfft = 64
    input_onesided = True
    boundary = False
    time_axis = 2
    freq_axis = 0
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    # Case 10
    Zxx = (np.random.randn(128, 5) + 1j * np.random.randn(128, 5)).astype(np.complex64)
    fs = 500.0
    window = np.hanning(32).astype(np.float32)
    nperseg = 32
    noverlap = 16
    nfft = 128
    input_onesided = False
    boundary = True
    time_axis = -1
    freq_axis = -2
    list_of_inputs.append({
        'Zxx': Zxx, 'fs': fs, 'window': window, 'nperseg': nperseg,
        'noverlap': noverlap, 'nfft': nfft, 'input_onesided': input_onesided,
        'boundary': boundary, 'time_axis': time_axis, 'freq_axis': freq_axis
    })

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_3"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_3'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_3'], lib="jax", suffix=3)
