
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def istft_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32/complex64 with standard Hann-like window
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 10
    Zxx = (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64/complex128 2D with rectangular window
    nperseg = 32
    noverlap = 16
    nfft = 32
    nfreq = nfft // 2 + 1
    ntime = 12
    Zxx = (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)).astype(np.complex128)
    fs = np.array(44100.0, dtype=np.float64)
    window = np.ones(nperseg, dtype=np.float64)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Transposed axes (time_axis=0, freq_axis=1)
    nperseg = 8
    noverlap = 4
    nfft = 8
    nfreq = nfft // 2 + 1
    ntime = 8
    Zxx = (np.random.randn(ntime, nfreq) + 1j * np.random.randn(ntime, nfreq)).astype(np.complex64)
    fs = np.array(16000.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": 0,
        "freq_axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D input with batch dimension (batch, freq, time)
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 8
    batch = 3
    Zxx = (np.random.randn(batch, nfreq, ntime) + 1j * np.random.randn(batch, nfreq, ntime)).astype(np.complex64)
    fs = np.array(100.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Two-sided STFT (input_onesided=False)
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft
    ntime = 8
    Zxx = (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    window = np.ones(nperseg, dtype=np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": False,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boundary=False
    nperseg = 16
    noverlap = 8
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 10
    Zxx = (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)).astype(np.complex64)
    fs = np.array(8000.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Over-sampled FFT (nfft > nperseg)
    nperseg = 16
    noverlap = 8
    nfft = 32
    nfreq = nfft // 2 + 1
    ntime = 10
    Zxx = (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)).astype(np.complex64)
    fs = np.array(2.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: No overlap (noverlap=0) with rectangular window and boundary=False
    nperseg = 16
    noverlap = 0
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 6
    Zxx = (np.random.randn(nfreq, ntime) + 1j * np.random.randn(nfreq, ntime)).astype(np.complex64)
    fs = np.array(1.0, dtype=np.float32)
    window = np.ones(nperseg, dtype=np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": False,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensor with multiple batch dimensions
    nperseg = 8
    noverlap = 4
    nfft = 8
    nfreq = nfft // 2 + 1
    ntime = 10
    Zxx = (np.random.randn(2, 4, nfreq, ntime) + 1j * np.random.randn(2, 4, nfreq, ntime)).astype(np.complex64)
    fs = np.array(50.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": -1,
        "freq_axis": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor with custom axes (ntime, batch, nfreq)
    nperseg = 16
    noverlap = 12
    nfft = 16
    nfreq = nfft // 2 + 1
    ntime = 15
    batch = 2
    Zxx = (np.random.randn(ntime, batch, nfreq) + 1j * np.random.randn(ntime, batch, nfreq)).astype(np.complex64)
    fs = np.array(250.0, dtype=np.float32)
    window = np.hamming(nperseg).astype(np.float32)
    input_dict = {
        "Zxx": Zxx,
        "fs": fs,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "nfft": nfft,
        "input_onesided": True,
        "boundary": True,
        "time_axis": 0,
        "freq_axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.istft_6"] = istft_inputs()

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
