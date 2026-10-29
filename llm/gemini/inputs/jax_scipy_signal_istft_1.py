
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import scipy.signal


def istft_inputs():
    list_of_inputs = []

    # Case 1: Standard 1D signal, default hann window
    x1 = np.random.randn(64).astype(np.float64)
    fs1 = 1.0
    window1 = "hann"
    nperseg1 = 16
    noverlap1 = 8
    nfft1 = 16
    onesided1 = True
    boundary1 = True
    _, _, zxx1 = scipy.signal.stft(
        x1,
        fs=fs1,
        window=window1,
        nperseg=nperseg1,
        noverlap=noverlap1,
        nfft=nfft1,
        return_onesided=onesided1,
        boundary="zeros",
    )
    list_of_inputs.append(
        {
            "Zxx": zxx1,
            "fs": float(fs1),
            "window": window1,
            "nperseg": int(nperseg1),
            "noverlap": int(noverlap1),
            "nfft": int(nfft1),
            "input_onesided": bool(onesided1),
            "boundary": bool(boundary1),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 2: boxcar window with boundary=False and higher sampling rate
    x2 = np.random.randn(128).astype(np.float32)
    fs2 = 100.0
    window2 = "boxcar"
    nperseg2 = 32
    noverlap2 = 16
    nfft2 = 32
    onesided2 = True
    boundary2 = False
    _, _, zxx2 = scipy.signal.stft(
        x2,
        fs=fs2,
        window=window2,
        nperseg=nperseg2,
        noverlap=noverlap2,
        nfft=nfft2,
        return_onesided=onesided2,
        boundary=None,
    )
    list_of_inputs.append(
        {
            "Zxx": zxx2,
            "fs": float(fs2),
            "window": window2,
            "nperseg": int(nperseg2),
            "noverlap": int(noverlap2),
            "nfft": int(nfft2),
            "input_onesided": bool(onesided2),
            "boundary": bool(boundary2),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 3: Two-sided STFT (input_onesided=False)
    x3 = (np.random.randn(80) + 1j * np.random.randn(80)).astype(np.complex128)
    fs3 = 2.0
    window3 = "hann"
    nperseg3 = 16
    noverlap3 = 8
    nfft3 = 16
    onesided3 = False
    boundary3 = True
    _, _, zxx3 = scipy.signal.stft(
        x3,
        fs=fs3,
        window=window3,
        nperseg=nperseg3,
        noverlap=noverlap3,
        nfft=nfft3,
        return_onesided=onesided3,
        boundary="zeros",
    )
    list_of_inputs.append(
        {
            "Zxx": zxx3,
            "fs": float(fs3),
            "window": window3,
            "nperseg": int(nperseg3),
            "noverlap": int(noverlap3),
            "nfft": int(nfft3),
            "input_onesided": bool(onesided3),
            "boundary": bool(boundary3),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 4: Transposed axes (time_axis=-2, freq_axis=-1)
    x4 = np.random.randn(96).astype(np.float64)
    fs4 = 8.0
    window4 = "hann"
    nperseg4 = 16
    noverlap4 = 8
    nfft4 = 16
    onesided4 = True
    boundary4 = True
    _, _, zxx4 = scipy.signal.stft(
        x4,
        fs=fs4,
        window=window4,
        nperseg=nperseg4,
        noverlap=noverlap4,
        nfft=nfft4,
        return_onesided=onesided4,
        boundary="zeros",
    )
    zxx4_t = np.swapaxes(zxx4, -1, -2)
    list_of_inputs.append(
        {
            "Zxx": zxx4_t,
            "fs": float(fs4),
            "window": window4,
            "nperseg": int(nperseg4),
            "noverlap": int(noverlap4),
            "nfft": int(nfft4),
            "input_onesided": bool(onesided4),
            "boundary": bool(boundary4),
            "time_axis": -2,
            "freq_axis": -1,
        }
    )

    # Case 5: Positive axes indices (time_axis=1, freq_axis=0)
    x5 = np.random.randn(64).astype(np.float64)
    fs5 = 16.0
    window5 = "hamming"
    nperseg5 = 16
    noverlap5 = 8
    nfft5 = 16
    onesided5 = True
    boundary5 = True
    _, _, zxx5 = scipy.signal.stft(
        x5,
        fs=fs5,
        window=window5,
        nperseg=nperseg5,
        noverlap=noverlap5,
        nfft=nfft5,
        return_onesided=onesided5,
        boundary="zeros",
    )
    list_of_inputs.append(
        {
            "Zxx": zxx5,
            "fs": float(fs5),
            "window": window5,
            "nperseg": int(nperseg5),
            "noverlap": int(noverlap5),
            "nfft": int(nfft5),
            "input_onesided": bool(onesided5),
            "boundary": bool(boundary5),
            "time_axis": 1,
            "freq_axis": 0,
        }
    )

    # Case 6: 3D batched input (batch, freq, time)
    x6 = np.random.randn(3, 100).astype(np.float64)
    fs6 = 1.0
    window6 = "hann"
    nperseg6 = 20
    noverlap6 = 10
    nfft6 = 20
    onesided6 = True
    boundary6 = True
    _, _, zxx6 = scipy.signal.stft(
        x6,
        fs=fs6,
        window=window6,
        nperseg=nperseg6,
        noverlap=noverlap6,
        nfft=nfft6,
        return_onesided=onesided6,
        boundary="zeros",
        axis=-1,
    )
    list_of_inputs.append(
        {
            "Zxx": zxx6,
            "fs": float(fs6),
            "window": window6,
            "nperseg": int(nperseg6),
            "noverlap": int(noverlap6),
            "nfft": int(nfft6),
            "input_onesided": bool(onesided6),
            "boundary": bool(boundary6),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 7: 4D batched input (batch1, batch2, freq, time)
    x7 = np.random.randn(2, 2, 64).astype(np.float32)
    fs7 = 4.0
    window7 = "bartlett"
    nperseg7 = 16
    noverlap7 = 8
    nfft7 = 16
    onesided7 = True
    boundary7 = True
    _, _, zxx7 = scipy.signal.stft(
        x7,
        fs=fs7,
        window=window7,
        nperseg=nperseg7,
        noverlap=noverlap7,
        nfft=nfft7,
        return_onesided=onesided7,
        boundary="zeros",
        axis=-1,
    )
    list_of_inputs.append(
        {
            "Zxx": zxx7,
            "fs": float(fs7),
            "window": window7,
            "nperseg": int(nperseg7),
            "noverlap": int(noverlap7),
            "nfft": int(nfft7),
            "input_onesided": bool(onesided7),
            "boundary": bool(boundary7),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 8: Zero-padded FFT (nfft > nperseg)
    x8 = np.random.randn(128).astype(np.float64)
    fs8 = 200.0
    window8 = "hann"
    nperseg8 = 16
    noverlap8 = 8
    nfft8 = 32
    onesided8 = True
    boundary8 = True
    _, _, zxx8 = scipy.signal.stft(
        x8,
        fs=fs8,
        window=window8,
        nperseg=nperseg8,
        noverlap=noverlap8,
        nfft=nfft8,
        return_onesided=onesided8,
        boundary="zeros",
    )
    list_of_inputs.append(
        {
            "Zxx": zxx8,
            "fs": float(fs8),
            "window": window8,
            "nperseg": int(nperseg8),
            "noverlap": int(noverlap8),
            "nfft": int(nfft8),
            "input_onesided": bool(onesided8),
            "boundary": bool(boundary8),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 9: Blackman window with larger overlap
    x9 = np.random.randn(120).astype(np.float64)
    fs9 = 50.0
    window9 = "blackman"
    nperseg9 = 24
    noverlap9 = 18
    nfft9 = 24
    onesided9 = True
    boundary9 = True
    _, _, zxx9 = scipy.signal.stft(
        x9,
        fs=fs9,
        window=window9,
        nperseg=nperseg9,
        noverlap=noverlap9,
        nfft=nfft9,
        return_onesided=onesided9,
        boundary="zeros",
    )
    list_of_inputs.append(
        {
            "Zxx": zxx9,
            "fs": float(fs9),
            "window": window9,
            "nperseg": int(nperseg9),
            "noverlap": int(noverlap9),
            "nfft": int(nfft9),
            "input_onesided": bool(onesided9),
            "boundary": bool(boundary9),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 10: Two-sided STFT with no boundary extension and float32 dtype
    x10 = (np.random.randn(128) + 1j * np.random.randn(128)).astype(
        np.complex64
    )
    fs10 = 0.5
    window10 = "hann"
    nperseg10 = 32
    noverlap10 = 16
    nfft10 = 32
    onesided10 = False
    boundary10 = False
    _, _, zxx10 = scipy.signal.stft(
        x10,
        fs=fs10,
        window=window10,
        nperseg=nperseg10,
        noverlap=noverlap10,
        nfft=nfft10,
        return_onesided=onesided10,
        boundary=None,
    )
    list_of_inputs.append(
        {
            "Zxx": zxx10,
            "fs": float(fs10),
            "window": window10,
            "nperseg": int(nperseg10),
            "noverlap": int(noverlap10),
            "nfft": int(nfft10),
            "input_onesided": bool(onesided10),
            "boundary": bool(boundary10),
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.signal.istft_1"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_1'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_1'], lib="jax", suffix=1)
