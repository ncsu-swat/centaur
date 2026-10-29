
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import scipy.signal as signal


def istft_inputs():
    list_of_inputs = []

    # Case 1: Standard 1D real signal, Hann window, 50% overlap, onesided, boundary=True
    nperseg = 16
    noverlap = 8
    nfft = 16
    fs = 1.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(128).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 2: 1D signal, float64, Hamming window, audio sampling rate (16000.0 Hz)
    nperseg = 32
    noverlap = 16
    nfft = 32
    fs = 16000.0
    win = signal.windows.hamming(nperseg, sym=False).astype(np.float64)
    x = np.random.randn(256).astype(np.float64)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 3: Two-sided STFT (input_onesided=False), complex STFT
    nperseg = 16
    noverlap = 8
    nfft = 16
    fs = 2.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(128).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=False,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": False,
            "boundary": True,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 4: Truncated boundaries (boundary=False), Bartlett window
    nperseg = 16
    noverlap = 8
    nfft = 16
    fs = 100.0
    win = signal.windows.bartlett(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(128).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": False,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 5: nfft > nperseg (frequency oversampling)
    nperseg = 16
    noverlap = 8
    nfft = 32
    fs = 8000.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(128).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 6: 2D batched signal (shape: batch x freq x time)
    nperseg = 16
    noverlap = 8
    nfft = 16
    fs = 44100.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(4, 128).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 7: Transposed axes: time_axis=0, freq_axis=1 (shape: time x freq)
    nperseg = 24
    noverlap = 12
    nfft = 24
    fs = 1.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(144).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    Zxx_t = np.transpose(Zxx, (1, 0))
    list_of_inputs.append(
        {
            "Zxx": Zxx_t,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": 0,
            "freq_axis": 1,
        }
    )

    # Case 8: 3D batched signal (shape: batch1 x batch2 x freq x time), boxcar window, boundary=False, two-sided
    nperseg = 8
    noverlap = 4
    nfft = 8
    fs = 10.0
    win = signal.windows.boxcar(nperseg).astype(np.float32)
    x = np.random.randn(2, 3, 64).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        return_onesided=False,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": False,
            "boundary": False,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 9: 75% overlap (noverlap = 24 with nperseg = 32)
    nperseg = 32
    noverlap = 24
    nfft = 32
    fs = 22050.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(256).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

    # Case 10: Batched with non-standard axes order: shape (freq, batch, time)
    nperseg = 16
    noverlap = 8
    nfft = 16
    fs = 500.0
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    x = np.random.randn(5, 128).astype(np.float32)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        return_onesided=True,
    )
    # Zxx is (5, freq, time) -> transpose to (freq, 5, time)
    Zxx_perm = np.transpose(Zxx, (1, 0, 2))
    list_of_inputs.append(
        {
            "Zxx": Zxx_perm,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": True,
            "time_axis": 2,
            "freq_axis": 0,
        }
    )

    # Case 11: Zero overlap (noverlap=0) with boxcar window, boundary=False
    nperseg = 16
    noverlap = 0
    nfft = 16
    fs = 1.0
    win = signal.windows.boxcar(nperseg).astype(np.float64)
    x = np.random.randn(128).astype(np.float64)
    _, _, Zxx = signal.stft(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        return_onesided=True,
    )
    list_of_inputs.append(
        {
            "Zxx": Zxx,
            "fs": float(fs),
            "window": win,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": True,
            "boundary": False,
            "time_axis": -1,
            "freq_axis": -2,
        }
    )

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
