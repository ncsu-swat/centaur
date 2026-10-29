
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.scipy.signal


def istft_inputs():
    list_of_inputs = []

    def make_input(
        sig_len,
        fs_val,
        window,
        nperseg,
        noverlap,
        nfft,
        input_onesided,
        boundary,
        time_axis,
        freq_axis,
        extra_dims=(),
    ):
        x = np.random.randn(*extra_dims, sig_len).astype(np.float32)
        stft_boundary = "zeros" if boundary else None
        _, _, Zxx = jax.scipy.signal.stft(
            x,
            fs=float(fs_val),
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=input_onesided,
            boundary=stft_boundary,
            axis=-1,
        )
        Zxx_np = np.asarray(Zxx)

        ndim = Zxx_np.ndim
        norm_time = time_axis % ndim
        norm_freq = freq_axis % ndim
        curr_freq = (ndim - 2) % ndim
        curr_time = (ndim - 1) % ndim

        if (norm_time, norm_freq) != (curr_time, curr_freq):
            axes = [i for i in range(ndim) if i not in (curr_freq, curr_time)]
            perm = [0] * ndim
            perm[norm_time] = curr_time
            perm[norm_freq] = curr_freq
            remaining_targets = [
                i for i in range(ndim) if i not in (norm_time, norm_freq)
            ]
            for t_idx, src_idx in zip(remaining_targets, axes):
                perm[t_idx] = src_idx
            inv_perm = [perm.index(i) for i in range(ndim)]
            Zxx_np = np.transpose(Zxx_np, inv_perm)

        return {
            "Zxx": Zxx_np,
            "fs": np.array(fs_val, dtype=np.float32),
            "window": window,
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
            "nfft": int(nfft),
            "input_onesided": bool(input_onesided),
            "boundary": bool(boundary),
            "time_axis": int(time_axis),
            "freq_axis": int(freq_axis),
        }

    # Input 1: Basic standard configuration (1D signal, hann window)
    inp1 = make_input(
        sig_len=128,
        fs_val=1.0,
        window="hann",
        nperseg=16,
        noverlap=8,
        nfft=16,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )
    list_of_inputs.append(copy.deepcopy(inp1))

    # Input 2: Higher sampling frequency, hamming window
    inp2 = make_input(
        sig_len=256,
        fs_val=16000.0,
        window="hamming",
        nperseg=32,
        noverlap=16,
        nfft=32,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )
    list_of_inputs.append(copy.deepcopy(inp2))

    # Input 3: input_onesided = False (two-sided STFT)
    inp3 = make_input(
        sig_len=64,
        fs_val=8000.0,
        window="hann",
        nperseg=16,
        noverlap=8,
        nfft=16,
        input_onesided=False,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )
    list_of_inputs.append(copy.deepcopy(inp3))

    # Input 4: boundary = False
    inp4 = make_input(
        sig_len=128,
        fs_val=1.0,
        window="hann",
        nperseg=16,
        noverlap=8,
        nfft=16,
        input_onesided=True,
        boundary=False,
        time_axis=-1,
        freq_axis=-2,
    )
    list_of_inputs.append(copy.deepcopy(inp4))

    # Input 5: nfft > nperseg (zero-padding in frequency domain)
    inp5 = make_input(
        sig_len=128,
        fs_val=100.0,
        window="hann",
        nperseg=16,
        noverlap=8,
        nfft=32,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )
    list_of_inputs.append(copy.deepcopy(inp5))

    # Input 6: Transposed axes (time_axis=-2, freq_axis=-1)
    inp6 = make_input(
        sig_len=64,
        fs_val=1.0,
        window="hann",
        nperseg=16,
        noverlap=8,
        nfft=16,
        input_onesided=True,
        boundary=True,
        time_axis=-2,
        freq_axis=-1,
    )
    list_of_inputs.append(copy.deepcopy(inp6))

    # Input 7: Batched signal (3D tensor input)
    inp7 = make_input(
        sig_len=128,
        fs_val=44100.0,
        window="hann",
        nperseg=32,
        noverlap=16,
        nfft=32,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
        extra_dims=(2,),
    )
    list_of_inputs.append(copy.deepcopy(inp7))

    # Input 8: boxcar window with 75% overlap
    inp8 = make_input(
        sig_len=128,
        fs_val=2.0,
        window="boxcar",
        nperseg=16,
        noverlap=12,
        nfft=16,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )
    list_of_inputs.append(copy.deepcopy(inp8))

    # Input 9: Positive axis indices (freq_axis=0, time_axis=1)
    inp9 = make_input(
        sig_len=80,
        fs_val=50.0,
        window="hann",
        nperseg=8,
        noverlap=4,
        nfft=8,
        input_onesided=True,
        boundary=True,
        time_axis=1,
        freq_axis=0,
    )
    list_of_inputs.append(copy.deepcopy(inp9))

    # Input 10: Multi-channel (4D tensor) with two-sided STFT and boundary=False
    inp10 = make_input(
        sig_len=96,
        fs_val=1.0,
        window="hann",
        nperseg=16,
        noverlap=8,
        nfft=16,
        input_onesided=False,
        boundary=False,
        time_axis=-1,
        freq_axis=-2,
        extra_dims=(2, 3),
    )
    list_of_inputs.append(copy.deepcopy(inp10))

    return list_of_inputs


generated_inputs["jax.scipy.signal.istft_4"] = istft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.istft_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.istft_4'.")


check_valid('jax.scipy.signal.istft', generated_inputs['jax.scipy.signal.istft_4'], lib="jax", suffix=4)
