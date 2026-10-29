
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic even size, default-like spacing, float32
    list_of_inputs.append(
        {"n": int(8), "d": float(1.0), "dtype": np.dtype(np.float32)}
    )

    # Input 2: Odd size, float64
    list_of_inputs.append(
        {"n": int(9), "d": float(1.0), "dtype": np.dtype(np.float64)}
    )

    # Input 3: Fractional sample spacing
    list_of_inputs.append(
        {"n": int(16), "d": float(0.5), "dtype": np.dtype(np.float32)}
    )

    # Input 4: Large sample spacing
    list_of_inputs.append(
        {"n": int(32), "d": float(10.0), "dtype": np.dtype(np.float64)}
    )

    # Input 5: Small sample spacing (high sampling rate)
    list_of_inputs.append(
        {"n": int(64), "d": float(0.001), "dtype": np.dtype(np.float32)}
    )

    # Input 6: Single element window
    list_of_inputs.append(
        {"n": int(1), "d": float(1.0), "dtype": np.dtype(np.float32)}
    )

    # Input 7: Zero length window
    list_of_inputs.append(
        {"n": int(0), "d": float(1.0), "dtype": np.dtype(np.float64)}
    )

    # Input 8: Negative sample spacing
    list_of_inputs.append(
        {"n": int(10), "d": float(-0.25), "dtype": np.dtype(np.float32)}
    )

    # Input 9: Large FFT window, float64
    list_of_inputs.append(
        {"n": int(1024), "d": float(0.1), "dtype": np.dtype(np.float64)}
    )

    # Input 10: Non-power-of-two size with arbitrary float spacing
    list_of_inputs.append(
        {"n": int(127), "d": float(3.14159), "dtype": np.dtype(np.float32)}
    )

    # Input 11: Float16 dtype
    list_of_inputs.append(
        {"n": int(20), "d": float(2.0), "dtype": np.dtype(np.float16)}
    )

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftfreq_1"] = fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftfreq_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftfreq_1'.")


check_valid('jax.numpy.fft.fftfreq', generated_inputs['jax.numpy.fft.fftfreq_1'], lib="jax", suffix=1)
