
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fftfreq_inputs():
    list_of_inputs = []

    # Input 1: Basic positive even n, unit d, float32
    input_dict = {"n": int(8), "d": int(1), "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Odd n, unit d, float64
    input_dict = {"n": int(9), "d": int(1), "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger even n, non-unit positive d, float32
    input_dict = {"n": int(64), "d": int(2), "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative d, even n, float64
    input_dict = {"n": int(16), "d": int(-1), "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Minimal positive n = 1, d = 1, float32
    input_dict = {"n": int(1), "d": int(1), "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero n, positive d, float64
    input_dict = {"n": int(0), "d": int(1), "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Odd n, large positive d, float32
    input_dict = {"n": int(15), "d": int(10), "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large n, d = -5, float64
    input_dict = {"n": int(128), "d": int(-5), "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small odd n, negative d, float32
    input_dict = {"n": int(5), "d": int(-2), "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large odd n, unit d, float32
    input_dict = {"n": int(255), "d": int(3), "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Even n, larger spacing d, float64
    input_dict = {"n": int(32), "d": int(8), "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.fftfreq_2"] = fftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftfreq_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftfreq_2'.")


check_valid('jax.numpy.fft.fftfreq', generated_inputs['jax.numpy.fft.fftfreq_2'], lib="jax", suffix=2)
