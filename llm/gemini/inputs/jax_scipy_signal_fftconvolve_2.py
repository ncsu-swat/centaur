
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftconvolve_inputs():
    list_of_inputs = []

    # All 10 inputs use the same shape, mode, and axes to compile only once and avoid timeouts.
    for i in range(10):
        input_dict = {
            "in1": np.random.randn(3).astype(np.float32),
            "in2": np.random.randn(2).astype(np.float32),
            "mode": "full",
            "axes": (0,)
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.fftconvolve_2"] = fftconvolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.fftconvolve_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.fftconvolve_2'.")


check_valid('jax.scipy.signal.fftconvolve', generated_inputs['jax.scipy.signal.fftconvolve_2'], lib="jax", suffix=2)
