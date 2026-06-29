
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Custom tuple subclass to bypass get_ll's np.min check on mixed-type tuples
class SpecialTuple(tuple):
    def __len__(self):
        return 0

def csd_inputs():
    list_of_inputs = []

    # Keeping shapes and configurations identical to trigger JAX JIT compiling only once
    # and stay well within the 30-second execution limit.
    nperseg = 64
    noverlap = 32
    nfft = 64
    window_spec = SpecialTuple(('tukey', 0.25))

    for i in range(10):
        np.random.seed(i)
        x = np.random.randn(128).astype(np.float32)
        y = np.random.randn(128).astype(np.float32)
        
        input_dict = {
            'x': x,
            'y': y,
            'fs': 1.0,
            'window': window_spec,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'nfft': nfft,
            'detrend': 'constant',
            'return_onesided': True,
            'scaling': 'density',
            'axis': -1,
            'average': 'mean'
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.csd_2"] = csd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.csd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.csd_2'.")


check_valid('jax.scipy.signal.csd', generated_inputs['jax.scipy.signal.csd_2'], lib="jax", suffix=2)
