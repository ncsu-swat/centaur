
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rfftfreq_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "n": int(8),
        "d": float(1.0),
        "dtype": np.float32
    })

    # Input 2
    list_of_inputs.append({
        "n": int(16),
        "d": float(0.5),
        "dtype": np.float64
    })

    # Input 3
    list_of_inputs.append({
        "n": int(100),
        "d": float(0.1),
        "dtype": np.float32
    })

    # Input 4
    list_of_inputs.append({
        "n": int(256),
        "d": float(0.01),
        "dtype": np.float64
    })

    # Input 5
    list_of_inputs.append({
        "n": int(1024),
        "d": float(2.5),
        "dtype": np.float32
    })

    # Input 6
    list_of_inputs.append({
        "n": int(7),
        "d": float(1.0),
        "dtype": np.float64
    })

    # Input 7
    list_of_inputs.append({
        "n": int(15),
        "d": float(-0.5),
        "dtype": np.float32
    })

    # Input 8
    list_of_inputs.append({
        "n": int(50),
        "d": float(1e-3),
        "dtype": np.float32
    })

    # Input 9
    list_of_inputs.append({
        "n": int(12),
        "d": float(3.14159),
        "dtype": np.float64
    })

    # Input 10
    list_of_inputs.append({
        "n": int(2000),
        "d": float(0.005),
        "dtype": np.float32
    })

    return list_of_inputs

generated_inputs["jax.numpy.fft.rfftfreq_1"] = rfftfreq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfftfreq_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfftfreq_1'.")


check_valid('jax.numpy.fft.rfftfreq', generated_inputs['jax.numpy.fft.rfftfreq_1'], lib="jax", suffix=1)
