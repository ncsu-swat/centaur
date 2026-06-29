
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 positive values
    x = np.array([0.0, 1.0, 2.5, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float32 negative values
    x = np.array([-0.5, -1.2, -3.0, -15.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float32 mixed values
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.5]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D float32 scalar tensor
    x = np.array(4.2, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float64 values
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float64 single-element
    x = np.array([100.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float32 large magnitude values
    x = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float32 tensor
    x = np.random.normal(0.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D float32 near-zero values
    x = np.array([-1e-5, 0.0, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D float64 random tensor
    x = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.lax.bessel_i0e_1"] = bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_1'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_1'], lib="jax", suffix=1)
