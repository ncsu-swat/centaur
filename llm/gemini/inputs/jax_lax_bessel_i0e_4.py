
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bessel_i0e_inputs():
    list_of_inputs = []

    # Input 1: 0D float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D float64
    x = np.array(-2.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D float32 positive
    x = np.array([0.0, 1.0, 2.0, 3.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D float32 negative and zero
    x = np.array([-1.2, 0.0, -0.5, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D float64
    x = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D float32
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Scalar float
    x = np.float32(0.5)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Scalar float64
    x = np.float64(-1.5)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large 1D float32
    x = np.linspace(-10.0, 10.0, 100).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D float32
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.bessel_i0e_4"] = bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bessel_i0e_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bessel_i0e_4'.")


check_valid('jax.lax.bessel_i0e', generated_inputs['jax.lax.bessel_i0e_4'], lib="jax", suffix=4)
