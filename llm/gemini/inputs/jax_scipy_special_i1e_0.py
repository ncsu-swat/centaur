
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def i1e_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, standard range
    list_of_inputs.append({"x": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)})

    # Input 2: 2D array, float32, positive values
    list_of_inputs.append({"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    # Input 3: 0D array (scalar tensor), float64
    list_of_inputs.append({"x": np.array(1.5, dtype=np.float64)})

    # Input 4: 1D array, zeros
    list_of_inputs.append({"x": np.zeros(5, dtype=np.float32)})

    # Input 5: 3D array, float64
    list_of_inputs.append({"x": np.random.randn(2, 3, 4).astype(np.float64)})

    # Input 6: 1D array, large positive and negative values (tests exponential scaling)
    list_of_inputs.append({"x": np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0], dtype=np.float64)})

    # Input 7: 1D array, very small values close to zero
    list_of_inputs.append({"x": np.array([-1e-5, -1e-15, 0.0, 1e-15, 1e-5], dtype=np.float32)})

    # Input 8: 2D array, uniform random in [-10, 10], float32
    list_of_inputs.append({"x": np.random.uniform(-10.0, 10.0, size=(5, 5)).astype(np.float32)})

    # Input 9: 4D array, float32
    list_of_inputs.append({"x": np.random.randn(2, 2, 2, 2).astype(np.float32)})

    # Input 10: 1D array, float64, evenly spaced
    list_of_inputs.append({"x": np.linspace(-50.0, 50.0, 20, dtype=np.float64)})

    return list_of_inputs

generated_inputs["jax.scipy.special.i1e"] = i1e_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i1e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i1e'.")


check_valid('jax.scipy.special.i1e', generated_inputs['jax.scipy.special.i1e'], lib="jax", suffix=0)
