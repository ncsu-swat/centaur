
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array with standard normal distribution
    x = np.random.randn(3, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array
    x = np.random.uniform(-3.0, 3.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D float32 array (scalar)
    x = np.array(0.75, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float16 array
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D float64 array with extreme values (erf saturates to -1 and 1)
    x = np.array([[-10.0, -5.0], [5.0, 10.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array with zeros and small values
    x = np.array([-1e-5, 0.0, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D float32 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array containing only negative values
    x = np.array([-3.0, -2.5, -1.2, -0.1], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.erf_1"] = erf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_1'.")


check_valid('jax.lax.erf', generated_inputs['jax.lax.erf_1'], lib="jax", suffix=1)
