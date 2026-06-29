
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array, x is 0.0
    x = 0.0
    y = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: 2D float32 array with negative and positive values, x is -1.5
    x = -1.5
    y = np.array([[-2.0, -1.5, -1.0], [0.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: 3D float64 array, x is 2.5
    x = 2.5
    y = np.random.randn(2, 3, 4).astype(np.float64) * 5.0
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: 4D float32 array, x is 100.0
    x = 100.0
    y = np.random.uniform(50.0, 150.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: 1D float64 array, x is -0.0
    x = -0.0
    y = np.array([-0.0, 0.0, -1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: 2D float32 array, x is 3.14159 (pi)
    x = 3.14159
    y = np.array([[3.14, 3.15], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: 1D float32 array with small values, x is 1e-5
    x = 1e-5
    y = np.array([1e-6, 1e-5, 1e-4, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: 2D float64 array with extreme values, x is 1e10
    x = 1e10
    y = np.array([[1e9, 1e10], [1e11, -1e12]], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: 5D float32 array, x is -10.0
    x = -10.0
    y = np.random.uniform(-20.0, 0.0, size=(2, 1, 3, 1, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: 3D float32 array with values between 0 and 1, x is 0.5
    x = 0.5
    y = np.random.rand(3, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.le_7"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_7'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_7'], lib="jax", suffix=7)
