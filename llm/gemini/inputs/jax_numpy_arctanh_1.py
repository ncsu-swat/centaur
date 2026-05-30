
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arctanh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array within (-1, 1)
    x = np.array([-0.8, -0.5, 0.0, 0.5, 0.8], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array with boundaries [-1, 1]
    x = np.array([[-1.0, -0.2], [0.2, 1.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array within (-1, 1)
    x = np.random.uniform(-0.99, 0.99, (2, 3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D float32 array (scalar tensor)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array
    x = np.array([-2.0 + 0j, 3.0 + 0j, 4.0 - 1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D complex128 array
    x = np.array([[0.5 + 0.5j, -0.5 - 0.5j], [1.5 + 0j, -1.5 + 0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float32 array with negative values
    x = np.array([-0.9, -0.7, -0.3, -0.1], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float32 array
    x = np.random.uniform(-0.5, 0.5, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D float16 array
    x = np.array([-0.99, 0.0, 0.99], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D float32 array with values outside [-1, 1]
    x = np.array([[-2.0, -1.5], [1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.arctanh_1"] = arctanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arctanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arctanh_1'.")


check_valid('jax.numpy.arctanh', generated_inputs['jax.numpy.arctanh_1'], lib="jax", suffix=1)
