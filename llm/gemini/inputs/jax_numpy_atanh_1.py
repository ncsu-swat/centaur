
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atanh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values in (0, 1)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float32 array with negative values in (-1, 0)
    x = np.array([-0.1, -0.5, -0.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0D array) float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D float64 array with mixed signs in (-1, 1)
    x = np.array([[-0.8, 0.2], [0.7, -0.3]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float32 array with small values
    x = np.random.uniform(-0.1, 0.1, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D complex64 array (atanh accepts complex values outside (-1, 1))
    x = np.array([0.5 + 0.5j, 1.5 - 2.0j, -0.2 + 0.1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex128 array
    x = np.array([[1.0 + 1.0j, -2.0 - 0.5j], [0.1 + 0.9j, -0.9 + 0.1j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float16 array
    x = np.array([-0.75, 0.0, 0.75], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float32 array with elements in (-0.95, 0.95)
    x = np.random.uniform(-0.95, 0.95, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array with values close to boundary
    x = np.array([-0.999, -0.99, 0.0, 0.99, 0.999], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 2D float32 array of zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.atanh_1"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_1'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_1'], lib="jax", suffix=1)
