
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def real_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array with positive and negative parts
    val = np.array([1 + 2j, -3 - 4j, 5 - 6j, -7 + 8j], dtype=np.complex64)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 2: 2D complex128 matrix
    val = np.array([[1.5 + 0.5j, -2.5 - 1.5j], [3.5 + 2.5j, -4.5 - 3.5j]], dtype=np.complex128)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 3: 0D complex64 array (scalar-like)
    val = np.array(-2.5 + 3.5j, dtype=np.complex64)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 4: 3D complex64 array
    val = np.random.randn(2, 3, 4).astype(np.float32) + 1j * np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 5: 1D float32 array (purely real input)
    val = np.array([-1.2, 0.0, 3.4], dtype=np.float32)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 6: 2D int32 array
    val = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 7: 1D complex128 array containing special values (inf, nan)
    val = np.array([complex(np.inf, -np.inf), complex(np.nan, 1.0), complex(0.0, np.nan)], dtype=np.complex128)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 8: 4D complex64 array
    val = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 9: 1D bool array
    val = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 10: Empty complex128 array with shape (0, 5)
    val = np.empty((0, 5), dtype=np.complex128)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    # Input 11: 1D float64 array with extreme values
    val = np.array([-1e30, 1e30, 0.0], dtype=np.float64)
    list_of_inputs.append({"val": copy.deepcopy(val)})

    return list_of_inputs

generated_inputs["jax.numpy.real_1"] = real_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.real_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.real_1'.")


check_valid('jax.numpy.real', generated_inputs['jax.numpy.real_1'], lib="jax", suffix=1)
