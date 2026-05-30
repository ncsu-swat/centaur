
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diagflat_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, k=0
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    k = 0
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 2: 1D int array, k=1 (positive offset)
    v = np.array([5, 6, 7, 8], dtype=np.int32)
    k = 1
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 3: 1D bool array, k=-1 (negative offset)
    v = np.array([True, False, True], dtype=bool)
    k = -1
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 4: 2D float array, k=2
    v = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k = 2
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 5: 2D int array, k=-2
    v = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    k = -2
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 6: 3D float array, k=0
    v = np.random.randn(2, 2, 2).astype(np.float32)
    k = 0
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 7: 1D double array, k=5
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    k = 5
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 8: 1D int array, k=-5
    v = np.array([10, 20, 30], dtype=np.int16)
    k = -5
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 9: 4D int8 array, k=1
    v = np.arange(12, dtype=np.int8).reshape(2, 1, 3, 2)
    k = 1
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 10: 1D complex array, k=-1
    v = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    k = -1
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    # Input 11: 0-dimensional array (scalar-like), k=0
    v = np.array(42)
    k = 0
    list_of_inputs.append({"v": copy.deepcopy(v), "k": k})

    return list_of_inputs

generated_inputs["jax.numpy.diagflat"] = diagflat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diagflat' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diagflat'.")


check_valid('jax.numpy.diagflat', generated_inputs['jax.numpy.diagflat'], lib="jax", suffix=0)
