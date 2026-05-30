
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D complex64 array
    x = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D complex128 array
    x = np.array([[1.0 + 1.0j, 2.0], [3.0j, 4.0 - 2.0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int32 array
    x = np.ones((2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D complex64 array
    x = np.array(1.0 + 0.0j, dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D float64 array
    x = np.array(-5.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D boolean array
    x = np.zeros((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D complex64 array with zero imaginary parts
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D int16 array with negative values
    x = np.array([-10, 0, 10], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 3D float64 array with random values
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplexobj_1"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_1'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_1'], lib="jax", suffix=1)
