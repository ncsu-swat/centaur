
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cumprod_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int array
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = 0
    dtype = np.int32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 2: 2D float array along axis 0
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 0
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 3: 2D float array along axis 1
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 1
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 4: 3D float64 array along axis 2
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 2
    dtype = np.float64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 5: 3D int64 array along negative axis
    a = np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int64)
    axis = -1
    dtype = np.int64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 6: 1D negative values float32
    a = np.array([-1.0, -2.0, -3.0, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 7: 4D float32 array along axis 2
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = 2
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 8: Complex64 array along axis 0
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    axis = 0
    dtype = np.complex64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 9: Cast integer array to float64 output dtype
    a = np.random.randint(1, 10, size=(3, 3)).astype(np.int32)
    axis = 1
    dtype = np.float64
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    # Input 10: Cast float16 array to float32 output dtype
    a = np.random.randn(4, 2).astype(np.float16)
    axis = 0
    dtype = np.float32
    list_of_inputs.append({"a": a, "axis": axis, "dtype": dtype})

    return copy.deepcopy(list_of_inputs)

generated_inputs["jax.numpy.cumprod"] = cumprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cumprod'.")


check_valid('jax.numpy.cumprod', generated_inputs['jax.numpy.cumprod'], lib="jax", suffix=0)
