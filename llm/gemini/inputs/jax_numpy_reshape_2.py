
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D, C-order, copy=True
    a = np.arange(6, dtype=np.float32)
    shape = (2, 3)
    order = 'C'
    copy_val = True
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 2: 2D to 1D, C-order, copy=False
    a = np.random.randint(0, 10, size=(3, 2)).astype(np.int32)
    shape = (6,)
    order = 'C'
    copy_val = False
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 3: Using -1 in shape, F-order, copy=True
    a = np.random.randn(4, 4).astype(np.float64)
    shape = (2, -1)
    order = 'F'
    copy_val = True
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 4: 3D to 2D, C-order, copy=False
    a = np.random.randn(2, 3, 4).astype(np.float32)
    shape = (6, 4)
    order = 'C'
    copy_val = False
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 5: 1D to 3D, F-order, copy=True
    a = np.arange(24, dtype=np.int64)
    shape = (2, 3, 4)
    order = 'F'
    copy_val = True
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 6: Shape with -1 in 3D, C-order, copy=False
    a = np.random.randn(2, 3, 2, 2).astype(np.float32)
    shape = (3, -1, 4)
    order = 'C'
    copy_val = False
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 7: Squeezing/unsqueezing, boolean type, C-order, copy=True
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    shape = (1, 5, 1)
    order = 'C'
    copy_val = True
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 8: Larger dimensions, F-order, copy=True
    a = np.random.randn(10, 10, 10).astype(np.float32)
    shape = (100, 10)
    order = 'F'
    copy_val = True
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 9: Int16 type, C-order, copy=False
    a = np.random.randint(-100, 100, size=(8, 8)).astype(np.int16)
    shape = (4, 16)
    order = 'C'
    copy_val = False
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    # Input 10: Float16 type, 1D to 1D (no-op), F-order, copy=True
    a = np.random.randn(12).astype(np.float16)
    shape = (12,)
    order = 'F'
    copy_val = True
    list_of_inputs.append({"a": a, "shape": shape, "order": order, "copy": copy_val})

    return list_of_inputs

generated_inputs["jax.numpy.reshape_2"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reshape_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reshape_2'.")


check_valid('jax.numpy.reshape', generated_inputs['jax.numpy.reshape_2'], lib="jax", suffix=2)
