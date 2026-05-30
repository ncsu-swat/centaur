
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_amin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, no keepdims
    a = np.array([1.5, -2.3, 0.0, 4.2, -1.1], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, True, True, True])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 2: 2D float32 array, axis 1, keepdims True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = 1
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 3: 3D int32 array, axis 2, keepdims False
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    axis = 2
    keepdims = False
    initial = np.array(100, dtype=np.int32)
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 4: 2D float64 array, axis 0, keepdims True
    a = np.random.randn(5, 5).astype(np.float64)
    axis = 0
    keepdims = True
    initial = np.array(1e5, dtype=np.float64)
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 5: 4D float32 array, axis -1, keepdims False
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = -1
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 6: 1D int64 array, axis -1, keepdims True, partially masked
    a = np.array([10, 20, 30], dtype=np.int64)
    axis = -1
    keepdims = True
    initial = np.array(999, dtype=np.int64)
    where = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 7: 2D float16 array, axis 0, keepdims False
    a = np.random.randn(3, 3).astype(np.float16)
    axis = 0
    keepdims = False
    initial = np.array(50.0, dtype=np.float16)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 8: 3D int16 array with negative values, axis 1, keepdims True
    a = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int16)
    axis = 1
    keepdims = True
    initial = np.array(32767, dtype=np.int16)
    where = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 9: 2D float32 array, axis 0, keepdims False, with a custom boolean mask
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    # Input 10: Single-element 1D float32 array, axis 0, keepdims True
    a = np.array([5.5], dtype=np.float32)
    axis = 0
    keepdims = True
    initial = np.array(100.0, dtype=np.float32)
    where = np.array([True], dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.amin_1"] = jax_numpy_amin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.amin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.amin_1'.")


check_valid('jax.numpy.amin', generated_inputs['jax.numpy.amin_1'], lib="jax", suffix=1)
