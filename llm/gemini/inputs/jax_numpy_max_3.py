
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = [0]
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([True, True, False, True, False])
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 2: 2D float32 array, keepdims=True, reduce axis 0
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [0]
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 3: 2D int32 array, reduce axis 1, random mask
    a = np.random.randint(-10, 10, size=(4, 5)).astype(np.int32)
    axis = [1]
    keepdims = False
    initial = np.array(-20, dtype=np.int32)
    where = np.random.choice([True, False], size=(4, 5))
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 4: 3D float32 array, reduce multiple axes
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = [0, 2]
    keepdims = True
    initial = np.array(-50.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 5: 2D int64 array with even numbers masked
    a = np.arange(12).reshape(3, 4).astype(np.int64)
    axis = [1]
    keepdims = True
    initial = np.array(-5, dtype=np.int64)
    where = (a % 2 == 0)
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 6: 4D float32 array, reduce axis 2
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = [2]
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 7: 1D float32 array where initial is larger than elements
    a = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    axis = [0]
    keepdims = False
    initial = np.array(-5.0, dtype=np.float32)
    where = np.array([True, True, True])
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 8: 3D float64 array, reduce axis 1
    a = np.random.randn(3, 3, 3).astype(np.float64)
    axis = [1]
    keepdims = True
    initial = np.array(-10.0, dtype=np.float64)
    where = np.ones((3, 3, 3), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 9: 2D float16 array, reduce all axes
    a = np.random.randn(5, 5).astype(np.float16)
    axis = [0, 1]
    keepdims = False
    initial = np.array(-100.0, dtype=np.float16)
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    # Input 10: 2D float32 array with broadcasted where mask
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [0]
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([[True], [False], [True]])
    list_of_inputs.append({
        "a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where
    })

    return list_of_inputs

generated_inputs["jax.numpy.max_3"] = jax_numpy_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.max_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.max_3'.")


check_valid('jax.numpy.max', generated_inputs['jax.numpy.max_3'], lib="jax", suffix=3)
