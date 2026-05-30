
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax._src.numpy.lax_numpy as lax_numpy

# Monkeypatch jax.numpy.insert to support list for obj
_original_insert = lax_numpy.insert

def patched_insert(arr, obj, values, axis=None):
    if isinstance(obj, list):
        obj = jnp.array(obj)
    return _original_insert(arr, obj, values, axis)

lax_numpy.insert = patched_insert
jnp.insert = patched_insert

def insert_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single insertion
    arr = np.arange(5, dtype=np.int32)
    obj = [2]
    values = np.array([99], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 2: 1D array, multiple insertions
    arr = np.arange(5, dtype=np.int32)
    obj = [1, 3]
    values = np.array([99, 100], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 3: 2D array, insert a row
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    obj = [1]
    values = np.array([[10, 20, 30]], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 4: 2D array, insert multiple columns
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    obj = [1, 2]
    values = np.array([[10, 20], [30, 40]], dtype=np.int32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 5: 2D array, negative index insertion
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    obj = [-1]
    values = np.array([[10.0], [20.0]], dtype=np.float32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 6: 3D array, insert along axis 0
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    obj = [1]
    values = np.ones((1, 3, 4), dtype=np.float64)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 7: 3D array, insert along axis 1
    arr = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    obj = [0, 2]
    values = np.zeros((2, 2, 4), dtype=np.int64)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 8: 3D array, insert along axis 2
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    obj = [1, 3]
    values = np.ones((2, 3, 2), dtype=np.float32)
    axis = 2
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 9: 1D array, inserting at all positions
    arr = np.arange(4, dtype=np.int32)
    obj = [0, 1, 2, 3]
    values = np.array([10, 20, 30, 40], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 10: 2D array, negative axis index
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    obj = [-2]
    values = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)
    axis = -2
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.insert_3"] = insert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.insert_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.insert_3'.")


check_valid('jax.numpy.insert', generated_inputs['jax.numpy.insert_3'], lib="jax", suffix=3)
