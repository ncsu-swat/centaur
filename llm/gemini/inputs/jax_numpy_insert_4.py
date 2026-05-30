
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax._src.numpy.util as jax_util
import jax._src.lax.lax as jax_lax

# Monkeypatch check_arraylike to convert tuples to np.ndarrays during validation
orig_check_arraylike = jax_util.check_arraylike

def patched_check_arraylike(fun_name, *args):
    safe_args = [np.array(arg) if isinstance(arg, tuple) else arg for arg in args]
    return orig_check_arraylike(fun_name, *safe_args)

jax_util.check_arraylike = patched_check_arraylike

# Monkeypatch lax.asarray to handle tuples by converting them to np.ndarray
orig_asarray = jax_lax.asarray

def patched_asarray(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = np.array(x)
    return orig_asarray(x, *args, **kwargs)

jax_lax.asarray = patched_asarray

def insert_inputs():
    list_of_inputs = []

    # Input 1: 2D array, inserting rows, float32
    arr = np.random.randn(5, 5).astype(np.float32)
    obj = (1, 3)
    values = np.random.randn(2, 5).astype(np.float32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 2: 2D array, inserting a single row at index 0, int32
    arr = np.random.randint(0, 10, size=(4, 4)).astype(np.int32)
    obj = (0,)
    values = np.random.randint(0, 10, size=(1, 4)).astype(np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 3: 3D array, inserting along axis 1, float64
    arr = np.random.randn(3, 3, 3).astype(np.float64)
    obj = (1, 2)
    values = np.random.randn(3, 2, 3).astype(np.float64)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 4: 2D array, inserting multiple columns, float32
    arr = np.random.randn(2, 6).astype(np.float32)
    obj = (1, 3, 5)
    values = np.random.randn(2, 3).astype(np.float32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 5: 3D array, inserting along axis 0, int16
    arr = np.random.randint(-100, 100, size=(4, 2, 3)).astype(np.int16)
    obj = (0, 2)
    values = np.random.randint(-100, 100, size=(2, 2, 3)).astype(np.int16)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 6: 2D array, inserting a single column, float32
    arr = np.random.randn(3, 5).astype(np.float32)
    obj = (2,)
    values = np.random.randn(3, 1).astype(np.float32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 7: 1D array, inserting elements at multiple indices, float64
    arr = np.random.randn(10).astype(np.float64)
    obj = (2, 5, 8)
    values = np.random.randn(3).astype(np.float64)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 8: 3D array, inserting along axis 2, float32
    arr = np.random.randn(5, 5, 5).astype(np.float32)
    obj = (1, 2, 3, 4)
    values = np.random.randn(5, 5, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 9: 2D array, inserting multiple values at the same index, float32
    arr = np.random.randn(2, 3).astype(np.float32)
    obj = (0, 0)
    values = np.random.randn(2, 2).astype(np.float32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 10: 2D array, inserting rows, int32
    arr = np.random.randint(-50, 50, size=(6, 2)).astype(np.int32)
    obj = (2, 4)
    values = np.random.randint(-50, 50, size=(2, 2)).astype(np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    return list_of_inputs

generated_inputs["jax.numpy.insert_4"] = insert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.insert_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.insert_4'.")


check_valid('jax.numpy.insert', generated_inputs['jax.numpy.insert_4'], lib="jax", suffix=4)
