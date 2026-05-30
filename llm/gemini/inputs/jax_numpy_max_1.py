
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def max_inputs():
    list_of_inputs = []

    # Input 1: Float32, 2D, axis 0, keepdims=False
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([[True, False, True], [False, True, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, 2D, axis 1, keepdims=True
    a = np.array([[-1.5, -2.5], [-3.5, -4.5]], dtype=np.float64)
    axis = 1
    keepdims = True
    initial = np.array(-100.0, dtype=np.float64)
    where = np.array([[True, True], [True, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, 3D, axis 2, keepdims=False
    a = np.random.randint(-10, 10, size=(3, 3, 3)).astype(np.int32)
    axis = 2
    keepdims = False
    initial = np.array(-99, dtype=np.int32)
    where = np.ones((3, 3, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float32, 2D with random boolean mask, axis -1, keepdims=True
    a = np.random.randn(5, 5).astype(np.float32)
    axis = -1
    keepdims = True
    initial = np.array(-10.0, dtype=np.float32)
    where = np.random.rand(5, 5) > 0.5
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64, 1D, axis 0, keepdims=False
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis = 0
    keepdims = False
    initial = np.array(0, dtype=np.int64)
    where = np.array([True, True, True, False, False], dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32, 3D, axis 1, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 1
    keepdims = True
    initial = np.array(-5.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int16, 3D, axis -2, keepdims=False
    a = np.random.randint(-50, 50, size=(4, 2, 3)).astype(np.int16)
    axis = -2
    keepdims = False
    initial = np.array(-100, dtype=np.int16)
    where = np.ones((4, 2, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64, 3D, axis 2, keepdims=True
    a = np.random.randn(3, 3, 3).astype(np.float64)
    axis = 2
    keepdims = True
    initial = np.array(-10.0, dtype=np.float64)
    where = np.ones((3, 3, 3), dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32, 2D, axis 1, keepdims=False
    a = np.random.randn(1, 5).astype(np.float32)
    axis = 1
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([[True, True, False, False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int32, 1D, axis 0, keepdims=True
    a = np.random.randint(-100, 100, size=(6,)).astype(np.int32)
    axis = 0
    keepdims = True
    initial = np.array(-105, dtype=np.int32)
    where = np.array([True, True, True, True, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": axis,
        "keepdims": keepdims,
        "initial": initial,
        "where": where
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.max_1"] = max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.max_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.max_1'.")


check_valid('jax.numpy.max', generated_inputs['jax.numpy.max_1'], lib="jax", suffix=1)
