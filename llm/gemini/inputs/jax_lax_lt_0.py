
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, same shape, mixed positive/negative
    x = np.array([-1.0, 0.0, 2.5, -3.2], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.5, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32, same shape
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[2, 2], [2, 5]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, same shape, random values
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalar tensors)
    x = np.array(5.0, dtype=np.float32)
    y = np.array(10.0, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32, broadcasting (same rank: (1, 3) and (2, 3))
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[2.0, 1.0, 4.0], [0.0, 5.0, 3.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32, broadcasting (same rank: (2, 1, 3) and (2, 4, 3))
    x = np.random.randn(2, 1, 3).astype(np.float32)
    y = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int64, same shape
    x = np.array([-10, 20, -30], dtype=np.int64)
    y = np.array([-5, 15, -35], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32, same shape, larger values
    x = np.array([[100, 200], [300, 400]], dtype=np.int32)
    y = np.array([[50, 250], [350, 450]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64, same shape
    x = np.array([-0.5, 0.5], dtype=np.float64)
    y = np.array([-0.1, 0.1], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int64, same shape
    x = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int64)
    y = np.random.randint(-10, 10, size=(2, 2, 2, 2)).astype(np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.lt"] = lt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lt'.")


check_valid('jax.lax.lt', generated_inputs['jax.lax.lt'], lib="jax", suffix=0)
