
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hyp2f1_inputs():
    list_of_inputs = []

    # Input 1: 0D arrays (scalars), float32, positive
    a = np.array(0.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    c = np.array(2.5, dtype=np.float32)
    x = np.array(0.2, dtype=np.float32)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32, positive
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([0.5, 1.5], dtype=np.float32)
    c = np.array([3.0, 4.0], dtype=np.float32)
    x = np.array([0.1, 0.5], dtype=np.float32)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64, positive
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    c = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays, with broadcasted dimensions
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.5], dtype=np.float32)
    c = np.array([4.0], dtype=np.float32)
    x = np.array([0.5], dtype=np.float32)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, float32
    a = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    b = np.ones((2, 2, 2), dtype=np.float32) * 1.5
    c = np.ones((2, 2, 2), dtype=np.float32) * 3.5
    x = np.ones((2, 2, 2), dtype=np.float32) * 0.25
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting with 2D and 1D arrays
    a = np.array([[1.0], [2.0]], dtype=np.float32)
    b = np.array([0.5, 1.0], dtype=np.float32)
    c = np.array([[3.0], [4.0]], dtype=np.float32)
    x = np.array([0.2, 0.3], dtype=np.float32)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive values, float64
    a = np.array([1e-3, 1e-2], dtype=np.float64)
    b = np.array([1e-3, 1e-2], dtype=np.float64)
    c = np.array([1.0, 1.5], dtype=np.float64)
    x = np.array([0.01, 0.02], dtype=np.float64)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large positive values for parameter c, float32
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([2.0, 3.0], dtype=np.float32)
    c = np.array([50.0, 100.0], dtype=np.float32)
    x = np.array([0.8, 0.9], dtype=np.float32)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small x values, float32
    a = np.array([2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    c = np.array([4.0], dtype=np.float32)
    x = np.array([1e-5], dtype=np.float32)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Randomized positive matrices, float64
    a = np.random.uniform(1.0, 5.0, (5, 5)).astype(np.float64)
    b = np.random.uniform(1.0, 5.0, (5, 5)).astype(np.float64)
    c = np.random.uniform(5.0, 10.0, (5, 5)).astype(np.float64)
    x = np.random.uniform(0.01, 0.9, (5, 5)).astype(np.float64)
    input_dict = {"a": a, "b": b, "c": c, "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.hyp2f1"] = hyp2f1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.hyp2f1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.hyp2f1'.")


check_valid('jax.scipy.special.hyp2f1', generated_inputs['jax.scipy.special.hyp2f1'], lib="jax", suffix=0)
