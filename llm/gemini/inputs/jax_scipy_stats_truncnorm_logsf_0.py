
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logsf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard symmetric bounds
    x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    a = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 2: 0D arrays (scalars as tensors), float32, asymmetric bounds
    x = np.array(1.5, dtype=np.float32)
    a = np.array(-1.0, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    loc = np.array(1.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64, mixed bounds and locations
    x = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64)
    a = np.array([[-2.0, -3.0], [-1.5, -2.5]], dtype=np.float64)
    b = np.array([[2.0, 3.0], [1.5, 2.5]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    scale = np.array([[1.0, 1.5], [0.8, 1.2]], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 4: 3D arrays, float32, shape (2, 2, 2)
    shape_3d = (2, 2, 2)
    x = np.random.uniform(-1.0, 1.0, size=shape_3d).astype(np.float32)
    a = np.full(shape_3d, -2.0, dtype=np.float32)
    b = np.full(shape_3d, 2.0, dtype=np.float32)
    loc = np.zeros(shape_3d, dtype=np.float32)
    scale = np.ones(shape_3d, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 5: Broadcasting test, mixing 1D and 2D shapes
    x = np.array([0.0, 0.5], dtype=np.float32)  # shape (2,)
    a = np.array([[-1.0], [-2.0]], dtype=np.float32)  # shape (2, 1)
    b = np.array([[1.0], [2.0]], dtype=np.float32)  # shape (2, 1)
    loc = np.array([0.0], dtype=np.float32)  # shape (1,)
    scale = np.array([1.0], dtype=np.float32)  # shape (1,)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 6: Negative bounds (entirely negative truncated normal distribution)
    x = np.array([-11.0], dtype=np.float32)
    a = np.array([-5.0], dtype=np.float32)
    b = np.array([-2.0], dtype=np.float32)
    loc = np.array([-10.0], dtype=np.float32)
    scale = np.array([0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 7: Large bounds approximating standard normal, float64
    x = np.array([0.0], dtype=np.float64)
    a = np.array([-10.0], dtype=np.float64)
    b = np.array([10.0], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 8: Highly asymmetric positive-only bounds
    x = np.array([3.0], dtype=np.float32)
    a = np.array([2.0], dtype=np.float32)
    b = np.array([5.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 9: Tiny scale parameter (narrow distribution)
    x = np.array([0.0001], dtype=np.float32)
    a = np.array([-1.0], dtype=np.float32)
    b = np.array([1.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([0.001], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    # Input 10: 4D arrays, float32, randomized bounds with shape (1, 3, 2, 2)
    shape_4d = (1, 3, 2, 2)
    a = np.random.uniform(-3.0, -1.0, size=shape_4d).astype(np.float32)
    b = np.random.uniform(1.0, 3.0, size=shape_4d).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=shape_4d).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=shape_4d).astype(np.float32)
    x = np.random.uniform(-1.0, 1.0, size=shape_4d).astype(np.float32)
    list_of_inputs.append({"x": x, "a": a, "b": b, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.truncnorm.logsf"] = logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.logsf'.")


check_valid('jax.scipy.stats.truncnorm.logsf', generated_inputs['jax.scipy.stats.truncnorm.logsf'], lib="jax", suffix=0)
