
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def standardize_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis (0,), algorithm fast
    x1 = np.array([1.0, -2.0, 3.0, 4.0, -5.0], dtype=np.float32)
    axis1 = (0,)
    mean1 = np.array([0.2], dtype=np.float32)
    var1 = np.array([11.76], dtype=np.float32)
    where1 = np.ones_like(x1, dtype=bool)
    list_of_inputs.append({
        "x": x1,
        "axis": axis1,
        "mean": mean1,
        "variance": var1,
        "epsilon": 1e-5,
        "where": where1,
        "algorithm": "fast"
    })

    # Input 2: 2D array, axis (-1,), algorithm stable, float64
    x2 = np.random.randn(3, 4).astype(np.float64)
    axis2 = (-1,)
    mean2 = np.mean(x2, axis=axis2, keepdims=True)
    var2 = np.var(x2, axis=axis2, keepdims=True)
    where2 = np.ones(x2.shape, dtype=bool)
    list_of_inputs.append({
        "x": x2,
        "axis": axis2,
        "mean": mean2,
        "variance": var2,
        "epsilon": 1e-6,
        "where": where2,
        "algorithm": "stable"
    })

    # Input 3: 2D array, axis (0,), algorithm fast
    x3 = np.array([[-10.0, 20.0], [30.0, -40.0], [0.0, 5.0]], dtype=np.float32)
    axis3 = (0,)
    mean3 = np.mean(x3, axis=axis3, keepdims=True)
    var3 = np.var(x3, axis=axis3, keepdims=True)
    where3 = np.array([[True, True], [True, False], [False, True]], dtype=bool)
    list_of_inputs.append({
        "x": x3,
        "axis": axis3,
        "mean": mean3,
        "variance": var3,
        "epsilon": 1e-4,
        "where": where3,
        "algorithm": "fast"
    })

    # Input 4: 3D array, axis (1, 2), algorithm stable
    x4 = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    axis4 = (1, 2)
    mean4 = np.mean(x4, axis=axis4, keepdims=True)
    var4 = np.var(x4, axis=axis4, keepdims=True)
    where4 = np.ones(x4.shape, dtype=bool)
    list_of_inputs.append({
        "x": x4,
        "axis": axis4,
        "mean": mean4,
        "variance": var4,
        "epsilon": 1e-5,
        "where": where4,
        "algorithm": "stable"
    })

    # Input 5: 3D array, axis (0, 1), algorithm fast, non-uniform variance
    x5 = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    axis5 = (0, 1)
    mean5 = np.mean(x5, axis=axis5, keepdims=True)
    var5 = np.var(x5, axis=axis5, keepdims=True)
    where5 = (x5 > 0.0)
    list_of_inputs.append({
        "x": x5,
        "axis": axis5,
        "mean": mean5,
        "variance": var5,
        "epsilon": 1e-3,
        "where": where5,
        "algorithm": "fast"
    })

    # Input 6: 4D array (e.g. NCHW layout), axis (1,), algorithm stable
    x6 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    axis6 = (1,)
    mean6 = np.mean(x6, axis=axis6, keepdims=True)
    var6 = np.var(x6, axis=axis6, keepdims=True)
    where6 = np.ones((2, 3, 4, 4), dtype=bool)
    list_of_inputs.append({
        "x": x6,
        "axis": axis6,
        "mean": mean6,
        "variance": var6,
        "epsilon": 1e-5,
        "where": where6,
        "algorithm": "stable"
    })

    # Input 7: 4D array, axis (1, 2, 3), layer-norm style, algorithm fast
    x7 = np.random.randn(4, 8, 8, 3).astype(np.float64)
    axis7 = (1, 2, 3)
    mean7 = np.mean(x7, axis=axis7, keepdims=True)
    var7 = np.var(x7, axis=axis7, keepdims=True)
    where7 = np.ones(x7.shape, dtype=bool)
    list_of_inputs.append({
        "x": x7,
        "axis": axis7,
        "mean": mean7,
        "variance": var7,
        "epsilon": 1e-7,
        "where": where7,
        "algorithm": "fast"
    })

    # Input 8: 1D array, large epsilon, algorithm stable
    x8 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    axis8 = (0,)
    mean8 = np.array([200.0], dtype=np.float32)
    var8 = np.array([6666.6665], dtype=np.float32)
    where8 = np.array([True, True, True], dtype=bool)
    list_of_inputs.append({
        "x": x8,
        "axis": axis8,
        "mean": mean8,
        "variance": var8,
        "epsilon": 0.1,
        "where": where8,
        "algorithm": "stable"
    })

    # Input 9: 5D array, axis (-1,), algorithm fast
    x9 = np.ones((2, 2, 2, 2, 2), dtype=np.float32) * -0.5
    axis9 = (-1,)
    mean9 = np.mean(x9, axis=axis9, keepdims=True)
    var9 = np.var(x9, axis=axis9, keepdims=True)
    where9 = np.ones(x9.shape, dtype=bool)
    list_of_inputs.append({
        "x": x9,
        "axis": axis9,
        "mean": mean9,
        "variance": var9,
        "epsilon": 1e-5,
        "where": where9,
        "algorithm": "fast"
    })

    # Input 10: 2D array, all axes (0, 1), algorithm stable
    x10 = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    axis10 = (0, 1)
    mean10 = np.mean(x10, axis=axis10, keepdims=True)
    var10 = np.var(x10, axis=axis10, keepdims=True)
    where10 = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append({
        "x": x10,
        "axis": axis10,
        "mean": mean10,
        "variance": var10,
        "epsilon": 1e-5,
        "where": where10,
        "algorithm": "stable"
    })

    return list_of_inputs

generated_inputs["jax.nn.standardize_2"] = standardize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.standardize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.standardize_2'.")


check_valid('jax.nn.standardize', generated_inputs['jax.nn.standardize_2'], lib="jax", suffix=2)
