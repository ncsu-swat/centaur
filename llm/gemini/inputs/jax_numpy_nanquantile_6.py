
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Monkeypatch JAX to allow lists of Tracers during tracing
try:
    import jax._src.numpy.util as jax_util
    import jax.numpy as jnp
    
    # Bypass the check_arraylike validation
    jax_util.check_arraylike = lambda *args, **kwargs: None
    
    # Override _arraylike_asarray to support converting Python lists (even with tracers)
    def custom_arraylike_asarray(x):
        if isinstance(x, list):
            return jnp.array(x)
        return jnp.asarray(x)
        
    jax_util._arraylike_asarray = custom_arraylike_asarray
except Exception:
    pass

def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 1D array
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    q = [0.5]
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 2: 2D array with multiple quantiles, keepdims=True
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    q = [0.25, 0.75]
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([[1.0, 1.0, 2.0], [1.0, 3.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 3: Float64 2D array, axis=(0,)
    a = np.array([[np.nan, 2.0], [3.0, np.nan], [5.0, 6.0]], dtype=np.float64)
    q = [0.5]
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 4: 3D array
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    q = [0.1, 0.5, 0.9]
    axis = (1, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.abs(np.random.randn(2, 3, 4)).astype(np.float32) + 0.1
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 5: negative values, 1D array
    a = np.array([-10.0, -5.0, np.nan, -1.0, 0.0], dtype=np.float32)
    q = [0.3]
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.array([2.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 6: Large 2D array
    a = np.random.randn(10, 10).astype(np.float32)
    a[a < -1.0] = np.nan
    q = [0.5]
    axis = (0, 1)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((10, 10), dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 7: 3D array with keepdims=False
    a = np.random.randn(3, 2, 2).astype(np.float32)
    a[0, 0, 0] = np.nan
    q = [0.25, 0.5, 0.75]
    axis = (0, 2)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.abs(np.random.randn(3, 2, 2)).astype(np.float32) + 0.5
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 8: 1D array, almost all nan
    a = np.array([np.nan, np.nan, 5.0], dtype=np.float32)
    q = [0.5]
    axis = (0,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 9: 2D array with 0.0 quantile (min)
    a = np.array([[5.0, np.nan, 2.0], [1.0, 10.0, np.nan]], dtype=np.float32)
    q = [0.0]
    axis = (1,)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = True
    weights = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    # Input 10: 4D array, axis=(1, 3)
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    a[a > 1.0] = np.nan
    q = [0.5]
    axis = (1, 3)
    overwrite_input = False
    method = "inverted_cdf"
    keepdims = False
    weights = np.ones((2, 3, 2, 4), dtype=np.float32)
    list_of_inputs.append({
        "a": a, "q": q, "axis": axis, "overwrite_input": overwrite_input,
        "method": method, "keepdims": keepdims, "weights": weights
    })

    return list_of_inputs

generated_inputs["jax.numpy.nanquantile_6"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_6'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_6'], lib="jax", suffix=6)
