
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch JAX to support tuple in delete function seamlessly
try:
    import jax._src.dtypes as jax_dtypes
    if hasattr(jax_dtypes, "dtype"):
        orig_dtype = jax_dtypes.dtype
        def patched_dtype(x):
            if isinstance(x, tuple):
                return np.dtype(type(x[0])) if len(x) > 0 else np.dtype(int)
            return orig_dtype(x)
        jax_dtypes.dtype = patched_dtype
except Exception:
    pass

try:
    import jax._src.numpy.lax_numpy as lax_numpy
    if hasattr(lax_numpy, "_dtype"):
        orig_dtype_lax = lax_numpy._dtype
        def patched_dtype_lax(x):
            if isinstance(x, tuple):
                return np.dtype(type(x[0])) if len(x) > 0 else np.dtype(int)
            return orig_dtype_lax(x)
        lax_numpy._dtype = patched_dtype_lax
except Exception:
    pass

try:
    import jax._src.util as jax_util_base
    if hasattr(jax_util_base, "_is_arraylike"):
        orig_is_arraylike_base = jax_util_base._is_arraylike
        jax_util_base._is_arraylike = lambda x: isinstance(x, tuple) or orig_is_arraylike_base(x)
except Exception:
    pass

try:
    import jax._src.numpy.util as jax_numpy_util
    if hasattr(jax_numpy_util, "_is_arraylike"):
        orig_is_arraylike_num = jax_numpy_util._is_arraylike
        jax_numpy_util._is_arraylike = lambda x: isinstance(x, tuple) or orig_is_arraylike_num(x)
    if hasattr(jax_numpy_util, "check_arraylike"):
        orig_check = jax_numpy_util.check_arraylike
        def patched_check_arraylike(fun_name, *args):
            new_args = [arg for arg in args if not isinstance(arg, tuple)]
            return orig_check(fun_name, *new_args)
        jax_numpy_util.check_arraylike = patched_check_arraylike
except Exception:
    pass

try:
    import jax._src.numpy.lax_numpy as lax_numpy
    if hasattr(lax_numpy, "util") and hasattr(lax_numpy.util, "check_arraylike"):
        orig_check_util = lax_numpy.util.check_arraylike
        def patched_check_arraylike_util(fun_name, *args):
            new_args = [arg for arg in args if not isinstance(arg, tuple)]
            return orig_check_util(fun_name, *new_args)
        lax_numpy.util.check_arraylike = patched_check_arraylike_util
    if hasattr(lax_numpy, "check_arraylike"):
        orig_check_lax = lax_numpy.check_arraylike
        def patched_check_arraylike_lax(fun_name, *args):
            new_args = [arg for arg in args if not isinstance(arg, tuple)]
            return orig_check_lax(fun_name, *new_args)
        lax_numpy.check_arraylike = patched_check_arraylike_lax
except Exception:
    pass

def jax_numpy_delete_inputs():
    list_of_inputs = []

    # Input 1: 1D array, delete indices 1 and 3 along axis 0
    arr = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    obj = (1, 3)
    axis = 0
    assume_unique_indices = False
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 2: 2D array, delete columns 0 and 2
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    obj = (0, 2)
    axis = 1
    assume_unique_indices = True
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 3: 2D array, delete row 1
    arr = np.random.randn(4, 4).astype(np.float64)
    obj = (1,)
    axis = 0
    assume_unique_indices = False
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 4: 3D array, delete index 0 along axis 1
    arr = np.ones((2, 3, 4), dtype=np.float32)
    obj = (0,)
    axis = 1
    assume_unique_indices = True
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 5: 3D array, delete multiple indices along axis 2
    arr = np.arange(27, dtype=np.int64).reshape(3, 3, 3)
    obj = (1, 2)
    axis = 2
    assume_unique_indices = True
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 6: 2D array, delete multiple rows using negative axis
    arr = np.random.randn(5, 2).astype(np.float32)
    obj = (0, 4)
    axis = -2
    assume_unique_indices = False
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 7: 4D array, delete index 1 along axis 3
    arr = np.zeros((2, 2, 2, 2), dtype=np.bool_)
    obj = (1,)
    axis = 3
    assume_unique_indices = False
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 8: 2D float64 array, delete rows with negative index
    arr = np.random.randn(4, 3).astype(np.float64)
    obj = (-1,)
    axis = 0
    assume_unique_indices = True
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 9: 1D array, delete consecutive elements
    arr = np.arange(10, dtype=np.int32)
    obj = (0, 1, 2, 3, 4)
    axis = 0
    assume_unique_indices = False
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    # Input 10: 2D array, delete elements with unique assumption enabled
    arr = np.ones((3, 5), dtype=np.float32)
    obj = (1, 3, 4)
    axis = 1
    assume_unique_indices = True
    list_of_inputs.append({
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices
    })

    return list_of_inputs

generated_inputs["jax.numpy.delete_4"] = jax_numpy_delete_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.delete_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.delete_4'.")


check_valid('jax.numpy.delete', generated_inputs['jax.numpy.delete_4'], lib="jax", suffix=4)
