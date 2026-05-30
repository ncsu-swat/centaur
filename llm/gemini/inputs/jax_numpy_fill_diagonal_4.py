
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_DISABLE_PJRT_PLUGINS"] = "1"

import numpy as np
import copy

try:
    import jax._src.numpy.util as jax_util
    if hasattr(jax_util, 'is_arraylike'):
        original_is_arraylike = jax_util.is_arraylike
        def patched_is_arraylike(x):
            if isinstance(x, list):
                return True
            return original_is_arraylike(x)
        jax_util.is_arraylike = patched_is_arraylike
        
        try:
            import jax._src.util as jax_src_util
            if hasattr(jax_src_util, 'is_arraylike'):
                jax_src_util.is_arraylike = patched_is_arraylike
        except Exception:
            pass
except Exception:
    pass

def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D Square matrix of float32, list of length matching diagonal
    a = np.zeros((3, 3), dtype=np.float32)
    val = [1.0, 2.0, 3.0]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D Non-square matrix (rows < cols), list of integers
    a = np.zeros((3, 5), dtype=np.int32)
    val = [1, 2, 3]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D Square matrix, list with too many entries (truncated)
    a = np.zeros((3, 3), dtype=np.int32)
    val = [10, 20, 30, 40, 50]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D Square matrix, list with too few entries (repeated)
    a = np.zeros((4, 4), dtype=np.int32)
    val = [1, 2]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D Square tensor (all dimensions same size)
    a = np.zeros((2, 2, 2), dtype=np.float32)
    val = [1.5, 2.5]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D Square tensor (all dimensions same size)
    a = np.zeros((3, 3, 3, 3), dtype=np.float64)
    val = [10.0]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D Square matrix with negative values
    a = np.ones((4, 4), dtype=np.float64) * -1
    val = [-2.0, -3.0, -4.0, -5.0]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D Square matrix, int16 dtype
    a = np.zeros((5, 5), dtype=np.int16)
    val = [5, 4, 3, 2, 1]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D rectangular matrix (rows > cols)
    a = np.zeros((5, 3), dtype=np.float32)
    val = [9.0, 8.0, 7.0]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 2D square matrix
    a = np.zeros((100, 100), dtype=np.int32)
    val = [999]
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fill_diagonal_4"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_4'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_4'], lib="jax", suffix=4)
