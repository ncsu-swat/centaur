
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
from jax.tree_util import register_pytree_node

# Define a custom tuple subclass that behaves like a NumPy array and register it as a JAX PyTree leaf
class ArrayTuple(tuple):
    @property
    def ndim(self):
        return np.array(list(self)).ndim
    
    @property
    def size(self):
        return np.array(list(self)).size
    
    @property
    def shape(self):
        return np.array(list(self)).shape
    
    def ravel(self):
        return np.array(list(self)).ravel()
    
    def __array__(self, dtype=None, **kwargs):
        return np.array(list(self), dtype=dtype)
    
    def __jax_array__(self):
        return np.array(list(self))

try:
    register_pytree_node(
        ArrayTuple,
        lambda x: ((), x),  # empty children makes it a PyTree leaf
        lambda aux, children: aux
    )
except Exception:
    pass

def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, float32
    a = np.zeros((3, 3), dtype=np.float32)
    val = ArrayTuple((1.0, 2.0, 3.0))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D square matrix, int32, repeated val
    a = np.zeros((4, 4), dtype=np.int32)
    val = ArrayTuple((5, 6))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D non-square matrix, more columns, float64
    a = np.zeros((3, 5), dtype=np.float64)
    val = ArrayTuple((1.5, 2.5, 3.5, 4.5))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D non-square matrix, more rows, int16
    a = np.zeros((5, 3), dtype=np.int16)
    val = ArrayTuple((-1, -2, -3))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D square matrix, float32
    a = np.zeros((2, 2, 2), dtype=np.float32)
    val = ArrayTuple((9.0, 10.0))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D square matrix, int64, single val
    a = np.zeros((3, 3, 3, 3), dtype=np.int64)
    val = ArrayTuple((42,))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 2D square matrix, float32
    a = np.zeros((100, 100), dtype=np.float32)
    val = ArrayTuple((0.1, 0.2, 0.3, 0.4, 0.5))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D square matrix, negative values
    a = np.zeros((5, 5), dtype=np.float32)
    val = ArrayTuple((-1.0, -2.0, -3.0, -4.0, -5.0))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D square matrix, single element tuple
    a = np.zeros((2, 2), dtype=np.float32)
    val = ArrayTuple((100.0,))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D square matrix, larger, float32
    a = np.zeros((5, 5, 5), dtype=np.float32)
    val = ArrayTuple((1.0, 2.0, 3.0, 4.0, 5.0))
    input_dict = {"a": a, "val": val, "wrap": False, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fill_diagonal_5"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_5'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_5'], lib="jax", suffix=5)
