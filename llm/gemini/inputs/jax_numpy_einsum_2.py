
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CustomTensorList(list):
    @property
    def shape(self):
        return (len(self),)
    @property
    def size(self):
        return len(self)
    @property
    def dtype(self):
        return self[0].dtype
    def min(self, *args, **kwargs):
        return min(float(np.min(t)) for t in self)
    def max(self, *args, **kwargs):
        return max(float(np.max(t)) for t in self)

class PrecisionTuple(tuple):
    def __array__(self, dtype=None, *args, **kwargs):
        return np.array(list(self), dtype=object)

def einsum_inputs():
    list_of_inputs = []

    # Input 1: Matrix multiplication (float32)
    input_dict = {
        "subscripts": "ij,jk->ik",
        "operands": CustomTensorList([
            np.random.randn(3, 4).astype(np.float32),
            np.random.randn(4, 5).astype(np.float32)
        ]),
        "optimize": True,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Vector dot product (float32, negative values)
    input_dict = {
        "subscripts": "i,i->",
        "operands": CustomTensorList([
            np.random.uniform(-10, 10, size=(10,)).astype(np.float32),
            np.random.uniform(-10, 10, size=(10,)).astype(np.float32)
        ]),
        "optimize": False,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Outer product (float64)
    input_dict = {
        "subscripts": "i,j->ij",
        "operands": CustomTensorList([
            np.random.randn(5).astype(np.float64),
            np.random.randn(6).astype(np.float64)
        ]),
        "optimize": True,
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Matrix trace (int32)
    input_dict = {
        "subscripts": "ii->",
        "operands": CustomTensorList([
            np.random.randint(-5, 5, size=(4, 4)).astype(np.int32)
        ]),
        "optimize": False,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch matrix multiplication
    input_dict = {
        "subscripts": "bij,bjk->bik",
        "operands": CustomTensorList([
            np.random.randn(2, 3, 4).astype(np.float32),
            np.random.randn(2, 4, 5).astype(np.float32)
        ]),
        "optimize": True,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Sum over axis
    input_dict = {
        "subscripts": "ij->i",
        "operands": CustomTensorList([
            np.random.randn(5, 5).astype(np.float32)
        ]),
        "optimize": False,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bilinear/Quadratic form (float64)
    input_dict = {
        "subscripts": "i,ij,j->",
        "operands": CustomTensorList([
            np.random.randn(3).astype(np.float64),
            np.random.randn(3, 3).astype(np.float64),
            np.random.randn(3).astype(np.float64)
        ]),
        "optimize": True,
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor contraction
    input_dict = {
        "subscripts": "ijk,jlk->il",
        "operands": CustomTensorList([
            np.random.randn(2, 3, 5).astype(np.float32),
            np.random.randn(3, 4, 5).astype(np.float32)
        ]),
        "optimize": True,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Matrix transpose
    input_dict = {
        "subscripts": "ij->ji",
        "operands": CustomTensorList([
            np.random.randn(3, 5).astype(np.float32)
        ]),
        "optimize": False,
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Sum of all elements
    input_dict = {
        "subscripts": "ij->",
        "operands": CustomTensorList([
            np.random.randn(10, 10).astype(np.float32)
        ]),
        "optimize": True,
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.einsum_2"] = einsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.einsum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.einsum_2'.")


check_valid('jax.numpy.einsum', generated_inputs['jax.numpy.einsum_2'], lib="jax", suffix=2)
