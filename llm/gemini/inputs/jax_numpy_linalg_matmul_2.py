
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.lax.Precision to support comparison for numpy validation
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if isinstance(other, jax.lax.Precision) else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if isinstance(other, jax.lax.Precision) else NotImplemented

def matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D matrix multiplication, float32
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(4, 5).astype(np.float32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 2: 1D vector dot product, int32
    x1 = np.random.randint(-10, 10, size=(5,)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(5,)).astype(np.int32)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    preferred_element_type = np.dtype('int32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 3: Batched 3D multiplication, float64
    x1 = np.random.randn(2, 4, 3).astype(np.float64)
    x2 = np.random.randn(2, 3, 5).astype(np.float64)
    precision = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    preferred_element_type = np.dtype('float64')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 4: Higher-dimensional batch with float16 inputs and float32 accumulation
    x1 = np.random.randn(2, 2, 3, 5).astype(np.float16)
    x2 = np.random.randn(2, 2, 5, 4).astype(np.float16)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 5: Multi-dimensional x1 and 1D x2
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 6: Broadcasting in leading dimensions
    x1 = np.random.randn(1, 3, 4).astype(np.float32)
    x2 = np.random.randn(5, 4, 2).astype(np.float32)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 7: Int32 inputs with float64 preferred accumulation
    x1 = np.random.randint(-5, 5, size=(2, 3)).astype(np.int32)
    x2 = np.random.randint(-5, 5, size=(3, 2)).astype(np.int32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
    preferred_element_type = np.dtype('float64')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 8: Square matrix with negative values
    x1 = np.random.uniform(-2.0, 0.0, size=(4, 4)).astype(np.float32)
    x2 = np.random.uniform(-2.0, 0.0, size=(4, 4)).astype(np.float32)
    precision = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 9: Large dimensions, mixed precision modes
    x1 = np.random.randn(16, 32).astype(np.float32)
    x2 = np.random.randn(32, 8).astype(np.float32)
    precision = (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    # Input 10: Multi-dimensional broadcasting, complex case
    x1 = np.random.randn(3, 1, 5, 4).astype(np.float32)
    x2 = np.random.randn(1, 2, 4, 3).astype(np.float32)
    precision = (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT)
    preferred_element_type = np.dtype('float32')
    list_of_inputs.append({
        'x1': x1,
        'x2': x2,
        'precision': precision,
        'preferred_element_type': preferred_element_type
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.matmul_2"] = matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.matmul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.matmul_2'.")


check_valid('jax.numpy.linalg.matmul', generated_inputs['jax.numpy.linalg.matmul_2'], lib="jax", suffix=2)
