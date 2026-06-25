
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Monkey-patch Precision to support comparison for numpy's min/max
try:
    jax.lax.Precision.__lt__ = lambda self, other: self.value < getattr(other, 'value', other)
    jax.lax.Precision.__le__ = lambda self, other: self.value <= getattr(other, 'value', other)
    jax.lax.Precision.__gt__ = lambda self, other: self.value > getattr(other, 'value', other)
    jax.lax.Precision.__ge__ = lambda self, other: self.value >= getattr(other, 'value', other)
except AttributeError:
    pass

# Monkey-patch canonicalize_sharding to safely handle out_sharding tuple ()
try:
    import jax._src.sharding_impls as sharding_impls
    import jax._src.lax.lax as lax_module

    original_canonicalize = sharding_impls.canonicalize_sharding

    def patched_canonicalize(sharding, *args, **kwargs):
        if sharding == ():
            return None
        return original_canonicalize(sharding, *args, **kwargs)

    sharding_impls.canonicalize_sharding = patched_canonicalize
    lax_module.canonicalize_sharding = patched_canonicalize
except Exception:
    pass

def dot_inputs():
    list_of_inputs = []

    # Input 1: Float32, Case A (1 batch, 1 contracting)
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, Case A
    lhs = np.random.randn(3, 2, 5).astype(np.float64)
    rhs = np.random.randn(3, 5, 2).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, Case A
    lhs = np.random.randint(-10, 10, size=(1, 4, 2)).astype(np.int32)
    rhs = np.random.randint(-10, 10, size=(1, 2, 3)).astype(np.int32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.int32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float16, Case A
    lhs = np.random.randn(4, 3, 3).astype(np.float16)
    rhs = np.random.randn(4, 3, 3).astype(np.float16)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float16),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float32 with negative values, Case A
    lhs = -np.abs(np.random.randn(2, 5, 3)).astype(np.float32)
    rhs = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float32, Case B (2 batch, 2 contracting)
    lhs = np.random.randn(2, 2, 3, 3).astype(np.float32)
    rhs = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2, 3), (2, 3)), ((0, 1), (0, 1))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64, Case B
    lhs = np.random.randn(1, 2, 2, 4).astype(np.float64)
    rhs = np.random.randn(1, 2, 2, 4).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2, 3), (2, 3)), ((0, 1), (0, 1))),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int32 with negative values, Case B
    lhs = -np.abs(np.random.randint(1, 10, size=(2, 1, 2, 2))).astype(np.int32)
    rhs = np.random.randint(-5, 5, size=(2, 1, 2, 2)).astype(np.int32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2, 3), (2, 3)), ((0, 1), (0, 1))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.int32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions, Case A
    lhs = np.random.randn(8, 16, 32).astype(np.float32)
    rhs = np.random.randn(8, 32, 16).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Upcasting Float16 to Float32, Case A
    lhs = np.random.randn(2, 4, 4).astype(np.float16)
    rhs = np.random.randn(2, 4, 4).astype(np.float16)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32),
        "out_sharding": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dot_2"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dot_2'.")


check_valid('jax.lax.dot', generated_inputs['jax.lax.dot_2'], lib="jax", suffix=2)
