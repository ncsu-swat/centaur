signatures = {}
signatures["jax.numpy.abs_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.abs_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.abs_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.abs_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.abs_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.absolute"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.acos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.acosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.add_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.add_2"] = {
    "args": {
        "x": "tensor",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.add_3"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.add_4"] = {
    "args": {
        "x": "float",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.add_5"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.all_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.all_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.all_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.allclose"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "rtol": "float",  # Can also be a tensor, but float is the most common type
        "atol": "float",  # Can also be a tensor, but float is the most common type
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.amax_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "initial": "tensor",  # Can also be a scalar (float/integer)
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.amax_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "initial": "tensor",  # Can also be a scalar (float/integer)
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.amin_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.amin_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.angle_1"] = {
    "args": {
        "z": "tensor"
    },
    "kwargs": {
        "deg": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.angle_2"] = {
    "args": {
        "z": "float"  # To represent a scalar complex number
    },
    "kwargs": {
        "deg": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.any_1"] = {
    "args": {
        "a": "tensor",
        "axis": "integer"
    },
    "kwargs": {
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.any_2"] = {
    "args": {
        "a": "tensor",
        "axis": "tuple"
    },
    "kwargs": {
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.any_3"] = {
    "args": {
        "a": "tensor",
        "axis": "list"
    },
    "kwargs": {
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.any_4"] = {
    "args": {
        "a": "tensor",
        "axis": "tensor"
    },
    "kwargs": {
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.append"] = {
    "args": {
        "arr": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.apply_along_axis"] = {
    "args": {
        "func1d": "string",  # "callable" would be more accurate
        "axis": "integer",
        "arr": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.apply_over_axes_1"] = {
    "args": {
        "func": "string",  # Should be Callable
        "a": "tensor",
        "axes": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.apply_over_axes_2"] = {
    "args": {
        "func": "string",  # Should be Callable
        "a": "tensor",
        "axes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.apply_over_axes_3"] = {
    "args": {
        "func": "string",  # Should be Callable
        "a": "tensor",
        "axes": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arange_1"] = {
    "args": {
        "start": "integer",
        "stop": "integer",
        "step": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.arange_2"] = {
    "args": {
        "start": "float",
        "stop": "float",
        "step": "float"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.arange_3"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor",
        "step": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.arccos_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arccos_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arccos_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arccosh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arccosh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arccosh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arccosh_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arccosh_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arcsin_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arcsin_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arcsin_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arcsinh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arcsinh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arcsinh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arctan_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arctan_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arctan_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arctan2"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.arctanh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arctanh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.arctanh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.argmax"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.argmin"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.argpartition"] = {
    "args": {
        "a": "tensor",
        "kth": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.argsort_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.argsort_2"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.argsort_3"] = {
    "args": {
        "a": "float"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.argsort_4"] = {
    "args": {
        "a": "boolean"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.argwhere_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.argwhere_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.argwhere_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}
signatures["jax.numpy.around"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "decimals": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.array_1"] = {
    "args": {
        "object": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "order": "string",
        "ndmin": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_2"] = {
    "args": {
        "object": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "order": "string",
        "ndmin": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_3"] = {
    "args": {
        "object": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "order": "string",
        "ndmin": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_4"] = {
    "args": {
        "object": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "order": "string",
        "ndmin": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_5"] = {
    "args": {
        "object": "float"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "order": "string",
        "ndmin": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_6"] = {
    "args": {
        "object": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "order": "string",
        "ndmin": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.array_equal"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.array_equiv"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.array_split_1"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_split_2"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_split_3"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.array_split_4"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.asarray_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.asarray_2"] = {
    "args": {
        "a": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.asarray_3"] = {
    "args": {
        "a": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.asarray_4"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.asarray_5"] = {
    "args": {
        "a": "float"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.asarray_6"] = {
    "args": {
        "a": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.asarray_7"] = {
    "args": {
        "a": "string"
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.asin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.asinh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.asinh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.asinh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.asinh_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.asinh_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.astype_1"] = {
    "args": {
        "x": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.astype_2"] = {
    "args": {
        "x": "tensor",
        "dtype": "string"
    },
    "kwargs": {
        "copy": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.atan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.atan2"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.atanh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atanh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atanh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atanh_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atanh_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.atleast_1d_1"] = {
    "args": {
        "arys": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atleast_1d_2"] = {
    "args": {
        "arys": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atleast_1d_3"] = {
    "args": {
        "arys": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atleast_1d_4"] = {
    "args": {
        "arys": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.atleast_2d"] = {
    "args": {
        "arys": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.atleast_3d_1"] = {
    "args": {
        "arys": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.atleast_3d_2"] = {
    "args": {
        "arys": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.average_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "weights": "tensor",
        "returned": "boolean",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.average_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "weights": "tensor",
        "returned": "boolean",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.average_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "weights": "tensor",
        "returned": "boolean",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.bartlett"] = {
    "args": {
        "M": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bincount"] = {
    "args": {
        "x": "tensor",
        "weights": "tensor",
        "minlength": "integer"
    },
    "kwargs": {
        "length": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.bitwise_and"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_count"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_invert"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_left_shift"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_or_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_or_2"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_or_3"] = {
    "args": {
        "x": "tensor",
        "y": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_or_4"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_or_5"] = {
    "args": {
        "x": "boolean",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_right_shift"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.bitwise_xor_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_xor_2"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_xor_3"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_xor_4"] = {
    "args": {
        "x": "tensor",
        "y": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.bitwise_xor_5"] = {
    "args": {
        "x": "boolean",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.blackman"] = {
    "args": {
        "M": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.block_1"] = {
    "args": {
        "arrays": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.block_2"] = {
    "args": {
        "arrays": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.block_3"] = {
    "args": {
        "arrays": "list"  # represents nested lists of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.broadcast_arrays"] = {
    "args": {
        "args": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.broadcast_shapes_1"] = {
    "args": {
        "shapes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.broadcast_shapes_2"] = {
    "args": {
        "shapes": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.broadcast_to_1"] = {
    "args": {
        "array": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.broadcast_to_2"] = {
    "args": {
        "array": "tensor",
        "shape": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.broadcast_to_3"] = {
    "args": {
        "array": "tensor",
        "shape": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.can_cast_1"] = {
    "args": {
        "from_": "dtype",
        "to": "dtype"
    },
    "kwargs": {
        "casting": "string"
    },
    "inner": {}
}

signatures["jax.numpy.can_cast_2"] = {
    "args": {
        "from_": "dtype",
        "to": "string"
    },
    "kwargs": {
        "casting": "string"
    },
    "inner": {}
}

signatures["jax.numpy.can_cast_3"] = {
    "args": {
        "from_": "string",
        "to": "dtype"
    },
    "kwargs": {
        "casting": "string"
    },
    "inner": {}
}

signatures["jax.numpy.can_cast_4"] = {
    "args": {
        "from_": "string",
        "to": "string"
    },
    "kwargs": {
        "casting": "string"
    },
    "inner": {}
}

signatures["jax.numpy.can_cast_5"] = {
    "args": {
        "from_": "tensor",
        "to": "dtype"
    },
    "kwargs": {
        "casting": "string"
    },
    "inner": {}
}

signatures["jax.numpy.can_cast_6"] = {
    "args": {
        "from_": "tensor",
        "to": "string"
    },
    "kwargs": {
        "casting": "string"
    },
    "inner": {}
}
signatures["jax.numpy.cbrt_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cbrt_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cbrt_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cbrt_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cbrt_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.ceil"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.choose_1"] = {
    "args": {
        "a": "tensor",
        "choices": "tensor_list"
    },
    "kwargs": {
        "mode": "string"
    },
    "inner": {}
}

signatures["jax.numpy.choose_2"] = {
    "args": {
        "a": "tensor",
        "choices": "tensor"
    },
    "kwargs": {
        "mode": "string"
    },
    "inner": {}
}
signatures["jax.numpy.clip_1"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {
        "min": "tensor",
        "max": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.clip_2"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {
        "min": "float",
        "max": "float"
    },
    "inner": {}
}

signatures["jax.numpy.clip_3"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {
        "min": "integer",
        "max": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.column_stack_1"] = {
    "args": {
        "tup": "tensor_list"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.column_stack_2"] = {
    "args": {
        "tup": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.compress_1"] = {
    "args": {
        "condition": "tensor",
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.compress_2"] = {
    "args": {
        "condition": "tensor",
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.compress_3"] = {
    "args": {
        "condition": "tensor",
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.concat_1"] = {
    "args": {
        "arrays": "tensor_list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.concat_2"] = {
    "args": {
        "arrays": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.concatenate_1"] = {
    "args": {
        "arrays": "tensor_list"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.concatenate_2"] = {
    "args": {
        "arrays": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.conj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.conjugate_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.conjugate_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.conjugate_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.conjugate_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.conjugate_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.convolve_1"] = {
    "args": {
        "a": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "precision": "string",  # Precision can be a string (e.g., 'DEFAULT', 'HIGH')
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.convolve_2"] = {
    "args": {
        "a": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "precision": "tuple",   # Precision can be a tuple of precisions
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.copy"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "order": "string"
    },
    "inner": {}
}
signatures["jax.numpy.copysign"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.corrcoef"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "y": "tensor",
        "rowvar": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.correlate_1"] = {
    "args": {
        "a": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.correlate_2"] = {
    "args": {
        "a": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.cos_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cos_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cos_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.cosh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cosh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cosh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cosh_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.cosh_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.count_nonzero_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.count_nonzero_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.count_nonzero_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.cov"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "y": "tensor",
        "rowvar": "boolean",
        "bias": "boolean",
        "ddof": "integer",
        "fweights": "tensor",
        "aweights": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.cross"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "axisa": "integer",
        "axisb": "integer",
        "axisc": "integer",
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.cumprod"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.cumsum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.cumulative_prod"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "include_initial": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.cumulative_sum"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "include_initial": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.deg2rad_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.deg2rad_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.deg2rad_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.degrees_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.degrees_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.degrees_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.degrees_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.degrees_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.delete_1"] = {
    "args": {
        "arr": "tensor",
        "obj": "integer"
    },
    "kwargs": {
        "axis": "integer",
        "assume_unique_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.delete_2"] = {
    "args": {
        "arr": "tensor",
        "obj": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "assume_unique_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.delete_3"] = {
    "args": {
        "arr": "tensor",
        "obj": "list"
    },
    "kwargs": {
        "axis": "integer",
        "assume_unique_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.delete_4"] = {
    "args": {
        "arr": "tensor",
        "obj": "tuple"
    },
    "kwargs": {
        "axis": "integer",
        "assume_unique_indices": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.diag"] = {
    "args": {
        "v": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.diag_indices"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "ndim": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.diag_indices_from"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.diagflat"] = {
    "args": {
        "v": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.diagonal"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "axis1": "integer",
        "axis2": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.diff"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "axis": "integer",
        "prepend": "tensor",  # Can be a scalar or array, mapped to "tensor"
        "append": "tensor"    # Can be a scalar or array, mapped to "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.digitize"] = {
    "args": {
        "x": "tensor",
        "bins": "tensor"
    },
    "kwargs": {
        "right": "boolean",
        "method": "string"
    },
    "inner": {}
}
signatures["jax.numpy.divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.divmod_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.divmod_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.divmod_3"] = {
    "args": {
        "x1": "tensor",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.divmod_4"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.divmod_5"] = {
    "args": {
        "x1": "float",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.divmod_6"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.divmod_7"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.dot_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.dot_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.dsplit_1"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.dsplit_2"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.dsplit_3"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.dsplit_4"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.dstack_1"] = {
    "args": {
        "tup": "tensor_list"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.dstack_2"] = {
    "args": {
        "tup": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.ediff1d_1"] = {
    "args": {
        "ary": "tensor"
    },
    "kwargs": {
        "to_end": "tensor",
        "to_begin": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.ediff1d_2"] = {
    "args": {
        "ary": "tensor"
    },
    "kwargs": {
        "to_end": "integer",
        "to_begin": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.ediff1d_3"] = {
    "args": {
        "ary": "tensor"
    },
    "kwargs": {
        "to_end": "float",
        "to_begin": "float"
    },
    "inner": {}
}

signatures["jax.numpy.ediff1d_4"] = {
    "args": {
        "ary": "list"
    },
    "kwargs": {
        "to_end": "tensor",
        "to_begin": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.ediff1d_5"] = {
    "args": {
        "ary": "tuple"
    },
    "kwargs": {
        "to_end": "tensor",
        "to_begin": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.einsum_1"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list"
    },
    "kwargs": {
        "optimize": "string",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.einsum_2"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list"
    },
    "kwargs": {
        "optimize": "boolean",
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.einsum_3"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list"
    },
    "kwargs": {
        "optimize": "list",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.einsum_path_1"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor"  # representing *operands which are multiple tensors
    },
    "kwargs": {
        "optimize": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.einsum_path_2"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor"
    },
    "kwargs": {
        "optimize": "string"
    },
    "inner": {}
}

signatures["jax.numpy.einsum_path_3"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor"
    },
    "kwargs": {
        "optimize": "list"
    },
    "inner": {}
}
signatures["jax.numpy.empty_1"] = {
    "args": {
        "shape": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out_sharding": "tuple"  # NamedSharding / PartitionSpec, mapped to tuple
    },
    "inner": {}
}

signatures["jax.numpy.empty_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "out_sharding": "tuple"  # NamedSharding / PartitionSpec, mapped to tuple
    },
    "inner": {}
}

signatures["jax.numpy.empty_3"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "out_sharding": "tuple"  # NamedSharding / PartitionSpec, mapped to tuple
    },
    "inner": {}
}

signatures["jax.numpy.empty_4"] = {
    "args": {
        "shape": "integer"
    },
    "kwargs": {
        "dtype": "string",
        "out_sharding": "tuple"  # NamedSharding / PartitionSpec, mapped to tuple
    },
    "inner": {}
}

signatures["jax.numpy.empty_5"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "string",
        "out_sharding": "tuple"  # NamedSharding / PartitionSpec, mapped to tuple
    },
    "inner": {}
}

signatures["jax.numpy.empty_6"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "string",
        "out_sharding": "tuple"  # NamedSharding / PartitionSpec, mapped to tuple
    },
    "inner": {}
}
signatures["jax.numpy.empty_like_1"] = {
    "args": {
        "prototype": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.empty_like_2"] = {
    "args": {
        "prototype": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.empty_like_3"] = {
    "args": {
        "prototype": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "list"
    },
    "inner": {}
}
signatures["jax.numpy.equal_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.equal_2"] = {
    "args": {
        "x": "tensor",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.equal_3"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.equal_4"] = {
    "args": {
        "x": "float",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.equal_5"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.equal_6"] = {
    "args": {
        "x": "float",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.equal_7"] = {
    "args": {
        "x": "integer",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.exp_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.exp_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.exp_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.exp2_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.exp2_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.exp2_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.expand_dims_1"] = {
    "args": {
        "a": "tensor",
        "axis": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.expand_dims_2"] = {
    "args": {
        "a": "tensor",
        "axis": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.expand_dims_3"] = {
    "args": {
        "a": "tensor",
        "axis": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.expm1_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.expm1_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.expm1_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.expm1_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.expm1_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.extract"] = {
    "args": {
        "condition": "tensor",
        "arr": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.eye_1"] = {
    "args": {
        "N": "integer"
    },
    "kwargs": {
        "M": "integer",
        "k": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.eye_2"] = {
    "args": {
        "N": "integer"
    },
    "kwargs": {
        "M": "integer",
        "k": "tensor",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.fabs_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fabs_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fabs_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fabs_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.fill_diagonal_1"] = {
    "args": {
        "a": "tensor",
        "val": "tensor",
    },
    "kwargs": {
        "wrap": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}

signatures["jax.numpy.fill_diagonal_2"] = {
    "args": {
        "a": "tensor",
        "val": "integer",
    },
    "kwargs": {
        "wrap": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}

signatures["jax.numpy.fill_diagonal_3"] = {
    "args": {
        "a": "tensor",
        "val": "float",
    },
    "kwargs": {
        "wrap": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}

signatures["jax.numpy.fill_diagonal_4"] = {
    "args": {
        "a": "tensor",
        "val": "list",
    },
    "kwargs": {
        "wrap": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}

signatures["jax.numpy.fill_diagonal_5"] = {
    "args": {
        "a": "tensor",
        "val": "tuple",
    },
    "kwargs": {
        "wrap": "boolean",
        "inplace": "boolean",
    },
    "inner": {},
}
signatures["jax.numpy.flatnonzero_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.flatnonzero_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.flatnonzero_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.flatnonzero_4"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tuple"
    },
    "inner": {}
}
signatures["jax.numpy.flip_1"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.flip_2"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}

signatures["jax.numpy.flip_3"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}
signatures["jax.numpy.fliplr"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.flipud"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.float_power_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.float_power_2"] = {
    "args": {
        "x": "float",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.float_power_3"] = {
    "args": {
        "x": "integer",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.floor_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.floor_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.floor_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.floor_divide_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.floor_divide_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.floor_divide_3"] = {
    "args": {
        "x1": "tensor",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.floor_divide_4"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.floor_divide_5"] = {
    "args": {
        "x1": "float",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.fmax_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmax_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmax_3"] = {
    "args": {
        "x1": "float",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmax_4"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmax_5"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmax_6"] = {
    "args": {
        "x1": "tensor",
        "x2": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmax_7"] = {
    "args": {
        "x1": "boolean",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.fmin"] = {
    "args": {
        "x1": "tensor", # Can also be a scalar (integer/float), represented here as tensor
        "x2": "tensor"  # Can also be a scalar (integer/float), represented here as tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.fmod_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmod_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmod_3"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmod_4"] = {
    "args": {
        "x1": "float",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmod_5"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmod_6"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.fmod_7"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.frexp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.from_dlpack"] = {
    "args": {
        "x": "tensor"  # DLPack tensor representation
    },
    "kwargs": {
        "copy": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.fromfunction_1"] = {
    "args": {
        "function": "string",  # should be "callable"
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.fromfunction_2"] = {
    "args": {
        "function": "string",  # should be "callable"
        "shape": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.full_1"] = {
    "args": {
        "shape": "integer",
        "fill_value": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_2"] = {
    "args": {
        "shape": "integer",
        "fill_value": "float"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_3"] = {
    "args": {
        "shape": "integer",
        "fill_value": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_4"] = {
    "args": {
        "shape": "integer",
        "fill_value": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_5"] = {
    "args": {
        "shape": "tuple",
        "fill_value": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_6"] = {
    "args": {
        "shape": "tuple",
        "fill_value": "float"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_7"] = {
    "args": {
        "shape": "tuple",
        "fill_value": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.full_8"] = {
    "args": {
        "shape": "tuple",
        "fill_value": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.full_like_1"] = {
    "args": {
        "a": "tensor",
        "fill_value": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.full_like_2"] = {
    "args": {
        "a": "tensor",
        "fill_value": "float"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.full_like_3"] = {
    "args": {
        "a": "tensor",
        "fill_value": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.full_like_4"] = {
    "args": {
        "a": "tensor",
        "fill_value": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "tuple"
    },
    "inner": {}
}
signatures["jax.numpy.gcd_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.gcd_2"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.geomspace_1"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.geomspace_2"] = {
    "args": {
        "start": "float",
        "stop": "float"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.geomspace_3"] = {
    "args": {
        "start": "integer",
        "stop": "integer"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.gradient_1"] = {
    "args": {
        "f": "tensor",
        "varargs": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.gradient_2"] = {
    "args": {
        "f": "tensor",
        "varargs": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.gradient_3"] = {
    "args": {
        "f": "tensor",
        "varargs": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.gradient_4"] = {
    "args": {
        "f": "tensor",
        "varargs": "float"
    },
    "kwargs": {
        "axis": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.gradient_5"] = {
    "args": {
        "f": "tensor",
        "varargs": "float"
    },
    "kwargs": {
        "axis": "tuple",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.gradient_6"] = {
    "args": {
        "f": "tensor",
        "varargs": "float"
    },
    "kwargs": {
        "axis": "list",
        "edge_order": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.greater_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.greater_2"] = {
    "args": {
        "x": "tensor",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.greater_3"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.greater_4"] = {
    "args": {
        "x": "float",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.greater_5"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.greater_6"] = {
    "args": {
        "x": "float",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.greater_7"] = {
    "args": {
        "x": "integer",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.greater_equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.hamming"] = {
    "args": {
        "M": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.hanning"] = {
    "args": {
        "M": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.heaviside_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_3"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_4"] = {
    "args": {
        "x1": "float",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_5"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_6"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_7"] = {
    "args": {
        "x1": "float",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_8"] = {
    "args": {
        "x1": "integer",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.heaviside_9"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.histogram_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "bins": "tensor",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.histogram2d_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_2"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "tensor",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_3"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "list",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_4"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "tuple",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_5"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "tensor_list",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_6"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_7"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "tensor",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_8"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "list",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_9"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "tuple",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogram2d_10"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "bins": "tensor_list",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.histogram_bin_edges_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "tuple",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.histogram_bin_edges_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "bins": "tensor",
        "range": "tuple",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.histogram_bin_edges_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "list",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.histogram_bin_edges_4"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "tensor",
        "weights": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.histogramdd_1"] = {
    "args": {
        "sample": "tensor"
    },
    "kwargs": {
        "bins": "integer",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogramdd_2"] = {
    "args": {
        "sample": "tensor"
    },
    "kwargs": {
        "bins": "tensor",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogramdd_3"] = {
    "args": {
        "sample": "tensor"
    },
    "kwargs": {
        "bins": "list",
        "range": "list",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.histogramdd_4"] = {
    "args": {
        "sample": "tensor"
    },
    "kwargs": {
        "bins": "tuple",
        "range": "tuple",
        "weights": "tensor",
        "density": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.hsplit_1"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.hsplit_2"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.hsplit_3"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.hsplit_4"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.hstack_1"] = {
    "args": {
        "tup": "tensor_list"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.hstack_2"] = {
    "args": {
        "tup": "tuple"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.hstack_3"] = {
    "args": {
        "tup": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.hypot_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.hypot_2"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.hypot_3"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.i0_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.i0_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.i0_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.identity"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.imag_1"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.imag_2"] = {
    "args": {
        "val": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.imag_3"] = {
    "args": {
        "val": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.indices_1"] = {
    "args": {
        "dimensions": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "sparse": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.indices_2"] = {
    "args": {
        "dimensions": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "sparse": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.inner_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.inner_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.inner_3"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "string"
    },
    "inner": {}
}

signatures["jax.numpy.inner_4"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "string"
    },
    "inner": {}
}
signatures["jax.numpy.insert_1"] = {
    "args": {
        "arr": "tensor",
        "obj": "integer",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.insert_2"] = {
    "args": {
        "arr": "tensor",
        "obj": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.insert_3"] = {
    "args": {
        "arr": "tensor",
        "obj": "list",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.insert_4"] = {
    "args": {
        "arr": "tensor",
        "obj": "tuple",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.interp_1"] = {
    "args": {
        "x": "tensor",
        "xp": "tensor",
        "fp": "tensor"
    },
    "kwargs": {
        "left": "tensor",
        "right": "tensor",
        "period": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.interp_2"] = {
    "args": {
        "x": "tensor",
        "xp": "tensor",
        "fp": "tensor"
    },
    "kwargs": {
        "left": "string",
        "right": "string",
        "period": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.intersect1d"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "return_indices": "boolean",
        "size": "integer",
        "fill_value": "tensor"  # Also accepts scalar float or integer
    },
    "inner": {}
}
signatures["jax.numpy.invert"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isclose"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "rtol": "float",  # Note: Can also be 'tensor'
        "atol": "float",  # Note: Can also be 'tensor'
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.iscomplex"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iscomplexobj_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iscomplexobj_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iscomplexobj_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iscomplexobj_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iscomplexobj_5"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iscomplexobj_6"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isdtype_1"] = {
    "args": {
        "dtype": "dtype",
        "kind": "string"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isdtype_2"] = {
    "args": {
        "dtype": "dtype",
        "kind": "dtype"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isdtype_3"] = {
    "args": {
        "dtype": "dtype",
        "kind": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isfinite_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isfinite_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isfinite_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isfinite_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isfinite_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isin"] = {
    "args": {
        "element": "tensor",
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean",
        "method": "string"
    },
    "inner": {}
}
signatures["jax.numpy.isinf_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isinf_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isinf_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isinf_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isnan_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isnan_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isnan_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isneginf_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.isneginf_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.isneginf_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.isposinf_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.isposinf_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.isposinf_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.isreal_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isreal_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isreal_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isreal_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isreal_5"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isreal_6"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isrealobj_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isrealobj_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isrealobj_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isrealobj_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isrealobj_5"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isrealobj_6"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.isscalar_1"] = {
    "args": {
        "element": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_2"] = {
    "args": {
        "element": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_3"] = {
    "args": {
        "element": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_4"] = {
    "args": {
        "element": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_5"] = {
    "args": {
        "element": "string"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_6"] = {
    "args": {
        "element": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_7"] = {
    "args": {
        "element": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.isscalar_8"] = {
    "args": {
        "element": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.issubdtype_1"] = {
    "args": {
        "arg1": "dtype",
        "arg2": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.issubdtype_2"] = {
    "args": {
        "arg1": "string",
        "arg2": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.issubdtype_3"] = {
    "args": {
        "arg1": "dtype",
        "arg2": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.issubdtype_4"] = {
    "args": {
        "arg1": "string",
        "arg2": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.iterable_1"] = {
    "args": {
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.iterable_2"] = {
    "args": {
        "y": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.iterable_3"] = {
    "args": {
        "y": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.iterable_4"] = {
    "args": {
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.iterable_5"] = {
    "args": {
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.iterable_6"] = {
    "args": {
        "y": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.iterable_7"] = {
    "args": {
        "y": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.ix_"] = {
    "args": {
        "args": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.kaiser_1"] = {
    "args": {
        "M": "integer",
        "beta": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.kaiser_2"] = {
    "args": {
        "M": "integer",
        "beta": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.kaiser_3"] = {
    "args": {
        "M": "integer",
        "beta": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.kaiser_4"] = {
    "args": {
        "M": "integer",
        "beta": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.kron"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.lcm"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.ldexp_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.ldexp_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.left_shift_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.left_shift_2"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.left_shift_3"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.left_shift_4"] = {
    "args": {
        "x": "integer",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.less_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.less_2"] = {
    "args": {
        "x": "tensor",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.less_3"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.less_4"] = {
    "args": {
        "x": "float",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.less_5"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.less_6"] = {
    "args": {
        "x": "float",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.less_7"] = {
    "args": {
        "x": "integer",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.less_equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.lexsort_1"] = {
    "args": {
        "keys": "tensor_list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.lexsort_2"] = {
    "args": {
        "keys": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.linspace_1"] = {
    "args": {
        "start": "float",
        "stop": "float"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "retstep": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.linspace_2"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "retstep": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.log_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.log_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.log_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.log_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.log_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.log10"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.log1p_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.log1p_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.log1p_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.log2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logaddexp"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logaddexp2_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.logaddexp2_2"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.logaddexp2_3"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logical_and"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logical_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logical_or"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logical_xor"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.logspace_1"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "base": "tensor",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.logspace_2"] = {
    "args": {
        "start": "float",
        "stop": "float"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "base": "float",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.logspace_3"] = {
    "args": {
        "start": "integer",
        "stop": "integer"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "base": "integer",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.mask_indices"] = {
    "args": {
        "n": "integer",
        "mask_func": "string",  # should be callable (Callable[[ArrayLike, int], Array])
        "k": "integer"
    },
    "kwargs": {
        "size": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.matmul_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.matmul_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.matrix_transpose"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.matvec"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.max_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "initial": "tensor",  # ArrayLike, mapped to tensor
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.max_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.max_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.maximum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.mean_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.mean_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.mean_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.median_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.median_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.meshgrid"] = {
    "args": {
        "xi": "tensor"  # Represents *xi variadic positional arguments of tensor type
    },
    "kwargs": {
        "copy": "boolean",
        "sparse": "boolean",
        "indexing": "string"
    },
    "inner": {}
}
signatures["jax.numpy.min_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.min_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.min_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.min_4"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tensor",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.minimum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.mod"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.modf_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.modf_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.modf_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.modf_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.modf_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.moveaxis_1"] = {
    "args": {
        "a": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.moveaxis_2"] = {
    "args": {
        "a": "tensor",
        "source": "list",
        "destination": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.moveaxis_3"] = {
    "args": {
        "a": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.multiply"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.nan_to_num_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "copy": "boolean",
        "nan": "float",
        "posinf": "float",
        "neginf": "float"
    },
    "inner": {}
}

signatures["jax.numpy.nan_to_num_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "copy": "boolean",
        "nan": "tensor",
        "posinf": "tensor",
        "neginf": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanargmax"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.nanargmin"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.nancumprod"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.nancumsum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.nanmax_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanmax_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanmax_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanmean_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanmean_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanmean_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "keepdims": "boolean",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanmedian_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "overwrite_input": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.nanmedian_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "overwrite_input": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.nanmin_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "initial": "tensor",  # initial can be integer, float, or tensor. Using tensor as general.
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanmin_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanmin_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanpercentile_1"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanpercentile_2"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanprod_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",  # Can also be float or integer
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanprod_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanprod_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanquantile_1"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanquantile_2"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanquantile_3"] = {
    "args": {
        "a": "tensor",
        "q": "float"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanquantile_4"] = {
    "args": {
        "a": "tensor",
        "q": "float"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanquantile_5"] = {
    "args": {
        "a": "tensor",
        "q": "list"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanquantile_6"] = {
    "args": {
        "a": "tensor",
        "q": "list"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanstd_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanstd_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanstd_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nansum_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nansum_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "integer",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nansum_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "float",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.nanvar_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanvar_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nanvar_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.ndim_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.ndim_2"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.ndim_3"] = {
    "args": {
        "a": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.ndim_4"] = {
    "args": {
        "a": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.negative_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.negative_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.negative_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.nextafter_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.nextafter_2"] = {
    "args": {
        "x": "float",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.nextafter_3"] = {
    "args": {
        "x": "integer",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.nonzero_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.nonzero_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tuple"
    },
    "inner": {}
}
signatures["jax.numpy.not_equal_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.not_equal_2"] = {
    "args": {
        "x": "tensor",
        "y": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.not_equal_3"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.not_equal_4"] = {
    "args": {
        "x": "float",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.not_equal_5"] = {
    "args": {
        "x": "integer",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.not_equal_6"] = {
    "args": {
        "x": "tensor",
        "y": "boolean"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.not_equal_7"] = {
    "args": {
        "x": "boolean",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.ones_1"] = {
    "args": {
        "shape": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.ones_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.ones_3"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.ones_like_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.ones_like_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.ones_like_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "list"
    },
    "inner": {}
}
signatures["jax.numpy.outer"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.packbits"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "bitorder": "string"
    },
    "inner": {}
}
signatures["jax.numpy.pad_1"] = {
    "args": {
        "array": "tensor",
        "pad_width": "integer"
    },
    "kwargs": {
        "mode": "string",
        "constant_values": "float",
        "stat_length": "integer",
        "end_values": "float",
        "reflect_type": "string"
    },
    "inner": {}
}

signatures["jax.numpy.pad_2"] = {
    "args": {
        "array": "tensor",
        "pad_width": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "constant_values": "float",
        "stat_length": "tuple",
        "end_values": "float",
        "reflect_type": "string"
    },
    "inner": {}
}

signatures["jax.numpy.pad_3"] = {
    "args": {
        "array": "tensor",
        "pad_width": "list"
    },
    "kwargs": {
        "mode": "string",
        "constant_values": "float",
        "stat_length": "integer",
        "end_values": "float",
        "reflect_type": "string"
    },
    "inner": {}
}
signatures["jax.numpy.partition"] = {
    "args": {
        "a": "tensor",
        "kth": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.percentile_1"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.percentile_2"] = {
    "args": {
        "a": "tensor",
        "q": "float"
    },
    "kwargs": {
        "axis": "integer",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.percentile_3"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.percentile_4"] = {
    "args": {
        "a": "tensor",
        "q": "float"
    },
    "kwargs": {
        "axis": "tuple",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.permute_dims"] = {
    "args": {
        "a": "tensor",
        "axes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.piecewise_1"] = {
    "args": {
        "x": "tensor",
        "condlist": "tensor_list",
        "funclist": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.piecewise_2"] = {
    "args": {
        "x": "tensor",
        "condlist": "tensor",
        "funclist": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.place_1"] = {
    "args": {
        "arr": "tensor",
        "mask": "tensor",
        "vals": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.place_2"] = {
    "args": {
        "arr": "tensor",
        "mask": "tensor",
        "vals": "integer"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.place_3"] = {
    "args": {
        "arr": "tensor",
        "mask": "tensor",
        "vals": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.place_4"] = {
    "args": {
        "arr": "tensor",
        "mask": "tensor",
        "vals": "list"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.place_5"] = {
    "args": {
        "arr": "tensor",
        "mask": "tensor",
        "vals": "tuple"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.poly_1"] = {
    "args": {
        "seq_of_zeros": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.poly_2"] = {
    "args": {
        "seq_of_zeros": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.poly_3"] = {
    "args": {
        "seq_of_zeros": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.polyadd"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.polyder"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "m": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.polydiv"] = {
    "args": {
        "u": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "trim_leading_zeros": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.polyfit_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor",
        "deg": "integer"
    },
    "kwargs": {
        "rcond": "float",
        "full": "boolean",
        "w": "tensor",
        "cov": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.polyfit_2"] = {
    "args": {
        "x": "tensor",
        "y": "tensor",
        "deg": "integer"
    },
    "kwargs": {
        "rcond": "float",
        "full": "boolean",
        "w": "tensor",
        "cov": "string"
    },
    "inner": {}
}
signatures["jax.numpy.polyint_1"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "m": "integer",
        "k": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.polyint_2"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "m": "integer",
        "k": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.polyint_3"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "m": "integer",
        "k": "list"
    },
    "inner": {}
}

signatures["jax.numpy.polyint_4"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "m": "integer",
        "k": "tuple"
    },
    "inner": {}
}
signatures["jax.numpy.polymul"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {
        "trim_leading_zeros": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.polysub"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.polyval_1"] = {
    "args": {
        "p": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "unroll": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.polyval_2"] = {
    "args": {
        "p": "tensor",
        "x": "float"
    },
    "kwargs": {
        "unroll": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.polyval_3"] = {
    "args": {
        "p": "tensor",
        "x": "integer"
    },
    "kwargs": {
        "unroll": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.positive_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.positive_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.positive_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.pow"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.power_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.power_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.power_3"] = {
    "args": {
        "x1": "tensor",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.power_4"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.power_5"] = {
    "args": {
        "x1": "float",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.power_6"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.power_7"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.prod_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",  # Can also be integer or float; represented here as tensor (ArrayLike)
        "where": "tensor",
        "promote_integers": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.prod_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor",
        "promote_integers": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.prod_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor",
        "promote_integers": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.promote_types_1"] = {
    "args": {
        "a": "dtype",
        "b": "dtype"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.promote_types_2"] = {
    "args": {
        "a": "string",
        "b": "string"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.promote_types_3"] = {
    "args": {
        "a": "dtype",
        "b": "string"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.promote_types_4"] = {
    "args": {
        "a": "string",
        "b": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.ptp_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.ptp_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.ptp_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.put_1"] = {
    "args": {
        "a": "tensor",
        "ind": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.put_2"] = {
    "args": {
        "a": "tensor",
        "ind": "list",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.put_3"] = {
    "args": {
        "a": "tensor",
        "ind": "tuple",
        "v": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.put_4"] = {
    "args": {
        "a": "tensor",
        "ind": "tensor",
        "v": "list"
    },
    "kwargs": {
        "mode": "string",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.put_5"] = {
    "args": {
        "a": "tensor",
        "ind": "tensor",
        "v": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.put_along_axis_1"] = {
    "args": {
        "arr": "tensor",
        "indices": "tensor",
        "values": "tensor",
        "axis": "integer"
    },
    "kwargs": {
        "inplace": "boolean",
        "mode": "string"
    },
    "inner": {}
}

signatures["jax.numpy.put_along_axis_2"] = {
    "args": {
        "arr": "tensor",
        "indices": "tensor",
        "values": "float",
        "axis": "integer"
    },
    "kwargs": {
        "inplace": "boolean",
        "mode": "string"
    },
    "inner": {}
}

signatures["jax.numpy.put_along_axis_3"] = {
    "args": {
        "arr": "tensor",
        "indices": "tensor",
        "values": "integer",
        "axis": "integer"
    },
    "kwargs": {
        "inplace": "boolean",
        "mode": "string"
    },
    "inner": {}
}
signatures["jax.numpy.quantile_1"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.quantile_2"] = {
    "args": {
        "a": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.quantile_3"] = {
    "args": {
        "a": "tensor",
        "q": "float"
    },
    "kwargs": {
        "axis": "integer",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.quantile_4"] = {
    "args": {
        "a": "tensor",
        "q": "float"
    },
    "kwargs": {
        "axis": "tuple",
        "overwrite_input": "boolean",
        "method": "string",
        "keepdims": "boolean",
        "weights": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.rad2deg_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.rad2deg_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.rad2deg_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.radians_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.radians_2"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.radians_3"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.radians_4"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.radians_5"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.ravel"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "order": "string"
    },
    "inner": {}
}
signatures["jax.numpy.ravel_multi_index_1"] = {
    "args": {
        "multi_index": "tensor_list",
        "dims": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "order": "string",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.ravel_multi_index_2"] = {
    "args": {
        "multi_index": "tensor_list",
        "dims": "list"
    },
    "kwargs": {
        "mode": "string",
        "order": "string",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.real_1"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.real_2"] = {
    "args": {
        "val": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.real_3"] = {
    "args": {
        "val": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.reciprocal_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.reciprocal_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.reciprocal_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.remainder_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.remainder_2"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.remainder_3"] = {
    "args": {
        "x1": "float",
        "x2": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.repeat_1"] = {
    "args": {
        "a": "tensor",
        "repeats": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "total_repeat_length": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.repeat_2"] = {
    "args": {
        "a": "tensor",
        "repeats": "integer"
    },
    "kwargs": {
        "axis": "integer",
        "total_repeat_length": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.repeat_3"] = {
    "args": {
        "a": "tensor",
        "repeats": "list"
    },
    "kwargs": {
        "axis": "integer",
        "total_repeat_length": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.reshape_1"] = {
    "args": {
        "a": "tensor",
        "shape": "integer"
    },
    "kwargs": {
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.reshape_2"] = {
    "args": {
        "a": "tensor",
        "shape": "tuple"
    },
    "kwargs": {
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.reshape_3"] = {
    "args": {
        "a": "tensor",
        "shape": "list"
    },
    "kwargs": {
        "order": "string",
        "copy": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.resize_1"] = {
    "args": {
        "a": "tensor",
        "new_shape": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.resize_2"] = {
    "args": {
        "a": "tensor",
        "new_shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.result_type_1"] = {
    "args": {
        "args": "dtype"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.result_type_2"] = {
    "args": {
        "args": "string"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.result_type_3"] = {
    "args": {
        "args": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.result_type_4"] = {
    "args": {
        "args": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.result_type_5"] = {
    "args": {
        "args": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.result_type_6"] = {
    "args": {
        "args": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.right_shift_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.right_shift_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.right_shift_3"] = {
    "args": {
        "x1": "integer",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.right_shift_4"] = {
    "args": {
        "x1": "integer",
        "x2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.rint"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.roll_1"] = {
    "args": {
        "a": "tensor",
        "shift": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.roll_2"] = {
    "args": {
        "a": "tensor",
        "shift": "list"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}

signatures["jax.numpy.roll_3"] = {
    "args": {
        "a": "tensor",
        "shift": "tuple"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.roll_4"] = {
    "args": {
        "a": "tensor",
        "shift": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.rollaxis"] = {
    "args": {
        "a": "tensor",
        "axis": "integer"
    },
    "kwargs": {
        "start": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.roots"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "strip_zeros": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.rot90"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "axes": "tuple"
    },
    "inner": {}
}
signatures["jax.numpy.round_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "decimals": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.round_2"] = {
    "args": {
        "a": "float"
    },
    "kwargs": {
        "decimals": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.round_3"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {
        "decimals": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.searchsorted"] = {
    "args": {
        "a": "tensor",
        "v": "tensor"
    },
    "kwargs": {
        "side": "string",
        "sorter": "tensor",
        "method": "string"
    },
    "inner": {}
}
signatures["jax.numpy.select_1"] = {
    "args": {
        "condlist": "tensor_list",
        "choicelist": "tensor_list"
    },
    "kwargs": {
        "default": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.select_2"] = {
    "args": {
        "condlist": "tensor_list",
        "choicelist": "tensor_list"
    },
    "kwargs": {
        "default": "float"
    },
    "inner": {}
}

signatures["jax.numpy.select_3"] = {
    "args": {
        "condlist": "tensor_list",
        "choicelist": "tensor_list"
    },
    "kwargs": {
        "default": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.select_4"] = {
    "args": {
        "condlist": "tensor_list",
        "choicelist": "tensor_list"
    },
    "kwargs": {
        "default": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.setdiff1d_1"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor",
        "assume_unique": "boolean"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.setdiff1d_2"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor",
        "assume_unique": "boolean"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.setdiff1d_3"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor",
        "assume_unique": "boolean"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.setxor1d"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "size": "integer",
        "fill_value": "tensor"  # can also be float or integer
    },
    "inner": {}
}
signatures["jax.numpy.shape_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.shape_2"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.shape_3"] = {
    "args": {
        "a": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.shape_4"] = {
    "args": {
        "a": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.sign_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sign_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sign_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sign_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sign_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.signbit"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.sin_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sin_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sin_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sin_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sin_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.sinc"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.sinh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sinh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sinh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.size_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.size_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.size_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}
signatures["jax.numpy.sort_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.sort_2"] = {
    "args": {
        "a": "list"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.sort_3"] = {
    "args": {
        "a": "tuple"
    },
    "kwargs": {
        "axis": "integer",
        "stable": "boolean",
        "descending": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.sort_complex"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.spacing"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.split_1"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.split_2"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.split_3"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.split_4"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.sqrt_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sqrt_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.sqrt_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.square_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.square_2"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.square_3"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.square_4"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.square_5"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.squeeze_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.squeeze_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.squeeze_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}
signatures["jax.numpy.stack_1"] = {
    "args": {
        "arrays": "tensor_list"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.stack_2"] = {
    "args": {
        "arrays": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.std_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor",
        "correction": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.std_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor",
        "correction": "float"
    },
    "inner": {}
}

signatures["jax.numpy.std_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean",
        "where": "tensor",
        "mean": "tensor",
        "correction": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.subtract"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.sum_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor",
        "promote_integers": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.sum_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor",
        "promote_integers": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.sum_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "dtype": "dtype",
        "keepdims": "boolean",
        "initial": "tensor",
        "where": "tensor",
        "promote_integers": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.swapaxes"] = {
    "args": {
        "a": "tensor",
        "axis1": "integer",
        "axis2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.take_1"] = {
    "args": {
        "a": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "mode": "string",
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.take_2"] = {
    "args": {
        "a": "tensor",
        "indices": "integer"
    },
    "kwargs": {
        "axis": "integer",
        "mode": "string",
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.take_3"] = {
    "args": {
        "a": "tensor",
        "indices": "list"
    },
    "kwargs": {
        "axis": "integer",
        "mode": "string",
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "fill_value": "boolean"
    },
    "inner": {}
}

signatures["jax.numpy.take_4"] = {
    "args": {
        "a": "tensor",
        "indices": "tuple"
    },
    "kwargs": {
        "axis": "integer",
        "mode": "string",
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "fill_value": "float"
    },
    "inner": {}
}
signatures["jax.numpy.take_along_axis_1"] = {
    "args": {
        "arr": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "mode": "string",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.take_along_axis_2"] = {
    "args": {
        "arr": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "mode": "string",
        "fill_value": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.tan_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.tan_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.tan_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.tanh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.tanh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.tanh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.tensordot_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "axes": "integer",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.tensordot_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "axes": "list",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.tensordot_3"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "axes": "tuple",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.tile_1"] = {
    "args": {
        "A": "tensor",
        "reps": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.tile_2"] = {
    "args": {
        "A": "tensor",
        "reps": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.tile_3"] = {
    "args": {
        "A": "tensor",
        "reps": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.trace_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "axis1": "integer",
        "axis2": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.trace_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "offset": "tensor",
        "axis1": "integer",
        "axis2": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.transpose_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axes": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.transpose_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axes": "list"
    },
    "inner": {}
}
signatures["jax.numpy.trapezoid_1"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "dx": "float",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.trapezoid_2"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "dx": "integer",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.trapezoid_3"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "dx": "tensor",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.trapezoid_4"] = {
    "args": {
        "y": "tensor"
    },
    "kwargs": {
        "x": "tensor",
        "dx": "float",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.trapezoid_5"] = {
    "args": {
        "y": "tensor"
    },
    "kwargs": {
        "x": "tensor",
        "dx": "integer",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.trapezoid_6"] = {
    "args": {
        "y": "tensor"
    },
    "kwargs": {
        "x": "tensor",
        "dx": "tensor",
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.tri"] = {
    "args": {
        "N": "integer"
    },
    "kwargs": {
        "M": "integer",
        "k": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.tril"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.tril_indices"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "k": "integer",
        "m": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.tril_indices_from"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.trim_zeros_1"] = {
    "args": {
        "filt": "tensor"
    },
    "kwargs": {
        "trim": "string",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.trim_zeros_2"] = {
    "args": {
        "filt": "tensor"
    },
    "kwargs": {
        "trim": "string",
        "axis": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.trim_zeros_3"] = {
    "args": {
        "filt": "tensor"
    },
    "kwargs": {
        "trim": "string",
        "axis": "list"
    },
    "inner": {}
}
signatures["jax.numpy.triu"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.triu_indices"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "k": "integer",
        "m": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.triu_indices_from"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.true_divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.trunc_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.trunc_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.trunc_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.union1d_1"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.union1d_2"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.union1d_3"] = {
    "args": {
        "ar1": "tensor",
        "ar2": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}
signatures["jax.numpy.unique"] = {
    "args": {
        "ar": "tensor"
    },
    "kwargs": {
        "return_index": "boolean",
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "axis": "integer",
        "equal_nan": "boolean",
        "size": "integer",
        "fill_value": "tensor",  # can also be float or integer
        "sorted": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.unique_all_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.unique_all_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.unique_all_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}
signatures["jax.numpy.unique_counts_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.unique_counts_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.unique_counts_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.unique_counts_4"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.unique_inverse_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.unique_inverse_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.unique_inverse_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.numpy.unique_inverse_4"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.unique_values_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "tensor"
    },
    "inner": {}
}

signatures["jax.numpy.unique_values_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.unique_values_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "float"
    },
    "inner": {}
}
signatures["jax.numpy.unpackbits"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "count": "integer",
        "bitorder": "string"
    },
    "inner": {}
}
signatures["jax.numpy.unravel_index_1"] = {
    "args": {
        "indices": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.unravel_index_2"] = {
    "args": {
        "indices": "tensor",
        "shape": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.unravel_index_3"] = {
    "args": {
        "indices": "integer",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.unravel_index_4"] = {
    "args": {
        "indices": "integer",
        "shape": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.unstack"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.unwrap_1"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "discont": "float",
        "axis": "integer",
        "period": "float"
    },
    "inner": {}
}

signatures["jax.numpy.unwrap_2"] = {
    "args": {
        "p": "tensor"
    },
    "kwargs": {
        "discont": "tensor",
        "axis": "integer",
        "period": "tensor"
    },
    "inner": {}
}
signatures["jax.numpy.vander"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "N": "integer",
        "increasing": "boolean"
    },
    "inner": {}
}
signatures["jax.numpy.var_1"] = {
    "args": {
        "a": "tensor",
        "axis": "integer",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean"
    },
    "kwargs": {
        "where": "tensor",
        "mean": "tensor",
        "correction": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.var_2"] = {
    "args": {
        "a": "tensor",
        "axis": "tuple",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean"
    },
    "kwargs": {
        "where": "tensor",
        "mean": "tensor",
        "correction": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.var_3"] = {
    "args": {
        "a": "tensor",
        "axis": "integer",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean"
    },
    "kwargs": {
        "where": "tensor",
        "mean": "tensor",
        "correction": "float"
    },
    "inner": {}
}

signatures["jax.numpy.var_4"] = {
    "args": {
        "a": "tensor",
        "axis": "tuple",
        "dtype": "dtype",
        "ddof": "integer",
        "keepdims": "boolean"
    },
    "kwargs": {
        "where": "tensor",
        "mean": "tensor",
        "correction": "float"
    },
    "inner": {}
}
signatures["jax.numpy.vdot_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.vdot_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.vdot_3"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "string"
    },
    "inner": {}
}

signatures["jax.numpy.vdot_4"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "string"
    },
    "inner": {}
}
signatures["jax.numpy.vecdot_1"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "precision": "string",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.vecdot_2"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.vecmat"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.vectorize_1"] = {
    "args": {
        "pyfunc": "tuple"  # 'callable' is not in allowed types
    },
    "kwargs": {
        "excluded": "tuple",
        "signature": "string"
    },
    "inner": {
        "args": {
            "args": "tensor"
        },
        "kwargs": {}
    }
}

signatures["jax.numpy.vectorize_2"] = {
    "args": {
        "pyfunc": "tuple"  # 'callable' is not in allowed types
    },
    "kwargs": {
        "excluded": "list",
        "signature": "string"
    },
    "inner": {
        "args": {
            "args": "tensor"
        },
        "kwargs": {}
    }
}
signatures["jax.numpy.vsplit_1"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.vsplit_2"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.vsplit_3"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.vsplit_4"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.numpy.vstack_1"] = {
    "args": {
        "tup": "tensor_list"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}

signatures["jax.numpy.vstack_2"] = {
    "args": {
        "tup": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["jax.numpy.where_1"] = {
    "args": {
        "condition": "tensor",
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.numpy.where_2"] = {
    "args": {
        "condition": "tensor"
    },
    "kwargs": {
        "size": "integer",
        "fill_value": "integer"
    },
    "inner": {}
}
signatures["jax.numpy.zeros_1"] = {
    "args": {
        "shape": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out_sharding": "tuple"  # Should be jax.sharding.Sharding or PartitionSpec
    },
    "inner": {}
}

signatures["jax.numpy.zeros_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "out_sharding": "tuple"  # Should be jax.sharding.Sharding or PartitionSpec
    },
    "inner": {}
}

signatures["jax.numpy.zeros_3"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "out_sharding": "tuple"  # Should be jax.sharding.Sharding or PartitionSpec
    },
    "inner": {}
}
signatures["jax.numpy.zeros_like_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.zeros_like_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.zeros_like_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "list"
    },
    "inner": {}
}

signatures["jax.numpy.zeros_like_4"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "string",
        "shape": "tuple"
    },
    "inner": {}
}

signatures["jax.numpy.zeros_like_5"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "string",
        "shape": "integer"
    },
    "inner": {}
}

signatures["jax.numpy.zeros_like_6"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "string",
        "shape": "list"
    },
    "inner": {}
}
signatures["jax.nn.celu_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "alpha": "tensor"
    },
    "inner": {}
}

signatures["jax.nn.celu_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "alpha": "float"
    },
    "inner": {}
}
signatures["jax.nn.dot_product_attention_1"] = {
    "args": {
        "query": "tensor",
        "key": "tensor",
        "value": "tensor",
        "bias": "tensor",
        "mask": "tensor"
    },
    "kwargs": {
        "scale": "float",
        "is_causal": "boolean",
        "query_seq_lengths": "tensor",
        "key_value_seq_lengths": "tensor",
        "local_window_size": "integer",
        "implementation": "string",
        "return_residual": "boolean"
    },
    "inner": {}
}

signatures["jax.nn.dot_product_attention_2"] = {
    "args": {
        "query": "tensor",
        "key": "tensor",
        "value": "tensor",
        "bias": "tensor",
        "mask": "tensor"
    },
    "kwargs": {
        "scale": "float",
        "is_causal": "boolean",
        "query_seq_lengths": "tensor",
        "key_value_seq_lengths": "tensor",
        "local_window_size": "tuple",
        "implementation": "string",
        "return_residual": "boolean"
    },
    "inner": {}
}
signatures["jax.nn.elu_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "alpha": "float"
    },
    "inner": {}
}

signatures["jax.nn.elu_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "alpha": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.gelu"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "approximate": "boolean"
    },
    "inner": {}
}
signatures["jax.nn.get_scaled_dot_general_config"] = {
    "args": {
        "mode": "string"
    },
    "kwargs": {
        "global_scale": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.glu"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.nn.hard_sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.hard_silu"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.hard_swish"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.hard_tanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.identity"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.leaky_relu_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "negative_slope": "float"
    },
    "inner": {}
}

signatures["jax.nn.leaky_relu_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "negative_slope": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.log1mexp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.log_sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.log_softmax_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.nn.log_softmax_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.logmeanexp_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "where": "tensor",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.nn.logmeanexp_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "where": "tensor",
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["jax.nn.logmeanexp_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "where": "tensor",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["jax.nn.logsumexp_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "b": "tensor",
        "keepdims": "boolean",
        "return_sign": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.nn.logsumexp_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "b": "tensor",
        "keepdims": "boolean",
        "return_sign": "boolean",
        "where": "tensor"
    },
    "inner": {}
}

signatures["jax.nn.logsumexp_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "b": "tensor",
        "keepdims": "boolean",
        "return_sign": "boolean",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.mish"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.one_hot_1"] = {
    "args": {
        "x": "tensor",
        "num_classes": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}

signatures["jax.nn.one_hot_2"] = {
    "args": {
        "x": "tensor",
        "num_classes": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "axis": "string"  # To represent AxisName
    },
    "inner": {}
}
signatures["jax.nn.relu"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.relu6"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.scaled_dot_general"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "dimension_numbers": "tuple"
    },
    "kwargs": {
        "preferred_element_type": "dtype",
        "configs": "list",
        "implementation": "string"
    },
    "inner": {}
}
signatures["jax.nn.scaled_matmul"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "lhs_scales": "tensor",
        "rhs_scales": "tensor"
    },
    "kwargs": {
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.nn.selu"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.silu"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.soft_sign"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.softmax_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "where": "tensor" # can also be None, but represented as tensor here
    },
    "inner": {}
}

signatures["jax.nn.softmax_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "where": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.softplus"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.sparse_plus"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.sparse_sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.squareplus_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "b": "integer"
    },
    "inner": {}
}

signatures["jax.nn.squareplus_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "b": "float"
    },
    "inner": {}
}

signatures["jax.nn.squareplus_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "b": "tensor"
    },
    "inner": {}
}
signatures["jax.nn.standardize_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "mean": "tensor",
        "variance": "tensor",
        "epsilon": "float",
        "where": "tensor",
        "algorithm": "string"
    },
    "inner": {}
}

signatures["jax.nn.standardize_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "mean": "tensor",
        "variance": "tensor",
        "epsilon": "float",
        "where": "tensor",
        "algorithm": "string"
    },
    "inner": {}
}
signatures["jax.nn.swish"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.nn.tanh_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.nn.tanh_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.nn.tanh_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.conv_1"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "window_strides": "list",
        "padding": "string"
    },
    "kwargs": {
        "precision": "string",  # Can be Precision enum (mapped to string) or tuple of enums
        "preferred_element_type": "dtype"
    },
    "inner": {}
}

signatures["jax.lax.conv_2"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "window_strides": "tuple",
        "padding": "string"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "dtype"
    },
    "inner": {}
}
signatures["jax.lax.conv_general_dilated_1"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "window_strides": "list",
        "padding": "string"
    },
    "kwargs": {
        "lhs_dilation": "list",
        "rhs_dilation": "list",
        "dimension_numbers": "tuple",
        "feature_group_count": "integer",
        "batch_group_count": "integer",
        "precision": "string",
        "preferred_element_type": "dtype",
        "out_sharding": "string"
    },
    "inner": {}
}

signatures["jax.lax.conv_general_dilated_2"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "window_strides": "list",
        "padding": "list"
    },
    "kwargs": {
        "lhs_dilation": "list",
        "rhs_dilation": "list",
        "dimension_numbers": "tuple",
        "feature_group_count": "integer",
        "batch_group_count": "integer",
        "precision": "string",
        "preferred_element_type": "dtype",
        "out_sharding": "string"
    },
    "inner": {}
}
signatures["jax.lax.conv_transpose_1"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "strides": "tuple",
        "padding": "string"
    },
    "kwargs": {
        "rhs_dilation": "tuple",
        "dimension_numbers": "tuple",
        "transpose_kernel": "boolean",
        "precision": "string",
        "preferred_element_type": "dtype",
        "use_consistent_padding": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.conv_transpose_2"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "strides": "list",
        "padding": "list"
    },
    "kwargs": {
        "rhs_dilation": "list",
        "dimension_numbers": "tuple",
        "transpose_kernel": "boolean",
        "precision": "string",
        "preferred_element_type": "dtype",
        "use_consistent_padding": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.conv_transpose_3"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "strides": "tuple",
        "padding": "tuple"
    },
    "kwargs": {
        "rhs_dilation": "tuple",
        "dimension_numbers": "tuple",
        "transpose_kernel": "boolean",
        "precision": "string",
        "preferred_element_type": "dtype",
        "use_consistent_padding": "boolean"
    },
    "inner": {}
}
signatures["jax.lax.dot_1"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "dimension_numbers": "tuple",
        "precision": "string",  # Precision can be a string (e.g., 'default', 'high', 'highest')
        "preferred_element_type": "dtype",
        "out_sharding": "tuple"  # Sharding specification mapped to tuple
    },
    "inner": {}
}

signatures["jax.lax.dot_2"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "dimension_numbers": "tuple",
        "precision": "tuple",  # Precision can also be a tuple of two Precision enums
        "preferred_element_type": "dtype",
        "out_sharding": "tuple"  # Sharding specification mapped to tuple
    },
    "inner": {}
}
signatures["jax.lax.dot_general_1"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "dimension_numbers": "tuple"  # Expected: ((lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch))
    },
    "kwargs": {
        "precision": "string",
        "preferred_element_type": "dtype",
        "out_sharding": "tuple"  # Expected: Sharding object or None, mapped to tuple as closest match
    },
    "inner": {}
}

signatures["jax.lax.dot_general_2"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor",
        "dimension_numbers": "tuple"
    },
    "kwargs": {
        "precision": "tuple",
        "preferred_element_type": "dtype",
        "out_sharding": "tuple"
    },
    "inner": {}
}
signatures["jax.lax.batch_matmul_1"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "precision": "string"
    },
    "inner": {}
}

signatures["jax.lax.batch_matmul_2"] = {
    "args": {
        "lhs": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "precision": "tuple"
    },
    "inner": {}
}
signatures["jax.lax.reduce_sum_1"] = {
    "args": {
        "operand": "tensor",
        "axes": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reduce_sum_2"] = {
    "args": {
        "operand": "tensor",
        "axes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.reduce_max_1"] = {
    "args": {
        "operand": "tensor",
        "axes": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reduce_max_2"] = {
    "args": {
        "operand": "tensor",
        "axes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.reduce_min_1"] = {
    "args": {
        "operand": "tensor",
        "axes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reduce_min_2"] = {
    "args": {
        "operand": "tensor",
        "axes": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.reduce_prod_1"] = {
    "args": {
        "operand": "tensor",
        "axes": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reduce_prod_2"] = {
    "args": {
        "operand": "tensor",
        "axes": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.gather_1"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "dimension_numbers": "tuple",  # GatherDimensionNumbers is a namedtuple
        "slice_sizes": "tuple"
    },
    "kwargs": {
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "mode": "string",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.lax.gather_2"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "dimension_numbers": "tuple",  # GatherDimensionNumbers is a namedtuple
        "slice_sizes": "tuple"
    },
    "kwargs": {
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "mode": "string",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.lax.gather_3"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "dimension_numbers": "tuple",  # GatherDimensionNumbers is a namedtuple
        "slice_sizes": "tuple"
    },
    "kwargs": {
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "mode": "string",
        "fill_value": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.gather_4"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "dimension_numbers": "tuple",  # GatherDimensionNumbers is a namedtuple
        "slice_sizes": "list"
    },
    "kwargs": {
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "mode": "string",
        "fill_value": "float"
    },
    "inner": {}
}

signatures["jax.lax.gather_5"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "dimension_numbers": "tuple",  # GatherDimensionNumbers is a namedtuple
        "slice_sizes": "list"
    },
    "kwargs": {
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "mode": "string",
        "fill_value": "integer"
    },
    "inner": {}
}

signatures["jax.lax.gather_6"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "dimension_numbers": "tuple",  # GatherDimensionNumbers is a namedtuple
        "slice_sizes": "list"
    },
    "kwargs": {
        "unique_indices": "boolean",
        "indices_are_sorted": "boolean",
        "mode": "string",
        "fill_value": "boolean"
    },
    "inner": {}
}
signatures["jax.lax.scatter"] = {
    "args": {
        "operand": "tensor",
        "scatter_indices": "tensor",
        "updates": "tensor",
        "dimension_numbers": "tuple"  # Actually a lax.ScatterDimensionNumbers object, which resembles a tuple/struct of dimension mappings
    },
    "kwargs": {
        "indices_are_sorted": "boolean",
        "unique_indices": "boolean",
        "mode": "string"  # Can be a string or GatherScatterMode enum
    },
    "inner": {}
}
signatures["jax.lax.scatter_add"] = {
    "args": {
        "operand": "tensor",
        "scatter_indices": "tensor",
        "updates": "tensor",
        "dimension_numbers": "tuple"  # closest type for ScatterDimensionNumbers
    },
    "kwargs": {
        "indices_are_sorted": "boolean",
        "unique_indices": "boolean",
        "mode": "string"  # can be string or GatherScatterMode enum
    },
    "inner": {}
}
signatures["jax.lax.scatter_mul"] = {
    "args": {
        "operand": "tensor",
        "scatter_indices": "tensor",
        "updates": "tensor",
        "dimension_numbers": "tuple" # ScatterDimensionNumbers is a structured tuple-like object
    },
    "kwargs": {
        "indices_are_sorted": "boolean",
        "unique_indices": "boolean",
        "mode": "string"
    },
    "inner": {}
}
signatures["jax.lax.dynamic_slice_1"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tuple",
        "slice_sizes": "tuple"
    },
    "kwargs": {
        "allow_negative_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_slice_2"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "list",
        "slice_sizes": "list"
    },
    "kwargs": {
        "allow_negative_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_slice_3"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tensor",
        "slice_sizes": "tuple"
    },
    "kwargs": {
        "allow_negative_indices": "boolean"
    },
    "inner": {}
}
signatures["jax.lax.dynamic_update_slice_1"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "list"
    },
    "kwargs": {
        "allow_negative_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_2"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "tuple"
    },
    "kwargs": {
        "allow_negative_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_3"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "tensor"
    },
    "kwargs": {
        "allow_negative_indices": "boolean"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_4"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "list"
    },
    "kwargs": {
        "allow_negative_indices": "list"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_5"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "tuple"
    },
    "kwargs": {
        "allow_negative_indices": "list"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_6"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "tensor"
    },
    "kwargs": {
        "allow_negative_indices": "list"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_7"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "list"
    },
    "kwargs": {
        "allow_negative_indices": "tuple"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_8"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "tuple"
    },
    "kwargs": {
        "allow_negative_indices": "tuple"
    },
    "inner": {}
}

signatures["jax.lax.dynamic_update_slice_9"] = {
    "args": {
        "operand": "tensor",
        "update": "tensor",
        "start_indices": "tensor"
    },
    "kwargs": {
        "allow_negative_indices": "tuple"
    },
    "inner": {}
}
signatures["jax.lax.pad_1"] = {
    "args": {
        "operand": "tensor",
        "padding_value": "tensor",
        "padding_config": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.pad_2"] = {
    "args": {
        "operand": "tensor",
        "padding_value": "tensor",
        "padding_config": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.broadcast_in_dim_1"] = {
    "args": {
        "operand": "tensor",
        "shape": "tuple",
        "broadcast_dimensions": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.broadcast_in_dim_2"] = {
    "args": {
        "operand": "tensor",
        "shape": "tuple",
        "broadcast_dimensions": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.broadcast_in_dim_3"] = {
    "args": {
        "operand": "tensor",
        "shape": "list",
        "broadcast_dimensions": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.broadcast_in_dim_4"] = {
    "args": {
        "operand": "tensor",
        "shape": "list",
        "broadcast_dimensions": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.reshape_1"] = {
    "args": {
        "operand": "tensor",
        "new_sizes": "tuple",
        "dimensions": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reshape_2"] = {
    "args": {
        "operand": "tensor",
        "new_sizes": "tuple",
        "dimensions": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reshape_3"] = {
    "args": {
        "operand": "tensor",
        "new_sizes": "list",
        "dimensions": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.reshape_4"] = {
    "args": {
        "operand": "tensor",
        "new_sizes": "list",
        "dimensions": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.transpose_1"] = {
    "args": {
        "operand": "tensor",
        "permutation": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.transpose_2"] = {
    "args": {
        "operand": "tensor",
        "permutation": "tuple"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.transpose_3"] = {
    "args": {
        "operand": "tensor",
        "permutation": "tensor"  # np.ndarray mapped to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.concatenate_1"] = {
    "args": {
        "operands": "tensor_list",
        "dimension": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.concatenate_2"] = {
    "args": {
        "operands": "tensor",
        "dimension": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.slice_1"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "list",
        "limit_indices": "list",
    },
    "kwargs": {
        "strides": "list"
    },
    "inner": {}
}

signatures["jax.lax.slice_2"] = {
    "args": {
        "operand": "tensor",
        "start_indices": "tuple",
        "limit_indices": "tuple",
    },
    "kwargs": {
        "strides": "tuple"
    },
    "inner": {}
}
signatures["jax.lax.select"] = {
    "args": {
        "pred": "tensor",
        "on_true": "tensor",
        "on_false": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.top_k"] = {
    "args": {
        "operand": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["jax.lax.sort_1"] = {
    "args": {
        "operand": "tensor"
    },
    "kwargs": {
        "dimension": "integer",
        "is_stable": "boolean",
        "num_keys": "integer"
    },
    "inner": {}
}

signatures["jax.lax.sort_2"] = {
    "args": {
        "operand": "tensor_list"
    },
    "kwargs": {
        "dimension": "integer",
        "is_stable": "boolean",
        "num_keys": "integer"
    },
    "inner": {}
}
signatures["jax.lax.sort_key_val"] = {
    "args": {
        "keys": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "dimension": "integer",
        "is_stable": "boolean"
    },
    "inner": {}
}
signatures["jax.lax.convert_element_type_1"] = {
    "args": {
        "operand": "tensor",
        "new_dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.convert_element_type_2"] = {
    "args": {
        "operand": "integer",
        "new_dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.convert_element_type_3"] = {
    "args": {
        "operand": "float",
        "new_dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.convert_element_type_4"] = {
    "args": {
        "operand": "boolean",
        "new_dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.clamp_1"] = {
    "args": {
        "min": "tensor",
        "x": "tensor",
        "max": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.clamp_2"] = {
    "args": {
        "min": "float",
        "x": "tensor",
        "max": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.clamp_3"] = {
    "args": {
        "min": "integer",
        "x": "tensor",
        "max": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.rsqrt_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "string"  # AccuracyMode representing implementation choice
    },
    "inner": {}
}

signatures["jax.lax.rsqrt_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "tuple"  # lax.Tolerance which is a named tuple of floats
    },
    "inner": {}
}
signatures["jax.lax.integer_pow"] = {
    "args": {
        "x": "tensor",
        "y": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.sqrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "string"  # Should be lax.Tolerance or lax.AccuracyMode custom object
    },
    "inner": {}
}
signatures["jax.lax.exp_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "string"  # AccuracyMode representing the implementation mode
    },
    "inner": {}
}

signatures["jax.lax.exp_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "tuple"  # Tolerance object (typically represented as a tuple of floats)
    },
    "inner": {}
}
signatures["jax.lax.log_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "string"  # For lax.AccuracyMode enum
    },
    "inner": {}
}

signatures["jax.lax.log_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "tuple"  # For lax.Tolerance (relative, absolute) tolerance pair
    },
    "inner": {}
}
signatures["jax.lax.tanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "accuracy": "string"  # accuracy can be Tolerance or AccuracyMode. Choosing "string" as a close match for AccuracyMode enums.
    },
    "inner": {}
}
signatures["jax.lax.erf_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.erf_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.erf_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.erf_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.lgamma_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.lgamma_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.lgamma_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.lgamma_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.digamma_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.digamma_2"] = {
    "args": {
        "x": "float"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.digamma_3"] = {
    "args": {
        "x": "integer"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.digamma_4"] = {
    "args": {
        "x": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.igamma"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.fft_1"] = {
    "args": {
        "x": "tensor",
        "fft_type": "string",
        "fft_lengths": "list"
    },
    "kwargs": {},
    "inner": {}
}

signatures["jax.lax.fft_2"] = {
    "args": {
        "x": "tensor",
        "fft_type": "string",
        "fft_lengths": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["jax.lax.cumsum"] = {
    "args": {
        "operand": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "reverse": "boolean"
    },
    "inner": {}
}
signatures["jax.lax.cumprod"] = {
    "args": {
        "operand": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "reverse": "boolean"
    },
    "inner": {}
}
signatures["jax.lax.iota"] = {
    "args": {
        "dtype": "dtype",
        "size": "integer"
    },
    "kwargs": {},
    "inner": {}
}
