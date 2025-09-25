signatures = {}
signatures["torch.special.xlog1py_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.xlog1py_2"] = {
    "args": {
        "input": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.floor_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.sparse__1"] = {
    "args": {
        "input": "tensor",
        "sparsity": "float",
        "dimension": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nansum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"  # Could be None as well
    },
    "inner": {}
}

signatures["torch.nansum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # Could be a list of ints as well
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"  # Could be None as well
    },
    "inner": {}
}
signatures["torch.sqrt_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could be None
    },
    "inner": {}
}
signatures["torch.nn.MaxPool3d_1"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple"  # Output size documentation gives a tuple
    },
    "inner": {}
}

signatures["torch.nn.MaxPool3d_2"] = {
    "args": {
        "kernel_size": "tuple",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple"  # Output size documentation gives a tuple
    },
    "inner": {}
}

signatures["torch.nn.MaxPool3d_3"] = {
    "args": {
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple"  # Output size documentation gives a tuple
    },
    "inner": {}
}

signatures["torch.nn.MaxPool3d_4"] = {
    "args": {
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple"  # Output size documentation gives a tuple
    },
    "inner": {}
}

signatures["torch.nn.MaxPool3d_5"] = {
    "args": {
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "output_size": "tuple"  # Output size documentation gives a tuple
    },
    "inner": {}
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"  # index should be a LongTensor so can be tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.arccosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
# The documentation says it's an alias for torch.acosh(). No other parameters.
signatures["torch.multiply_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.isposinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.acos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple" # could be a list of integers but tuple seems more appropriate
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.special.ndtri"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.set_autocast_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.logsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.get_rng_state"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
