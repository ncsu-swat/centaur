signatures = {}
signatures["tf.raw_ops.GreaterEqual"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_saturation"] = {
    "args": {
        "image": "tensor", # RGB image or images
        "saturation_factor": "float"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.categorical_column_with_identity"] = {
    "args": {
        "key": "string",
        "num_buckets": "integer" # could be Int32Tensor which matches integer
    },
    "kwargs": {
        "default_value": "integer" # could be Int32Tensor which matches integer
    },
    "inner": {}
}
signatures["tf.io.serialize_tensor"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.atan2"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Empty"] = {
    "args": {
        "shape": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "init": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseReduceSumSparse"] = {
    "args": {
        "input_indices": "tensor",
        "input_values": "tensor",
        "input_shape": "tensor",
        "reduction_axes": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isfinite"] = {
    "args": {
        "x": "tensor"  # Could also be a tensor_list, but tensor feels more general
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.raw_ops.Relu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sysconfig.get_include"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.get_static_value"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "partial": "boolean"
    },
    "inner": {}
}
signatures["tf.compat.path_to_str"] = {
    "args": {
        "path": "string"  # It can be a PathLike object, can be converted automatically to a string
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = {
    "args": {
        "key": "string",
        "hash_bucket_size": "integer"
    },
    "kwargs": {
        "dtype": "dtype" # Could be a string or integer
    },
    "inner": {}
}
signatures["tf.identity"] = {
    "args": {
        "input": "tensor"  # Could be a Variable or CompositeTensor but behaves like a tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ComputeAccidentalHits"] = {
    "args": {
        "true_classes": "tensor",
        "sampled_candidates": "tensor",
        "num_true": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
