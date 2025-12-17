### Want functions to do the following:
# Take Data4D, Data3D, etc
# And perform operations (keeping it the same shape) via multiply, divide, add, subtract, etc.

# Should the data objects be in HTBAM_analysis? I feel like maybe...

# def transform_data(data: Data4D, apply_to: str, store_as: str, function: callable, flatten: bool = False, data_type: str = None) -> dict:
#     """
#     Transform a data dictionary by applying a function to the specified dependent variable.

#     TODO: Currently, this uses a single lambda to apply to all values. Would be nice to pass in a matrix or something too, so we can divide all chamber values by standard curve slope, for example.
#     Parameters
#     ----------
#     data : dict
#         Dictionary in "RFU_data" format (see htbam_db_api.py).
#     apply_to : str
#         Key in data["dep_vars"] to apply the function to.
#     store_as : str
#         Key in data["dep_vars"] to store the transformed variable.
#     function : callable
#         Function to apply to the dependent variable.
#     flatten : bool, optional
#         If True, flatten the output array (default False).

#     Returns
#     -------
#     result : dict
#         New data dictionary with transformed dependent variable.
#     """
#     if apply_to not in data.dep_var_type:
#         raise KeyError(f"'{apply_to}' not in data['dep_vars']")

#     transformed_data = deepcopy(data)
#     transformed_data["dep_vars"][store_as] = function(transformed_data["dep_vars"][apply_to])
#     transformed_data.pop(apply_to, None)  # Remove the original variable if needed

#     if flatten:
#         transformed_data["dep_vars"][apply_to] = transformed_data["dep_vars"][apply_to].flatten()

#     if data_type is not None:
#         transformed_data["data_type"] = data_type

#     return transformed_data


import numpy as np
from copy import deepcopy
import pint

def transform_data(
    data_objs: list,        # list of Data2D, Data3D, Data4D instances (all same class and shape)
    expr: str,             # e.g. "(a_luminance - b_luminance)"
    output_name: str,      # name of the new field, e.g. "difference"
    expression_vars: dict = None,  # optional mapping of name -> object to be available in expr
    keep_existing: bool = False,  # if True, keep existing dep_var fields and append/replace the new field
):
    """
    Return a new object of the same class as data_objs[0] in which:
      - The indep_vars are deep-copied from data_objs[0].
      - We evaluate `expr` elementwise, treating each field in each object's
        dep_var_type as a NumPy array slice.  Each object is assigned a prefix
        ("a_", "b_", "c_", ...) in the namespace.  For example, if the first
        object has dep_var_type=["luminance",...], then `a_luminance` maps to
        data_objs[0].dep_var[..., idx], etc.
      - The result (shape = dep_var.shape[:-1]) is expanded along a new final axis
        of size 1 and stored in the new .dep_var.  The new .dep_var_type = [output_name].

      Example usage:
        result_obj = transform_data(
            data_objs = [objA, objB],
            expr      = "(a_luminance - b_luminance)",
            output_name = "difference"
        )
    If `keep_existing` is True, the returned object will retain all original
    dependent variables from `data_objs[0]` and append (or replace) the
    transformed variable named `output_name`. If False (default), the returned
    object contains only the transformed variable.
    """

    # 1) Check that at least one object was passed
    if not data_objs:
        raise ValueError("Must pass a non-empty list of data objects.")

    # 2) Ensure all objects are same class and same dep_var shape
    base_class = type(data_objs[0])
    base_shape = data_objs[0].dep_var.shape
    # TODO: I'm skipping this check. Will probably be useful to have some double-checking.
    print(f"Transforming {len(data_objs)} objects of type {[type(o) for o in data_objs]} with shapes {[o.dep_var.shape for o in data_objs]}.")
    # for idx, obj in enumerate(data_objs):
    #     if type(obj) is not base_class:
    #         raise TypeError(
    #             f"All data objects must be the same class. "
    #             f"Object 0 is {base_class.__name__!r}, but object {idx} is {type(obj).__name__!r}."
    #         )
    #     if obj.dep_var.shape != base_shape:
    #         raise ValueError(
    #             f"All data objects must have the same dep_var.shape. "
    #             f"Object 0 shape = {base_shape}, but object {idx} shape = {obj.dep_var.shape}."
    #         )

    # 3) Build a namespace mapping each prefixed field name → its NumPy array slice
    #    Prefixes: "a_", "b_", "c_", ... up to as many objects as in list
    #    Also expose single-letter proxies (a, b, ...) that support dot-syntax
    #    e.g. a.luminance, a.sample.luminance and intercept NumPy functions
    namespace = {}

    class GroupedField:
        """Wraps a per-chamber array and a sample-ID vector to support grouped
        reductions. Intercepts NumPy functions via __array_function__ so calls
        like np.mean(a.sample.luminance) compute the per-sample reduction and
        map the result back to chambers.
        """
        # Ensure NumPy prefers __array_function__ on this object
        __array_priority__ = 1000

        def __init__(self, arr, sample_ids):
            self.arr = np.asarray(arr)
            self.sample_ids = np.asarray(sample_ids)
            if self.arr.shape[-1] != self.sample_ids.shape[0]:
                raise ValueError("sample_IDs length must match chamber axis length")
            # precompute unique samples and inverse mapping
            self._uniques, self._inv = np.unique(self.sample_ids, return_inverse=True)

        def _reduce_per_sample(self, func, *args, **kwargs):
            """Apply func to each group's slice along the chamber axis (last axis)
            and map the per-sample result back to chambers.
            """
            arr = self.arr
            inv = self._inv
            uniques = self._uniques
            per_sample = []
            # Prepare kwargs so that axis defaults to -1 (the chamber axis inside group)
            kwargs2 = dict(kwargs)

            # only set axis kw if user didn't provide any extra positional args
            # (so we won't override a user-provided positional axis/q)
            if 'axis' not in kwargs2 and len(args) <= 1:
                kwargs2['axis'] = -1

            for s_idx in range(uniques.shape[0]):
                mask = (inv == s_idx)
                group = arr[..., mask]
                # Replace the original first positional arg (the array) with the group
                # so we don't accidentally pass the old GroupedField or duplicate args.
                if len(args) >= 1:
                    new_args = list(args)
                    new_args[0] = group
                else:
                    new_args = [group]

                # Call the reduction function on the group along its last axis
                res = func(*new_args, **kwargs2)
                per_sample.append(np.asarray(res))

            # Stack results into shape (..., n_samples, ...extra)
            stacked = np.stack(per_sample, axis=-1)
            # Map per-sample results back to chambers using inverse indices
            mapped = stacked[..., inv]
            return mapped

        # Intercept NumPy functions (np.mean, np.median, np.percentile, etc.)
        def __array_function__(self, func, types, args, kwargs):
            # We'll accept typical reduction functions from numpy
            try:
                name = func.__name__
            except Exception:
                name = None

            # Special-case percentile as it expects 'q' as second positional arg
            if func is np.percentile or name == 'percentile':
                # np.percentile(a, q, axis=None, **kwargs)
                # args may contain q as args[0]
                return self._reduce_per_sample(func, *args, **kwargs)

            # For other reductions, forward axis=-1 if not provided
            return self._reduce_per_sample(func, *args, **kwargs)

        # Fallback conversion to ndarray: return the raw per-chamber array
        def __array__(self, dtype=None):
            return np.asarray(self.arr, dtype=dtype)

    class GroupAccessor:
        """Accessor returned by proxy.sample — used to fetch grouped fields."""
        def __init__(self, obj):
            self._obj = obj

        def __getattr__(self, field_name):
            if field_name not in self._obj.dep_var_type:
                raise AttributeError(field_name)
            idx = self._obj.dep_var_type.index(field_name)
            arr = self._obj.dep_var[..., idx]
            return GroupedField(arr, self._obj.indep_vars.sample_IDs)

    class DeviceField:
        """Wraps a per-chamber array and provides reductions across the
        entire device (i.e., across the chamber axis). This intercepts
        NumPy reduction functions so calls like `np.mean(a.device.luminance)`
        reduce over chambers and return a scalar or appropriately-shaped
        array without mapping back to chambers.
        """
        __array_priority__ = 1000

        def __init__(self, arr):
            self.arr = np.asarray(arr)

        def _reduce_across_device(self, func, *args, **kwargs):
            # Default to axis=-1 (chamber axis) if not provided
            kwargs2 = dict(kwargs)
            if 'axis' not in kwargs2 and len(args) <= 1:
                kwargs2['axis'] = -1

            # Ensure the first positional arg is the raw array
            if len(args) >= 1:
                new_args = list(args)
                new_args[0] = self.arr
            else:
                new_args = [self.arr]

            # Call the reduction
            res = func(*new_args, **kwargs2)
            res = np.asarray(res)

            # Figure out which axis the reduction used (normalize negative indices)
            axis_used = kwargs2.get('axis', None)
            # If axis_used is a tuple/list, we won't try to map back
            if isinstance(axis_used, (list, tuple, np.ndarray)):
                return res

            if axis_used is None:
                return res

            # Normalize axis to positive index
            try:
                axis_norm = int(axis_used)
            except Exception:
                return res

            if axis_norm < 0:
                axis_norm = axis_norm % self.arr.ndim

            # If the reduction removed the chamber axis (last axis), map the
            # per-device result back to chambers by inserting a chamber axis
            # and repeating across chambers so the returned array has the
            # same leading shape plus a chamber axis at the same position.
            chamber_axis = self.arr.ndim - 1
            if axis_norm == chamber_axis:
                n_chambers = self.arr.shape[-1]
                # res.shape should start with arr.shape[:-1]
                prefix = self.arr.shape[:-1]
                if res.shape[:len(prefix)] != prefix:
                    # Unexpected shape; just return the raw result
                    return res

                # Insert chamber axis before any extra tail dims
                insert_at = len(prefix)
                expanded = np.expand_dims(res, axis=insert_at)
                mapped = np.repeat(expanded, n_chambers, axis=insert_at)
                return mapped

            # Reduction wasn't along chamber axis — return as-is
            return res

        def __array_function__(self, func, types, args, kwargs):
            try:
                name = func.__name__
            except Exception:
                name = None

            # Special-case percentile which expects q as second positional arg
            if func is np.percentile or name == 'percentile':
                return self._reduce_across_device(func, *args, **kwargs)

            return self._reduce_across_device(func, *args, **kwargs)

        def __array__(self, dtype=None):
            return np.asarray(self.arr, dtype=dtype)

    class DeviceAccessor:
        """Accessor returned by proxy.device — used to fetch device-wide fields."""
        def __init__(self, obj):
            self._obj = obj

        def __getattr__(self, field_name):
            if field_name not in self._obj.dep_var_type:
                raise AttributeError(field_name)
            idx = self._obj.dep_var_type.index(field_name)
            arr = self._obj.dep_var[..., idx]
            return DeviceField(arr)

    class DataProxy:
        """Proxy for a DataND-like object that exposes dot syntax.
        Example: a.luminance, a.sample.luminance, a.chamber.luminance
        """
        def __init__(self, obj):
            self._obj = obj
            self.sample = GroupAccessor(obj)
            self.device = DeviceAccessor(obj)
            # chamber accessor is an alias to raw attributes
            self.chamber = self

        def __getattr__(self, name):
            # Return raw per-chamber array for dep_var fields
            if name in self._obj.dep_var_type:
                idx = self._obj.dep_var_type.index(name)
                return self._obj.dep_var[..., idx]
            raise AttributeError(name)

    for i, obj in enumerate(data_objs):
        prefix = f"{chr(ord('a') + i)}_"  # 'a_', 'b_', 'c_', ...
        # expose both flat names (a_luminance) and single-letter proxy (a)
        proxy_name = f"{chr(ord('a') + i)}"
        if proxy_name in namespace:
            raise ValueError(f"Namespace conflict: '{proxy_name}' already exists.")
        namespace[proxy_name] = DataProxy(obj)
        for idx, field_name in enumerate(obj.dep_var_type):
            key = prefix + field_name
            if key in namespace:
                raise ValueError(f"Namespace conflict: '{key}' already exists.")
            # Adding the array, with units!
            namespace[key] = obj.dep_var[..., idx] * obj.dep_var_units[idx]
    
    # If the caller provided named variables to be available in the expression,
    # merge them into the namespace. This lets callers pass real objects (e.g.
    # NumPy arrays) instead of embedding their stringified representations into
    # the expression.
    #
    # We intentionally validate keys so callers can't override internal
    # proxies (a, b, a_luminance, etc.) or inject underscored names.
    # Merge provided expression_vars (if any) into the namespace so they are
    # available when evaluating `expr`. This is the preferred mechanism for
    # passing arrays or other objects into expressions without stringifying
    # them.
    if expression_vars is not None:
        if not isinstance(expression_vars, dict):
            raise TypeError("expression_vars must be a dict mapping names to objects")
        for k, v in expression_vars.items():
            if not isinstance(k, str):
                raise TypeError("expression_vars keys must be strings")
            if not k.isidentifier() or k.startswith("_"):
                raise ValueError(f"Invalid expression variable name: {k!r}")
            if k in namespace or k in ("np",):
                raise ValueError(f"expression_vars name '{k}' conflicts with existing namespace")
            namespace[k] = v

    # 4) Evaluate the expression using NumPy
    safe_globals = {"__builtins__": None, "np": np}
    print(f"Evaluating expression: {expr}")
    try:
        result = eval(expr, safe_globals, namespace)
    except Exception as e:
        raise RuntimeError(f"Error evaluating expression {expr!r}: {e}")

    # 5) Ensure result is a NumPy array of the correct shape
    if not isinstance(result, np.ndarray) and not isinstance(result, pint.Quantity):
        result = np.array(result)

    expected_shape = base_shape[:-1]
    if result.shape != expected_shape:
        raise ValueError(
            f"After evaluating {expr!r}, got result.shape = {result.shape}, "
            f"but expected {expected_shape}."
        )

    # 6) Expand the last axis so that new dep_var has final dim = 1
    new_transformed = result[..., np.newaxis]
    new_transformed_units = result.units
    new_transformed = new_transformed.magnitude # Proper way to convert to np array (without warnings)

    # If caller wants to keep existing dep_vars, combine them with the new one
    if keep_existing:
        # original dep_var and dep_var_type from first object
        orig_dep = data_objs[0].dep_var
        orig_types = list(data_objs[0].dep_var_type)
        orig_units = list(data_objs[0].dep_var_units)

        # If output_name already exists, replace that column; otherwise append
        if output_name in orig_types:
            replace_idx = orig_types.index(output_name)
            # Ensure shapes match (all dep_vars share same leading shape)
            if orig_dep.shape[:-1] != new_transformed.shape[:-1]:
                raise ValueError(
                    "Transformed result shape does not match original dependent variable shape."
                )
            combined = orig_dep.copy()
            combined[..., replace_idx] = new_transformed[..., 0]
            combined = combined  # shape unchanged
            combined_types = orig_types
            combined_units = orig_units
        else:
            # concatenate along final axis
            if orig_dep.shape[:-1] != new_transformed.shape[:-1]:
                raise ValueError(
                    "Transformed result shape does not match original dependent variable shape."
                )
            combined = np.concatenate([orig_dep, new_transformed], axis=-1)
            combined_types = orig_types + [output_name]
            combined_units = orig_units + [new_transformed_units]

        final_dep_var = combined
        final_dep_var_type = combined_types
        final_dep_var_units = combined_units

    else:
        final_dep_var = np.array(new_transformed)
        final_dep_var_type = [output_name]
        final_dep_var_units = [new_transformed_units]

    # 7) Build the output object—same class as first, same indep_vars, new dep_var + dep_var_type
    NewClass = base_class

    try:
        new_instance = NewClass(
            indep_vars   = data_objs[0].indep_vars, # deep-copied by DataND constructor
            dep_var      = final_dep_var,
            dep_var_type = final_dep_var_type,
            dep_var_units = final_dep_var_units,
            meta         = deepcopy(data_objs[0].meta)
        )
    except TypeError:
        raise TypeError(
            f"Cannot create new instance of {NewClass.__name__} with the arguments indep_vars, dep_var, dep_var_type, and meta. "
            "Check the constructor signature."
        )

    return new_instance
