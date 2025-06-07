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

import numpy as np
from copy import deepcopy


def transform_data(
    data_objs: list,        # list of Data2D, Data3D, Data4D instances (all same class and shape)
    expr: str,             # e.g. "(a_luminance - b_luminance)"
    output_name: str       # name of the new field, e.g. "difference"
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
    namespace = {}
    for i, obj in enumerate(data_objs):
        prefix = f"{chr(ord('a') + i)}_"  # 'a_', 'b_', 'c_', ...
        for idx, field_name in enumerate(obj.dep_var_type):
            key = prefix + field_name
            if key in namespace:
                raise ValueError(f"Namespace conflict: '{key}' already exists.")
            namespace[key] = obj.dep_var[..., idx]

    # 4) Evaluate the expression using NumPy
    safe_globals = {"__builtins__": None, "np": np}
    try:
        result = eval(expr, safe_globals, namespace)
    except Exception as e:
        raise RuntimeError(f"Error evaluating expression {expr!r}: {e}")

    # 5) Ensure result is a NumPy array of the correct shape
    if not isinstance(result, np.ndarray):
        result = np.array(result)

    expected_shape = base_shape[:-1]
    if result.shape != expected_shape:
        raise ValueError(
            f"After evaluating {expr!r}, got result.shape = {result.shape}, "
            f"but expected {expected_shape}."
        )

    # 6) Expand the last axis so that new dep_var has final dim = 1
    new_dep_var = result[..., np.newaxis]

    # 7) Build the output object—same class as first, same indep_vars, new dep_var + dep_var_type
    NewClass = base_class

    try:
        new_instance = NewClass(
            indep_vars   = data_objs[0].indep_vars, # deep-copied by DataND constructor
            dep_var      = new_dep_var,
            dep_var_type = [output_name],
            meta         = deepcopy(data_objs[0].meta)
        )
    except TypeError:
        raise TypeError(
            f"Cannot create new instance of {NewClass.__name__} with the arguments indep_vars, dep_var, dep_var_type, and meta. "
            "Check the constructor signature."
        )

    return new_instance

# def transform_data(
#     data_obj,              # e.g. a Data2D or Data3D or Data4D instance
#     standard_obj,          # same class as data_obj, same shape in .dep_var
#     expr: str,             # e.g. "(luminance - intercept) / slope"
#     output_name: str       # name of the new field, e.g. "concentration"
# ):
#     """
#     Return a new object of the same class as `data_obj` in which:
#       - The indep_vars are deep-copied from data_obj.
#       - We evaluate `expr` elementwise, treating each name in
#         data_obj.dep_var_type  and standard_obj.dep_var_type
#         as a NumPy array of shape  = data_obj.dep_var[..., i].
#       - The result (shape = data_obj.dep_var.shape[:-1]) is
#         expanded along a new final axis of size 1, and stored
#         in the new .dep_var.  The new .dep_var_type = [output_name].
#     """

#     # 1) Must be exactly the same class and same dep_var shape
#     if type(data_obj) is not type(standard_obj):
#         raise TypeError(
#             f"Cannot transform between different classes: "
#             f"{type(data_obj).__name__!r} vs {type(standard_obj).__name__!r}"
#         )
#     if data_obj.dep_var.shape != standard_obj.dep_var.shape:
#         raise ValueError(
#             f"data_obj.dep_var.shape = {data_obj.dep_var.shape} but "
#             f"standard_obj.dep_var.shape = {standard_obj.dep_var.shape}. "
#             "They must match exactly."
#         )

#     # 2) Build a local namespace mapping each field name → its NumPy array slice
#     namespace = {}
#     # — from data_obj:
#     for idx, field_name in enumerate(data_obj.dep_var_type):
#         if field_name in namespace:
#             raise ValueError(f"Duplicate dep_var_type {field_name!r} in data_obj")
#         namespace[field_name] = data_obj.dep_var[..., idx]
#     # — from standard_obj:
#     for idx, field_name in enumerate(standard_obj.dep_var_type):
#         # if field_name in namespace:
#         #     raise ValueError(
#         #         f"Name conflict: {field_name!r} appears in both "
#         #         "data_obj.dep_var_type and standard_obj.dep_var_type"
#         #     )
#         namespace[field_name] = standard_obj.dep_var[..., idx]

#     # 3) Evaluate the expression using NumPy
#     #    We explicitly disable builtins for safety, but allow 'np' for math.
#     safe_globals = {"__builtins__": None, "np": np}

#     try:
#         result = eval(expr, safe_globals, namespace)
#     except Exception as e:
#         raise RuntimeError(f"Error evaluating expression {expr!r}: {e}")

#     # 4) The result should be a NumPy array of shape = data_obj.dep_var.shape[:-1].
#     if not isinstance(result, np.ndarray):
#         # If the user wrote something that yields a scalar or Python list, convert to np.array:
#         result = np.array(result)

#     expected_shape = data_obj.dep_var.shape[:-1]
#     if result.shape != expected_shape:
#         raise ValueError(
#             f"After evaluating {expr!r}, got result.shape = {result.shape}, "
#             f"but expected shape = {expected_shape}.  Check that your field names "
#             "align correctly."
#         )

#     # 5) Expand the last axis so that the new dep_var has final dim = 1
#     new_dep_var = result[..., np.newaxis]  # shape = (*expected_shape, 1)

#     # 6) Build the output object—same class, same indep_vars, new dep_var + dep_var_type
#     NewClass = type(data_obj)
#     new_indep = deepcopy(data_obj.indep_vars)

#     # If the original class signature is (indep_vars, dep_var, dep_var_type, meta),
#     # we pass them in accordingly.  If your class signature differs, you may adapt this.
#     try:
#         new_instance = NewClass(
#             indep_vars = new_indep,
#             dep_var = new_dep_var,
#             dep_var_type = [output_name],
#             meta = deepcopy(data_obj.meta)
#         )
#     except TypeError:
#         # In case your DataND constructor does not take a 'meta' argument in that position,
#         # try omitting it (or adjust as needed).
#         new_instance = NewClass(
#             indep_vars = new_indep,
#             dep_var = new_dep_var,
#             dep_var_type = [output_name]
#         )

#     return new_instance
