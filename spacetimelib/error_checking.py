# Error checking functions

# Check a condition and raise an error if it is False
#
# Args:
#   condition (bool):
#     The condition to check.
#
#   error_type (type):
#     The type of error to raise.
#
#   message (callable):
#     The message for the error.
def check(condition, error_type, message):
    assert callable(message), '`message` must be callable'
    if not condition:
        raise error_type(message())

# Check if the object has the expected type and raise an error if not
#
# Args:
#   obj (object):
#     The object to check.
#
#   expected_type (type):
#     The expected type of the object.
#
#   obj_name (str):
#     The name of the object, included in error message.
def check_type(obj, expected_type, obj_name):
    check(isinstance(obj, expected_type), TypeError, lambda: (
        f"expected '{obj_name}' to be type {expected_type}, but got {type(obj)}"))

# Check a condition and raise an internal assert if it is False
#
# Args:
#   condition (bool):
#     The condition to check.
#
#   message (list of strings):
#     The message for the error.
#     Variable arg list of objects that have `__str__` methods.
def internal_assert(condition, message=None):
    if message is not None:
        assert callable(message), '`message` must be callable'
    if not condition:
        message_final = 'Internal assert failed'

        if message is not None:
            message_final += ': ' + message()
        raise AssertionError(
            f"{message_final}\n"
            "Please report an issue with the error message and traceback here: "
            "https://github.com/kurtamohler/spacetimelib/issues")

# Wrap negative indices and throw an error if index is out of range.
#
# Args:
#  idx (int): Index to wrap.
#
#  length (int): Length of array.
def maybe_wrap_index(idx, length):
    internal_assert(isinstance(idx, int))
    internal_assert(isinstance(length, int))

    if idx < 0:
        idx_wrapped = idx + length
    else:
        idx_wrapped = idx

    check(idx_wrapped >= 0 and idx_wrapped < length, ValueError, lambda: (
        f"index '{idx}' is out of range for container of length '{length}'"))

    return idx_wrapped
