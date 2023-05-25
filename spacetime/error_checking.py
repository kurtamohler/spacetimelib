# Check a condition and raise an error if it is False
#
# Args:
#   condition : bool
#       The condition to check.
#
#   error_type : type
#       The type of error to raise.
#
#   message : variable arg list of objects that have `__str__` methods
#       The message for the error.
def check(condition, error_type, *message):
    internal_assert(len(message), '`message` cannot be empty')
    if not condition:
        message_joined = ''.join([str(m) for m in message])
        raise error_type(message_joined)

def check_type(obj, expected_type, obj_name):
    check(isinstance(obj, expected_type), TypeError,
        f"expected '{obj_name}' to be type {expected_type}, but got {type(obj)}")

def internal_assert(condition, *message):
    if not condition:
        message_final = 'Internal assert failed'

        if len(message):
            message_final += ': ' + ''.join([str(m) for m in message]) if len(message) else ''
        raise AssertionError(
            f"{message_final}\n"
            "Please report an issue with the error message and traceback here: "
            "https://github.com/kurtamohler/spacetimelib/issues")

def maybe_wrap_index(idx, length):
    internal_assert(isinstance(idx, int))
    internal_assert(isinstance(length, int))

    if idx < 0:
        idx_wrapped = idx + length
    else:
        idx_wrapped = idx

    check(idx_wrapped >= 0 and idx_wrapped < length, ValueError,
        f"index '{idx}' is out of range for container of length '{length}'")

    return idx_wrapped
