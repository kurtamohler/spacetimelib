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

def internal_assert(condition, *message):
    if not condition:
        message_final = 'Internal assert failed'

        if len(message):
            message_final += ': ' + ''.join([str(m) for m in message]) if len(message) else ''
        raise AssertionError(
            f"{message_final}\n"
            "Please report an issue with the error message and traceback here: "
            "https://github.com/kurtamohler/spacetimelib/issues")

