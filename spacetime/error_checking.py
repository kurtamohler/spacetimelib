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
    assert len(message) > 0, 'Internal error: `message` cannot be empty'
    if not condition:
        message_joined = ''.join([str(m) for m in message])
        raise error_type(message_joined)
