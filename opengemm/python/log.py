"""Library messages, on stderr with the ``[opengemm]`` prefix so a caller's
stdout stays its own.
"""
import sys

def log(message):
    print(f"[opengemm] {message}", file=sys.stderr, flush=True)
