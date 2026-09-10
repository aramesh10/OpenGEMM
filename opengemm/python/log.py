import sys

def log(message):
    print(f"[opengemm] {message}", file=sys.stderr, flush=True)
