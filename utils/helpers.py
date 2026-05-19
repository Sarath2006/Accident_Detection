import datetime


def log(message, level="INFO"):
    """
    Simple logging utility
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{level}] [{timestamp}] {message}")


def clamp(value, min_value, max_value):
    """
    Clamp a value between min and max
    """
    return max(min_value, min(value, max_value))


def safe_divide(a, b):
    """
    Prevent division by zero
    """
    return a / b if b != 0 else 0
