import hashlib
import pickle
from functools import wraps

import diskcache as dc


def diskcache_cache(cache_dir, cache_size):
    """
    Decorator to cache function outputs persistently using diskcache.

    Args:
        cache_dir (str): Directory where the cache is stored.
        cache_size (int): Maximum size of the cache in bytes.
    """
    cache = dc.Cache(cache_dir, size_limit=cache_size)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a unique key based on function name and arguments
            key_source = (func.__name__, args[1:], frozenset(kwargs.items()))
            key = hashlib.sha256(pickle.dumps(key_source)).hexdigest()

            # Attempt to retrieve from cache
            result = cache.get(key)
            if result:
                return result

            # Execute the function and cache the result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper

    return decorator
