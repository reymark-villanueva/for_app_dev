import hashlib

from django.core.cache import cache


def client_ip(request):
    forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR', '')
    if forwarded_for:
        return forwarded_for.split(',', 1)[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def is_rate_limited(request, scope, limit, window_seconds, *parts):
    identity = ':'.join(str(part).strip().lower() for part in parts if part)
    raw_key = f'{scope}:{client_ip(request)}:{identity}'
    cache_key = f'rl:{hashlib.sha256(raw_key.encode("utf-8")).hexdigest()}'

    attempts = cache.get(cache_key, 0)
    if attempts >= limit:
        return True

    if not cache.add(cache_key, 1, timeout=window_seconds):
        try:
            cache.incr(cache_key)
        except ValueError:
            cache.set(cache_key, 1, timeout=window_seconds)
    return False
