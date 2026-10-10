"""Operator service: the dashboard plus a small JSON API.

Deliberately thin. Nothing here recomputes anything per request except the
current-window snapshot, which is built once at startup from the cached
pipeline output. A competition demo that needs a database and a queue is a demo
that will be broken by the time it is set up.
"""
