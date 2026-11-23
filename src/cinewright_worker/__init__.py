"""Cinewright worker: pulls jobs from the core over the worker protocol and drives one engine adapter.

Standard library only, Python 3.10+, so it runs inside engine environments (see `adapter.py`).
Start it with `python -m cinewright_worker --engine mock|wan2gp`.
"""
