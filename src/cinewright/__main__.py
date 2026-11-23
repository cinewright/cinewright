"""`python -m cinewright` runs the launcher; `python -m cinewright --version` is the CI smoke test."""

from cinewright.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
