"""Run with Python after installing the packages listed in README.md."""

import logging
from time import perf_counter

from delta_classification import delta_classification


def main():
    logging.getLogger().setLevel(logging.WARNING)
    start = perf_counter()
    result = delta_classification(m=2, Delta=5, mode="delta")
    print(f"Classes: {len(result)}")
    maximum = max(len(p.integral_points()) for p in result)
    print(f"Maximum lattice points (including the origin): {maximum}")
    print(f"Maximum columns up to sign: {(maximum - 1) // 2}")
    for index, polytope in enumerate(result, start=1):
        print(f"{index}: {list(map(tuple, polytope.vertices()))}")
    print(f"Elapsed: {perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
