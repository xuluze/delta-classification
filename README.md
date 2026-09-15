# delta-classification

This repository builds on and improves the implementation by the authors of [mschymura/delta-classification](https://github.com/mschymura/delta-classification). It provides classification code used by [xuluze/proximity](https://github.com/xuluze/proximity).

The previous Sage code and classification data implement the algorithms in [On the maximal number of columns of a Δ-modular matrix](https://arxiv.org/abs/2111.06294), by Gennadiy Averkov and Matthias Schymura, for classifying centrally symmetric lattice polytopes.

This follow-up provides Python versions using [passagemath](https://github.com/passagemath/passagemath), a pip-installable, modularized fork of SageMath.

The original Sage sources are `delta-classification.sage` and `polytopes.sage`. The Python versions are `delta_classification.py` and `polytopes.py`. The `data/` folder contains the classification data.

## Clone and install

Run the following commands in Linux or an Ubuntu terminal in WSL on Windows. Use Python 3.14 for the package versions below.

```bash
git clone https://github.com/xuluze/delta-classification.git
cd delta-classification

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install passagemath-polyhedra==10.8.11 passagemath-combinat==10.8.11 passagemath-groups==10.8.11 passagemath-flint==10.8.11
```

These modular passagemath packages provide the mathematical functionality needed for classification. Their dependencies are installed automatically; a full SageMath installation is not required for the Python code.

The `python3.14` command selects Python 3.14 when creating the environment. After activation, `python` refers to that environment's interpreter. In a new terminal, return to the repository and activate the environment with `source .venv/bin/activate` before running the code.

## Usage

Run the included example for dimension `m=2` and `Delta=5`:

```bash
python example_delta_2_5.py
```

It lists the 16 classes and reports a maximum of 23 lattice points, including the origin.

To call the function yourself, start Python from the repository directory:

```bash
python
```

Then enter:

```python
from delta_classification import delta_classification

# Classify all centrally symmetric lattice polytopes for m=2, Delta=5.
polytopes = delta_classification(m=2, Delta=5, mode="delta")
print(len(polytopes))  # 16
print(max(len(p.integral_points()) for p in polytopes))  # 23

# Classify only the extremal examples.
extremal = delta_classification(m=2, Delta=5, mode="delta_ext")
print(len(extremal))  # 1
```

The function returns a list of polyhedra. Use `p.vertices()` to inspect a polyhedron's vertices and `p.integral_points()` to obtain its lattice points.

## GPT assistance

GPT assisted with adapting the Sage code for Python and modular passagemath, checking dependencies, writing tests, and preparing documentation. The original mathematical algorithms and classification data are credited to the upstream authors.
