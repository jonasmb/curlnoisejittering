# Curl Noise Jittering

This repository contains the code associated with the article:

---

**Curl Noise Jittering**

[J. Andreas Bærentzen](http://www2.compute.dtu.dk/~janba/), [Jeppe Revall Frisvad](http://www.imm.dtu.dk/~jerf/), [Jonàs Martínez](https://sites.google.com/site/jonasmartinezbayona/)

Siggraph Asia 2023 Conference Track

---

This repository specifically contains the Python code used for the 2D direct method and comparisons with other methods.
A ShaderToy producing images qualitatively identical to Figure 5 using our implicit method is available [here](https://www.shadertoy.com/view/Dd3yW4).

## Dependencies

- Anaconda, Miniconda (https://anaconda.org), or Pip (https://pypi.org/project/pip/)
- CMake (https://cmake.org)
- libcairo (https://www.cairographics.org)
- pkg-config (https://www.freedesktop.org/wiki/Software/pkg-config/)

## Installation using Anaconda

```
conda env create -f environment.yml
conda activate cnj
./build_external.sh
```

## Installation using Pip

```
pip install --user numpy matplotlib numba scipy
./build_external.sh
```

# Usage 

To generate the results of Figure 4 in the paper, run the command:

```
python3 points.py
```

If everything goes well, an output figure_4.pdf will be generated.

# External libraries

This repository includes code from:

* Capacity Constrained Voronoi Diagrams https://github.com/michaelbalzer/ccvt (subfolder ccvt_mod)

* psa - Point Set Analysis https://github.com/nodag/psa/tree/master (subfolder psa_mod)

* Optimizing dyadic nets http://abdallagafar.com/publications/dyadic-nets/files/dyadic-nets-code.zip (subfolder dyadic_mod)