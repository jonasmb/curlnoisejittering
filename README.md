# Curl Noise Jittering


This repository contains the code associated with the article:

---

**Curl Noise Jittering**

[J. Andreas Bærentzen](http://www2.compute.dtu.dk/~janba/), [Jeppe Revall Frisvad](http://www.imm.dtu.dk/~jerf/), [Jonàs Martínez](https://sites.google.com/site/jonasmartinezbayona/)

Siggraph Asia 2023 Conference Track

---

This repository contains the Python code used for the 2D direct method and comparisons with other methods. The comparisons are shown in Figure 4.
A ShaderToy demo able to produce the images in Figure 5 of the paper (using our implicit method) is available [here](https://www.shadertoy.com/view/Dd3yW4).

# Description

![][curl_noise_jittering.jpg]

We propose a method for implicitly generating blue noise point sets. Our method is based on the observations that curl noise vector fields are volume-preserving and that jittering can be construed as moving points along the streamlines of a vector field. We demonstrate that the volume preservation keeps the points well separated when jittered using a curl noise vector field. At the same time, the anisotropy that stems from regular lattices is significantly reduced by such jittering. In combination, these properties entail that jittering by curl noise effectively transforms a regular lattice into a point set with blue noise properties. Our implicit method does not require computing the point set in advance. This makes our technique valuable when an arbitrarily large set of points with blue noise properties is needed.


# Installation


## Dependencies

- Anaconda, Miniconda (https://anaconda.org), or Pip (https://pypi.org/project/pip/)
- CMake (https://cmake.org)

## Installation using Anaconda

```
conda env create -f environment.yml
conda activate cnj
./build_external.sh
```

On Windows, run build_external.sh in the Developer PowerShell and the rest in the Anaconda Prompt.

## Installation using Pip

```
python3 -m pip install --user numpy matplotlib numba scipy
./build_external.sh
```

# Usage 

To generate the results of Figure 4 in the paper, run the command:

```
python3 points.py
```

If everything goes well, an output figure_4.pdf will be generated.
This figure is a comparison of our method (far left) to the methods discussed in the paper. We show point sets (top) as well as the power spectrum (middle) and the radial spectrum for each point set (bottom).

# External libraries

This repository includes code from:

* Capacity Constrained Voronoi Diagrams https://github.com/michaelbalzer/ccvt (subfolder ccvt_mod)

* psa - Point Set Analysis https://github.com/nodag/psa/tree/master (subfolder psa_mod)

* Optimizing dyadic nets http://abdallagafar.com/publications/dyadic-nets/files/dyadic-nets-code.zip (subfolder dyadic_mod)
