# Curl Noise Jittering

This repository contains the code associated with the article:

---

**Curl Noise Jittering**

[J. Andreas Bærentzen](http://www2.compute.dtu.dk/~janba/), [Jeppe Revall Frisvad](http://www.imm.dtu.dk/~jerf/), [Jonàs Martínez](https://sites.google.com/site/jonasmartinezbayona/)

Siggraph Asia 2023 Conference Track

---

Note that this repository specifically contains the Python code used for the 2D direct method and comparisons with other methods. There is also a ShaderToy example which produces images qualitatively identical to Figure 5 using our implicit method. This example is available [here](https://www.shadertoy.com/view/Dd3yW4).

## Dependencies

- Anaconda or Miniconda
- CMake
- libcairo (https://www.cairographics.org)

## Installation using Anaconda

```
conda env create -f environment.yml
conda activate cnj
./build_external.sh
```

# Usage 

To generate the results of Figure 4 in the paper run the command:

```
python points.py
```

If everything goes well an output figure_4.pdf will be generated.

# External libraries

This repository includes code from:

* Capacity Constrained Voronoi Diagrams https://github.com/michaelbalzer/ccvt (subfolder ccvt_mod)

*  psa - Point Set Analysis https://github.com/nodag/psa/tree/master (subfolder psa_mod)

* Optimizing dyadic nets http://abdallagafar.com/publications/dyadic-nets/files/dyadic-nets-code.zip (subfolder dyadic_mod)