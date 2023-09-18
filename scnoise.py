import numpy as np
import matplotlib.pyplot as plt
import math
import time
from numba import jit, njit, float64, uint64, prange
import numba as nb

vector = nb.types.Array(dtype=float64, ndim=1, layout="C")
image = nb.types.Array(dtype=float64, ndim=3, layout="C")

@jit(vector(float64, float64, float64))
def vec3(x,y,z):
  return np.array((x,y,z))

@jit(vector(vector))
def normalize(v):
  len = math.sqrt(np.dot(v,v))
  return v/len

@jit(float64(vector))
def cubic(v):
  x = 1.0 - np.dot(v, v)*4.0
  return x*x*x if x > 0.0 else 0.0

@jit(vector(vector))
def cubic_grad(v):
  x = 1.0 - np.dot(v, v)*4.0
  return v*(x*x*(-24.0) if x > 0.0 else 0.0)

#sources = 30;
#a = 0.8*pow(sources, -1.0/3.0);
# Use the following if using impulse magnitudes in [0.5,1] to manage with fewer impulses

@jit(float64(vector))
def scnoise(p):
  sources = 10
  a = 0.5*pow(sources, -1.0/3.0)
  pi0 = np.floor(p - 0.5)
  result = 0.0
  for i in range(8):
    corner = np.mod(vec3(i, np.floor(i/2), np.floor(i/4)), 2)
    pi = pi0 + corner
    t = np.uint32(int(4*sources*(pi[0] + pi[1]*1000 + pi[2]*576 + pi[0]*pi[1]*pi[2]*3)))
    np.random.seed(t)
    n = np.random.random(size=(sources,4))
    for j in range(sources):
      #c = a*(n[j,0] - 0.5)
      # Use the following for c to manage with fewer impulses (e.g. sources = 10)
      c = n[j,0] - 0.5
      c = a*(np.copysign(0.5, c) + c)*0.5
      xi = vec3(n[j,1], n[j,2], n[j,3])
      x = pi + xi
      result += c*cubic(x - p)
  return result + 0.5

@jit(vector(vector))
def scnoise_grad(p):
  sources = 10
  a = 0.5*pow(sources, -1.0/3.0)
  pi0 = np.floor(p - 0.5)
  result = vec3(0.0, 0.0, 0.0)
  for i in range(8):
    corner = np.mod(vec3(i, np.floor(i/2), np.floor(i/4)), 2)
    pi = pi0 + corner
    t = np.uint32(int(4*sources*(pi[0] + pi[1]*1000 + pi[2]*576 + pi[0]*pi[1]*pi[2]*3)))
    np.random.seed(t)
    n = np.random.random(size=(sources,4))
    for j in range(sources):
      #c = a*(n[j,0] - 0.5)
      # Use the following for c to manage with fewer impulses (e.g. sources = 10)
      c = n[j,0] - 0.5
      c = a*(np.copysign(0.5, c) + c)*0.5
      xi = vec3(n[j,1], n[j,2], n[j,3])
      x = pi + xi
      result += c*cubic_grad(x - p)
  return result

@jit(vector(vector))
def scnoise_curl(p):
  sources = 10
  a = 0.5*pow(sources, -1.0/3.0)
  pi0 = np.floor(p - 0.5)
  g1 = vec3(0.0, 0.0, 0.0)
  g2 = vec3(0.0, 0.0, 0.0)
  g3 = vec3(0.0, 0.0, 0.0)
  for i in range(8):
    corner = vec3(np.bitwise_and(i, 1), np.bitwise_and(np.right_shift(i, 1), 1), np.bitwise_and(np.right_shift(i, 2), 1))
    pi = pi0 + corner
    t = np.uint32(int(6*sources*(pi[0] + pi[1]*1000 + pi[2]*576 + pi[0]*pi[1]*pi[2]*3)))
    np.random.seed(t)
    n = np.random.random(size=(sources,6))
    for j in range(sources):
      c = a*(vec3(n[j,0], n[j,1], n[j,2]) - 0.5)
      # Use the following for c to manage with fewer impulses (e.g. sources = 10)
      #c = n[j,0] - 0.5
      #c = a*(np.copysign(0.5, c) + c)*0.5
      xi = vec3(n[j,3], n[j,4], n[j,5])
      x = pi + xi
      g1 += c[0]*cubic_grad(x - p)
      g2 += c[1]*cubic_grad(x - p)
      g3 += c[2]*cubic_grad(x - p)
  return vec3(g3[1] - g2[2], g1[2] - g3[0], g2[0] - g1[1])

@jit(image(), nopython=True, parallel=False)
def render():
  noise_scale = 20.0
  width = 512
  height = 384
  aspect = width/height
  im = np.zeros(shape=(height, width, 3))
  z = -2.0
  for row in prange(height):
    for column in range(width):
      u = (column + 0.5)/width
      v = (row + 0.5)/height
      ray_direction = vec3(aspect*(2.0*u-1.0), 2.0*v-1.0, z)
      #im[row,column,:] = scnoise(ray_direction*noise_scale)
      #im[row,column,:] = scnoise_grad(ray_direction*noise_scale)*0.5 + 0.5
      im[row,column,:] = scnoise_curl(ray_direction*noise_scale)*0.5 + 0.5
      im[row,column,2] = 0.0
  return im

if __name__ == "__main__":
  t0 = time.perf_counter()
  im = render()
  t1 = time.perf_counter()
  print(t1 - t0)

  plt.imshow(im)
  plt.show()
