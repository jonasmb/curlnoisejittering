from numba import njit
from math import cos, exp, sin, pi
from random import gauss, seed, uniform
from common import vec2, vec4
from numpy import empty, dot

dir = []

def init_noise_vectors(N):
	'''Initialize random vectors used to generate the 2D sine waves that form the basis of the noise'''
	# The sum-of-sines noise depends on random vectors. This seed has been observed to work 
	# well as the basis of the noise function that we use to produce the curl noise
	seed(25)
	global dir
	dir = empty((N,4))
	sigma = 7
	for i in range(N):
		alpha = uniform(0, 2*pi)
		w = gauss(mu=0,sigma=sigma)
		v_2d = vec2(cos(alpha), sin(alpha))*(sigma+w)
		v = vec4(v_2d[0], v_2d[1],exp(-w*w/sigma**2), uniform(0,2*pi))
		dir[i] = v

init_noise_vectors(256)

@njit
def sos_noise(x):
	'''Noise function. Simply sums a number of sine functions'''
	I = 0.0
	for d in dir:
		I += sin(dot(x,d[0:2])+d[3])*d[2]
	return 0.5 + 0.5 * (I / float(len(dir)))

@njit
def sos_noise_grad(p):
	x = p[0:2]
	'''Gradient of noise function computed analytically'''
	g = vec2(0,0)
	for d in dir:
		g += d[2]*d[0:2]*cos(dot(x,d[0:2])+d[3])
	return 0.5 * g / len(dir)	