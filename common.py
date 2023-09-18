from numba import njit
from numpy import empty
from math import sqrt
from random import uniform

@njit
def vec2(x,y):
	'''Convert two numbers to a 2D numpy array '''
	v = empty((2,),dtype=float)
	v[0] = x
	v[1] = y
	return v

@njit
def vec3(x,y,z):
	'''Convert three numbers to a 2D numpy array '''
	v = empty((3,),dtype=float)
	v[0] = x
	v[1] = y
	v[2] = z
	return v

@njit
def vec4(x,y,z,w):
	'''Convert four numbers to a 2D numpy array '''
	v = empty((4,),dtype=float)
	v[0] = x
	v[1] = y
	v[2] = z
	v[3] = w
	return v

# @njit	
# def V(x, inv_noise_scale, offset):
# 	''' V returns a curl noise vector field'''
# 	p = empty(3)
# 	p[0:2] = x*inv_noise_scale+offset
# 	return scnoise_curl(p)[0:2]

@njit	
def V(x, inv_noise_scale, offset, noise_grad):
	''' V returns a curl noise vector field'''
	p = empty(3)
	p[0:2] = x*inv_noise_scale+offset
	p[2] = 98.45
	g = noise_grad(p)
	gh = vec2(-g[1], g[0])
	return gh

@njit
def Euler(x, dt, inv_noise_scale, offset, noise_grad):
	'''Perform one Euler step of length dt from x along the curl noise vector field'''
	return x + dt * V(x, inv_noise_scale, offset, noise_grad)

@njit
def RK4(x, dt, inv_noise_scale, offset, noise_grad):
	'''Perform one Runge-Kutta 4th order step of length dt from x along the curl noise vector field'''
	a = dt * V(x, inv_noise_scale, offset, noise_grad)
	b = dt * V(x+a/2, inv_noise_scale, offset, noise_grad)
	c = dt * V(x+b/2, inv_noise_scale, offset, noise_grad)
	d = dt * V(x+c, inv_noise_scale, offset, noise_grad)
	return x + (a+2*b+2*c+d)/6



@njit
def rand_vec2():
	'''Convert two numbers to a 2D numpy array '''
	v = empty((2,),dtype=float)
	v[0] = uniform(0,1)
	v[1] = uniform(0,1)
	return v

## UNUSED

@njit
def smoothstep(a,b,x):
	t = x-a/b-a
	if t < 0: return 0
	if 1 < t: return 1
	return 3*t**2-2*t**3

@njit
def Rand(x, dt, inv_noise_scale, offset):
	'''Perform a step of length dt from x in a random direction.'''
	return x + dt * rand_vec2() 

@njit
def NoiseV(x, dt, inv_noise_scale, offset, noise_grad):
	'''Perform a step of length dt from x in a noise vector direction.'''
	p = vec3(0,0,offset)
	p[0:2] = x*inv_noise_scale
	g = noise_grad(p)
	return x + dt * g[0:2]

def L2_star_disc(P):
	n = P.shape[0]
	l2sd = (1/3)**2 
	l2sd -= (2/n) * sum([ 0.25 * (1 - x[0]**2) * (1 - x[1]**2) for x in P])
	s = 0
	for xi in P:
		for xk in P:
			s += (1-max(xi[0], xk[0]))*(1-max(xi[1], xk[1]))
	l2sd += s/(n**2)
	return sqrt(l2sd)
