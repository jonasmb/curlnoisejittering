from numba import njit
import numpy as np
from common import vec2

@njit
def hash2d(x):
	p1 = 73856093
	p2 = 19349663
	p3 = 83492791
	K = 93856263
	i = int(x[0])
	j = int(x[1])
	h1 = ((i * p1) ^ (j * p2)) % K
	h2 = ((j * p1) ^ (i * p3)) % K
	return vec2(h1,h2) / float(K) - 0.5

@njit	
def perlin_noise_grad( p ):
	x = p[0:2]
	'''returns 3D value noise (in [0])  and its derivatives (in .yz)'''
	i = np.floor(x)
	f = x-i	

	u = f*f*f*(f*(f*6.0-15.0)+10.0)
	du = 30.0*f*f*(f*(f-2.0)+1.0)
    
	ga = hash2d( i + vec2(0.0,0.0) )
	gb = hash2d( i + vec2(1.0,0.0) )
	gc = hash2d( i + vec2(0.0,1.0) )
	gd = hash2d( i + vec2(1.0,1.0) )
	#print(ga, gb,gc,gd)
    
	va = np.dot( ga, f - vec2(0.0,0.0) )
	vb = np.dot( gb, f - vec2(1.0,0.0) )
	vc = np.dot( gc, f - vec2(0.0,1.0) )
	vd = np.dot( gd, f - vec2(1.0,1.0) )

	grd =  ga + u[0]*(gb-ga) + u[1]*(gc-ga) + u[0]*u[1]*(ga-gb-gc+gd) + (vec2(u[1],u[0])*(va-vb-vc+vd) + vec2(vb,vc) - va)*du
	return grd
	# return vec3( va + u[0]*(vb-va) + u[1]*(vc-va) + u[0]*u[1]*(va-vb-vc+vd),
    #             grd[0], grd[1])