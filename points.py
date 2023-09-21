#!/opt/local/bin/python
from math import atan2, ceil, cos, exp, log, pi, sin, sqrt
from random import gauss, seed, shuffle, uniform

import numpy as np
from matplotlib import cm, pyplot
from numba import njit
from numpy import array, dot, empty, linspace, mean, ndarray, ndindex, zeros
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
from scipy.stats.qmc import PoissonDisk
from time import time

from psa_wrapper import psa_wrapper
from ccvt_wrapper import ccvt_wrapper
from dyadic_wrapper import dyadic_wrapper
from common import vec2, vec3, rand_vec2, V, Euler, RK4, L2_star_disc
from scnoise import scnoise, scnoise_grad, scnoise_curl
from sosnoise import sos_noise, sos_noise_grad
from perlinnoise import perlin_noise_grad

def Fourier_spectrum(P, sz=2048):
	'''Create fourier transform of image generated from points'''
	I = zeros((sz,sz))
	for p in P:
		i = max(0, min(sz-1, int(sz*p[0])))
		j = max(0, min(sz-1, int(sz*p[1])))
		I[i,j] = 1
	ft = np.fft.fft2(I)
	ft = np.fft.fftshift(ft)
	ft_mag = np.clip(np.abs(ft), 0.0, 75)
	return ft_mag

def psa_point_stats(P):
	'''Use the PSA program to compute a range of statistics for a a point set. Specifically,
	the function returns effective Nyquist, anisotropy, quality, and radial spectrum. '''
	(effnyquist, oscillations, data_rp, data_ani, data_rdf) = psa_wrapper([P])
	aniso = array(data_rp[1][1:])**2 @ array(data_ani[1][1:])
	q = effnyquist/aniso
	return effnyquist, aniso, q, data_rp

def show_Fourier_spectrum(P,ax):
	'''Matplotlib display of Fourier spectrum.'''
	ft_mag = Fourier_spectrum(P)
	ax.imshow(ft_mag, cmap='gray')
	ax.set_xlim(924, 1124)
	ax.set_ylim(924, 1124)

def show_points(P,ax):
	'''Matplotlib display of point cloud.'''
	ax.set_aspect('equal')
	ax.scatter(P[:,0], P[:,1], color='k', s=0.5)#, s=0.25, marker='.')
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)

# Constants used for various visualization modes.
POINTS = 1
FOURIER = 2
STATS = 3
ALL = 4

def Poisson_disk_points(N=1000):
	'''Compute the Poisson Disk Sampling for given number of points.'''
	r = 0.8/(ceil(sqrt(N)))
	eng = PoissonDisk(d=2,radius=r)
	return eng.random(N)

def ccvt_points(N=1000):
	'''Compute the Capacity Constrained Voronoi Tessellation for given number of points.'''
	return ccvt_wrapper(N)

def regular_shuffle_2d(n=64,i0=0,j0=0):
	'''Shuffles the points by random descent in an quadtree. This is not strictly needed 
	but provides a random ordering of the points completely independent of curl noise
	jittering. Can be used for progressive generation.'''
	if n==1:
		return [ (i0,j0) ]
	n2 = n//2
	L_leaves = [regular_shuffle_2d(n2, i0+i*n2, j0+j*n2) for i,j in ndindex((2,2))]
	L = []
	order = [0, 1, 2, 3]
	for k in range(len(L_leaves[0])):
		shuffle(order)
		L += [ L_leaves[o][k] for o in order ]
	return L

def coherently_jittered_points(N=1000):
	'''This function performs coherent jittering'''
	P = []	
	Ny = int(ceil(sqrt(N)))
	Nx = Ny
	shiftx = [ uniform(0,1) for _ in range(Ny+4)]
	shifty = [ uniform(0,1) for _ in range(Nx+4)]
	for i,j in ndindex(Nx+4, Ny+4):
		p = vec2(i-2+shiftx[j],j-2+shifty[i]) / Nx
		if 0 < p[0] < 1 and 0 < p[1] < 1:
			P.append(p)
	return array(P)

def smoothly_jittered_points(N=1000):
	'''This function jitters by smoothing jittered positions'''
	P = []
	delta_x = cos(pi/3)
	delta_y = sin(pi/3)
	Ny = int(ceil(sqrt(N/delta_y)))
	Nx = int(Ny * delta_y)
	indices = list(ndindex(Nx+4,Ny+4))
	A = empty((Nx+4,Ny+4,2))
	for _i,_j in indices:
		i = _i - 2
		j = _j - 2
		alpha = uniform(0,2*pi)
		p = (1.1/Nx)*uniform(0,1)*vec2(cos(alpha),sin(alpha))
		A[i,j,:] = p 
	A_new = array(A)
	for _i,_j in ndindex(Nx+2,Ny+2):
		i = _i - 1
		j = _j - 1
		A_new[i,j] += A[i-1,j]+A[i+1,j]+A[i-1,j+1]+A[i+1,j+1]+A[i-1,j-1]+A[i+1,j-1]
		A_new[i,j] /= 7
	A = A_new
	for _i,_j in indices:
		i = _i - 2
		j = _j - 2
		p = vec2(i+(j%2)*delta_x,(j+0.5)*delta_y) / Nx
		p += A[i,j]
		if 0 < p[0] < 1 and 0 < p[1] < 1:
			P.append(p)
	return array(P)

def curl_noise_jittered_points_iterative(noise_grad=perlin_noise_grad, N=1000, triangular_grid=True, step_fun=RK4, dt=0.9, noise_scale=4, iter=1, offset=vec2(0,0), fraction=1):
	'''This function actually performs curl noise jittering - iterative fashion'''
	P = []
	delta_x = cos(pi/3) if triangular_grid else 0
	delta_y = sin(pi/3) if triangular_grid else 1
	Ny = int(ceil(sqrt(N/delta_y)))
	Nx = int(Ny * delta_y)
	indices = regular_shuffle_2d(2**ceil(log(max(Nx,Ny))/log(2)))
	jitter_dists = []
	for _i,_j in indices[0:int(fraction*len(indices))]:
		i = _i - 2
		j = _j - 2
		p = vec2(i+(j%2)*delta_x,(j+0.5)*delta_y) / Nx
		p_old = p
		for k in range(iter):
			p = step_fun(p, dt/(Nx*2**k), Nx/noise_scale, offset=offset+vec2(k*31.42, k*53.3), noise_grad=noise_grad)
		if 0 < p[0] < 1 and 0 < p[1] < 1:
			P.append(p)
			jitter_dists.append(norm(p-p_old))
	return array(P), array(jitter_dists)*Nx


def curl_noise_jittered_points(noise_grad=perlin_noise_grad, N=1000, triangular_grid=True, step_fun=RK4, dt=0.9, noise_scale=4, offset=vec2(0,0), fraction=1):
	'''This function actually performs curl noise jittering'''
	P = []
	delta_x = cos(pi/3) if triangular_grid else 0
	delta_y = sin(pi/3) if triangular_grid else 1
	Ny = int(ceil(sqrt(N/delta_y)))
	Nx = int(Ny * delta_y)
	indices = regular_shuffle_2d(2**ceil(log(max(Nx,Ny))/log(2)))
	jitter_dists = []
	for _i,_j in indices[0:int(fraction*len(indices))]:
		i = _i - 2
		j = _j - 2
		p = vec2(i+(j%2)*delta_x,(j+0.5)*delta_y) / Nx
		p_old = p
		p = step_fun(p, dt/Nx, Nx/noise_scale, offset=offset, noise_grad=noise_grad)
		if 0 < p[0] < 1 and 0 < p[1] < 1:
			P.append(p)
			jitter_dists.append(norm(p-p_old))
	return array(P), array(jitter_dists)*Nx


def point_generation(ax, N=1000, triangular_grid=True, step_fun=RK4, dt=0.9, noise_scale=4, show=POINTS, iter=1):
	''' Produce a point set using any of the available methods and show a figure.'''
	str = ""
	if iter>0:
		if iter>1:
			P,_ = curl_noise_jittered_points_iterative(N=N, triangular_grid=triangular_grid, step_fun=step_fun, dt=dt, noise_scale=noise_scale, iter=iter)
		else:
			P,_ = curl_noise_jittered_points(N=N, triangular_grid=triangular_grid, step_fun=step_fun, dt=dt, noise_scale=noise_scale)
		str = f"CNJ s: {noise_scale:5.3f}, t: {dt:5.3f}"
		if iter > 1:
			str += f", #iter: {iter} "
	elif iter==0:
		P = Poisson_disk_points(N)
		str = "PDS"
	elif iter==-1:
		P = coherently_jittered_points(N)
		str = "Coherent Jittering"
	elif iter==-2:
		P = array(ccvt_points(N))
		str = "CCVT"
	elif iter==-3:
		P = smoothly_jittered_points(N)
		str = "Smoothed Jittering"
	elif iter==-4:
		P = array(dyadic_wrapper(10,sigma=0.5))
		str = "Smoothed Jittering"

	if show==FOURIER:
		show_Fourier_spectrum(P, ax)
	elif show==POINTS:
		e,a,q,_ = psa_point_stats(P)
		str += f", Q: {q:5.3}"		
		show_points(P,ax)
		ax.set_title(str)
	print(str)
	return P

###### Visualization functions

def table2():
	'''Produces the raw output for generating Table 2 in "Curl Noise Jittering" 
	by Bærentzen, Frisvad, and Martinez'''
	N=1000
	dt = 1.05862
	s = 2.86207

	dt_iter = 1.0
	s_iter = 3.75
	iterations=64

	f = open("table2_data.txt", 'w')

	def output(P, str, t):
		e,a,q,_ = psa_point_stats(P)
		l2sd = L2_star_disc(P)
		f.write(str + f"| {e} & {a} & {q} & {l2sd} & {t}\n")
		f.flush()

	for i in range(100):
		seed(i)
		offs = vec2(uniform(0,10),uniform(0,10))
		T = time()
		P,_ = curl_noise_jittered_points(N=N, dt=dt, noise_scale=s, offset=offs)
		str = f"CNJ s: {s:5.3f}, t: {dt:5.3f}"
		output(P, str, time()-T)

		T = time()
		P,_ = curl_noise_jittered_points_iterative(N=N, dt=dt_iter, noise_scale=s_iter, iter=iterations, offset=offs)
		str = f"CNJ s: {s_iter:5.3f}, t: {dt_iter:5.3f}, #iter: {iterations} "
		output(P, str, time()-T)

		T = time()
		P = Poisson_disk_points(N)
		str = "PDS"
		output(P, str, time()-T)

		T = time()
		P = coherently_jittered_points(N)
		str = "Coherent Jittering"
		output(P, str, time()-T)

		T = time()
		P = array(ccvt_wrapper(N, random_seed=i))
		str = "CCVT"
		output(P, str, time()-T)

		T = time()
		P = smoothly_jittered_points(N)
		str = "Smoothed Jittering"
		output(P, str, time()-T)

		T = time()
		P = array(dyadic_wrapper(10,sigma=0.5))
		str = "Blue Nets"
		output(P, str, time()-T)


def figure4(N=1000):
	'''Produces the PDF for Figure 4 in "Curl Noise Jittering" 
	by Bærentzen, Frisvad, and Martinez'''
	dt = 1.05862
	s = 2.86207
	dt_iter = 1.0
	s_iter = 3.75
	iter=64

	fig, axs = pyplot.subplots(3,7, tight_layout=True)
	fig.set_size_inches(w=17.5,h=7.5)
	fig.set_dpi(320)
	for i,j in np.ndindex((3,7)):
		axs[i,j].set_xticks([])
		axs[i,j].set_yticks([])
	for i in range(7):
		axs[2,i].set_ylim((0,3))
	point_generation(axs[0,0], N=N, show=POINTS, dt=dt, noise_scale=s)
	point_generation(axs[0,1], N=N, show=POINTS, dt=dt_iter, iter=iter, noise_scale=s_iter)
	point_generation(axs[0,2], N=N, iter=-1, show=POINTS)
	point_generation(axs[0,3], N=N, iter=-3, show=POINTS)
	point_generation(axs[0,4], N=N, iter=-4, show=POINTS)
	point_generation(axs[0,5], N=N, iter=-2, show=POINTS)
	point_generation(axs[0,6], N=N, iter=0, show=POINTS)
	axs[0,0].set_title("CNJ")
	axs[0,1].set_title("Iterative CNJ")
	axs[0,2].set_title("Correlated Multi-Jittering")
	axs[0,3].set_title("Smoothed Jittering")
	axs[0,4].set_title("Blue Nets")
	axs[0,5].set_title("CCVT")
	axs[0,6].set_title("Poisson Disk Sampling")
	P = point_generation(axs[1,0], N=N, dt=dt, noise_scale=s, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,0].plot(rp[0],rp[1])
	P = point_generation(axs[1,1], N=N, dt=dt_iter, iter=iter, noise_scale=s_iter, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,1].plot(rp[0],rp[1])
	P = point_generation(axs[1,2], N=N, iter=-1, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,2].plot(rp[0],rp[1])
	P = point_generation(axs[1,3], N=N, iter=-3, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,3].plot(rp[0],rp[1])
	P = point_generation(axs[1,4], N=N, iter=-4, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,4].plot(rp[0],rp[1])
	P = point_generation(axs[1,5], N=N, iter=-2, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,5].plot(rp[0],rp[1])
	P = point_generation(axs[1,6], N=N, iter=0, show=FOURIER)
	e,a,q,rp = psa_point_stats(P)
	axs[2,6].plot(rp[0],rp[1])
	pyplot.savefig("figure_4.pdf")

if __name__ == "__main__":
	##### Code for various experiments below. Uncomment as appropriate.
	figure4(N=1000)
	# table2()