import os
import subprocess
import numpy

def dyadic_wrapper(power_of_two_num_points, sigma, opt_steps=1000):
    dyadicbin_path = os.path.join(os.path.dirname(__file__), 'dyadic_mod/build/net-optimize-pointers')
    assert os.path.exists(dyadicbin_path) or os.path.exists(dyadicbin_path + '.exe')
    tempfile_sites = 'net.txt'
    if os.path.exists(tempfile_sites):
        os.remove(tempfile_sites)
    cmd_str = dyadicbin_path + " -q v -s " + str(sigma) + " -n " + str(opt_steps) + " " + str(2**power_of_two_num_points)
    print(subprocess.run(cmd_str, shell=True))
    points = []
    with open(tempfile_sites) as filesites:
    	lines = filesites.readlines()
    	for line in lines[1:]:
    		fields = line.strip('\n').split(' ')
    		if len(fields) == 2:
    			points.append(numpy.array([float(fields[0]), float(fields[1])]))
    
    assert len(points) == 2**power_of_two_num_points
    return points


if __name__ == '__main__':
    import psa_wrapper
    num_samples = 20
    samples = []
    for sample in range(num_samples):
        points = dyadic_wrapper(power_of_two_num_points=10, sigma=0.5)
        samples.append(points)
    (effnyquist, oscillations, data_rp, data_ani, data_rdf) = psa_wrapper.psa_wrapper(samples)
    psa_wrapper.plot_data(data_ani, 'Frequency', 'Anisotropy', 'plot_ani')