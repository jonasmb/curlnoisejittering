import os
import subprocess
import numpy

def ccvt_wrapper(num_points, random_seed=42):
    ccvtbin_path = os.path.join(os.path.dirname(__file__), 'ccvt_mod/build/ccvt')
    tempfile_sites = 'sites.txt'
    print(os.path.curdir,  os.path.exists(ccvtbin_path))
    assert os.path.exists(ccvtbin_path)
    if os.path.exists(tempfile_sites):
        os.remove(tempfile_sites)
    cmd_str = ccvtbin_path + " " + str(num_points) + " " + str(random_seed)
    print(subprocess.run(cmd_str, shell=True))
    points = []
    with open(tempfile_sites) as filesites:
    	lines = filesites.readlines()
    	for line in lines:
    		fields = line.strip('\n').split(' ')
    		if len(fields) == 2:
    			points.append(numpy.array([float(fields[0]), float(fields[1])]))
    
    assert len(points) == num_points
    return points


if __name__ == '__main__':
    import psa_wrapper
    num_samples = 100
    samples = []
    for sample in range(num_samples):
        points = ccvt_wrapper(num_points=512, random_seed=sample)
        samples.append(points)
    (effnyquist, oscillations, data_rp, data_ani, data_rdf) = psa_wrapper.psa_wrapper(samples)
    psa_wrapper.plot_data(data_ani, 'Frequency', 'Anisotropy', 'plot_ani')