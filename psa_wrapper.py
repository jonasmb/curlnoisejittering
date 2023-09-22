import subprocess
import os
import numpy
import shutil
import matplotlib
import matplotlib.pyplot as plt


def read_psa_raw_data(filename):
    field_1 = []
    field_2 = []
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip('\n').split(' ')
            assert len(fields) == 2
            field_1.append(float(fields[0]))
            field_2.append(float(fields[1]))
    return (field_1, field_2)


def psa_wrapper(samples, temp_post_fix = "", points_id=0):
    psabin_path = os.path.join(os.path.dirname(__file__), 'psa_mod/build/psabin')
    temp_path = os.path.join(os.path.dirname(__file__), 'temp' + temp_post_fix)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    print(os.path.curdir, os.path.exists(psabin_path))
    assert os.path.exists(psabin_path) or os.path.exists(psabin_path + '.exe')
    points_name = 'points' + str(points_id) + '_'
    points_path = os.path.join(temp_path, points_name)
    for i in range(len(samples)):
        points = samples[i]
        points_psa_txt = points_path + str(i) + '.txt'
        with open(points_psa_txt, 'w') as f:
            f.write(str(len(points)) + '\n')
            for point in points:
                f.write(str(point[0]) + ' ' + str(point[1]) + '\n')
    average = len(samples) > 1
    flags = ' '
    if average:
        flags +='--avg'
    flags += ' --raw --ani --rp --rdf --spectral '
    if average:
        path_ani = os.path.join(os.path.dirname(__file__), 'avg_ani.txt')
    else:
        path_ani = os.path.join(os.path.dirname(__file__), points_name + '0_ani.txt')
    if os.path.exists(path_ani):
        os.remove(path_ani)
    if average:
        path_rdf = os.path.join(os.path.dirname(__file__), 'avg_rdf.txt')
    else:
        path_rdf = os.path.join(os.path.dirname(__file__), points_name + '0_rdf.txt')
    if os.path.exists(path_rdf):
        os.remove(path_rdf)
    if average:
        path_rp = os.path.join(os.path.dirname(__file__), 'avg_rp.txt')
    else:
        path_rp = os.path.join(os.path.dirname(__file__), points_name + '0_rp.txt')
    if os.path.exists(path_rp):
        os.remove(path_rp)
    if average:
        path_spectral = os.path.join(os.path.dirname(__file__), 'avg_spectral.txt')
    else:
        path_spectral = os.path.join(os.path.dirname(__file__), points_name + '0_spectral.txt')
    if os.path.exists(path_spectral):
        os.remove(path_spectral)
    if average:
        cmd_str = psabin_path + flags + points_path + '*.txt'
    else:
        cmd_str = psabin_path + flags + points_path + '0.txt'
    print(subprocess.run(cmd_str, shell=True))
    # read back data from files
    with open(path_spectral, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2
        effnyquist = float(lines[0].strip('\n'))
        oscillations = float(lines[1].strip('\n'))
        print("effnyquist = " + str(effnyquist))
        print("oscillations = " + str(oscillations))
    rp_raw_data = read_psa_raw_data(path_rp)
    ani_raw_data = read_psa_raw_data(path_ani)
    rdf_raw_data = read_psa_raw_data(path_rdf)
    os.remove(path_ani)
    os.remove(path_rdf)
    os.remove(path_rp)
    os.remove(path_spectral)
    if not average:
        #os.remove(os.path.join(os.path.dirname(__file__), points_name + '0_spec.png'))
        os.remove(points_path + '0.txt')
    return (
        effnyquist,
        oscillations,
        rp_raw_data,
        ani_raw_data,
        rdf_raw_data,
    )

def plot_data(data, xlabel, ylabel, name, to_pgf=False):
    if to_pgf:
        matplotlib.use("pgf") # for latex (see https://ctan.org/pkg/pgf)
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    fig, ax = plt.subplots()
    ax.plot(data[0], data[1], c='k')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(left=0.0)
    if to_pgf:
        fig.savefig(filename + ".pgf")
    else:
        plt.show()

if __name__ == '__main__':
    numpy.random.seed(42)
    samples = []
    num_points = 128
    num_samples = 1
    for i in range(num_samples):
        samples.append(numpy.random.random_sample((num_points, 2)))
    print(len(samples), samples[0].shape)
    (effnyquist, oscillations, data_rp, data_ani, data_rdf) = psa_wrapper(samples)
    plot_data(data_rp, 'Frequency', 'Radial Power', 'plot_rp')
    plot_data(data_ani, 'Frequency', 'Anisotropy', 'plot_ani')
    plot_data(data_rdf, 'Distance', 'RDF', 'plot_rdf')
    # heuristic measure of anisotropy derived from raw data PSA
    print("anisotropy heuristic mean =" + str(numpy.mean(data_ani[1][1:])))
