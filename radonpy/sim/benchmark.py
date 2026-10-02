"""Opt-in, repeated MD timing from a supplied equilibrated periodic cell.

Both engines use the portable Hamiltonian, identical output frequency, timestep,
constraints and resources. Native engine throughput and complete wall time are
reported separately. This command does not assert scientific equivalence.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import shutil
import time

from ..core import poly, utils
from . import md
from .gromacs import _read_molecule
from .md_wrapper import MD_solver
from .validation_campaign import sha


def run(source, destination, sizes=(10000, 100000), repeats=3, steps=100000,
        warmup=10000, omp=1, mpi=0, gpu=0):
    if repeats < 3 or steps < 1 or warmup < 1 or not sizes or min(sizes) < 1:
        raise ValueError('Use >=3 repeats and positive warmup/production lengths')
    if omp < 1 or mpi < 0 or gpu not in (0, 1):
        raise ValueError('Use omp >=1, mpi >=0 and gpu=0 or 1')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    original = _read_molecule(source)
    report = {'fixture_sha256': sha(source), 'platform': platform.platform(),
              'resources': {'omp': omp, 'mpi': mpi, 'gpu': gpu},
              'time_step_fs': 1, 'constraints': 'bonds to mass-1 hydrogens',
              'interaction_profile': 'portable', 'output_interval_steps': 1000,
              'scientific_equivalence': 'not established by timing', 'systems': []}
    report['cpu'] = platform.processor()
    if shutil.which('nvidia-smi'):
        report['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
    for target in sizes:
        tiles = [1, 1, 1]
        while original.GetNumAtoms()*tiles[0]*tiles[1]*tiles[2] < target:
            axis = min(range(3), key=lambda i: tiles[i])
            tiles[axis] += 1
        molecule = poly.super_cell(original, x=tiles[0], y=tiles[1], z=tiles[2])
        system = {'target_atoms': target, 'atoms': molecule.GetNumAtoms(), 'tiles': tiles, 'engines': {}}
        for engine in ('lammps', 'gromacs'):
            trials = []
            for repeat in range(repeats):
                work = destination/('%s_%s_%s' % (target, engine, repeat))
                work.mkdir()
                solver = MD_solver(engine, work_dir=str(work))
                if engine == 'lammps' and ((gpu and not solver.package['gpu']) or
                        (omp > 1 and not solver.package['omp'])):
                    raise ValueError('LAMMPS build cannot use the requested GPU/OpenMP resources')
                state = utils.deepcopy_mol(molecule)
                for phase, length in [('warmup', warmup), ('production', steps)]:
                    options = md.MD(mol=state, interaction_profile='portable',
                        input_file=phase+'.in', dat_file=phase+'.data', log_file=phase+'.log',
                        outstr=phase+'.dump', write_data=phase+'_last.data',
                        dump_file=None, rst=False, thermo_freq=1000)
                    options.add_md('nvt', length, time_step=1, shake=True)
                    solver.make_dat(state, file_name=options.dat_file)
                    start = time.perf_counter()
                    state = solver.run(options, mol=state, input_file=phase+'.in', omp=omp, mpi=mpi, gpu=gpu)
                    wall = time.perf_counter()-start
                    if phase == 'production':
                        if engine == 'gromacs':
                            text = (work/'production.in.gmx/stage_000/run.log').read_text()
                            match = re.search(r'Performance:\s+([\d.]+)', text)
                            native = float(match[1]) if match else None
                        else:
                            text = (work/'production.log').read_text()
                            loops = re.findall(r'Loop time of ([\d.eE+-]+) on .*? for (\d+) steps', text)
                            seconds = sum(float(t) for t, n in loops if int(n) > 0)
                            native = steps*1e-6/seconds*86400 if seconds else None
                        trials.append({'wall_seconds': wall, 'native_ns_day': native,
                                       'end_to_end_ns_day': steps*1e-6/wall*86400})
            system['engines'][engine] = {'version': str(solver.get_version()), 'trials': trials,
                'median_end_to_end_ns_day': statistics.median(t['end_to_end_ns_day'] for t in trials)}
            native_rates = [t['native_ns_day'] for t in trials if t['native_ns_day'] is not None]
            system['engines'][engine]['median_native_ns_day'] = statistics.median(native_rates) if native_rates else None
        report['systems'].append(system)
        (destination/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('equilibrated_cell', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[10000, 100000])
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--omp', type=int, default=1)
    parser.add_argument('--mpi', type=int, default=0)
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    report = run(args.equilibrated_cell, args.output, args.sizes, args.repeats,
                 args.steps, args.warmup, args.omp, args.mpi, args.gpu)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
