"""GROMACS backend for explicitly selected portable GAFF simulations.

Native top/mdp/tpr/edr/trr/cpt files are retained beside a JSON run manifest.
Public results use RadonPy's real units (angstrom, fs, kcal/mol, atm).
GROMACS is an optional executable, selected with GROMACS_EXEC or solver_path.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem

from ..core import calc, utils
from . import lammps, analysis
from .profiles import apply_profile

KCAL = 4.184
ATM = 1.01325


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _write_molecule(mol, path):
    data = utils.MolToJSON_dict(mol)
    for record, conformer in zip(data['molecules'][0]['conformers'], mol.GetConformers()):
        record['coords'] = conformer.GetPositions().tolist()
    data['gromacs_conformer_ids'] = [c.GetId() for c in mol.GetConformers()]
    _json(path, data)


def _read_molecule(path):
    # RDKit's interchange reader casts coordinates to float32. Restore the
    # original doubles after reading graph and RadonPy parameter extensions.
    data = json.loads(Path(path).read_text())
    mol = utils.JSONToMol_dict(data)
    ids = data.get('gromacs_conformer_ids', range(mol.GetNumConformers()))
    for cid, conformer, record in zip(ids, mol.GetConformers(), data['molecules'][0]['conformers']):
        conformer.SetId(cid)
        for i, xyz in enumerate(record['coords']):
            conformer.SetAtomPosition(i, xyz)
    return mol


def clear_run(manifest_path):
    """Remove only artifacts owned by this run, for the explicit tmp_clear API."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    base = Path(data['work_dir'])
    opts = data['options']
    paths = [Path(data['molecule']), base/(opts['log_file']+'.gmx.json'),
             base/((opts['outstr'] or 'final_state')+'.npz')]
    paths += [base/opts[k] for k in ('write_data', 'xtc_file') if opts[k]]
    for path in paths:
        path.unlink(missing_ok=True)
    shutil.rmtree(str(manifest_path)+'.gmx')
    manifest_path.unlink()


def _pairs14(mol):
    """Shortest graph distance, including rings; do not infer from torsions."""
    neighbors = [set(a.GetIdx() for a in atom.GetNeighbors()) for atom in mol.GetAtoms()]
    for start in range(mol.GetNumAtoms()):
        seen, frontier = {start}, {start}
        for _ in range(3):
            frontier = set().union(*(neighbors[i] for i in frontier)) - seen
            seen.update(frontier)
        for end in sorted(frontier):
            if end > start:
                yield start, end


def validate_molecule(mol):
    if not hasattr(mol, 'cell'):
        raise NotImplementedError('GROMACS currently requires a periodic orthorhombic cell')
    if mol.HasProp('ff_name') and mol.GetProp('ff_name').lower() not in ('gaff', 'gaff2', 'gaff2_mod'):
        raise NotImplementedError('Only GAFF, GAFF2 and GAFF2_mod are supported')
    for key, expected in [('pair_style', 'lj'), ('bond_style', 'harmonic'),
                          ('angle_style', 'harmonic'), ('dihedral_style', 'fourier'),
                          ('improper_style', 'cvff')]:
        if mol.HasProp(key) and mol.GetProp(key) != expected:
            raise NotImplementedError('%s=%s is unsupported' % (key, mol.GetProp(key)))
    if getattr(mol, 'cmaps', {}):
        raise NotImplementedError('CMAP is not supported')
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H' and atom.GetIsotope() >= 3:
            raise NotImplementedError('Linker/isotope-labelled hydrogens require explicit mass handling')
        if any(atom.HasProp(k) and atom.GetBoolProp(k) for k in ('CL_react', 'CL_remove')):
            raise NotImplementedError('Reactive atoms are not supported')
        for key in ('AtomicCharge', 'ff_sigma', 'ff_epsilon'):
            if not atom.HasProp(key) or not np.isfinite(atom.GetDoubleProp(key)):
                raise ValueError('Missing or invalid atom parameter: %s' % key)
        if atom.GetMass() <= 0 or atom.GetDoubleProp('ff_sigma') < 0 or atom.GetDoubleProp('ff_epsilon') < 0:
            raise ValueError('Invalid mass or LJ parameters')
    if abs(sum(a.GetDoubleProp('AtomicCharge') for a in mol.GetAtoms())) > 1e-6:
        raise NotImplementedError('Only charge-neutral cells are supported')
    if not np.all(np.isfinite(mol.GetConformer().GetPositions())):
        raise ValueError('Non-finite coordinates')


def topology(mol, electrostatics=True, shake=False):
    """Convert assigned parameters without retyping atoms or assigning charges."""
    validate_molecule(mol)
    rows = ['; Generated from RadonPy parameters; nm, kJ/mol', '[ defaults ]',
            '1 2 yes 0.5 0.8333333333333333', '', '[ atomtypes ]']
    types, atom_types = {}, []
    for atom in mol.GetAtoms():
        key = (atom.GetMass(), atom.GetDoubleProp('ff_sigma'), atom.GetDoubleProp('ff_epsilon'))
        if key not in types:
            name = 'R%d' % len(types)
            types[key] = name
            rows.append('%s %.16g 0 A %.16g %.16g' % (name, key[0], key[1]/10, key[2]*KCAL))
        atom_types.append(types[key])
    rows += ['', '[ moleculetype ]', 'RADON 3', '', '[ atoms ]']
    for i, atom in enumerate(mol.GetAtoms()):
        charge = atom.GetDoubleProp('AtomicCharge') if electrostatics else 0.0
        rows.append('%d %s 1 MOL %s %d %.16g %.16g' %
                    (i+1, atom_types[i], atom.GetSymbol(), i+1, charge, atom.GetMass()))
    rows += ['', '[ bonds ]']
    constrained = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0, k = bond.GetDoubleProp('ff_r0')/10, bond.GetDoubleProp('ff_k')*2*KCAL*100
        # LAMMPS "shake ... m 1.0" uses a mass window, not an element label.
        if shake and any(abs(mol.GetAtomWithIdx(n).GetMass()-1.0) <= 0.1 for n in (i, j)):
            constrained.append('%d %d 1 %.16g' % (i+1, j+1, r0))
            # Function 5 preserves connectivity/exclusions without a bond force.
            rows.append('%d %d 5' % (i+1, j+1))
        else:
            rows.append('%d %d 1 %.16g %.16g' % (i+1, j+1, r0, k))
    rows += ['', '[ pairs ]']
    rows += ['%d %d 1' % (a+1, b+1) for a, b in _pairs14(mol)]
    rows += ['', '[ angles ]']
    for term in getattr(mol, 'angles', {}).values():
        rows.append('%d %d %d 1 %.16g %.16g' %
                    (term.a+1, term.b+1, term.c+1, term.ff.theta0, term.ff.k*2*KCAL))
    rows += ['', '[ dihedrals ]']
    for term in getattr(mol, 'dihedrals', {}).values():
        for k, n, phase in zip(term.ff.k, term.ff.n, term.ff.d0):
            rows.append('%d %d %d %d 9 %.16g %.16g %d' %
                        (term.a+1, term.b+1, term.c+1, term.d+1, phase, k*KCAL, n))
    for term in getattr(mol, 'impropers', {}).values():
        if term.ff.d0 not in (-1, 1):
            raise ValueError('CVFF improper sign must be -1 or 1')
        rows.append('%d %d %d %d 4 %g %.16g %d' %
                    (term.a+1, term.b+1, term.c+1, term.d+1,
                     180 if term.ff.d0 == -1 else 0, term.ff.k*KCAL, term.ff.n))
    if constrained:
        rows += ['', '[ constraints ]'] + constrained
    rows += ['', '[ system ]', 'RadonPy portable cell', '', '[ molecules ]', 'RADON 1', '']
    return '\n'.join(rows)


def write_gro(mol, path, confId=0):
    """Extended precision GRO input; stock three-decimal GRO loses force accuracy."""
    cell = mol.cell
    origin = np.array([cell.xlo, cell.ylo, cell.zlo])
    xyz = (mol.GetConformer(confId).GetPositions()-origin)/10
    rows = ['RadonPy', str(mol.GetNumAtoms())]
    for i, atom in enumerate(mol.GetAtoms()):
        vel = [atom.GetDoubleProp(k)*100 if atom.HasProp(k) else 0 for k in ('vx', 'vy', 'vz')]
        rows.append('%5d%-5s%5s%5d' % (1, 'MOL', atom.GetSymbol(), (i+1) % 100000)
                    + ''.join('%15.10f' % x for x in xyz[i])
                    + ''.join('%15.11f' % x for x in vel))
    rows.append(' '.join('%.12f' % (x/10) for x in (cell.dx, cell.dy, cell.dz)))
    Path(path).write_text('\n'.join(rows)+'\n')


def read_g96(path):
    blocks, section = {}, None
    for line in Path(path).read_text().splitlines():
        if line.strip() == 'END':
            section = None
        elif line.strip() in ('POSITION', 'POSITIONRED', 'VELOCITY', 'VELOCITYRED', 'BOX'):
            section = line.strip()
            blocks[section] = []
        elif section and line.strip() and not line.startswith('#'):
            blocks[section].append([float(x) for x in line.split()[-3:]])
    pos = np.asarray(blocks.get('POSITION', blocks.get('POSITIONRED')))*10
    vel = np.asarray(blocks.get('VELOCITY', blocks.get('VELOCITYRED', np.zeros_like(pos))))/100
    box = np.asarray(blocks['BOX']).ravel()*10
    if len(box) != 3:
        raise NotImplementedError('Only orthorhombic boxes are supported')
    return pos, vel, box


def whole_positions(mol, coordinates, lengths):
    """Reconstruct each bonded molecule without changing atom order."""
    result = np.array(coordinates, copy=True)
    visited = set()
    for fragment in Chem.GetMolFrags(mol):
        pending = [fragment[0]]
        visited.add(fragment[0])
        while pending:
            i = pending.pop()
            for atom in mol.GetAtomWithIdx(i).GetNeighbors():
                j = atom.GetIdx()
                if j not in visited:
                    delta = coordinates[j]-coordinates[i]
                    result[j] = result[i]+delta-lengths*np.rint(delta/lengths)
                    visited.add(j)
                    pending.append(j)
    return result


def read_xvg(path):
    names, data = {}, []
    for line in Path(path).read_text().splitlines():
        match = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', line)
        if match:
            names[int(match[1])] = match[2]
        elif line.strip() and not line.startswith(('#', '@', '&')):
            data.append([float(x) for x in line.split()])
    if not data:
        raise ValueError('No numeric data in %s' % path)
    array = np.asarray(data)
    return array, [names.get(i, str(i)) for i in range(array.shape[1]-1)]


class Gromacs:
    def __init__(self, work_dir=None, solver_path=None, **kwargs):
        self.work_dir = str(Path(work_dir or '.').resolve())
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.solver_path = str(solver_path or os.environ.get('GROMACS_EXEC', 'gmx'))
        idx = kwargs.get('idx')
        self.input_file = 'radon_gmx%s.json' % ('' if idx is None else '_%d' % idx)
        self.output_file = 'gromacs.stdout'
        self.dat_file = None

    @property
    def get_name(self):
        return 'GROMACS'

    def _path(self, name):
        return Path(self.work_dir) / name

    def _command(self, args, cwd, stdin=None, mpi=0, omp=1):
        executable = shutil.which(self.solver_path)
        if executable is None:
            raise FileNotFoundError('GROMACS executable not found: %s; set GROMACS_EXEC' % self.solver_path)
        command = [executable] + [str(a) for a in args]
        if mpi > 1:
            command = ['mpirun', '-np', str(mpi)] + command
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(max(1, omp))
        cp = subprocess.run(command, cwd=cwd, input=stdin, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        with (Path(cwd)/'commands.log').open('a') as stream:
            stream.write(json.dumps(command)+'\n'+cp.stdout+'\n')
        if cp.returncode:
            raise RuntimeError('GROMACS failed (%s); see %s\n%s' %
                               (cp.returncode, Path(cwd)/'commands.log', cp.stdout[-4000:]))
        return cp

    def get_version(self):
        return self._command(['--version'], self.work_dir).stdout

    def make_dat(self, mol, confId=0, file_name=None, dir_name=None, velocity=True, temp=300, **kwargs):
        validate_molecule(mol)
        copied = utils.deepcopy_mol(mol)
        if velocity and not all(a.HasProp(k) for a in copied.GetAtoms() for k in ('vx', 'vy', 'vz')):
            calc.set_velocity(copied, temp)
        self.dat_file = str(Path(dir_name or self.work_dir) / ((file_name or 'radon_gmx_molecule')+'.json'))
        _write_molecule(copied, self.dat_file)
        self.confId = confId
        return self.dat_file

    def make_input(self, md, file_name=None, dir_name=None):
        apply_profile(md)
        if md.interaction_profile != 'portable':
            raise ValueError("GROMACS requires interaction_profile='portable'; legacy CHARMM switching is not equivalent")
        if not md.pbc or md.units != 'real' or md.boundary != 'p p p':
            raise NotImplementedError('Only periodic real-unit simulations are supported')
        if md.drude or md.add or md.add_f:
            raise NotImplementedError('Drude and raw LAMMPS commands are unsupported')
        if md.atom_style != 'full' or md.bc_flag or any(getattr(md, key) != value for key, value in
                [('bond_style', 'harmonic'), ('angle_style', 'harmonic'),
                 ('dihedral_style', 'fourier'), ('improper_style', 'cvff')]):
            raise NotImplementedError('Unsupported topology or bonded interaction settings')
        if md.thermo_freq < 1 or md.dump_freq < 1:
            raise ValueError('Output frequencies must be positive')
        if md.pair_style not in ('lj/cut/coul/long', 'lj/cut') or md.dielectric != 1.0:
            raise NotImplementedError('Unsupported nonbonded settings')
        for wf in md.wf:
            self._validate_workflow(wf)
        if self.dat_file is None:
            raise ValueError('Call make_dat before make_input')
        name = str(Path(dir_name or self.work_dir) / (file_name or self.input_file))
        if Path(name).resolve() == Path(self.dat_file).resolve():
            raise ValueError('Input manifest and molecule snapshot must use different paths')
        options = {k: v for k, v in vars(md).items() if k not in ('mol', 'wf') and not k.startswith('_')}
        _json(name, {'schema': 1, 'engine': 'gromacs', 'molecule': self.dat_file,
                     'work_dir': self.work_dir,
                     'confId': self.confId, 'options': options,
                     'workflow': [vars(w) for w in md.wf]})
        self.input_file = name
        md._gromacs_manifest = name
        return name

    @staticmethod
    def _validate_workflow(wf):
        if wf.type == 'minimize':
            if wf.min_style not in ('cg', 'sd'):
                raise NotImplementedError('GROMACS minimization supports cg and sd')
            return
        if wf.ensemble not in ('nve', 'nvt', 'npt'):
            raise NotImplementedError('Unsupported ensemble: %s' % wf.ensemble)
        for flag in ('bcreate', 'efield', 'dipole', 'variable', 'timeave', 'rerun', 'rattle', 'momentum', 'p_aniso'):
            if getattr(wf, flag, False):
                raise NotImplementedError('Unsupported workflow option: %s' % flag)
        if wf.nve_limit or wf.add or wf.add_f:
            raise NotImplementedError('nve/limit and raw LAMMPS commands are unsupported')
        if wf.thermo_freq is not None or wf.thermo_style:
            raise NotImplementedError('Set output frequency on MD; per-stage thermo overrides are unsupported')
        if wf.p_couple is not None or any(getattr(wf, k, None) is not None
            for k in ('px_start', 'py_start', 'pz_start', 'px_stop', 'py_stop', 'pz_stop')):
            raise NotImplementedError('Only isotropic pressure control is supported')
        if wf.ensemble != 'nve' and wf.thermostat != 'Nose-Hoover':
            raise NotImplementedError('Select Nose-Hoover for this backend')
        if wf.ensemble == 'npt' and wf.barostat != 'Parrinello-Rahman':
            raise ValueError("GROMACS NPT requires barostat='Parrinello-Rahman' explicitly")
        if wf.ensemble == 'npt' and wf.p_start != wf.p_stop:
            raise NotImplementedError('Pressure ramps are unsupported')
        if wf.ensemble == 'npt' and (not np.isfinite(wf.compressibility) or wf.compressibility <= 0):
            raise ValueError('Compressibility must be finite and positive (bar^-1)')
        if wf.deform and (wf.deform != 'final' or wf.deform_axis != 'xyz' or wf.ensemble != 'nvt'):
            raise NotImplementedError('Only isotropic final-box packing deformation is supported')
        if wf.step <= 0 or wf.time_step <= 0:
            raise ValueError('MD step count and timestep must be positive')

    def _mdp(self, md, wf, mol, initial, elapsed):
        energy = md.pair_style != 'lj/cut'
        cutoff = float(md.cutoff_in)/10
        lengths = np.array([mol.cell.dx, mol.cell.dy, mol.cell.dz])/10
        if np.min(lengths) <= 2*cutoff:
            raise ValueError('Each box length must exceed twice the cutoff')
        sample_stride = math.gcd(md.dump_freq, md.thermo_freq)
        need_observables = getattr(wf, 'rg', False) or getattr(wf, 'msd', False)
        opts = {'cutoff-scheme': 'Verlet', 'pbc': 'xyz', 'rlist': cutoff,
                'verlet-buffer-tolerance': -1, 'nstlist': 1,
                'coulombtype': 'PME' if energy else 'Cut-off', 'rcoulomb': cutoff,
                'coulomb-modifier': 'None', 'vdwtype': 'Cut-off', 'vdw-modifier': 'None',
                'rvdw': cutoff, 'DispCorr': 'no', 'fourierspacing': 0.08,
                'pme-order': 6, 'ewald-rtol': 1e-8, 'constraints': 'none',
                'constraint-algorithm': 'lincs', 'lincs-order': 8, 'lincs-iter': 2,
                'nstenergy': md.thermo_freq, 'nstcalcenergy': 1, 'nstlog': md.thermo_freq,
                'nstxout-compressed': sample_stride if md.xtc_file or md.dump_file or need_observables else 0,
                'compressed-x-precision': 1000000, 'comm-mode': 'Linear',
                'nstcomm': 1000, 'tinit': elapsed}
        if wf.type == 'minimize':
            opts.update(integrator='cg' if wf.min_style == 'cg' else 'steep',
                        nsteps=wf.maxiter, emtol=wf.ftol*KCAL*10, emstep=0.001,
                        nstxout=1, nstvout=1, nstfout=1, nstenergy=1,
                        **{'nstxout-compressed': 0})
        else:
            opts.update(integrator='md-vv' if wf.ensemble == 'nve' else 'md',
                        nsteps=wf.step, dt=wf.time_step/1000,
                        nstxout=md.dump_freq if md.dump_file else 0,
                        nstvout=md.dump_freq if md.dump_file else 0, nstfout=1 if wf.step == 0 else 0,
                        continuation='no',
                        **{'gen-vel': 'no'})
            if wf.ensemble != 'nve':
                opts.update({'tcoupl': 'nose-hoover', 'tc-grps': 'System',
                             'tau-t': wf.t_dump/1000, 'ref-t': wf.t_start,
                             'nh-chain-length': 1})
                if wf.t_start != wf.t_stop:
                    opts.update(annealing='single', **{'annealing-npoints': 2,
                        'annealing-time': '%g %g' % (elapsed, elapsed+wf.step*wf.time_step/1000),
                        'annealing-temp': '%g %g' % (wf.t_start, wf.t_stop)})
            if wf.ensemble == 'npt':
                opts.update({'pcoupl': 'Parrinello-Rahman', 'pcoupltype': 'isotropic',
                             'tau-p': wf.p_dump/1000, 'ref-p': wf.p_start*ATM,
                             'compressibility': getattr(wf, 'compressibility', 4.5e-5)})
            if wf.deform:
                target = (wf.deform_fin_hi-wf.deform_fin_lo)/10
                if target <= 2*cutoff:
                    raise ValueError('Final packing cell is smaller than twice the cutoff')
                rates = (target-lengths)/(wf.step*wf.time_step/1000)
                opts['deform'] = ' '.join('%g' % x for x in [*rates, 0, 0, 0])
        return '\n'.join('%s = %s' % item for item in opts.items())+'\n'

    def exec(self, input_file=None, output_file=None, omp=1, mpi=0, gpu=0,
             return_cmd=False, intel='off', opt='off', resume=False):
        if gpu not in (0, 1):
            raise NotImplementedError('The initial GROMACS backend supports zero or one GPU')
        if return_cmd:
            raise NotImplementedError('Use the manifest and native stage files for GROMACS execution')
        if isinstance(input_file, list):
            raise NotImplementedError('Use quick_min_all for conformers')
        manifest_path = self._path(input_file or self.input_file)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') != 1 or manifest.get('engine') != 'gromacs':
            raise ValueError('Unsupported GROMACS manifest schema')
        md = SimpleNamespace(**manifest['options'])
        mol = _read_molecule(manifest['molecule'])
        cid = manifest['confId']
        root = Path(str(manifest_path)+'.gmx')
        root.mkdir(exist_ok=True)
        digest = hashlib.sha256(manifest_path.read_bytes()+Path(manifest['molecule']).read_bytes()).hexdigest()
        identity = root/'input.sha256'
        if resume and (not identity.exists() or identity.read_text() != digest):
            raise ValueError('Cannot resume: inputs have changed or run identity is missing')
        if not resume:
            identity.write_text(digest)
        # Never read a previous successful run's result after a failed new run.
        result_path = self._path(md.log_file+'.gmx.json')
        result_path.unlink(missing_ok=True)
        stages = manifest['workflow'] or [{'type': 'md', 'ensemble': 'nve', 'step': 0,
                                           'time_step': 1, 'shake': False, 'set_init_velocity': None,
                                           'deform': None}]
        dfs, xtcs, observables, elapsed, step_offset = [], [], [], 0.0, 0
        origin = np.array([mol.cell.xlo, mol.cell.ylo, mol.cell.zlo])
        init_temp = md.set_init_velocity
        version = self.get_version()
        if mpi > 1 and 'thread_mpi' in version:
            raise ValueError('External MPI requires an MPI-enabled GROMACS executable')
        for index, spec in enumerate(stages):
            wf = SimpleNamespace(**spec)
            stage = root/('stage_%03d' % index)
            stage.mkdir(exist_ok=True)
            completed = stage/'completed.json'
            continuing = resume and (stage/'run.cpt').exists()
            if not resume and any(stage.glob('run.*')):
                raise FileExistsError('Run already exists: %s; use resume=True or a new directory' % stage)
            if not (resume and completed.exists()):
                requested_temp = getattr(wf, 'set_init_velocity', None)
                if isinstance(requested_temp, (int, float)) and not isinstance(requested_temp, bool):
                    calc.set_velocity(mol, requested_temp)
                elif index == 0 and isinstance(init_temp, (int, float)) and not isinstance(init_temp, bool):
                    calc.set_velocity(mol, init_temp)
                if not continuing:
                    (stage/'system.top').write_text(topology(mol, md.pair_style != 'lj/cut', getattr(wf, 'shake', False)))
                    write_gro(mol, stage/'input.gro', cid)
                    text = self._mdp(md, wf, mol, index == 0, elapsed)
                    (stage/'run.mdp').write_text(text)
                    self._command(['grompp', '-f', 'run.mdp', '-c', 'input.gro', '-p', 'system.top',
                                   '-o', 'run.tpr'], stage)
                command = ['mdrun', '-s', 'run.tpr', '-deffnm', 'run', '-c', 'final.g96',
                           '-ntomp', str(max(1, omp)), '-nb', 'gpu' if gpu else 'cpu']
                if 'thread_mpi' in version:
                    command += ['-ntmpi', '1']
                if continuing:
                    command += ['-cpi', 'run.cpt', '-append']
                cp = self._command(command, stage, mpi=mpi, omp=omp)
                menu = self._command(['energy', '-f', 'run.edr', '-o', 'energy.xvg', '-dp'],
                                     stage, stdin='Potential\n0\n').stdout
                terms = re.findall(r'(?<!\S)(\d+)\s+(?=[A-Za-z])',
                                   menu.split('Select the terms')[1].split('Last energy frame')[0])
                self._command(['energy', '-f', 'run.edr', '-o', 'energy.xvg', '-dp'],
                              stage, stdin=' '.join(terms)+'\n0\n')
                if not (stage/'energy.xvg').exists():
                    # Select all terms by names discovered from the energy menu.
                    raise RuntimeError('GROMACS energy extraction produced no data')
            # Extraction is repeatable; checkpoint resume preserves native time.
            df = self._energy_frame(stage, mol, wf, elapsed, step_offset)
            if wf.type == 'md' and abs(df['Time'].iat[-1] -
                    (elapsed*1000+wf.step*wf.time_step)) > max(.001, wf.time_step*.25):
                raise RuntimeError('GROMACS stopped before the requested final time; use resume=True')
            obs = self._observables(stage, md, wf, mol, df, cid) if (stage/'run.xtc').exists() else {}
            observables.append(obs)
            dfs.append(df)
            xyz, vel, lengths = read_g96(stage/'final.g96')
            if (stage/'whole.xtc').exists():
                import mdtraj
                with mdtraj.formats.XTCTrajectoryFile(str(stage/'whole.xtc')) as trajectory:
                    trajectory.seek(len(trajectory)-1)
                    reference = trajectory.read(1)[0][0].astype(float)*10
                # Use the trajectory only to recover image counters; retain
                # the higher-precision final G96 coordinates themselves.
                xyz += lengths*np.rint((reference-xyz)/lengths)
            else:
                xyz = whole_positions(mol, xyz, lengths)
            if wf.type == 'minimize':
                vel = np.array([[a.GetDoubleProp(k) if a.HasProp(k) else 0
                                 for k in ('vx', 'vy', 'vz')] for a in mol.GetAtoms()])
            xyz += origin
            for i, atom in enumerate(mol.GetAtoms()):
                mol.GetConformer(cid).SetAtomPosition(i, xyz[i].tolist())
                for axis, value in zip(('vx', 'vy', 'vz'), vel[i]):
                    atom.SetDoubleProp(axis, float(value))
            mol.cell = utils.Cell(origin[0]+lengths[0], origin[0], origin[1]+lengths[1], origin[1],
                                  origin[2]+lengths[2], origin[2])
            if (stage/'whole.xtc').exists():
                xtcs.append(str(stage/'whole.xtc'))
            _write_molecule(mol, stage/'final.json')
            if wf.type == 'md':
                elapsed += wf.step*wf.time_step/1000
                step_offset += wf.step
            _json(completed, {'elapsed_ps': elapsed, 'step': step_offset})
        # Forces are required for single-point evaluation and minimization only.
        force = np.full((mol.GetNumAtoms(), 3), np.nan)
        if not manifest['workflow'] or stages[-1]['type'] == 'minimize':
            self._command(['traj', '-f', 'run.trr', '-s', 'run.tpr', '-of', 'force.xvg', '-fp'],
                          stage, stdin='System\n')
            data, _ = read_xvg(stage/'force.xvg')
            force = data[-1, 1:].reshape(-1, 3)/(KCAL*10)
        cell = np.column_stack([origin, origin+lengths])
        wrapped = (xyz-origin) % lengths + origin
        state_name = md.outstr or 'final_state'
        np.savez(self._path(state_name+'.npz'), unwrapped=xyz, wrapped=wrapped,
                 cell=cell, velocity=vel, force=force)
        if md.write_data:
            # A real LAMMPS data export supports existing structure/charge readers.
            lammps.MolToLAMMPSdata(mol, str(self._path(md.write_data)), confId=cid)
        if md.xtc_file and xtcs:
            self._command(['trjcat', '-f', *xtcs, '-o', str(self._path(md.xtc_file))], root)
        payload = {'schema': 1, 'engine': 'gromacs', 'version': version,
                   'interaction_profile': md.interaction_profile, 'manifest': str(manifest_path),
                   'charges': [a.GetDoubleProp('AtomicCharge') for a in mol.GetAtoms()],
                   'frames': [df.reset_index().to_dict(orient='list') for df in dfs],
                   'observables': observables,
                   'state': str(self._path(state_name+'.npz')),
                   'trajectory': str(self._path(md.xtc_file)) if md.xtc_file else None,
                   'molecule': str(stage/'final.json')}
        _json(result_path, payload)
        return subprocess.CompletedProcess([self.solver_path, 'mdrun'], 0)

    def _energy_frame(self, stage, mol, wf, elapsed, step_offset):
        values, names = read_xvg(stage/'energy.xvg')
        raw = pd.DataFrame(values[:, 1:], columns=names)
        def val(name, default=0):
            return raw[name].to_numpy() if name in raw else np.full(len(raw), default)
        result = pd.DataFrame({'Time': values[:, 0]*1000})
        mapping = {'Bond': 'E_bond', 'Angle': 'E_angle', 'Proper Dih.': 'E_dihed',
                   'Per. Imp. Dih.': 'E_impro', 'Potential': 'PotEng', 'Kinetic En.': 'KinEng',
                   'Total Energy': 'TotEng', 'Enthalpy': 'Enthalpy', 'Coul. recip.': 'E_long'}
        for key, target in mapping.items():
            result[target] = val(key)/KCAL
        result['E_vdwl'] = (val('LJ (SR)')+val('LJ-14'))/KCAL
        result['E_coul'] = (val('Coulomb (SR)')+val('Coulomb-14'))/KCAL
        result['Temp'] = val('Temperature')
        result['Press'] = val('Pressure')/ATM
        result['Volume'] = val('Volume', mol.cell.volume/1000)*1000
        result['Density'] = calc.mol_mass(mol)/6.02214076e23/result['Volume']*1e24
        scale = np.cbrt(result['Volume'].to_numpy()/mol.cell.volume)
        for axis, length in zip('XYZ', (mol.cell.dx, mol.cell.dy, mol.cell.dz)):
            result['L'+axis.lower()] = val('Box-'+axis)*10 if 'Box-'+axis in raw else length*scale
        for a, b in [('X', 'X'), ('Y', 'Y'), ('Z', 'Z'), ('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]:
            result['P'+a.lower()+b.lower()] = val('Pres-'+a+b)/ATM
        if 'Total Energy' not in raw:
            result['TotEng'] = result['PotEng']+result['KinEng']
        result.index = np.rint((values[:, 0]-elapsed)*1000/getattr(wf, 'time_step', 1)).astype(int)+step_offset
        result.index.name = 'Step'
        return result

    def _observables(self, stage, md, wf, mol, df, cid):
        """Sample trajectory observables at the recorded physical times.

        These are instantaneous samples, unlike LAMMPS's fix ave/time windows;
        reports must compare statistical estimates, never framewise MSD values.
        """
        import mdtraj
        self._command(['trjconv', '-s', 'run.tpr', '-f', 'run.xtc', '-pbc', 'nojump',
                       '-o', 'whole.xtc'], stage, stdin='System\n')
        if not (getattr(wf, 'rg', False) or getattr(wf, 'msd', False)):
            return {}
        pdb = stage/'topology.pdb'
        utils.MolToPDBFile(mol, str(pdb))
        fragments = [np.asarray(f) for f in Chem.GetMolFrags(mol)]
        masses = np.array([a.GetMass() for a in mol.GetAtoms()])
        origin = np.array([mol.cell.xlo, mol.cell.ylo, mol.cell.zlo])
        initial = mol.GetConformer(cid).GetPositions()-origin
        times, msd, rg = [], [], []
        for chunk in mdtraj.iterload(str(stage/'whole.xtc'), top=str(pdb), chunk=1000):
            for time_ps, xyz_nm in zip(chunk.time, chunk.xyz):
                xyz = xyz_nm.astype(float)*10
                delta = xyz-initial
                delta -= np.average(delta, axis=0, weights=masses)
                times.append(float(time_ps)*1000)
                msd.append(float(np.mean(np.sum(delta**2, axis=1))))
                rg.append([float(np.sqrt(np.average(np.sum((xyz[f]-np.average(xyz[f], axis=0,
                    weights=masses[f]))**2, axis=1), weights=masses[f]))) for f in fragments])
        if getattr(wf, 'msd', False):
            target_times = df['Time'].to_numpy()
            positions = np.searchsorted(times, target_times)
            positions = np.minimum(positions, len(times)-1)
            previous = np.maximum(positions-1, 0)
            positions = np.where(np.abs(np.asarray(times)[previous]-target_times) <
                                 np.abs(np.asarray(times)[positions]-target_times), previous, positions)
            if not np.allclose(np.asarray(times)[positions], df['Time'], atol=.01, rtol=1e-6):
                raise ValueError('MSD trajectory and energy samples do not share timestamps')
            df['v_msd'] = np.asarray(msd)[positions]
        return {'time_fs': times, 'rg_angstrom': rg, 'msd_angstrom2': msd,
                'estimator': 'instantaneous samples; mass-weighted molecular Rg; COM-corrected atom MSD'}

    def read_traj_simple(self, filename):
        with np.load(str(filename)+'.npz') as data:
            return tuple(data[k].copy() for k in ('unwrapped', 'wrapped', 'cell', 'velocity', 'force'))

    read_traj_last = read_traj_simple

    def run(self, md, mol=None, confId=0, input_file=None, output_file=None, last_data=None,
            last_str=None, omp=1, mpi=0, gpu=0, intel='off', opt='off', resume=False):
        if last_data:
            md.write_data = last_data
        if last_str:
            md.outstr = last_str
        if not resume:
            self.make_dat(mol, confId=confId, file_name=md.dat_file)
            self.make_input(md, file_name=input_file)
        self.exec(input_file=input_file, omp=omp, mpi=mpi, gpu=gpu, resume=resume)
        metadata = json.loads(self._path(md.log_file+'.gmx.json').read_text())
        return _read_molecule(metadata['molecule'])


class Analyze(analysis.Analyze):
    """GROMACS readers feeding the existing RadonPy property calculations."""
    def read_log(self, log_file, ignore_log=None):
        self.metadata = json.loads(Path(str(log_file)+'.gmx.json').read_text())
        return [pd.DataFrame(frame).set_index('Step') for frame in self.metadata['frames']]

    def get_partial_charges(self, dat_file=None):
        self.charges = np.asarray(self.metadata['charges'])
        return self.charges

    def calc_rg(self, rg_file=None, ave_type=0, init=-2000, last=None):
        values = np.asarray(self.metadata['observables'][-1].get('rg_angstrom', []))
        values = values[init:last]
        if len(values) < 2:
            raise ValueError('Insufficient Rg samples')
        mean, sd = values.mean(axis=ave_type), values.std(axis=ave_type)
        n = len(values)
        return {'mean': mean, 'sd': sd, 'se': sd/np.sqrt(n), 'sd_max': float(sd.max()),
                'se_max': float(sd.max()/np.sqrt(n)), 'mean_mean': float(mean.mean()),
                'mean_sd': float(mean.std()), 'mean_se': float(mean.std()/np.sqrt(len(mean)))}

    def has_rg_data(self):
        return bool(self.metadata.get('observables', [{}])[-1].get('rg_angstrom'))

    def read_traj(self, traj_file=None, pdb_file=None, traj_type=None):
        import mdtraj
        self.traj = mdtraj.load(traj_file or self.metadata['trajectory'], top=pdb_file or self.pdb_file)
        return self.traj
