"""Prepare independent cells and run prespecified A/B/C validation campaigns.

This is opt-in scientific validation, not part of the quick PR test suite.
The preparation command produces frozen inputs and a reviewable campaign file.
Inspect pilot convergence before freezing production lengths and margins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from rdkit.Chem import AllChem

from ..core import calc, poly, utils
from ..ff.gaff import GAFF
from ..ff.gaff2 import GAFF2
from ..ff.gaff2_mod import GAFF2_mod
from .gromacs import _write_molecule, _read_molecule
from .preset import eq
from .validation import autocorrelation_time, block_means, compare_properties


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(destination, degree=20, chains=16, replicas=5, seed=1701):
    if degree < 3 or chains < 1 or replicas < 5:
        raise ValueError('Use degree >=3, chains >=1 and at least 5 replicas')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    cases = []
    for name, smiles in {'pe': '*CC*', 'ps': '*CC(*)c1ccccc1', 'pmma': '*CC(*)(C)C(=O)OC'}.items():
        for ff_class in (GAFF, GAFF2, GAFF2_mod):
            ff = ff_class()
            case = {'name': name+'_'+ff.name, 'replicas': []}
            folder = destination/case['name']
            folder.mkdir()
            for replica in range(replicas):
                current_seed = seed+replica
                np.random.seed(current_seed)
                unit = utils.mol_from_smiles(smiles, coord=False)
                terminal = utils.mol_from_smiles('*C', coord=False)
                for molecule in (unit, terminal):
                    if AllChem.EmbedMolecule(molecule, randomSeed=current_seed) != 0:
                        raise RuntimeError('Embedding failed')
                chain = poly.terminate_mols(poly.polymerize_mols(unit, degree, random_rot=True), terminal)
                # Assembly can produce degenerate linker geometry for some
                # seeds. Re-embed the completed graph before optimizing it.
                if AllChem.EmbedMolecule(chain, randomSeed=current_seed, useRandomCoords=True) != 0:
                    raise RuntimeError('Completed-chain embedding failed')
                if AllChem.MMFFOptimizeMolecule(chain) < 0:
                    raise RuntimeError('MMFF parameters are unavailable')
                if not np.isfinite(chain.GetConformer().GetPositions()).all():
                    raise ValueError('Non-finite chain coordinates before cell packing')
                if not ff.ff_assign(chain, charge='gasteiger'):
                    raise RuntimeError('Force field assignment failed')
                cell = poly.amorphous_cell(chain, chains, density=.1)
                if not np.isfinite(cell.GetConformer().GetPositions()).all():
                    raise ValueError('Non-finite packed cell coordinates')
                calc.set_velocity(cell, 300)
                path = folder/('replica_%d.json' % replica)
                _write_molecule(cell, path)
                case['replicas'].append({'replica_id': str(replica), 'seed': current_seed,
                    'molecule': str(path.relative_to(destination)), 'sha256': sha(path)})
            cases.append(case)
    config = {'schema': 1, 'purpose': 'pilot', 'cases': cases,
        'preset': 'EQ21step', 'equilibration': {'eq_step': 5},
        'analysis': {'init': 2000, 'width': 1000, 'f_width': 1000},
        'block_size': 100, 'minimum_effective_samples': 200,
        'thermo_freq': 1000, 'dump_freq': 1000,
        'hardware': {'omp': 1, 'mpi': 0, 'gpu': 0},
        'absolute_margins': {'nematic_order_parameter': .02}, 'reference_scales': {},
        'diffusive_regime_confirmed': False}
    dump(destination/'campaign.json', config)
    return destination/'campaign.json'


def run(config_path, output, case_name=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if (config.get('schema') != 1 or config['preset'] not in ('Annealing', 'EQ21step')
            or config.get('purpose') not in ('pilot', 'production')):
        raise ValueError('Unsupported campaign schema or preset')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'campaign.json', config)
    summary = {'config_sha256': sha(config_path), 'purpose': config['purpose'],
               'platform': platform.platform(), 'cases': {}}
    cases = [c for c in config['cases'] if case_name is None or c['name'] == case_name]
    if not cases:
        raise ValueError('Unknown case')
    for case in cases:
        records = {label: [] for label in 'ABC'}
        if len(case['replicas']) < 5 or len({r['sha256'] for r in case['replicas']}) != len(case['replicas']):
            raise ValueError('Each case requires >=5 distinct frozen input cells')
        folder = output/case['name']
        folder.mkdir()
        for replica in case['replicas']:
            source = config_path.parent/replica['molecule']
            if sha(source) != replica['sha256']:
                raise ValueError('Fixture checksum changed: %s' % source)
            for label, engine, profile in [('A', 'lammps', None), ('B', 'lammps', 'portable'),
                                           ('C', 'gromacs', 'portable')]:
                work = folder/(label+'_'+str(replica['replica_id']))
                work.mkdir()
                np.random.seed(replica['seed'])
                molecule = _read_molecule(source)
                preset = getattr(eq, config['preset'])(molecule, work_dir=str(work), solver=engine,
                    interaction_profile=profile, thermo_freq=config['thermo_freq'], dump_freq=config['dump_freq'])
                start = time.perf_counter()
                record = {'replica_id': replica['replica_id'], 'input_sha256': replica['sha256'],
                          'converged': False, 'properties': {}}
                try:
                    preset.exec(**config['equilibration'], **config['hardware'])
                    analyzer = preset.analyze()
                    values = analyzer.get_all_prop(**config['analysis'])
                    record['properties'] = {k: float(v) if np.isfinite(v) else None for k, v in values.items()}
                    df = analyzer.dfs[-1].iloc[config['analysis']['init']:]
                    diagnostics = {}
                    sampled = True
                    for field in ('TotEng', 'Volume', 'Density'):
                        tau = autocorrelation_time(df[field].to_numpy())
                        n_eff = len(df)/tau
                        blocks = block_means(df[field].to_numpy(), config['block_size'])
                        diagnostics[field] = {'autocorrelation_samples': float(tau) if np.isfinite(tau) else None,
                                              'effective_samples': float(n_eff), 'blocks': len(blocks)}
                        sampled &= n_eff >= config['minimum_effective_samples'] and config['block_size'] >= 5*tau
                    record['diagnostics'] = diagnostics
                    record['converged'] = bool(sampled and analyzer.check_eq() and
                        all(v is not None for v in record['properties'].values()) and
                        config.get('diffusive_regime_confirmed', False))
                except Exception as error:
                    record['error'] = type(error).__name__+': '+str(error)
                record['wall_seconds'] = time.perf_counter()-start
                dump(work/'validation.json', record)
                records[label].append(record)
                dump(folder/(label+'.json'), records[label])
        comparisons = {}
        for a, b in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
            comparisons[a+'_vs_'+b] = compare_properties(records[a], records[b],
                absolute_margins=config['absolute_margins'], reference_scales=config['reference_scales'])
        summary['cases'][case['name']] = comparisons
        dump(output/'report.json', summary)
    # Pilot evidence can inform a production design, but cannot certify it.
    states = {result['status'] for case in summary['cases'].values() for result in case.values()}
    passed = config['purpose'] == 'production' and states == {'pass'}
    summary['status'] = 'fail' if 'fail' in states else ('pass' if passed else 'inconclusive')
    dump(output/'report.json', summary)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('destination', type=Path)
    prep.add_argument('--degree', type=int, default=20)
    prep.add_argument('--chains', type=int, default=16)
    prep.add_argument('--replicas', type=int, default=5)
    prep.add_argument('--seed', type=int, default=1701)
    campaign = commands.add_parser('run')
    campaign.add_argument('config', type=Path)
    campaign.add_argument('--output', type=Path, required=True)
    campaign.add_argument('--case')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.destination, args.degree, args.chains, args.replicas, args.seed))
        return 0
    return 0 if run(args.config, args.output, args.case) else 1


if __name__ == '__main__':
    raise SystemExit(main())
