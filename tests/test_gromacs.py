import json
from pathlib import Path

import numpy as np
import pytest

from radonpy.sim import gromacs, md
from radonpy.sim.md_wrapper import MD_solver, MD_analyzer
from radonpy.core import utils


def test_conversion(molecule):
    text = gromacs.topology(molecule)
    bond = molecule.GetBondWithIdx(0)
    assert '%.16g' % (bond.GetDoubleProp('ff_k')*836.8) in text
    assert '1 2 yes 0.5 0.8333333333333333' in text
    pairs = set(gromacs._pairs14(molecule))
    assert (0, 3) in pairs and (0, 2) not in pairs
    assert len(pairs) == sum(utils.Chem.GetDistanceMatrix(molecule)[i, j] == 3
                             for i in range(molecule.GetNumAtoms())
                             for j in range(i+1, molecule.GetNumAtoms()))


def test_profile_is_explicit(molecule, tmp_path):
    solver = gromacs.Gromacs(tmp_path)
    solver.make_dat(molecule)
    with pytest.raises(ValueError, match='interaction_profile'):
        solver.make_input(md.MD(mol=molecule))
    options = md.MD(mol=molecule, interaction_profile='portable')
    options.add_md('nve', 10)
    solver.make_input(options)
    options.wf[0].nve_limit = 0.1
    with pytest.raises(NotImplementedError, match='nve/limit'):
        solver.make_input(options)


def test_unsupported_and_unknown(molecule):
    molecule.GetAtomWithIdx(0).SetDoubleProp('AtomicCharge', 1)
    with pytest.raises(NotImplementedError, match='charge-neutral'):
        gromacs.topology(molecule)
    for factory in (MD_solver, MD_analyzer):
        with pytest.raises(ValueError, match='Unknown'):
            factory('typo')


def test_extended_precision_roundtrip(molecule, tmp_path):
    molecule.GetConformer().SetAtomPosition(0, [1.12345678912, 2.33333333333, -4.23456789123])
    solver = gromacs.Gromacs(tmp_path)
    path = solver.make_dat(molecule, velocity=False)
    np.testing.assert_allclose(gromacs._read_molecule(path).GetConformer().GetPositions(),
                               molecule.GetConformer().GetPositions(), atol=1e-10, rtol=0)


@pytest.mark.engines
@pytest.mark.parametrize('fixture', ['butane_gaff2'] + [p+'_'+f for p in ('pe', 'ps', 'pmma')
                                                       for f in ('gaff', 'gaff2', 'gaff2_mod')])
def test_single_point_parity(fixture, engines, tmp_path):
    molecule = utils.JSONToMol(str(Path(__file__).parent/'fixtures'/(fixture+'.json')))
    results = {}
    for engine, executable in engines.items():
        work = tmp_path/engine
        work.mkdir()
        results[engine] = md.quick_energy(utils.deepcopy_mol(molecule), solver=engine,
            solver_path=executable, work_dir=str(work), interaction_profile='portable')
    le, lf = results['lammps']; ge, gf = results['gromacs']
    np.testing.assert_allclose(ge*4.184/molecule.GetNumAtoms(), le*4.184/molecule.GetNumAtoms(),
                               atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(gf*41.84, lf*41.84, atol=1e-2, rtol=1e-4)


@pytest.mark.engines
def test_minimization_and_conformers(molecule, engines, tmp_path):
    from rdkit import Chem
    molecule.AddConformer(Chem.Conformer(molecule.GetConformer()), assignId=True)
    for conformer, cid in zip(molecule.GetConformers(), (7, 13)):
        conformer.SetId(cid)
    result, energies, xyz = md.quick_min_all(molecule, solver='gromacs',
        solver_path=engines['gromacs'], work_dir=str(tmp_path), mpi=0,
        interaction_profile='portable', ftol=.01, maxiter=500)
    assert result.GetNumConformers() == 2
    assert [c.GetId() for c in result.GetConformers()] == [7, 13]
    assert len(energies) == len(xyz) == 2
    assert energies[0] < 14
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-8)


@pytest.mark.engines
def test_inter_molecular_and_boundary_parity(molecule, engines, tmp_path):
    from radonpy.core import poly
    from radonpy.sim.validation import compare_static
    atoms = molecule.GetNumAtoms()
    cell = poly.super_cell(molecule, x=2, y=1, z=1)
    xyz = cell.GetConformer().GetPositions()
    xyz[atoms:] -= [molecule.cell.dx-6, 0, 0]
    # Wrap individual atoms: bonded interactions and exclusions must remain
    # valid even when a molecule straddles the periodic boundary.
    origin = np.array([cell.cell.xlo, cell.cell.ylo, cell.cell.zlo])
    lengths = np.array([cell.cell.dx, cell.cell.dy, cell.cell.dz])
    xyz += origin-xyz[0]
    xyz = (xyz-origin) % lengths+origin
    for i, position in enumerate(xyz):
        cell.GetConformer().SetAtomPosition(i, position)
    results = {}
    for engine, executable in engines.items():
        work = tmp_path/engine
        work.mkdir()
        results[engine] = md.quick_energy(utils.deepcopy_mol(cell), solver=engine,
            solver_path=executable, work_dir=str(work), interaction_profile='portable', tmp_clear=True)
        assert not (work/'radon_md_last.data').exists()
    assert compare_static(results['lammps'], results['gromacs'], cell.GetNumAtoms())['status'] == 'pass'


@pytest.mark.engines
def test_failure_cannot_reuse_previous_result(molecule, engines, tmp_path):
    solver = gromacs.Gromacs(tmp_path, solver_path=engines['gromacs'])
    solver.make_dat(molecule)
    options = md.MD(mol=molecule, interaction_profile='portable')
    solver.make_input(options)
    solver.exec()
    with pytest.raises(FileExistsError, match='already exists'):
        solver.exec()


def test_ring_pairs_are_not_torsion_counted():
    from rdkit import Chem
    ring = Chem.MolFromSmiles('C1CCCCC1')
    assert set(gromacs._pairs14(ring)) == {(0, 3), (1, 4), (2, 5)}


@pytest.mark.engines
@pytest.mark.parametrize('ensemble', ['nve', 'nvt', 'npt'])
def test_short_dynamics(molecule, engines, tmp_path, ensemble):
    # Explicit velocities keep the comparison independent of engine RNGs.
    np.random.seed(23)
    from radonpy.core import calc
    calc.set_velocity(molecule, 300)
    kwargs = {'barostat': 'Parrinello-Rahman'} if ensemble == 'npt' else {}
    result = getattr(md, 'quick_'+ensemble)(molecule, solver='gromacs',
        solver_path=engines['gromacs'], work_dir=str(tmp_path),
        interaction_profile='portable', step=10, time_step=0.1, **kwargs)
    assert result[0].GetNumAtoms() == molecule.GetNumAtoms()
    assert np.isfinite(result[1]).all()
    assert not np.array_equal(result[1], molecule.GetConformer().GetPositions())
