from pathlib import Path
import numpy as np
import pytest

from radonpy.core import utils, calc
from radonpy.sim import gromacs, lammps, md
from radonpy.sim.preset import eq
from radonpy.sim.validation import PROPERTY_MARGINS


def test_preset_profiles(tmp_path):
    molecule = utils.JSONToMol(str(Path(__file__).parent/'fixtures/pe_gaff2.json'))
    legacy = eq.EQ21step(molecule, work_dir=str(tmp_path))
    assert legacy.sampling().pair_style == 'lj/charmm/coul/long'
    portable = eq.EQ21step(molecule, work_dir=str(tmp_path), solver='gromacs',
                           interaction_profile='portable')
    schedule = portable.eq21step()
    assert len(schedule.wf) == 21
    assert schedule.pair_style == 'lj/cut/coul/long'
    assert not any(w.add for w in schedule.wf)
    assert all(w.barostat == 'Parrinello-Rahman' for w in schedule.wf if w.ensemble == 'npt')
    packing = portable.packing()
    assert packing.pair_style == 'lj/cut' and packing.cutoff_in == 3
    assert packing.wf[-1].deform == 'final'
    from radonpy.sim.preset import tc
    with pytest.raises(NotImplementedError, match='only by the eq'):
        tc.NEMD_MP(molecule, work_dir=str(tmp_path), solver='gromacs')


@pytest.mark.engines
@pytest.mark.slow
@pytest.mark.parametrize('preset_name', ['Annealing', 'EQ21step'])
def test_annealing_end_to_end(engines, tmp_path, preset_name):
    molecule = utils.JSONToMol(str(Path(__file__).parent/'fixtures/pe_gaff2.json'))
    preset = getattr(eq, preset_name)(molecule, work_dir=str(tmp_path), solver='gromacs',
        solver_path=engines['gromacs'], interaction_profile='portable',
        thermo_freq=1, dump_freq=1, no_traj_ann=True)
    density = calc.mol_mass(molecule)/6.02214076e23/(38**3*1e-24)
    schedule = {'annealing_pre_steps': (2, 2, 2), 'annealing_chunk_steps': 2,
                'ann_step': .000004} if preset_name == 'Annealing' else {
                'step_list': [[2, 2, 2]]*7, 'max_press': 2, 'time_step': .2}
    result = preset.exec(f_density=density, max_temp=300, temp=300, mpi=0,
        packing_steps=(2, 2, 2), eq_step=.00004, **schedule)
    assert result.GetNumAtoms() == molecule.GetNumAtoms()
    assert eq.get_final_idx(str(tmp_path)) == 3
    restored = eq.restore(preset.save_dir)
    assert restored.GetNumAtoms() == result.GetNumAtoms()
    analyzer = preset.analyze()
    props = analyzer.get_all_prop(init=10, width=5, f_width=10)
    assert set(props) == set(PROPERTY_MARGINS)
    assert all(np.isfinite(value) for value in props.values())
    assert np.isfinite(analyzer.dfs[-1]['v_msd']).all()
    extra = eq.Additional(result, work_dir=str(tmp_path), solver='gromacs',
        solver_path=engines['gromacs'], interaction_profile='portable', thermo_freq=1, dump_freq=1)
    assert extra.idx == 4
    extra.exec(eq_step=.00001, mpi=0)
    assert eq.get_final_idx(str(tmp_path)) == 4


@pytest.mark.engines
def test_checkpoint_replay(engines, tmp_path, molecule):
    solver = gromacs.Gromacs(str(tmp_path), solver_path=engines['gromacs'])
    options = md.MD(mol=molecule, interaction_profile='portable', thermo_freq=1, dump_freq=1)
    options.add_md('nve', 5, time_step=.1)
    first = solver.run(options, mol=molecule)
    first_frames = gromacs.Analyze(log_file=str(tmp_path/options.log_file)).dfs[-1]
    # Remove only the completion marker to exercise mdrun checkpoint replay.
    marker = Path(solver.input_file+'.gmx')/'stage_000'/'completed.json'
    marker.unlink()
    second = solver.run(options, mol=molecule, resume=True)
    second_frames = gromacs.Analyze(log_file=str(tmp_path/options.log_file)).dfs[-1]
    np.testing.assert_allclose(first.GetConformer().GetPositions(), second.GetConformer().GetPositions())
    np.testing.assert_allclose(first_frames, second_frames)
    assert second_frames.index.is_unique


@pytest.mark.engines
def test_incomplete_run_is_resumed_not_certified(engines, tmp_path, molecule, monkeypatch):
    solver = gromacs.Gromacs(str(tmp_path), solver_path=engines['gromacs'])
    options = md.MD(mol=molecule, interaction_profile='portable', thermo_freq=1, dump_freq=1)
    options.add_md('nve', 10, time_step=.1)
    command = solver._command
    def stop_early(args, *positional, **kwargs):
        if args[0] == 'mdrun':
            args = args + ['-nsteps', '5']
        return command(args, *positional, **kwargs)
    monkeypatch.setattr(solver, '_command', stop_early)
    with pytest.raises(RuntimeError, match='stopped before'):
        solver.run(options, mol=molecule)
    assert not (tmp_path/(options.log_file+'.gmx.json')).exists()
    monkeypatch.setattr(solver, '_command', command)
    solver.run(options, mol=molecule, resume=True)
    frame = gromacs.Analyze(log_file=str(tmp_path/options.log_file)).dfs[-1]
    assert frame.index[-1] == 10
    assert frame.index.is_unique


def test_shared_analysis_formulas():
    import pandas as pd
    from radonpy.core import const
    df = pd.DataFrame({'TotEng': [1., 2., 3., 4.], 'Temp': [300.]*4,
                       'Volume': [1000.]*4, 'Density': [1.]*4})
    expected = np.var(df.TotEng*4184/const.NA)/(1e-24*const.kB*300**2)
    assert lammps.Analyze.heat_capacity_Cp(df) == pytest.approx(expected)
    assert gromacs.Analyze.heat_capacity_Cp(df) == pytest.approx(expected)
