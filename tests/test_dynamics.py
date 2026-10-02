import numpy as np
import pytest
from radonpy.core import calc, utils
from radonpy.sim import md
from radonpy.sim.md_wrapper import MD_solver, MD_analyzer


@pytest.mark.engines
@pytest.mark.parametrize('engine', ['lammps', 'gromacs'])
def test_nve_drift_decreases_with_timestep(molecule, engines, tmp_path, engine):
    np.random.seed(130)
    calc.set_velocity(molecule, 300)
    drifts = []
    for timestep in (.1, .05):
        work = tmp_path/str(timestep)
        work.mkdir()
        state = utils.deepcopy_mol(molecule)
        options = md.MD(mol=state, interaction_profile='portable', thermo_freq=10, dump_freq=10)
        options.add_md('nve', int(20/timestep), time_step=timestep)
        solver = MD_solver(engine, work_dir=str(work), solver_path=engines[engine])
        solver.make_dat(state, file_name=options.dat_file)
        solver.run(options, mol=state, mpi=0, omp=1)
        frame = MD_analyzer(engine, log_file=str(work/options.log_file)).dfs[-1]
        energy = frame.TotEng.to_numpy()*4.184/state.GetNumAtoms()
        drifts.append(np.max(np.abs(energy-energy[0])))
    assert drifts[0] < .01
    assert drifts[1] < .6*drifts[0]+1e-5
