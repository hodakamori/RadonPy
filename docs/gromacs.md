# Experimental GROMACS backend and validation

GROMACS is available for basic MD and the `sim.preset.eq` module. LAMMPS
remains the default. The implementation is experimental: passing force tests
and short workflows does **not** establish that long-time polymer properties
match the existing LAMMPS defaults, or that GROMACS is faster.

## Supported scope

| Module | Implemented scope |
| --- | --- |
| `sim.gromacs` | Assigned GAFF, GAFF2 and GAFF2_mod parameters; native topology, input generation, execution, result reading and checkpoint continuation |
| `sim.md` | `quick_energy`, `quick_min`, `quick_min_all`, `quick_nve`, `quick_nvt`, `quick_npt`, and sequential `MD` workflows |
| `sim.preset.eq` | Packing, `Annealing`, `EQ21step`, sampling, `Additional`, restoration and all 14 analysis outputs |
| Validation | Frozen fixtures, legacy regression, A/B/C static reports, statistical comparison, opt-in bulk campaigns and timing |

Inputs must be charge neutral, parameterized, periodic and orthorhombic. Each
box edge must exceed twice the nonbonded cutoff. Atom order and assigned
charges are preserved; the exporter does not run GROMACS force-field typing.
Disconnected molecules and shortest-path 1–4 pairs, including rings, are
supported. Nonzero total charge, nonperiodic systems, Drude, CMAP, reactive
bond creation, anisotropic pressure control, pressure ramps, electric fields,
`nve/limit`, and raw LAMMPS commands are rejected. Other presets (`tc`, `tg`,
`sp`, `ef_dp`, `elong`, etc.) do not support GROMACS and reject that selection.

`quick_min_all` currently runs conformers sequentially, retaining their IDs.
CG and steepest descent use GROMACS's force stopping criterion and iteration
limit. LAMMPS's energy tolerance and maximum force-evaluation count do not have
identical GROMACS controls. Minima and trajectories need not be identical.

## Explicit interaction profile

Always specify `interaction_profile='portable'` when selecting GROMACS.
Selecting the executable alone must not silently change the physical model.

| Comparison arm | Engine | Interactions |
| --- | --- | --- |
| A | LAMMPS | Existing defaults: CHARMM LJ switch from 8 to 12 Å, PPPM |
| B | LAMMPS | Explicit portable profile |
| C | GROMACS | Explicit portable profile |

The portable profile uses LJ cut off at 12 Å, no shift, no switch, no tail
correction, arithmetic sigma/geometric epsilon mixing, LJ 1–4 scaling of 1/2
and electrostatic 1–4 scaling of 5/6. Bonded parameters are converted directly
from RadonPy (including the harmonic factor of two and Fourier/CVFF phases).
LAMMPS uses PPPM with accuracy `1e-8`; GROMACS uses PME with `ewald-rtol=1e-8`,
order 6 and 0.08 nm grid spacing. PME and PPPM are compared numerically, not
assumed to be the same algorithm. Packing retains the preset's short-range,
charge-free LJ interaction with a 3 Å cutoff.

Public results retain RadonPy units: Å, fs, kcal/mol, kcal/(mol Å), atm and
g/cm³. Native GROMACS files use nm, ps, kJ/mol and bar. Input coordinates and
final states preserve double precision; a conventional three-decimal GRO
file is insufficient for the force comparison.

GROMACS NVE uses velocity Verlet; NVT/NPT use leapfrog and Nosé–Hoover with a
single thermostat variable. NPT uses isotropic Parrinello–Rahman, with
`compressibility=4.5e-5` bar⁻¹ by default (configurable on `Dynamics`).
The eq presets select this barostat explicitly. Basic NPT calls require
`barostat='Parrinello-Rahman'`. These coupling algorithms differ from the
LAMMPS preset's Nosé–Hoover chains/barostat, particularly during compression.
The default high-pressure EQ21step protocol still needs production validation.

When `shake=True`, the exporter constrains the same mass-1 hydrogen bonds
selected by the LAMMPS mass window, using LINCS (order 8, two iterations).
GROMACS removes center-of-mass motion every 1,000 steps and uses the resulting
translational degrees of freedom. Constraint algorithms, thermostat degrees
of freedom and coupling details are reasons to compare ensemble statistics,
not stepwise coordinates. Native inputs record the actual settings.

The initial implementation rebuilds the neighbor list every step. It favors
numerical validation; acceleration settings must be validated separately
before making performance claims.

## Basic use

Set `GROMACS_EXEC` to a `gmx`/`gmx_mpi` executable, or pass `solver_path`.
Importing RadonPy does not require that executable to be installed.

```python
from radonpy.core import utils
from radonpy.sim import md
from radonpy.sim.preset import eq

# A small frozen fixture is suitable for a single-point example only.
mol = utils.JSONToMol('tests/fixtures/pmma_gaff2.json')
energy, force = md.quick_energy(
    mol, solver='gromacs', interaction_profile='portable',
    work_dir='single-point', mpi=0, omp=1,
)

# For production, use a separately prepared periodic bulk cell instead.
preset = eq.EQ21step(
    bulk_cell, solver='gromacs', interaction_profile='portable',
    work_dir='equilibration', thermo_freq=1000, dump_freq=1000,
)
equilibrated = preset.exec(mpi=0, omp=4, gpu=0)
properties = preset.analyze().get_all_prop()
```

AutoMD scripts `1_eq.py` and `2_rst_eq.py` accept environment variables
`RadonPy_MD_Solver=gromacs` and `RadonPy_Interaction_Profile=portable`.
Existing scripts without these variables continue to use LAMMPS defaults.

Each run needs its own directory or distinct filenames. The backend saves a
JSON manifest, an exact molecule snapshot, and a `<manifest>.gmx/` directory
containing each stage's TOP/GRO/MDP/TPR/EDR/TRR/XTC/CPT files and command logs.
The public analyzer reads `<log_file>.gmx.json`; the final arrays are in NPZ.
A genuine LAMMPS data file is also exported for existing structure readers.
Native artifacts and metadata contain absolute paths; keep the run directory
in place while analyzing or resuming it.

To resume an interrupted **basic MD workflow**, use the same `Gromacs` object
and call `solver.run(options, mol=mol, resume=True)` or
`solver.exec(input_file=manifest_path, resume=True)`. The manifest and input
snapshot hashes must match. Completed stages are reused; the interrupted
stage resumes from its native checkpoint with append semantics. A partial
run is not published as successful, even when `mdrun` exits with status zero.
Starting another run over existing native outputs raises an error.
This does not automatically resume an entire interrupted eq preset; use its
last completed saved structure with `eq.Additional`, or recover the native
stage explicitly. `tmp_clear=True` removes the artifacts owned by a quick run.

`make_lammps_input` remains a compatibility alias for LAMMPS eq workflows.
Full-preset `make_input` for GROMACS is deliberately unavailable: later inputs
depend on the preceding stage's final cell and coordinates. Use `preset.exec`
or the basic solver's `make_dat`/`make_input` for individual stages.

## Reproducible tests

The reference environment pins Python dependencies in
[`tests/requirements.lock`](../tests/requirements.lock), LAMMPS
29Aug2024 Update2 and double-precision CPU GROMACS 2024.3 in
[`tests/engines.Dockerfile`](../tests/engines.Dockerfile).
LAMMPS needs MOLECULE, EXTRA-MOLECULE, KSPACE, RIGID, EXTRA-DUMP and EXTRA-FIX;
OPENMP is enabled for CPU threading. In particular, Fourier dihedrals require
[EXTRA-MOLECULE](https://docs.lammps.org/stable/dihedral_fourier.html), and XTC
output requires [EXTRA-DUMP](https://docs.lammps.org/dump.html).

```bash
docker build -f tests/engines.Dockerfile -t radonpy-md-validation .
docker run --rm -v "$PWD:/work" radonpy-md-validation \
  python -m pytest --require-engines -q \
  --basetemp=/work/validation-results/pytest \
  --junitxml=/work/validation-results/results.xml
```

Or use local executables in an isolated Python 3.12 environment:

```bash
python3.12 -m venv .venv-md
.venv-md/bin/pip install -r tests/requirements.lock
export LAMMPS_EXEC=/path/to/lmp
export GROMACS_EXEC=/path/to/gmx_d
.venv-md/bin/python -m pytest --require-engines -q
# No executables required for this subset:
.venv-md/bin/python -m pytest -m 'not engines'
```

The PR workflow requires both executables; a missing executable cannot produce
a green run through skips. It uploads XML, inputs and native logs. The local
default permits engine tests to skip when binaries are absent.

Tests include ten frozen inputs (butane and PE/PS/PMMA × three force fields),
an interacting multiple-molecule system crossing a periodic boundary, unit
conversion and exclusion checks, minimization with nonconsecutive conformer
IDs, NVE timestep/drift checks, NVT/NPT execution, short complete Annealing and
EQ21step workflows, all 14 finite analysis outputs, Additional/restoration,
checkpoint replay, interrupted execution and stale-output protection.

The legacy LAMMPS energy/force baseline was generated with the unmodified
checkout `5d14893` and is stored with fixture checksums and engine version in
`tests/fixtures/lammps_reference.json`. Do not regenerate it using the new
backend and call that a regression reference. `tests/generate_fixtures.py`
documents deterministic fixture construction; changes to those fixtures
require explicit review of the baseline. These small oligomers are **not**
equilibrated bulk-property references.

Double precision is the numerical reference. Mixed-precision/GPU GROMACS
needs a separate accuracy assessment: the initial mixed-precision test
exceeded the force tolerance for a few components. The tolerance was not
relaxed to make that build pass.

Generate an inspectable A/B/C report with:

```bash
python -m radonpy.sim.validation static tests/fixtures/pmma_gaff2.json \
  --output validation-results/pmma-static
```

The report includes input hashes, engine versions, energy terms, force errors
and wall time. B/C acceptance requires energy per atom within
`1e-4 + 1e-5*abs(reference)` kJ/mol and every force component within
`0.01 + 1e-4*abs(reference)` kJ/(mol nm). A/B is reported separately and need
not pass these static tolerances because the Hamiltonian changed. A passing
B/C result cannot establish equivalence to A.

## Long-time property validation

All 14 eq properties are required; unavailable or unconverged values produce
`inconclusive`, never a reduced list of passing properties.

| Properties | Prespecified default equivalence margin |
| --- | --- |
| Density | 1% |
| Rg, r2 | 5% |
| Cp, Cv; isothermal/isentropic compressibility and bulk modulus; volume/linear expansion; static dielectric constant | 10% |
| Self-diffusion | 20% |
| Nematic order parameter | Absolute 0.02 |

The statistical unit is an independently prepared cell, with at least five
matched cells per engine pair. The whole paired 95% Student confidence
interval for the difference must lie inside the margin. Overlap with zero,
a nonsignificant difference, or a narrow interval computed from correlated
trajectory frames is insufficient. Intervals wholly outside the margin fail;
intervals crossing a margin are inconclusive. Near-zero reference values
require an absolute margin or reference scale fixed before production.

```bash
python -m radonpy.sim.validation_campaign prepare validation-results/pilot-inputs \
  --degree 20 --chains 16 --replicas 5 --seed 1701
python -m radonpy.sim.validation_campaign run \
  validation-results/pilot-inputs/campaign.json \
  --output validation-results/pilot-runs
```

Preparation freezes coordinates, parameters, charges and velocities for
PE/PS/PMMA × GAFF/GAFF2/GAFF2_mod. Gasteiger charges are used to avoid requiring
quantum calculations in this harness; other production charge assignment
methods require their own validation cases. Each A/B/C arm starts from the
same input within a replicate. The runner saves all 14 estimates and A/B,
B/C and A/C comparisons for each case. `--case pe_gaff2` selects one case.

First inspect pilot sampling, density, chain dimensions, dipole relaxation
and the MSD diffusive regime. The runner checks energy/volume/density
autocorrelation, effective sample counts, at least 20 blocks, block lengths
of at least five autocorrelation times, and the existing `check_eq` tests.
These automatic diagnostics do not establish that a polymer reached its
diffusive regime. Set `diffusive_regime_confirmed` only after examining that
evidence. GROMACS samples instantaneous Rg and COM-corrected atomic MSD,
whereas the LAMMPS preset uses averaging windows; compare statistical
estimates, not framewise equality. Shared analysis formulas retain the
existing RadonPy definitions and units.

Freeze production lengths, output strides, analysis windows, absolute margins
and any reference scales in a reviewed `campaign.json` before running a new
independent production set. Set `purpose` to `production`; pilot results
cannot certify a pass. Configuration and input hashes are retained. Lengths
may need to be substantially longer for glassy polymers and fluctuation
properties. Do not loosen the margin after inspecting a failed comparison.
Non-passing campaigns exit with status 1, while retaining their reports.

Existing separately generated records can also be compared using:

```bash
python -m radonpy.sim.validation properties A.json C.json \
  --margins margins.json --output comparison.json
```

Each input is a list of `{ "replica_id": "0", "converged": true,
"properties": { ...all 14 property names... } }`. The optional margins file
contains `absolute_margins` and/or `reference_scales` dictionaries. The manual
`md-campaign.yml` workflow requires a configured self-hosted `radonpy-md`
runner, its own engines and Python environment, and frozen local inputs.

## Performance experiment

After accuracy and property validation, measure both native throughput and
complete wall time on the same hardware. Use an equilibrated bulk cell,
systems of approximately 10,000 and 100,000 atoms, at least three repeats,
and a separate warmup:

```bash
python -m radonpy.sim.benchmark equilibrated-cell.json \
  --output validation-results/timing-cpu --sizes 10000 100000 \
  --warmup 10000 --steps 100000 --repeats 3 --omp 8 --mpi 0 --gpu 0
```

For a GPU experiment, use independently validated GPU builds of both engines
and `--gpu 1` in a new output directory. The CPU double-precision reference
image is not a GPU benchmark image. The runner refuses a LAMMPS build that
would silently fall back from the requested GPU/thread resources. Reports
include engine versions, resources, system sizes, hardware information where
available, individual measurements and medians. The current common profile
uses the same timestep, constraints and output stride, and includes topology
generation/result extraction in end-to-end timing. A timing run does not
certify scientific equivalence.

## Upstream PR boundaries and release gates

Submit dependency-ordered PRs, with each support PR delivering a usable module
slice plus its tests and documentation. This working tree contains the slices
together; it is not itself a set of published upstream PRs.

| PR | Reviewable deliverable | Gate before declaring support |
| --- | --- | --- |
| 0: validation foundation | Pinned CPU environment, fixtures and original LAMMPS regression, comparison definitions | Existing LAMMPS regression remains green |
| 1: `sim.md` energies/minimization | Public GROMACS solver/reader, exporter, explicit profile, quick energy/min APIs | Unit conversion, exclusions, periodic and multiple-molecule energy/force comparisons pass |
| 2: `sim.md` dynamics | NVE/NVT/NPT, constraints, trajectories, native checkpoint continuation | Drift/timestep, execution and restart tests pass; ensemble validation evidence attached |
| 3: `sim.preset.eq` | Shared analysis, packing/Annealing/EQ21step/Additional, 14 properties and AutoMD selection | Full production A/B/C campaigns pass for all agreed polymer/force-field cases |

Parts of the solver and analysis necessarily form dependencies between these
PRs. Split and review the changes in that order, keeping unrelated presets
out of each PR. Maintain experimental labeling while a gate is pending.
Future `tc`, `tg`, `sp`, `ef_dp` and `elong` support should each have its own
module PR and observable-specific validation, rather than inheriting an eq
pass as proof of correctness.

Local implementation checks have exercised the real reference engines,
legacy regressions, short eq workflows and a tiny timing-runner smoke test.
Long-time bulk equivalence, default high-pressure compression, full-size CPU
and GPU timing, external MPI behavior, and the hosted Docker/CI execution
remain unverified. No speedup or production-level LAMMPS equivalence is
claimed from the short tests.
