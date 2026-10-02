# GROMACS reference validation

This suite compares RadonPy's existing LAMMPS calculations with a GROMACS
reference. All conversion, execution and comparison helpers live in
`tests/helpers/`. RadonPy's package, public APIs, presets, AutoMD scripts and
default simulation settings are unchanged from `develop` (`5d14893`).

The scope is single-point energies and forces, periodic boundaries, short
unconstrained NVE runs, and regression against frozen legacy LAMMPS results.
It does not provide a production GROMACS backend, equilibration integration,
bulk-property campaigns or performance benchmarks.

## Run the tests

From the repository root, use Python 3.12 and the pinned test dependencies:

```bash
python -m pip install -r tests/requirements.lock
python -m pytest -c tests/pytest.ini tests -m 'not engines'
```

For numerical validation, use the double-precision GROMACS 2024.3 CPU build
and LAMMPS `stable_29Aug2024_update2` specified in `engines.Dockerfile`:

```bash
export LAMMPS_EXEC=/path/to/lmp
export GROMACS_EXEC=/path/to/gmx_d
python -m pytest -c tests/pytest.ini tests --require-engines
```

Without `--require-engines`, tests skip when their required executable is
missing. LAMMPS-only regression tests do not require GROMACS. Pass `tests`
explicitly to avoid collecting the repository's runnable sample scripts.

The Docker build context is the `tests` directory. Mount the repository when
running the image:

```bash
docker build -f tests/engines.Dockerfile -t radonpy-md-validation tests
docker run --rm -v "$PWD:/work" radonpy-md-validation
```

Pytest retains native inputs and logs in its temporary directories. To choose
an artifact directory, add `--basetemp=/tmp/radonpy-validation`; pytest clears
that directory at the beginning of the run, so use a dedicated location.

## What is compared

The static report separates the original settings from the common settings:

| Arm | Engine | Nonbonded interactions |
| --- | --- | --- |
| A | LAMMPS | Existing defaults, including CHARMM LJ switching at 8–12 Å |
| B | LAMMPS | Test-local LJ cutoff at 12 Å, PPPM accuracy `1e-8` |
| C | GROMACS | Same LJ cutoff, PME tolerance `1e-8`, order 6, 0.08 nm grid |

B and C use no LJ shift or tail correction, arithmetic sigma/geometric epsilon
mixing, and 1–4 scaling of 1/2 for LJ and 5/6 for electrostatics. Assigned GAFF,
GAFF2 and GAFF2_mod parameters and atom order are preserved. Inputs must be
neutral orthorhombic periodic cells with edges longer than 24 Å.

The comparison tolerances are `atol=1e-4, rtol=1e-5` for energy per atom in
kJ/mol and `atol=1e-2, rtol=1e-4` for each force component in kJ/(mol nm).
NVE tests check that energy drift decreases when the timestep is halved.
These short tests do not establish long-time bulk-property equivalence with
the original LAMMPS settings.

Generate a report in a new output directory using the same executable
environment variables:

```bash
python -m tests.helpers.validation static tests/fixtures/pmma_gaff2.json \
  --output /tmp/radonpy-static-report
```

The report retains fixture hashes, engine versions and A/B/C results. Its exit
status reflects B-versus-C parity; A-versus-B differences are reported
separately because the interaction settings differ.

`fixtures/lammps_reference.json` contains original-checkout results and fixture
checksums. Frozen butane and PE/PS/PMMA oligomers are small validation inputs,
not equilibrated bulk cells. `python -m tests.generate_fixtures` explicitly
regenerates the polymer fixtures and manifest; it does not update the legacy
reference results, which must be obtained from the original checkout if the
fixtures change. Tests never regenerate their expected results.

The statistical comparison utilities are also test-local. They can compare
separately prepared property records with
`python -m tests.helpers.validation properties A.json C.json --output comparison.json`.
Their unit tests use synthetic data and provide no physical-equivalence evidence.
