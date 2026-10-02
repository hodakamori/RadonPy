import os
from pathlib import Path
import shutil

import pytest
from radonpy.core import utils


def pytest_addoption(parser):
    parser.addoption('--require-engines', action='store_true',
                     help='Fail instead of skip when either MD engine is absent')


@pytest.fixture
def molecule():
    return utils.JSONToMol(str(Path(__file__).parent/'fixtures/butane_gaff2.json'))


@pytest.fixture
def engines(request):
    result = {'lammps': os.environ.get('LAMMPS_EXEC', 'lmp'),
              'gromacs': os.environ.get('GROMACS_EXEC', 'gmx')}
    missing = [name for name, exe in result.items() if shutil.which(exe) is None]
    if missing:
        message = 'Missing MD executables: '+', '.join(missing)
        if request.config.getoption('--require-engines'):
            pytest.fail(message)
        pytest.skip(message)
    return result
