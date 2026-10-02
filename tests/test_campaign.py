"""Campaign orchestration checks; synthetic values are not physical evidence."""
import json
import numpy as np
import pandas as pd
import pytest

from radonpy.sim import validation_campaign as campaign
from radonpy.sim.validation import PROPERTY_MARGINS


@pytest.mark.parametrize('purpose,shift,converged,expected', [
    ('production', 0, True, 'pass'),
    ('pilot', 0, True, 'inconclusive'),
    ('production', .5, True, 'fail'),
    ('production', 0, False, 'inconclusive'),
])
def test_campaign_preserves_fail_and_inconclusive(tmp_path, monkeypatch,
                                                purpose, shift, converged, expected):
    replicas = []
    for index in range(5):
        path = tmp_path/('%d.json' % index)
        path.write_text(str(index))
        replicas.append({'replica_id': str(index), 'seed': index,
                         'molecule': path.name, 'sha256': campaign.sha(path)})
    config = {'schema': 1, 'purpose': purpose, 'preset': 'EQ21step',
              'cases': [{'name': 'synthetic', 'replicas': replicas}],
              'equilibration': {}, 'analysis': {'init': 0}, 'hardware': {},
              'thermo_freq': 1, 'dump_freq': 1, 'block_size': 100,
              'minimum_effective_samples': 200, 'diffusive_regime_confirmed': True,
              'absolute_margins': {}, 'reference_scales': {}}
    source = tmp_path/'campaign.json'
    source.write_text(json.dumps(config))
    class FakePreset:
        def __init__(self, molecule, **kwargs):
            self.engine = kwargs['solver']
            noise = np.random.default_rng(31).normal(size=4000)
            self.dfs = [pd.DataFrame({k: noise for k in ('TotEng', 'Volume', 'Density')})]
        def exec(self, **kwargs):
            pass
        def analyze(self):
            return self
        def get_all_prop(self, **kwargs):
            return {k: 1+(shift if self.engine == 'gromacs' else 0) for k in PROPERTY_MARGINS}
        def check_eq(self):
            return converged
    monkeypatch.setattr(campaign.eq, 'EQ21step', FakePreset)
    monkeypatch.setattr(campaign, '_read_molecule', lambda path: None)
    output = tmp_path/'results'
    assert campaign.run(source, output) is (expected == 'pass')
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == expected
    assert set(report['cases']['synthetic']) == {'A_vs_B', 'B_vs_C', 'A_vs_C'}
