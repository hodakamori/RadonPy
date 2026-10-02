"""Explicit interaction profiles shared by MD backends.

The portable profile is a different Hamiltonian from RadonPy's historical
CHARMM energy switch. Selecting an engine never implicitly selects a profile.
"""


def apply_profile(md):
    profile = getattr(md, 'interaction_profile', None)
    if profile is None:
        return md
    if profile != 'portable':
        raise ValueError('Unknown interaction_profile: %s' % profile)
    if not md.pbc:
        raise NotImplementedError('portable requires a periodic cell')
    # Packing deliberately uses short-range LJ without electrostatics.
    if md.pair_style != 'lj/cut':
        md.pair_style = 'lj/cut/coul/long'
        md.cutoff_in = 12.0
        md.cutoff_out = 12.0
        md.kspace_style = 'pppm'
        md.kspace_style_accuracy = '1e-8'
    md.pair_modify = 'mix arithmetic shift no tail no'
    md.special_bonds = 'amber'
    return md
