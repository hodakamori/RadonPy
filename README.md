![logo](https://user-images.githubusercontent.com/83273612/160471242-40d7d7f1-d2cd-4658-b4e1-75f5e608665d.png)

## Overview
RadonPy is the first open-source Python library for fully automated calculation for a comprehensive set of polymer properties, using all-atom classical MD simulations. For a given polymer repeating unit with its chemical structure, the entire process of the MD simulation can be carried out fully automatically, including molecular modelling, equilibrium and non-equilibrium MD simulations, automatic determination of the completion of equilibration, scheduling of restarts in case of failure to converge, and property calculations in the post-process step. In this release, the library comprises the calculation of 62 properties at the amorphous state.

## Requirement
- Python 3.9, 3.10, 3.11, 3.12, 3.13
- LAMMPS >= 3Mar20
- rdkit >= 2020.03
- psi4 >= 1.5
- resp
- dftd3-python
- mdtraj >= 1.9
- scipy
- matplotlib

## Installation and usage
User manual and conda packages are currently in preparation.

[PyPI package](https://pypi.org/project/radonpy-pypi/) is available, but Psi4 can not be installed by pip install.

[PDF file](https://github.com/RadonPy/RadonPy/blob/develop/docs/RadonPy_tutorial_20220331.pdf) of RadonPy tutorial is available.

### Installation for conda (for python=3.13, psi4=1.10):
Notice: Runtime errors may occur in Psi4 when installed Python packages from different channels are mixed. Therefore, the option `-c conda-forge` should be added in the conda installation.

1. Create conda environment
```
conda create -n radonpy python=3.13 -c conda-forge
conda activate radonpy
```

2. Installation of requirement packages by conda
```
conda install -c conda-forge rdkit psi4 dftd3-python resp mdtraj psutil scipy pandas matplotlib pip
```

3. Installation of LAMMPS by conda
```
conda install -c conda-forge lammps
```

or manually build from source of [LAMMPS official site](https://www.lammps.org/).
The preset module of solubility parameters requires to install TALLY package in LAMMPS.
In this case, the environment variable must be set:
```
export LAMMPS_EXEC=<Path-to-LAMMPS-binary>
```

4. Installation of a beta test version of RadonPy (1.0b2) can be installed by
```
git clone https://github.com/RadonPy/RadonPy.git
pip install --no-index --find-links=./RadonPy/dist/ radonpy-pypi
```

or an old version of RadonPy (0.2.x) can be installed by
```
pip install radonpy-pypi
```


5. (Optional) RadonPy for Bio-polymers (peptides, polysaccharides, water models (tip3p, tip4p, tip5p))
```
conda install -c conda-forge ambertools intermol
```



### Installation from PyPI
RadonPy can be also installed by using only pip install. However, this intallation method can not install Psi4.

- Without LAMMPS installation
```
pip install radonpy-pypi
```
This is minimal installation of RadonPy. Many functions, such as polymer structure builder, force field assignment, force field descriptor, 
and tools for polymer informatics, are available, but automated DFT and MD simulations are not available.

- With LAMMPS installation
```
pip install radonpy-pypi[lammps]
```
MD simulations are available in this installation, but DFT calculations (conformation search, cherge calculation, and electronic property calculation) are not available.


## Features
- Fully automated all-atom classical MD calculation for polymeric materials
	- Conformation search
	- Cherge calculation (RESP, ESP, Mulliken, Lowdin, Gasteiger)
	- Electronic property calculation (HOMO, LUMO, dipole moment, polarizability)
	- Generation of a polymer chain
		- Homopolymer
		- Alternating copolymer
		- Random copolymer
		- Block copolymer
		- Branched polymer
	- Generation of a simulation cell
		- Amorphous
		- Polymer mixture
		- Polymer solution
		- Crystalline polymer
		- Oriented structure
    - Assignment of force field
		- GAFF
		- GAFF2
		- GAFF2_mod (J. Träg, D. Zahn, J. Mol. Model. 25:39 (2019))
		- Dreiding
		- Dreiding_UT (K. Sasaki, T. Yamashita, J. Chem. Inf. Model., 61:1172 (2021))
        - Amber (amber_ff19SB)
        - GLYCAM_06j
        - Water models (TIP3P, TIP4P, TIP5P)
	- Run for equilibration MD
	- Checking archivement of equilibrium
	- Run for non-equilibrium MD (NEMD)
	- Calculation of physical properties from the MD calculation results
		- Density
		- Cp
		- Cv
		- Linear expansion coefficient
		- Volumetric expansion coefficient
		- Compressibility
		- Bulk modulus
		- Isentropic compressibility
		- Isentropic bulk modulus
		- Static dielectric constant
		- Refractive index
		- Abbe number
		- Radius of gyration
		- End-to-end distance
		- Nematic order parameter
		- Thermal conductivity
		- Thermal diffusivity
		- Glass transition temperature (Tg)
        - Solubility parameters (Hildebrand, dispersion term, electrostatic term)
        - Cohesive energy density
        - Dynamic dielectric properties
	- Using LAMMPS and Psi4 as calculation engines of MD and DFT calculations
- Implementation of add-on like presets to allow for proper and easy execution of polymer MD calculations
	- Equilibration MD (sim/preset/eq.py)
	- Calculation of thermal conductivity with NEMD (sim/preset/tc.py)
    - Calculation of Tg (sim/preset/tg.py)
    - Calculation of solubility parameters (sim/preset/sp.py)
    - Calculation of dynamic dielectric properties (sim/preset/ef_dp.py)
    - Calculation of physical properties of stretched-oriented structures (sim/preset/elong.py)
- Easy installation
    - Only using open-source software
    - All package can be installed via conda or pip
- Tools for polymer informatics
	- Force field descriptor ([How to use](https://github.com/RadonPy/RadonPy/blob/develop/docs/FF-Descriptor_man.pdf))
	- Generator of macrocyclic oligomer for descriptor construction of polymers
	- Full and substruct match function for polymer SMILES
	- Extractor of mainchain in a polymer backbone
	- Monomerization of oligomer SMILES
	- Emulator of polymer classification in PoLyInfo

## MD calculated data
- [1070 amorphous polymers](https://github.com/RadonPy/RadonPy/blob/develop/data/PI1070.csv)

## Publications
1. Y. Hayashi, J. Shiomi, J. Morikawa, R. Yoshida, "RadonPy: Automated Physical Property Calculation using All-atom Classical Molecular Dynamics Simulations for Polymer Informatics," npj Comput. Mater. 8:222 (2022) \[[Link](https://www.nature.com/articles/s41524-022-00906-4)\]
2. M. Kusaba, Y. Hayashi, C. Liu, A. Wakiuchi, R. Yoshida, "Representation of materials by kernel mean embedding", Phys. Rev. B, 108:134107 (2023)\[[Link](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.134107)\]
3. R. Yoshida and Y. Hayashi et al., "Omics-scale polymer computational database transferable to real-world artificial intelligence applications", arXiv preprint, arXiv:2511.11626 (2025) \[[Link](https://arxiv.org/abs/2511.11626)\]

## Related publications
1. S. Nanjo, Arifin, H. Maeda, Y. Hayashi, K. Hatakeyama-Sato, R. Himeno, T. Hayakawa, R. Yoshida, "SPACIER: on-demand polymer design with fully automated all-atom classical molecular dynamics integrated into machine learning pipelines," npj Comput. Mater. 11:16 (2025) \[[Link](https://www.nature.com/articles/s41524-024-01492-3)\]
2. S. Minami, Y. Hayashi, S. Wu, K. Fukumizu, H. Sugisawa, M. Ishii, I. Kuwajima, K. Shiratori, R. Yoshida, "Scaling law of Sim2Real transfer learning in expanding computational materials databases for real-world predictions," npj Comput. Mater. 11:146 (2025) \[[Link](https://www.nature.com/articles/s41524-025-01606-5)\]

## Contributors
- Yoshihiro Hayashi (The Institute of Statistical Mathematics) (Organizer, Leading developer)
- Ryohei Hosoya (Institute of Science Tokyo) (Implementation of sim/preset/ef_dp.py, sim/preset/elong.py)
- Hidemine Furuya (Institute of Science Tokyo) (Implementation of sim/preset/ef_dp.py, sim/preset/elong.py)
- Hiroki Sugisawa (Mitsubishi Chemical Corporation) (Implementation of sim/preset/tg.py)
- Kazuyoshi Kaneko (The Yokohama Rubber Co., Ltd.) (Implementation of sim/preset/sp.py)
- Teruki Tsurimoto (Sekisui Chemical Co., Ltd.) (Implementation of ff/dreiding.py)
- Shun Nanjo (SOKENDAI) (Determination of calculation conditions for Abbe number)

## Related projects
- XenonPy (Machine learning tools for materials informatics) \[[Link](https://github.com/yoshida-lab/XenonPy)\]
- SMiPoly (Polymerization rule-based virtual polymer generator) \[[Link](https://github.com/PEJpOhno/SMiPoly)\]
- SPACIER (Efficient molecular exploration tool using Bayesian Optimization combined with RadonPy) \[[Link](https://github.com/s-nanjo/Spacier/)\]

## Acknowledgements
The development of RadonPy has been financially supported by the following grants:
- Japan Science and Technology Agency (JST) CREST (Grant Number: JPMJCR19I3)
- Ministry of Education, Culture, Sports, Science and Technology (MEXT) as “Program for Promoting Researches on the Supercomputer Fugaku” (Project ID: hp210264)
- JST as "Program on Open Innovation Platform for Industry-Academia Co-creation (COI-NEXT)" (Grant Number: JPMJPF2102)
- MEXT as "Data Creation and Utilization-Type Material Research and Development Project (DxMT)" (Grant Number: JPMXP1122714694)
- The Japan Society for the Promotion of Science (JSPS) as the Grant-in-Aid for Scientific Research (A) (Grant Number: 19H01132)
- JSPS as the Grant-in-Aid for Scientific Research (B) (Grant Number: 25K00147)
- JSPS as the Grant-in-Aid for Scientific Research (C) (Grant Number: 22K11949)

The numerical calculations were conducted on the following supercomputer systems:
- Fugaku at the RIKEN Center for Computational Science, Kobe, Japan (Project ID: hp210264, hp210213, hp220179, hp230190, hp240216, hp250235, hp250324, hp260172, hp260195)
- The supercomputer at the Research Center for Computational Science, Okazaki, Japan (Project ID: 21-IMS-C126, 22-IMS-C125, 23-IMS-C113, 24-IMS-C107, 25-IMS-C107, 26-ISM-C098)
- The supercomputer Ohtaka at the Supercomputer Center, the Institute for Solid State Physics, the University of Tokyo, Tokyo, Japan
- The supercomputer TSUBAME3.0 at the Tokyo Institute of Technology, Tokyo, Japan
- The supercomputer ABCI at the National Institute of Advanced Industrial Science and Technology, Tsukuba, Japan

The experimental varidation data was provided by:
- PoLyInfo \[[Link](https://polymer.nims.go.jp/)\] developed by National Institute for Materials Science (NIMS)

## Copyright and licence
©Copyright 2026 The RadonPy developers, all rights reserved.
Released under the `BSD-3 license`.


![Radon_ikaho](https://user-images.githubusercontent.com/83273612/158885745-224f6e7a-4b1d-46f4-b5c6-80455827c904.png)

