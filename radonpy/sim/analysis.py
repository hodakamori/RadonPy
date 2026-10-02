# Copyright (c) 2026. RadonPy developers. All rights reserved.
# Use of this source code is governed by the BSD-3 license in LICENSE.

"""Shared thermodynamic and trajectory property calculations.

Readers provide DataFrames in RadonPy real units. Engine-specific parsing is
kept in the backend modules; existing LAMMPS analysis APIs remain available.
"""
import os
import math
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as pp
from ..core import calc, poly, const, utils

try:
    import mdtraj
    mdtraj_avail = True
except ImportError:
    mdtraj_avail = False


class Analyze:
    def __init__(self, log_file='radon_md.log', ignore_log=[], **kwargs):
        self.dfs = self.read_log(log_file, ignore_log=ignore_log)
        self.log_file = log_file
        self.in_file = kwargs.get('in_file', 'radon_md.dump')
        self.dat_file = kwargs.get('dat_file', 'radon_md_lmp.data')
        self.traj_file = kwargs.get('traj_file', 'radon_md.dump')
        self.rg_file = kwargs.get('rg_file', 'rg.profile')
        self.pdb_file = kwargs.get('pdb_file', 'topology.pdb')

        self.traj = None
        self.charges = np.array([])

        self.totene_data = {}
        self.kinene_data = {}
        self.ebond_data = {}
        self.eangle_data = {}
        self.edihed_data = {}
        self.evdw_data = {}
        self.ecoul_data = {}
        self.elong_data = {}
        self.dens_data = {}
        self.temp_data = {}
        self.rg_data = {}
        self.msd_data = {}
        self.diffc_data = {}
        self.Cp_data = {}
        self.Cv_data = {}
        self.compress_T_data = {}
        self.compress_S_data = {}
        self.bulk_mod_T_data = {}
        self.bulk_mod_S_data = {}
        self.volume_exp_data = {}
        self.linear_exp_data = {}
        self.diele = {}
        self.diele_data = {}
        self.nop = {}
        self.nop_data = {}
        self.prop_df = pd.DataFrame([])
        self.conv_df = pd.DataFrame([])

        self.totene_sma_sd_crit = kwargs.get('totene_sma_sd_crit', 0.0005)
        self.kinene_sma_sd_crit = kwargs.get('kinene_sma_sd_crit', 0.0005)
        self.ebond_sma_sd_crit = kwargs.get('ebond_sma_sd_crit', 0.001)
        self.eangle_sma_sd_crit = kwargs.get('eangle_sma_sd_crit', 0.001)
        self.edihed_sma_sd_crit = kwargs.get('edihed_sma_sd_crit', 0.002)
        self.evdw_sma_sd_crit = kwargs.get('evdw_sma_sd_crit', 30.0)
        self.ecoul_sma_sd_crit = kwargs.get('ecoul_sma_sd_crit', None)
        self.elong_sma_sd_crit = kwargs.get('elong_sma_sd_crit', 0.001)
        self.dens_sma_sd_crit = kwargs.get('dens_sma_sd_crit', 0.001)
        self.rg_sd_crit = kwargs.get('rg_sd_crit', 0.01)
        self.diffc_sma_sd_crit = kwargs.get('diffc_sma_sd_crit', None)
        self.Cp_sma_sd_crit = kwargs.get('Cp_sma_sd_crit', None)
        self.compress_sma_sd_crit = kwargs.get('compress_sma_sd_crit', None)
        self.volexp_sma_sd_crit = kwargs.get('volexp_sma_sd_crit', None)


    def analyze_thermo(self, target, init=2000, last=None, width=2000,
                    ylabel=None, conv_a=1.0, conv_b=0.0, printout=False, save=None):
        """
        LAMMPS.analyze_thermo

        Analyze thermodynamic propeties in a log file

        Args:
            target: Target property

        Optional args:
            init: Initial step (int)
            last: Last step (int)
            width: Width of steps to use for the average, variance, and covariance calculations (int)
            ylabel: Label of y-axis in the output plot (str)
            timestep: Conversion factor of timestep -> ps (float)
            conv_a, conv_b: Conversion factor of property: prop_c = conv_a * prop + conv_b (float)
            printout: Printout analyzed data for STDOUT (boolean)
            save: Dir path to save analyzed data (str)

        Return:
            Analyzed results
        """

        thermo_df = self.dfs[-1]
        data = {}
        ps = 1e-3
        if not last: last = 0
        if len(thermo_df) < last: last = 0

        if init >= len(thermo_df):
            utils.radon_print('init=%i is out of range. Require init < %i' % (init, len(thermo_df)), level=3)
            return None

        data_conv = thermo_df[target] * conv_a + conv_b
        data_sma = data_conv.rolling(width).mean()
        data_sd = data_conv.rolling(width).std()
        data_se = data_sd / np.sqrt(width)

        data['init'] = thermo_df['Time'].values[init] * ps
        data['last'] = thermo_df['Time'].values[last-1] * ps
        data['width'] = width
        data['mean'] = data_sma.values[int(last-1)]
        data['sd'] = data_sd.values[int(last-1)]
        data['se'] = data_se.values[int(last-1)]
        data['sma_sd'] = data_sma.values[last-width-1:last].std() if last else data_sma.values[-width-1:].std()
        data['sma_se'] = data['sma_sd'] / np.sqrt(width)

        if printout or save:
            if not last: last = None
            fig, ax = pp.subplots(figsize=(6, 6))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
            ax.plot(thermo_df['Time'].values[init:last]*ps, data_conv.values[init:last], linewidth=0.1)
            ax.errorbar(thermo_df['Time'].values[init:last]*ps, data_sma.values[init:last], yerr=data_se.values[init:last]*2, linewidth=2.0)
            ax.set_xlabel('Time [ps]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            output = 'Accumulation of %f - %f ps\n' % (data['init'], data['last'])
            output += '%s = %e     SD = %e    SE = %e\n' % (ylabel, data['mean'], data['sd'], data['se'])
            output += 'SMA_SD = %e     SMA_SE = %e\n' % (data['sma_sd'], data['sma_se'])

            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, target+'.png'))
                with open(os.path.join(save, target+'.txt'), mode='w') as f:
                    f.write(output)

            pp.close(fig)

        return data


    def analyze_thermo_fluctuation(self, func, target=-1, temp=None, press=1.0, mass=None, f_width=2000,
            init=2000, last=None, width=2000, name='img', ylabel=None, conv_a=1.0, conv_b=0.0,
            printout=False, save=None):
        """
        lammps.Analyze.analyze_thermo_fluctuation

        Analyze thermodynamic fluctiation propeties in a log file

        Args:
            func: Function object of thermodynamic fluctiation propeties
                lammps.Analyze.heat_capacity_Cp
                lammps.Analyze.heat_capacity_Cv
                lammps.Analyze.heat_capacity_Cv_NVT
                lammps.Analyze.isothermal_compressibility
                lammps.Analyze.isentropic_compressibility
                lammps.Analyze.bulk_modulus
                lammps.Analyze.isentropic_bulk_modulus
                lammps.Analyze.speed_of_sound
                lammps.Analyze.volume_expansion
                lammps.Analyze.linear_expansion

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)
            width: Width of steps to use for the average, variance, and covariance calculations (int)
            name: Output file name (str)
            ylabel: Label of y-axis in the output plot (str)
            timestep: Conversion factor of timestep -> ps (float)
            conv_a, conv_b: Conversion factor of property: prop_c = conv_a * prop + conv_b (float)
            printout: Printout analyzed data for STDOUT (boolean)
            save: Dir path to save analyzed data (str)

        Return:
            Analyzed results
        """

        thermo_df = self.dfs[target]
        law_data = []
        data = {}
        ps = 1e-3

        if last and len(thermo_df) < last: last = None

        n = len(thermo_df) - init
        if n <= 0:
            utils.radon_print('init=%i is out of range. Require init > %i' % (init, len(thermo_df)), level=3)
            return None

        if init < f_width:
            utils.radon_print('init=%i, f_width=%i is out of range. Require init >= f_width' % (init, f_width), level=3)
            return None

        for i in range(n):
            f_last = i + init
            prop = func(thermo_df, temp=temp, press=press, mass=mass, init=init-f_width, last=f_last)
            law_data.append([thermo_df['Time'].values[f_last], prop])

        if not last: last = 0

        prop_df = pd.DataFrame(law_data, columns=['Time', 'Prop'])
        data_conv = prop_df['Prop'] * conv_a + conv_b
        data_sma = data_conv.rolling(width).mean()
        data_sd = data_conv.rolling(width).std()
        data_se = data_sd / np.sqrt(width)

        data['init'] = thermo_df['Time'].values[init] * ps
        data['last'] = thermo_df['Time'].values[last-1] * ps
        data['width'] = width
        data['mean'] = data_sma.values[int(last-1)]
        data['sd'] = data_sd.values[int(last-1)]
        data['se'] = data_se.values[int(last-1)]
        data['sma_sd'] = data_sma.values[last-width-1:last].std() if last else data_sma.values[-width-1:].std()
        data['sma_se'] = data['sma_sd'] / np.sqrt(width)

        if printout or save:
            if not last: last = None
            fig, ax = pp.subplots(figsize=(6, 6))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
            ax.plot(prop_df['Time'].values[:last]*ps, data_conv.values[:last], linewidth=1.0)
            ax.errorbar(prop_df['Time'].values[:last]*ps, data_sma.values, yerr=data_se.values*2, linewidth=2.0)
            ax.set_xlabel('Time [ps]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            output = 'Accumulation of %f - %f ps\n' % (data['init'], data['last'])
            output += '%s = %e     SD = %e    SE = %e\n' % (ylabel, data['mean'], data['sd'], data['se'])
            output += 'SMA_SD = %e     SMA_SE = %e\n' % (data['sma_sd'], data['sma_se'])

            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, name+'.png'))
                with open(os.path.join(save, name+'.txt'), mode='w') as f:
                    f.write(output)

            pp.close(fig)

        return data


    @classmethod
    def heat_capacity_Cp(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.heat_capacity_Cp

        Calculate isobaric specific heat capacity from thermodynamic data in a log file
        Cp = Var(H)/(m*kB*T**2)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isobaric heat capacity (float, J/(kg K))
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        M = V * (thermo_df['Density'].to_numpy() * 1e+3) # m**3 * (g/cm**3) -> m**3 * (kg/m**3) = kg

        T = thermo_df['Temp'].to_numpy() # K

        U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        P = press * const.atm2pa # Pa = J / m**3
        H = U + P * V

        mM = M[init:last].mean() if mass is None else mass
        mT = T[init:last].mean() if temp is None else temp
        H_var = np.var(H[init:last])

        Cp = H_var / (mM * const.kB * mT**2) # J**2 / (kg * J/K * K**2) -> J/(kg K)

        return Cp


    @classmethod
    def heat_capacity_Cv(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.heat_capacity_Cv

        Calculate isochoric specific heat capacity from thermodynamic data in a log file
        Cv = Cp - V*T*alpha_P**2 / (beta_T * m)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isochoric heat capacity (float, J/(kg K))
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        M = V * (thermo_df['Density'].to_numpy() * 1e+3) # m**3 * (g/cm**3) -> m**3 * (kg/m**3) = kg

        T = thermo_df['Temp'].to_numpy() # K

        mV = V[init:last].mean()
        mM = M[init:last].mean() if mass is None else mass
        mT = T[init:last].mean() if temp is None else temp

        Cp = cls.heat_capacity_Cp(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)
        alpha_P = cls.volume_expansion(thermo_df, temp=temp, press=press, init=init, last=last)
        beta_T = cls.isothermal_compressibility(thermo_df, temp=temp, init=init, last=last)

        Cv = Cp - mV * mT * alpha_P**2 / beta_T / mM # m**3 * K * K**-2 / (m s**2 / kg) / kg = m**2 * K**-1 * s**-2 * kg * kg**-1 = J/(kg K)

        return Cv


    @classmethod
    def heat_capacity_Cv_NVT(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.heat_capacity_Cv_NVT

        Calculate isochoric specific heat capacity from thermodynamic data in a log file (for NVT)
        Cv = Var(U)/(m*kB*T)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isobaric heat capacity (float, J/(kg K))
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        M = V * (thermo_df['Density'].to_numpy() * 1e+3) # m**3 * (g/cm**3) -> m**3 * (kg/m**3) = kg

        T = thermo_df['Temp'].to_numpy() # K

        if 'TotEng' in thermo_df.columns:
            U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        else:
            U = (thermo_df['PotEng'].to_numpy() + thermo_df['KinEng'].to_numpy()) * const.cal2j * 1000 / const.NA # kcal/mol -> J

        mM = M[init:last].mean() if mass is None else mass
        mT = T[init:last].mean() if temp is None else temp
        U_var = np.var(U[init:last])

        Cv = U_var / (mM * const.kB * mT**2) # J**2 / (kg * J/K * K**2) -> J/(kg K)

        return Cv


    @classmethod
    def isothermal_compressibility(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.isothermal_compressibility

        Calculate isothermal compressibility from thermodynamic data in a log file
        beta_T = Var(V)/(V*kB*T)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isothermal compressibility (float, Pa**-1 = m s**2 / kg)
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        T = thermo_df['Temp'].to_numpy() # K

        mV = V[init:last].mean()
        mT = T[init:last].mean() if temp is None else temp
        V_var = np.var(V[init:last])

        beta_T = V_var / (mV * const.kB * mT) # m**6 / (m**3 * J/K * K) = m**3/J = m**3 * (s**2/kg m**2) = m s**2 / kg

        return beta_T


    @classmethod
    def isentropic_compressibility(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.isentropic_compressibility

        Calculate isentropic (or adiabatic) compressibility from thermodynamic data in a log file
        beta_S = bata_T*Cv/Cp

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isentropic compressibility (float, Pa**-1 = m s**2 / kg)
        """

        Cp = cls.heat_capacity_Cp(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)
        Cv = cls.heat_capacity_Cv(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)
        beta_T = cls.isothermal_compressibility(thermo_df, temp=temp, init=init, last=last)

        beta_S = beta_T * Cv / Cp

        return beta_S


    @classmethod
    def bulk_modulus(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.bulk_modulus

        Calculate (isothermal) bulk modulus from thermodynamic data in a log file
        K_T = 1/beta_T

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            init: Initial step (int)
            last: Last step (int)

        Return:
            bulk modulus (float, Pa = kg / (m s**2))
        """

        return 1 / cls.isothermal_compressibility(thermo_df, temp=temp, init=init, last=last)


    @classmethod
    def isentropic_bulk_modulus(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.isentropic_bulk_modulus

        Calculate isentropic bulk modulus from thermodynamic data in a log file
        K_S = 1/beta_S

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isentropic bulk modulus (float, Pa = kg / (m s**2))
        """

        return 1 / cls.isentropic_compressibility(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)


    @classmethod
    def speed_of_sound_lq(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.speed_of_sound_lq

        Calculate speed of sound as a liquid from thermodynamic data in a log file
        The Newton–Laplace equation: c = sqrt(1/(rho*beta_S)) or c = sqrt(K_S/rho)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            speed of sound (float, m/s)
        """

        beta_S = cls.isentropic_compressibility(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last) # m s**2 / kg
        #K_S = cls.isentropic_bulk_modulus(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last) # kg / m s**2
        rho = thermo_df['Density'].to_numpy() * 1e+3 # g/cm**3 -> kg/m**3
        m_rho = rho[init:last].mean()

        c = math.sqrt(1/(m_rho*beta_S)) # sqrt( (m**3/kg) * (kg / (m s**2)) ) = sqrt(m**2 / s**2) = m/s
        #c = math.sqrt(K_S/m_rho) # sqrt( (kg / (m s**2)) / (kg/m**3) ) = sqrt(m**2 / s**2) = m/s

        return c


    @classmethod
    def volume_expansion(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.volume_expansion

        Calculate (isobaric volumetric) thermal expansion coefficient from thermodynamic data in a log file
        alpha_P = Cov(V, H) / (V*kB*T**2)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            volume expansion (float, K**-1)
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        T = thermo_df['Temp'].to_numpy() # K

        U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        P = press * const.atm2pa # Pa = J / m**3
        H = U + P * V

        mV = V[init:last].mean()
        mT = T[init:last].mean() if temp is None else temp
        VH_cov = np.sum((V[init:last] - mV)*(H[init:last] - H[init:last].mean())) / len(V[init:last])

        alpha_P = VH_cov / (mV * const.kB * mT**2) # m**3 * J / (m**3 * J/K * K**2) = 1/K

        return alpha_P


    @classmethod
    def linear_expansion(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.linear_expansion

        Calculate (isotropic, isobaric) linear expansion coefficient from thermodynamic data in a log file
        alpha_lP = alpha_P / 3

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            linear expansion (float, K**-1)
        """

        return cls.volume_expansion(thermo_df, temp=temp, press=press, init=init, last=last) / 3


    @classmethod
    def linear_expansion_aniso(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.linear_expansion_aniso

        Calculate anisotropic (isobaric) linear expansion coefficient from thermodynamic data in a log file
        alpha_lP = alpha_P / 3

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            linear expansion x, y, z (float, K**-1)
        """

        Lx = thermo_df['Lx'].to_numpy() * 1e-10 # Angstrom -> m
        Ly = thermo_df['Ly'].to_numpy() * 1e-10
        Lz = thermo_df['Lz'].to_numpy() * 1e-10

        T = thermo_df['Temp'].to_numpy() # K

        U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        P = press * const.atm2pa # Pa = J / m**3
        H = U + P * (Lx * Ly * Lz)

        mLx = Lx[init:last].mean()
        mLy = Ly[init:last].mean()
        mLz = Lz[init:last].mean()
        mT = T[init:last].mean() if temp is None else temp
        mH = H[init:last].mean()
        N = len(T[init:last])

        LxH_cov = np.sum((Lx[init:last] - mLx)*(H[init:last] - mH)) / N
        LyH_cov = np.sum((Ly[init:last] - mLy)*(H[init:last] - mH)) / N
        LzH_cov = np.sum((Lz[init:last] - mLz)*(H[init:last] - mH)) / N

        alpha_Lx = LxH_cov / (mLx * const.kB * mT**2) # m * J / (m * J/K * K**2) = 1/K
        alpha_Ly = LyH_cov / (mLy * const.kB * mT**2)
        alpha_Lz = LzH_cov / (mLz * const.kB * mT**2)

        return alpha_Lx, alpha_Ly, alpha_Lz


    @classmethod
    def self_diffusion(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.self_diffusion

        Calculate self-diffusion coefficient from thermodynamic data in a log file
        D = 1/6 * d(MSD)/dt

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            self-diffusion coefficient (float, m**2/s)
        """

        MSD = thermo_df['v_msd'].to_numpy() * 1e-20 # Angstrom^2 -> m^2
        t = thermo_df['Time'].to_numpy() * 1e-15 # fs -> s
        grad, k, r, p, se = stats.linregress(t[init:last], MSD[init:last])
        diffc = grad / 6

        return diffc


    def analyze_traj(self, traj, prop, conv_a=1.0, conv_b=0.0, ylabel=None, temp=300, periodic=False, charges=None, enthalpy=None,
                     init=1000, last=None, width=1000, printout=True, save=None):

        if not mdtraj_avail:
            utils.radon_print('mdtraj is not available. You can use analyze_traj by "conda install -c conda-forge mdtraj"', level=3)
            return None, None

        data = {}
        chain_prop_data = []
        chain_id_code = const.pdb_id

        if type(prop) is list:
            prop_data = pd.Series(prop, index=traj.time)

        elif prop == 'rmsd':
            #prop_data = pd.Series(mdtraj.rmsd(traj, traj, frame=0), index=traj.time)
            for i in range(traj.n_chains):
                atom_list = traj.topology.select("chainid %s" % (i))
                traj_tmp = traj.atom_slice(atom_list)
                prop_tmp = mdtraj.rmsd(traj_tmp, traj_tmp, frame=0)
                chain_prop_data.append(prop_tmp)

            chain_prop_data = np.array(chain_prop_data)
            prop_data = np.mean(chain_prop_data, axis=0)
            prop_data = pd.Series(prop_data, index=traj.time)

        elif prop == 'r2':
            res1 = 0
            res2 = 0
            for i in range(traj.n_chains):
                atom_list = traj.topology.select("chainid %s" % (i))
                traj_tmp = traj.atom_slice(atom_list)

                for j, r in enumerate(traj_tmp.topology.residues):
                    if r.name == 'TU0':
                        res1 = j
                    elif r.name == 'TU1':
                        res2 = j

                prop_tmp = mdtraj.compute_contacts(traj_tmp, contacts=[(res1, res2)], scheme='closest', periodic=periodic)
                chain_prop_data.append(prop_tmp[0].flatten())

            chain_prop_data = np.array(chain_prop_data)
            prop_data = np.mean(chain_prop_data**2, axis=0)
            prop_data = pd.Series(prop_data, index=traj.time)

        elif prop == 'rg':
            #prop_data = pd.Series(mdtraj.compute_rg(traj), index=traj.time)
            for i in range(traj.n_chains):
                atom_list = traj.topology.select("chainid %s" % (i))
                traj_tmp = traj.atom_slice(atom_list)
                prop_tmp = mdtraj.compute_rg(traj_tmp)
                chain_prop_data.append(prop_tmp)

            chain_prop_data = np.array(chain_prop_data)
            prop_data = np.mean(chain_prop_data, axis=0)
            prop_data = pd.Series(prop_data, index=traj.time)

        elif prop == 'density':
            prop_data = pd.Series(mdtraj.density(traj), index=traj.time)

        elif prop == 'order_param':
            prop_data = pd.Series(mdtraj.compute_nematic_order(traj, indices='residues'), index=traj.time)
            data['director'] = mdtraj.compute_directors(traj, indices='residues')

        elif prop == 'dipole_moments': # nm * e
            data['dipole'] = pd.DataFrame(mdtraj.dipole_moments(traj, charges), columns=['x', 'y', 'z'], index=traj.time)
            dipole_data = np.sqrt(data['dipole'].x.values**2 + data['dipole'].y.values**2 + data['dipole'].z.values**2)
            prop_data = pd.Series(dipole_data, index=traj.time)

        elif prop == 'dielectric':
            n = traj[init:last:width].n_frames
            prop_data_tmp = []
            time = []
            for i in range(n):
                diele = np.mean(mdtraj.static_dielectric(traj[init-width:init+i*width], charges, temp))
                prop_data_tmp.append(diele)
                time.append(traj.time[init+i*width])
            if traj.n_frames > init+(n-1)*width:
                diele = np.mean(mdtraj.static_dielectric(traj[-width:], charges, temp))
                prop_data_tmp.append(diele)
                time.append(traj.time[-1])
            prop_data = pd.Series(prop_data_tmp, index=time)

        elif prop == 'diffusion_coeff': # m**2/s
            n = traj[init:last].n_frames
            msd = mdtraj.rmsd(traj, traj, frame=0) ** 2
            prop_data_tmp = []
            for i in range(n):
                time = traj.time[init-width+i:init+1+i] - traj.time[init-width+i]
                grad, k, r, p, std = stats.linregress(msd[init-width+i:init+1+i], time)
                prop_data_tmp.append(grad/6 * 1e-8)  # angstrom**2/ps -> m**2/s
            prop_data = pd.Series(prop_data_tmp, index=traj.time[init:last])

        elif prop == 'compressibility': # bar^-1
            n = traj[init:last:width].n_frames
            prop_data_tmp = []
            time = []
            for i in range(n):
                kappa_T = np.mean(mdtraj.isothermal_compressability_kappa_T(traj[init-width:init+i*width], temp))
                prop_data_tmp.append(kappa_T)
                time.append(traj.time[init+i*width])
            if traj.n_frames > init+(n-1)*width:
                kappa_T = np.mean(mdtraj.isothermal_compressability_kappa_T(traj[-width:], temp))
                prop_data_tmp.append(kappa_T)
                time.append(traj.time[-1])
            prop_data = pd.Series(prop_data_tmp, index=time)

        elif prop == 'expansion': # K^-1
            n = traj[init:last:width].n_frames
            prop_data_tmp = []
            time = []
            for i in range(n):
                alpha_P = np.mean(mdtraj.thermal_expansion_alpha_P(traj[init-width:init+i*width], temp, enthalpy))
                prop_data_tmp.append(alpha_P)
                time.append(traj.time[init+i*width])
            if traj.n_frames > init+(n-1)*width:
                alpha_P = np.mean(mdtraj.thermal_expansion_alpha_P(traj[-width:], temp, enthalpy))
                prop_data_tmp.append(alpha_P)
                time.append(traj.time[-1])
            prop_data = pd.Series(prop_data_tmp, index=time)

        prop_data = prop_data * conv_a + conv_b
        data_sma = prop_data.rolling(width).mean()
        data_sd = prop_data.rolling(width).std()
        data_se = data_sd / np.sqrt(width)

        if not last: last = 0
        if traj.n_frames < last: last = 0

        data['init'] = traj.time[last-width-1]
        data['last'] = traj.time[last-1]
        data['width'] = width
        if prop in ['dielectric', 'compressibility', 'expansion']:
            data['mean'] = prop_data.values[-1]
        else:
            data['mean'] = data_sma.values[int(last-1)]
            data['sd'] = data_sd.values[int(last-1)]
            data['se'] = data_se.values[int(last-1)]
            data['sma_sd'] = data_sma.values[last-width-1:last].std() if last else data_sma.values[-width-1:].std()
            data['sma_se'] = data['sma_sd'] / np.sqrt(width)

        if printout or save:
            if not last: last = None
            fig, ax = pp.subplots(figsize=(6, 6))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))

            if prop in ['dielectric', 'compressibility', 'expansion']:
                ax.plot(time, prop_data.values, linewidth=2.0)
            else:
                ax.plot(traj.time[init:last], prop_data.values[init:last], linewidth=0.1)

            if prop in ['dielectric', 'compressibility', 'expansion']:
                pass
            elif prop in ['diffusion_coeff']:
                ax.errorbar(data_sma.index[init:last], data_sma.values[init:last], yerr=data_se.values[init+width:last]*2, linewidth=2.0)
            else:
                ax.errorbar(data_sma.index[init:last], data_sma.values[init:last], yerr=data_se.values[init:last]*2, linewidth=2.0)
            ax.set_xlabel('Time [ps]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            output = 'Accumulation of %f - %f ps\n' % (data['init'], data['last'])
            if prop in ['dielectric', 'compressibility', 'expansion']:
                output += '%s = %e' % (ylabel, data['mean'])
            else:
                output += '%s = %e     SD = %e    SE = %e' % (ylabel, data['mean'], data['sd'], data['se'])
                output += 'SMA_SD = %e     SMA_SE = %e' % (data['sma_sd'], data['sma_se'])

            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                label = ylabel if type(prop) is list else prop
                fig.savefig(os.path.join(save, label)+'.png')
                with open(os.path.join(save, label)+'.txt', mode='w') as f:
                    f.write(output)

            pp.close(fig)

        return prop_data, data


    def get_all_prop(self, temp=300.0, press=1.0, width=2000, init=2000, last=None, f_width=2000,
                printout=False, save=False, save_name='analyze', do_traj=True):

        thermo_df = self.dfs[-1]

        if save:
            save_dir = os.path.join(os.path.dirname(self.log_file), save_name)
        else:
            save_dir = None

        self.totene_data = self.analyze_thermo('TotEng', conv_a=const.cal2j, ylabel='Total energy [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.kinene_data = self.analyze_thermo('KinEng', conv_a=const.cal2j, ylabel='Kinetic energy [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.ebond_data = self.analyze_thermo('E_bond', conv_a=const.cal2j, ylabel='Potential energy of bonds [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.eangle_data = self.analyze_thermo('E_angle', conv_a=const.cal2j, ylabel='Potential energy of angles [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.edihed_data = self.analyze_thermo('E_dihed', conv_a=const.cal2j, ylabel='Potential energy of dihedrals [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.evdw_data = self.analyze_thermo('E_vdwl', conv_a=const.cal2j, ylabel='Potential energy of vdW [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.ecoul_data = self.analyze_thermo('E_coul', conv_a=const.cal2j, ylabel='Potential energy of coulomb [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.elong_data = self.analyze_thermo('E_long', conv_a=const.cal2j, ylabel='Potential energy of Kspace [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.temp_data = self.analyze_thermo('Temp', ylabel='Temperature [K]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.dens_data = self.analyze_thermo('Density', ylabel='Density [g/cm^3]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        if self.has_rg_data():
            self.rg_data = self.calc_rg(rg_file=self.rg_file, init=-width, last=last)

        if 'v_msd' in thermo_df.columns.tolist():
            self.msd_data = self.analyze_thermo('v_msd', ylabel='MSD [Angstrome^2]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)
            self.diffc_data = self.analyze_thermo_fluctuation(self.self_diffusion,
                            name='self_diffusion', ylabel='Self-diffusion coeffisient [m^2/s]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.Cp_data = self.analyze_thermo_fluctuation(self.heat_capacity_Cp,
                            name='heat_capacity_Cp', ylabel='Heat capacity Cp [J/(kg K)]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.Cv_data = self.analyze_thermo_fluctuation(self.heat_capacity_Cv,
                            name='heat_capacity_Cv', ylabel='Heat capacity Cv [J/(kg K)]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.compress_T_data = self.analyze_thermo_fluctuation(self.isothermal_compressibility,
                            name='isothermal_compressibility', ylabel='Isothermal compressibility [1/Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.compress_S_data = self.analyze_thermo_fluctuation(self.isentropic_compressibility,
                            name='isentropic_compressibility', ylabel='Isentropic compressibility [1/Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.bulk_mod_T_data = self.analyze_thermo_fluctuation(self.bulk_modulus,
                            name='bulk_modulus', ylabel='Bulk modulus [Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.bulk_mod_S_data = self.analyze_thermo_fluctuation(self.isentropic_bulk_modulus,
                            name='isentropic_bulk_modulus', ylabel='Isentropic bulk modulus [Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.volume_exp_data = self.analyze_thermo_fluctuation(self.volume_expansion,
                            name='volume_expansion', ylabel='Volume expansion [1/K]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.linear_exp_data = self.analyze_thermo_fluctuation(self.linear_expansion,
                            name='linear_expansion', ylabel='Linear expansion [1/K]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        prop_data = {
            'density': self.dens_data.get('mean', np.nan),
            'Rg': self.rg_data.get('mean_mean', np.nan),
            'self-diffusion': self.diffc_data.get('mean', np.nan),
            'Cp': self.Cp_data.get('mean', np.nan),
            'Cv': self.Cv_data.get('mean', np.nan),
            'compressibility': self.compress_T_data.get('mean', np.nan),
            'isentropic_compressibility': self.compress_S_data.get('mean', np.nan),
            'bulk_modulus': self.bulk_mod_T_data.get('mean', np.nan),
            'isentropic_bulk_modulus': self.bulk_mod_S_data.get('mean', np.nan),
            'volume_expansion': self.volume_exp_data.get('mean', np.nan),
            'linear_expansion': self.linear_exp_data.get('mean', np.nan),
        }

        conv_data = {
            'totene': self.totene_data.get('mean', np.nan),
            'totene_sd': self.totene_data.get('sd', np.nan),
            'totene_se': self.totene_data.get('se', np.nan),
            'totene_sma_sd': self.totene_data.get('sma_sd', np.nan),
            'totene_sma_se': self.totene_data.get('sma_se', np.nan),
            'kinene': self.kinene_data.get('mean', np.nan),
            'kinene_sd': self.kinene_data.get('sd', np.nan),
            'kinene_se': self.kinene_data.get('se', np.nan),
            'kinene_sma_sd': self.kinene_data.get('sma_sd', np.nan),
            'kinene_sma_se': self.kinene_data.get('sma_se', np.nan),
            'ebond': self.ebond_data.get('mean', np.nan),
            'ebond_sd': self.ebond_data.get('sd', np.nan),
            'ebond_se': self.ebond_data.get('se', np.nan),
            'ebond_sma_sd': self.ebond_data.get('sma_sd', np.nan),
            'ebond_sma_se': self.ebond_data.get('sma_se', np.nan),
            'eangle': self.eangle_data.get('mean', np.nan),
            'eangle_sd': self.eangle_data.get('sd', np.nan),
            'eangle_se': self.eangle_data.get('se', np.nan),
            'eangle_sma_sd': self.eangle_data.get('sma_sd', np.nan),
            'eangle_sma_se': self.eangle_data.get('sma_se', np.nan),
            'edihed': self.edihed_data.get('mean', np.nan),
            'edihed_sd': self.edihed_data.get('sd', np.nan),
            'edihed_se': self.edihed_data.get('se', np.nan),
            'edihed_sma_sd': self.edihed_data.get('sma_sd', np.nan),
            'edihed_sma_se': self.edihed_data.get('sma_se', np.nan),
            'evdw': self.evdw_data.get('mean', np.nan),
            'evdw_sd': self.evdw_data.get('sd', np.nan),
            'evdw_se': self.evdw_data.get('se', np.nan),
            'evdw_sma_sd': self.evdw_data.get('sma_sd', np.nan),
            'evdw_sma_se': self.evdw_data.get('sma_se', np.nan),
            'ecoul': self.ecoul_data.get('mean', np.nan),
            'ecoul_sd': self.ecoul_data.get('sd', np.nan),
            'ecoul_se': self.ecoul_data.get('se', np.nan),
            'ecoul_sma_sd': self.ecoul_data.get('sma_sd', np.nan),
            'ecoul_sma_se': self.ecoul_data.get('sma_se', np.nan),
            'elong': self.elong_data.get('mean', np.nan),
            'elong_sd': self.elong_data.get('sd', np.nan),
            'elong_se': self.elong_data.get('se', np.nan),
            'elong_sma_sd': self.elong_data.get('sma_sd', np.nan),
            'elong_sma_se': self.elong_data.get('sma_se', np.nan),
            'density_sd': self.dens_data.get('sd', np.nan),
            'density_se': self.dens_data.get('se', np.nan),
            'density_sma_sd': self.dens_data.get('sma_sd', np.nan),
            'density_sma_se': self.dens_data.get('sma_se', np.nan),
            'Rg_sd_max': self.rg_data.get('sd_max', np.nan),
            'Rg_se_max': self.rg_data.get('se_max', np.nan),
            'Rg_mean_sd': self.rg_data.get('mean_sd', np.nan),
            'Rg_mean_se': self.rg_data.get('mean_se', np.nan),
            'self-diff_sd': self.diffc_data.get('sd', np.nan),
            'self-diff_se': self.diffc_data.get('se', np.nan),
            'self-diff_sma_sd': self.diffc_data.get('sma_sd', np.nan),
            'self-diff_sma_se': self.diffc_data.get('sma_se', np.nan),
            'Cp_sd': self.Cp_data.get('sd', np.nan),
            'Cp_se': self.Cp_data.get('se', np.nan),
            'Cp_sma_sd': self.Cp_data.get('sma_sd', np.nan),
            'Cp_sma_se': self.Cp_data.get('sma_se', np.nan),
            'Cv_sd': self.Cv_data.get('sd', np.nan),
            'Cv_se': self.Cv_data.get('se', np.nan),
            'Cv_sma_sd': self.Cv_data.get('sma_sd', np.nan),
            'Cv_sma_se': self.Cv_data.get('sma_se', np.nan),
            'compress_T_sd': self.compress_T_data.get('sd', np.nan),
            'compress_T_se': self.compress_T_data.get('se', np.nan),
            'compress_T_sma_sd': self.compress_T_data.get('sma_sd', np.nan),
            'compress_T_sma_se': self.compress_T_data.get('sma_se', np.nan),
            'compress_S_sd': self.compress_S_data.get('sd', np.nan),
            'compress_S_se': self.compress_S_data.get('se', np.nan),
            'compress_S_sma_sd': self.compress_S_data.get('sma_sd', np.nan),
            'compress_S_sma_se': self.compress_S_data.get('sma_se', np.nan),
            'bulk_mod_T_sd': self.bulk_mod_T_data.get('sd', np.nan),
            'bulk_mod_T_se': self.bulk_mod_T_data.get('se', np.nan),
            'bulk_mod_T_sma_sd': self.bulk_mod_T_data.get('sma_sd', np.nan),
            'bulk_mod_T_sma_se': self.bulk_mod_T_data.get('sma_se', np.nan),
            'bulk_mod_S_sd': self.bulk_mod_S_data.get('sd', np.nan),
            'bulk_mod_S_se': self.bulk_mod_S_data.get('se', np.nan),
            'bulk_mod_S_sma_sd': self.bulk_mod_S_data.get('sma_sd', np.nan),
            'bulk_mod_S_sma_se': self.bulk_mod_S_data.get('sma_se', np.nan),
            'volume_exp_sd': self.volume_exp_data.get('sd', np.nan),
            'volume_exp_se': self.volume_exp_data.get('se', np.nan),
            'volume_exp_sma_sd': self.volume_exp_data.get('sma_sd', np.nan),
            'volume_exp_sma_se': self.volume_exp_data.get('sma_se', np.nan),
            'linear_exp_sd': self.linear_exp_data.get('sd', np.nan),
            'linear_exp_se': self.linear_exp_data.get('se', np.nan),
            'linear_exp_sma_sd': self.linear_exp_data.get('sma_sd', np.nan),
            'linear_exp_sma_se': self.linear_exp_data.get('sma_se', np.nan),
        }

        if do_traj and mdtraj_avail:
            self.read_traj()
            self.get_partial_charges()

            self.r2, self.r2_data = self.analyze_traj(self.traj, 'r2', ylabel='<R2> [nm^2]',
                                width=width, init=init, last=last, printout=printout, save=save_dir, temp=temp, charges=self.charges)

            self.diele, self.diele_data = self.analyze_traj(self.traj, 'dielectric', ylabel='Static dielectric constant',
                                width=width, init=init, last=last, printout=printout, save=save_dir, temp=temp, charges=self.charges)

            self.nop, self.nop_data = self.analyze_traj(self.traj, 'order_param', ylabel='Nematic order parameter',
                                width=width, init=init, last=last, printout=printout, save=save_dir)

            prop_data['r2'] = self.r2_data.get('mean', np.nan)
            conv_data['r2_sd'] = self.r2_data.get('sd', np.nan)
            conv_data['r2_se'] = self.r2_data.get('se', np.nan)
            conv_data['r2_sma_sd'] = self.r2_data.get('sma_sd', np.nan)
            conv_data['r2_sma_se'] = self.r2_data.get('sma_se', np.nan)

            prop_data['static_dielectric_const'] = self.diele_data.get('mean', np.nan)

            prop_data['nematic_order_parameter'] = self.nop_data.get('mean', np.nan)
            conv_data['nematic_order_parameter_sd'] = self.nop_data.get('sd', np.nan)
            conv_data['nematic_order_parameter_se'] = self.nop_data.get('se', np.nan)
            conv_data['nematic_order_parameter_sma_sd'] = self.nop_data.get('sma_sd', np.nan)
            conv_data['nematic_order_parameter_sma_se'] = self.nop_data.get('sma_se', np.nan)

        elif not mdtraj_avail:
            utils.radon_print('mdtraj is not available. You can use analyze_traj by "conda install -c conda-forge mdtraj"', level=3)

        self.prop_df = pd.DataFrame(prop_data, index=[0])
        self.conv_df = pd.DataFrame(conv_data, index=[0])
        if save:
            self.prop_df.to_csv(os.path.join(save_dir, 'eq_prop_data.csv'))
            self.conv_df.to_csv(os.path.join(save_dir, 'eq_conv_data.csv'))

        return prop_data


    def check_eq(self, do_analyze=False, temp=300.0, press=1.0, width=2000, init=2000, last=None, f_width=2000,
                printout=False, save=False, save_name='analyze'):

        if do_analyze:
            self.get_all_prop(temp=temp, press=press, printout=printout, width=width, init=init, last=last,
                f_width=f_width,save=save, save_name=save_name, do_traj=False)

        check = True
        conv_data = {}

        if self.totene_data.get('sma_sd') is not None and self.totene_data.get('mean') is not None and self.totene_sma_sd_crit is not None:
            if self.totene_data['sma_sd'] > abs(self.totene_data['mean']) * self.totene_sma_sd_crit:
                utils.radon_print('Total energy does not converge. mean = %f, sma_sd = %f'
                    % (self.totene_data['mean'], self.totene_data['sma_sd']), level=2)
                check = False
        elif self.totene_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of total energy.', level=2)
            return False

        if self.kinene_data.get('sma_sd') is not None and self.kinene_data.get('mean') and self.kinene_sma_sd_crit is not None:
            if self.kinene_data['sma_sd'] > abs(self.kinene_data['mean']) * self.kinene_sma_sd_crit:
                utils.radon_print('Kinetic energy does not converge. mean = %f, sma_sd = %f'
                    % (self.kinene_data['mean'], self.kinene_data['sma_sd']), level=2)
                check = False
        elif self.kinene_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of kinetic energy. Skip to check the kinetic energy convergence.', level=2)

        if self.ebond_data.get('sma_sd') is not None and self.ebond_data.get('mean') is not None and self.ebond_sma_sd_crit is not None:
            if self.ebond_data['sma_sd'] > abs(self.ebond_data['mean']) * self.ebond_sma_sd_crit:
                utils.radon_print('Potential energy of bonds does not converge. mean = %f, sma_sd = %f'
                    % (self.ebond_data['mean'], self.ebond_data['sma_sd']), level=2)
                check = False
        elif self.ebond_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of bonds.', level=2)
            return False

        if self.eangle_data.get('sma_sd') is not None and self.eangle_data.get('mean') is not None and self.eangle_sma_sd_crit is not None:
            if self.eangle_data['sma_sd'] > abs(self.eangle_data['mean']) * self.eangle_sma_sd_crit:
                utils.radon_print('Potential energy of angles does not converge. mean = %f, sma_sd = %f'
                    % (self.eangle_data['mean'], self.eangle_data['sma_sd']), level=2)
                check = False
        elif self.eangle_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of angles.', level=2)
            return False

        if self.edihed_data.get('sma_sd') is not None and self.edihed_data.get('mean') is not None and self.edihed_sma_sd_crit is not None:
            if self.edihed_data['sma_sd'] > abs(self.edihed_data['mean']) * self.edihed_sma_sd_crit:
                utils.radon_print('Potential energy of dihedrals does not converge. mean = %f, sma_sd = %f'
                    % (self.edihed_data['mean'], self.edihed_data['sma_sd']), level=2)
                check = False
        elif self.edihed_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of dihedrals.', level=2)
            return False

        if self.evdw_data.get('sma_sd') is not None and self.evdw_sma_sd_crit is not None:
            if self.evdw_data['sma_sd'] > self.evdw_sma_sd_crit:
                utils.radon_print('Potential energy of vdW interactions does not converge. mean = %f, sma_sd = %f'
                    % (self.evdw_data['mean'], self.evdw_data['sma_sd']), level=2)
                check = False
        elif self.evdw_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of vdW interactions.', level=2)
            return False

        if self.ecoul_data.get('sma_sd') is not None and self.ecoul_sma_sd_crit is not None:
            if self.ecoul_data['sma_sd'] > self.ecoul_sma_sd_crit:
                utils.radon_print('Potential energy of coulomb interactions does not converge. mean = %f, sma_sd = %f'
                    % (self.ecoul_data['mean'], self.ecoul_data['sma_sd']), level=2)
                check = False
        elif self.ecoul_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of coulomb interactions.', level=2)
            return False

        if self.elong_data.get('sma_sd') is not None and self.elong_data.get('mean') is not None and self.elong_sma_sd_crit is not None:
            if self.elong_data['sma_sd'] > abs(self.elong_data['mean']) * self.elong_sma_sd_crit:
                utils.radon_print('Potential energy of KSpace does not converge. mean = %f, sma_sd = %f'
                    % (self.elong_data['mean'], self.elong_data['sma_sd']), level=2)
                check = False
        elif self.elong_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of potential energy of KSpace. Skip to check the KSpace energy convergence.', level=2)

        if self.dens_data.get('sma_sd') is not None and self.dens_data.get('mean') is not None and self.dens_sma_sd_crit is not None:
            if self.dens_data['sma_sd'] > self.dens_data['mean'] * self.dens_sma_sd_crit:
                utils.radon_print('Density does not converge. mean = %f, sma_sd = %f'
                    % (self.dens_data['mean'], self.dens_data['sma_sd']), level=2)
                check = False
        elif self.dens_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the density.', level=2)
            return False

        if self.rg_data.get('sd_max') is not None and self.rg_data.get('mean_mean') is not None and self.rg_sd_crit is not None:
            if self.rg_data['sd_max'] > self.rg_data['mean_mean'] * self.rg_sd_crit:
                utils.radon_print('Radius of gyration does not converge. mean = %f, sd = %f'
                    % (self.rg_data['mean_mean'], self.rg_data['sd_max']), level=2)
                check = False
        elif self.rg_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of radius of gyration. Skip to check the Rg convergence.', level=2)

        if self.diffc_data.get('sma_sd') is not None and self.diffc_data.get('mean') is not None and self.diffc_sma_sd_crit is not None:
            if self.diffc_data['sma_sd'] > self.diffc_data['mean'] * self.diffc_sma_sd_crit:
                utils.radon_print('Self-diffusion coefficient does not converge. mean = %f, sma_sd = %f'
                    % (self.diffc_data['mean'], self.diffc_data['sma_sd']), level=2)
                check = False
        elif self.diffc_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of self-diffusion coefficient. Skip to check the self-diffusion coefficient convergence.', level=1)

        if self.Cp_data.get('sma_sd') is not None and self.Cp_data.get('mean') is not None and self.Cp_sma_sd_crit is not None:
            if self.Cp_data['sma_sd'] > self.Cp_data['mean'] * self.Cp_sma_sd_crit:
                utils.radon_print('Cp does not converge. mean = %f, sma_sd = %f'
                    % (self.Cp_data['mean'], self.Cp_data['sma_sd']), level=2)
                check = False
        elif self.Cp_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of Cp. Skip to check the Cp convergence.', level=2)

        if self.compress_T_data.get('sma_sd') is not None and self.compress_T_data.get('mean') is not None and self.compress_sma_sd_crit is not None:
            if self.compress_T_data['sma_sd'] > self.compress_T_data['mean'] * self.compress_sma_sd_crit:
                utils.radon_print('Compressibility does not converge. mean = %f, sma_sd = %f'
                    % (self.compress_T_data['mean'], self.compress_T_data['sma_sd']), level=2)
                check = False
        elif self.compress_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of compressibility. Skip to check the compressibility convergence.', level=2)

        if self.volume_exp_data.get('sma_sd') is not None and self.volume_exp_data.get('mean') is not None and self.volexp_sma_sd_crit is not None:
            if self.volume_exp_data['sma_sd'] > self.volume_exp_data['mean'] * self.volexp_sma_sd_crit:
                utils.radon_print('Volumetric thermal expantion coefficient does not converge. mean = %f, sma_sd = %f'
                    % (self.volume_exp_data['mean'], self.volume_exp_data['sma_sd']), level=2)
                check = False
        elif self.volexp_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of volumetric thermal expantion coefficient. Skip to check the volumetric thermal expantion coefficient convergence.', level=2)

        return check


    def has_rg_data(self):
        return os.path.exists(self.rg_file)
