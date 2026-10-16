#%%
import numpy as np
import matplotlib.pyplot as plt
import os 
from ase.io import read, write
# import ase.units as 
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Callable, Dict, List, Tuple, Union
R = 8.3145e-3  # kJ/mol/K
import sys, os
# Dynamically add the metal repo root to sys.path
repo_root = os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 
    'metal')
sys.path.append(repo_root)
sys.path.append("/cfs/klemming/projects/supr/naiss2023-5-521/stanicic/gitlab/metal")

# Now you can import modules using absolute imports from metal
from metal.applications.crystals.tools import (
    get_mu_oxygen_ref,
    get_mu_oxygen,
    get_mu_oxygen_sr_ref,
    get_mu_oxygen_sr
)

#        'AR': {'T': 1000 + 273.15, 'pO2': 0.05},
#        'FR': {'T': 900 + 273.15, 'pO2': 1e-16}, #previous po2 = 1e-16.  ratio 1e-6 corresponds to 99.99 % conversion to H2O. For 99.99% the ratio 0.0001 is sufficient.
#        'SR':{'T': 800 + 273.15, 'ratio': 0.61511}, # ratio = 0.61511, logmean concentraiton is 62% assuming we allow 35 vol% H2O in the outlet

#%% PLOTTING THE POTENTIAL AS FUNCTION OF TEMPERATURE
# Temperatures (K)
T = np.array([298.15+800, 298.15+900, 298.15+1000
              ])

# --- O2 route ---
pO2_vol = [0.05, 1e-16]  # 5%, 21%, 100%
o2_color = 'red'
o2_linestyles = ['dotted','--', '-', ':']

plt.figure(figsize=(8,6))

for pO2, ls in zip(pO2_vol, o2_linestyles):
    mu_O_list = get_mu_oxygen(T, [pO2]*len(T))

    plt.plot(T, mu_O_list, color=o2_color, linestyle=ls, marker='o',
             label=f'AR: O2 {pO2*100:.0f} vol%')




# --- FR route ---
ratios = [1.12]
h2_color = 'green'
h2_linestyles = ['-', '--', ':','-','-','-']

for ratio, ls in zip(ratios, h2_linestyles):
    mu_O_list = get_mu_oxygen_sr(T, ratio)
    plt.plot(T, mu_O_list, color=h2_color, linestyle=ls, marker='s',
             label=f'FR: H2O/H2 = {ratio}')

# --- H2 route ---
ratios = [1/0.615]
h2_color = 'blue'
h2_linestyles = ['-', '--', ':','-','-','-']

for ratio, ls in zip(ratios, h2_linestyles):
    mu_O_list = get_mu_oxygen_sr(T, ratio)
    plt.plot(T, mu_O_list, color=h2_color, linestyle=ls, marker='s',
             label=f'SR: H2O/H2 = {ratio}')

# Plot settings
plt.xlabel('Temperature [K]')
plt.ylabel('Oxygen chemical potential μ_O [kJ/mol]')
plt.title('μ_O for O2 pressures and H2O/H2 ratios')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('testing.png')
plt.show()

#%%
# 
# ratio=1.625722941

mu_O_list = get_mu_oxygen(900+273.15, [1e-16])
print('THIS IS mu_O for 1e-16 in FR')
print(mu_O_list)
mu_O_list_sr =  get_mu_oxygen_sr(800+273.15, 1.625722941)
print('THIS IS mu_O for ratio=0.615 in SR')
print(mu_O_list_sr)


mu_O_list_sr_ref=get_mu_oxygen_sr_ref(800+273.15)/96.485

pO2=np.exp(2 * (mu_O_list_sr - mu_O_list_sr_ref) / (R * (800+273.15)))

#%% THIS GIVES REASONABLE OUTPUT FOR THE PO2 as a function of the RATIO H2O/H2!
R = 8.617333262e-5  # eV/K per atom
NA = 6.02214076e23       # Avogadro number

# Temperatures (K)
T = np.array([298.15+800, 298.15+900, 298.15+1000])

# --- O2 route ---
pO2_vol = [1e-16]  # 5%, 21%, 100%
o2_color = 'red'
o2_linestyles = ['dotted','--', '-', ':']

plt.figure(figsize=(8,6))

for pO2, ls in zip(pO2_vol, o2_linestyles):
    mu_O_list = get_mu_oxygen(T, [pO2]*len(T))
    mu_ref = get_mu_oxygen_ref(T)
    d_mu=mu_O_list-mu_ref
    po2 = np.exp(2*d_mu/(R*T))
    plt.plot(T, np.log10(po2), color=o2_color, linestyle=ls, marker='o',
             label=f'AR: O2 {pO2:.0f} vol%')

# --- FR route ---
ratios = [11.20613368]
CONVFR = [1 / (1 + r) * 100 for r in ratios]
h2_color = 'green'
h2_linestyles = ['-', '--', ':','-','-','-']

for ratio, ls in zip(ratios, h2_linestyles):
    mu_O_list = get_mu_oxygen_sr(T, ratio)
    mu_ref = get_mu_oxygen_sr_ref(T)/96.485
    d_mu=mu_O_list-mu_ref
    po2 = np.exp(2*d_mu/(R*T))
    plt.plot(T, np.log10(po2), color=h2_color, linestyle=ls, marker='s',
             label=f'FR: H2O/H2 = {ratio} CONV = {CONVFR[0]:.6f}')



# --- H2 route ---
ratios = [1.625722941]
CONVSR = [1 / (1 + r) * 100 for r in ratios]
h2_color = 'blue'
h2_linestyles = ['-', '--', ':','-','-','-']

for ratio, ls in zip(ratios, h2_linestyles):
    mu_O_list = get_mu_oxygen_sr(T, ratio)
    mu_ref = get_mu_oxygen_sr_ref(T)/96.485
    d_mu=mu_O_list-mu_ref
    po2 = np.exp(2*d_mu/(R*T))
    plt.plot(T, np.log10(po2), color=h2_color, linestyle=ls, marker='s',
            label=f'SR: H2O/H2 = {ratio} [CONV = {CONVSR[0]:.6f}')

# Plot settings
plt.xlabel('Temperature [K]')
plt.ylabel('Logarithmic oxygen partial pressure [atm]')
plt.title('pO2 for H2O/H2 ratios')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('testing2.png')
plt.show()
# %%
