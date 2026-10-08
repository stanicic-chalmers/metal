from __future__ import annotations

import traceback
from typing import Union

from ase import units
from ase.calculators.mixing import SumCalculator

from mace.calculators import mace_mp, MACECalculator

import torch

try:
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    torch_dftd_error = None
except Exception:
    torch_dftd_error = traceback.format_exc()

from .torch_tools import cuda_devices_sorted_by_free_mem
from .base_optimizer import BaseStructOptimizer


class MACEOptimizer(BaseStructOptimizer):
    """The MACE optimizer is a class for structural relaxation using a
    MACECalculator (Adapted from `matgl.ext.ase.Relaxer`). Any additional
    keyword arguments are passed on to the MACECalculator, and
    TorchDFTD3Calculator if dispersion corrections are used.

    Parameters
    ----------
    model : str
        path to model or models if a committee is produced to make a committee
        use a wild card notation like mace_*.model. Defaults to None which
        first checks for a local model and then downloads the default
        pretrained model from figshare. Specify "small", "medium" or "large" to
        download a smaller or larger pretrained model from figshare.
    device : str
        device to run on (cuda or cpu)
    energy_units_to_eV : float
        conversion factor from model energy units to eV
    length_units_to_A: float
        conversion factor from model length units to Angstroms
    default_dtype : str
        default dtype for the model
    charges_key : str
        array field of atoms object where atomic charges are stored
    model_type : str
         type of model to load, can be either "MACE", "DipoleMACE"
         or "EnergyDipoleMACE"
    dispersion : bool
        whether to use D3 dispersion corrections
    dispersion_xc : str
        exchange-correlation functional for D3 dispersion corrections
    dispersion_cutoff : float
        cutoff radius in Bhor for D3 dispersion corrections
    optimizer : str
        the optimization algorithm
    relax_cell : bool
        whether to relax the lattice cell
    """

    def __init__(self,
                 model: Union[list, str] = None,
                 device: str = None,
                 energy_units_to_eV: float = 1.0,
                 length_units_to_A: float = 1.0,
                 default_dtype: str = "float64",
                 charges_key: str = "Qs",
                 model_type: str = "MACE",
                 dispersion: bool = False,
                 dispersion_xc: str = "pbe",
                 dispersion_cutoff: float = 40.0 * units.Bohr,
                 optimizer: str = "FIRE",
                 relax_cell: bool = True,
                 **kwargs):

        super().__init__(optimizer=optimizer, relax_cell=relax_cell)

        # check if torch-dftd is used and available
        gh_url = "https://github.com/pfnet-research/torch-dftd"
        if dispersion and torch_dftd_error is not None:
            raise ImportError('\n'.join([f'torch-dftd (see {gh_url}), which is'
                                         f' required to calculate dispersion'
                                         f' corrections is not available '
                                         f'because of',
                                         str(torch_dftd_error)]))

        # set device
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else
                                     "cpu")
            if self.device == "cuda":
                self.device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"

        # setup calculator
        if (model in (None, 'small', 'medium', 'large') or
                str(model).startswith("https:")): 
            mace_calc = mace_mp(
                model=model,
                device=self.device,
                energy_units_to_eV=energy_units_to_eV,
                length_units_to_A=length_units_to_A,
                default_dtype=default_dtype,
                charges_key=charges_key,
                model_type=model_type,
                **kwargs)
        else:
            mace_calc = MACECalculator(
                model_paths=model,
                device=self.device,
                energy_units_to_eV=energy_units_to_eV,
                length_units_to_A=length_units_to_A,
                default_dtype=default_dtype,
                charges_key=charges_key,
                model_type=model_type,
                **kwargs)

        # add disperion calculator
        if dispersion:
            print(f'Using TorchDFTD3Calculator for D3 dispersion corrections'
                  f' (see {gh_url})')
            dtype = (torch.float32 if default_dtype == "float32" else
                     torch.float64)
            d3_calc = TorchDFTD3Calculator(
                device=self.device,
                damping="bj",
                dtype=dtype,
                xc=dispersion_xc,
                cutoff=dispersion_cutoff,
                **kwargs)
            self.calculator = SumCalculator([mace_calc, d3_calc])
        else:
            self.calculator = mace_calc

