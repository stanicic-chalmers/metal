from __future__ import annotations

import collections
import contextlib
import inspect
import io
import numpy as np
import pandas as pd
import pickle
import sys
from typing import TYPE_CHECKING

from ase import Atoms
import ase.optimize as opt
from ase import filters
from ase.filters import Filter

from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from ase.optimize.optimize import Optimizer


valid_ase_filter_names = [
    name
    for name, cls in inspect.getmembers(filters, inspect.isclass)
    if issubclass(cls, Filter)
]

optimizer_keys = {
    'fire': 'FIRE',
    'bfgs': 'BFGS',
    'lbfgs': 'LBFGS',
    'lbfgslinesearch': 'LBFGSLineSearch',
    'mdmin': 'MDMin',
    'scipyfmincg': 'SciPyFminCG',
    'scipyfminbfgs': 'SciPyFminBFGS',
    'bfgslinesearch': 'BFGSLineSearch'}


OPTIMIZERS = dict(
    fire=opt.fire.FIRE,
    bfgs=opt.bfgs.BFGS,
    lbfgs=opt.lbfgs.LBFGS,
    lbfgslinesearch=opt.lbfgs.LBFGSLineSearch,
    mdmin=opt.mdmin.MDMin,
    scipyfmincg=opt.sciopt.SciPyFminCG,
    scipyfminbfgs=opt.sciopt.SciPyFminBFGS,
    bfgslinesearch=opt.bfgslinesearch.BFGSLineSearch)


class BaseStructOptimizer:
    """The base structure optimizer is a general class for structural
    relaxation. (Adapted from `matgl.ext.ase.Relaxer`)

    Parameters
    ----------
    optimizer : str
        the optimization algorithm
    relax_cell : bool
        whether to relax the lattice cell
    """

    def __init__(self,
                 optimizer: str = 'FIRE',
                 relax_cell: bool = True):

        if isinstance(optimizer, str):
            optimizer = optimizer.lower()
            if optimizer in OPTIMIZERS:
                optimizer = OPTIMIZERS[optimizer]
            else:
                raise ValueError(f"Optimizer instance not found. Select from"
                                 f" {list(OPTIMIZERS)}")

        self.optimizer: Optimizer = optimizer
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        ase_filter: str | None = "ExpCellFilter",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms
                the atoms for relaxation
            fmax
                total force tolerance for relaxation convergence; here fmax is
                a sum of force and stress forces
            steps
                max number of steps for relaxation
            traj_file
                the trajectory file for saving
            interval
                the step interval for saving the trajectories
            ase_filter
                filter which is applied to the Atoms object
            verbose
                whether to have verbose output.
            kwargs
                Kwargs pass-through to optimizer.
        """
        if isinstance(ase_filter, str):
            if ase_filter in valid_ase_filter_names:
                ase_filter = getattr(filters, ase_filter)
            else:
                raise ValueError(
                    f'Invalid {ase_filter=}, must be one of'
                    f' {valid_ase_filter_names}.'
                )
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = ase_filter(atoms)
            optimizer = self.optimizer(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, Filter):
            atoms = atoms.atoms

        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
        }


class TrajectoryObserver(collections.abc.Sequence):
    """The trajectory observer is a hook in the relaxation process that saves
    the intermediate structures. Adapted from
    `matgl.ext.ase.TrajectoryObserver`"""

    def __init__(self, atoms: Atoms) -> None:
        """
        Initiate the trajectory observer from an ASE Atoms object.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the
        relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __getitem__(self, item):
        return self.energies[item], self.forces[item], self.stresses[item],\
            self.cells[item], self.atom_positions[item]

    def __len__(self):
        return len(self.energies)

    def as_pandas(self) -> pd.DataFrame:
        """Returns dataframe with energies, forces, stresses, cells and
        atomic positions."""
        return pd.DataFrame(
            {
                "energies": self.energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "cells": self.cells,
                "atom_positions": self.atom_positions,
            }
        )

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Parameters
        ----------
        filename
            filename for the saved trajectory
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)
