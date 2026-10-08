from contextlib import nullcontext
from functools import cached_property
import json
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
from itertools import combinations_with_replacement, combinations, product
import tempfile
import traceback
from typing import Callable, Dict, List, Tuple, Union
import warnings
import psutil
import pymatgen.entries.computed_entries as computed_entries
import sys
sys.modules["pymatgen.core.entries"] = computed_entries


from ase import Atoms
from ase.db import connect
from ase.db.row import AtomsRow
from ase.units import kB

from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram

from scipy.optimize import minimize, milp, LinearConstraint, Bounds
import time

from sklearn.calibration import CalibratedClassifierCV

from .tools import _atoms_row_to_structure, get_row_ids, _add_structure_db
from .thermodynamics import get_computed_entry
from ...utils import decompress_db, compress_db, apply_function_db

np.random.seed(123)

# ionic valences from Wang et al (2022)
valences_Wang2022 = {
    'Sr': [2],
    'Ca': [2],
    'K': [1],
    'Y': [3],
    'Ba': [2],
    'La': [3],
    'Sm': [3],
    'Fe': [3, 4],
    'Co': [3, 4],
    'Cu': [3, 4],
    'Mn': [3, 4],
    'Ni': [3, 4],
    'Ti': [3, 4],
    'Mg': [2],
    'O': [-2],
    'XO': [0]  # oxygen vacancy
}

# ionic radii from Wang et al (2022)
radii_Wang2022 = {
    'Sr': [1.44],
    'Ca': [1.34],
    'K': [1.64],
    'Y': [1.196],
    'Ba': [1.61],
    'La': [1.36],
    'Sm': [1.24],
    'Fe': [0.645, 0.585],
    'Co': [0.61, 0.53],
    'Cu': [0.71, 0.69],
    'Mn': [0.645, 0.53],
    'Ni': [0.60, 0.48],
    'Ti': [0.67, 0.605],
    'Mg': [0.72],
    'O': [1.40],
    'XO': [1.31]  # oxygen vacancy
}


class PerovskiteTolerance(object):
    """Class for calculating the tolerance factor for a perovskite with the
    general formula ABX3, where the both the cation (A, B) and anion (X)
    sublattices can be occupied by multiple elements, that:
        - determines whether compound can be charge-balanced
        - assigns oxidation states for A and B
        - predicts radii
        - generates the Bartel and Goldschmidt tolerance factors
        - classifies as perovskite/nonperovskite
        - estimates the probability that the perovskite structure is stable

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    possible_a_elements : List[str]
        list of chemical symbols for cations that can occupy sublattice A;
        the default is 'Ca', 'K', 'Y', 'Ba', 'La', 'Sm', and 'Sr'
    possible_b_elements : List[str]
        list of chemical symbols for cations that can occupy sublattice B;
        the default is 'Co', 'Cu', 'Mn', 'Mg', 'Ni', 'Ti', 'Fe'
    possible_x_elements : List[str]
        list of chemical symbols for anions that can occupy sublattice X;
        the default is 'N', 'O', 'S', 'Se', 'F', 'Cl', 'Br', 'I'
    max_oxidation_diff
        maximum difference between the valences for each element
    strict
        strictly require that the perovskite is charge balanced and that the
        difference in the valence for each element is below the maximum; if
        False then a warning message is produced instead of an error
    reduce
        reduce the structure to the minimal size required to represent the
        composition
    oxidation_method
        method for determining oxidation states: 'auto' (default, chooses based
        on memory), 'vectorized' (fast but memory intensive), 'iterative' (slow
        but memory efficient), or 'ilp' (uses integer linear programming)
    """
    def __init__(
            self,
            atoms_row: Union[Atoms, AtomsRow],
            possible_a_elements: List[str] = None,
            possible_b_elements: List[str] = None,
            possible_x_elements: List[str] = None,
            max_oxidation_diff: int = 1,
            strict: bool = False,
            reduce: bool = False,
            oxidation_method: str = 'ilp',
            oxidation_print: bool = False
    ):
        structure = _atoms_row_to_structure(atoms_row)
        if reduce:
            self.structure = Atoms(structure.get_chemical_formula(
                empirical=True))
        else:
            self.structure = structure
        self.symbols = self.structure.get_chemical_symbols()
        self.elements = list(set(self.symbols))
        if possible_a_elements is None:
            possible_a_elements = ['Ca', 'Ce', 'K', 'Mg', 'Y', 'Ba', 'La', 'Sm', 'Sr']
        self.a_elements = [element for element in self.elements if element in
                           possible_a_elements]
        if possible_b_elements is None:
            possible_b_elements = ['Al','Cr','Co', 'Cu', 'Mn','Sn','V', 'W','Zr','Zn','Mg', 'Ni', 'Ti', 'Fe']
        self.b_elements = [element for element in self.elements if element in
                           possible_b_elements]
        if possible_x_elements is None:
            possible_x_elements = ['N', 'O', 'S', 'Se', 'F', 'Cl', 'Br', 'I']
        self.x_elements = [element for element in self.elements if element in
                           possible_x_elements]
        if (set(self.a_elements + self.b_elements + self.x_elements)
                != set(self.elements)):
            raise ValueError(
                ' '.join([', '.join(set(self.elements) -
                                    set(self.a_elements + self.b_elements +
                                    self.x_elements)),
                          'are not found among the lists of allowed A, B or X'
                          ' elements']))
        self.max_oxidation_diff = max_oxidation_diff
        self.strict = strict
        self.shannon_radii = read_shannon_radii()
        self.disallowed_oxidation_dict = {}
        self.oxidation_method = oxidation_method
        self.oxidation_print = oxidation_print

    @property
    def anion_oxidation_states(self):
        """
        returns {el (str): oxidation state (int)} for allowed anions
        """
        return {'N': -3,
                'O': -2,
                'S': -2,
                'Se': -2,
                'F': -1,
                'Cl': -1,
                'Br': -1,
                'I': -1}

    @property
    def cation_plus_one(self):
        """
        list of cations likely to have a oxidiation state of 1+
        """
        return ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Ag']

    @property
    def cation_plus_two(self):
        """
        list of cations likely to have a oxidiation state of 2+
        """
        return ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra','Ni']

    @property
    def cation_plus_three(self):
        """
        list of cations likely to have a oxidiation state of 3+
        """
        return ['Sc', 'Y', 'La', 'Al', 'Ga', 'In',
                'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

    @cached_property
    def anion_symbols(self):
        """
        list of all symbols for the anions, on sublattice X
        """
        return [symbol for symbol in self.symbols if symbol in self.x_elements]

    @cached_property
    def cation_symbols(self):
        """
        list of all symbols for cations, on sublattices A and B
        """
        return self.a_symbols + self.b_symbols

    @cached_property
    def a_symbols(self):
        """
        list of all symbols for elements on sublattice A
        """
        return [symbol for symbol in self.symbols if symbol in self.a_elements]

    @cached_property
    def b_symbols(self):
        """
        list of all symbols for elements on sublattice B
        """
        return [symbol for symbol in self.symbols if symbol in self.b_elements]

    @cached_property
    def element_indices(self):
        """atomic indices for each element"""
        return {element: [i for i, symbol in enumerate(self.symbols) if symbol
                          == element] for element in set(self.symbols)}

    @cached_property
    def abx_elements(self):
        """elements on the A, B and X sublattices"""
        return {'a': self.a_elements, 'b': self.b_elements,
                'x': self.x_elements}

    @cached_property
    def abx_symbols(self):
        """all symbols on the A, B and X sublattices"""
        return {'a': self.a_symbols, 'b': self.b_symbols,
                'x': self.anion_symbols}

    @cached_property
    def electronegativities(self):
        """
        Pauling electronegativity for all elements
        """
        electronegativities = {}
        dir_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(dir_path, 'data/electronegativities.csv')) as f:
            for line in f:
                line = line.split(',')
                if line[0] in self.elements:
                    electronegativities[line[0]] = float(line[1][:-1])
        return electronegativities

    @cached_property
    def possible_oxidation_states(self):
        """
        List of allowed oxidation states for each element, using a self-defined dictionary
        to filter unlikely oxidation states.
        """
        allowed_oxidation_dict = getattr(self, "allowed_oxidations", {})

        possible_oxidation_states = {}
        for cation in self.cation_symbols:
            if cation in self.cation_plus_one:
                possible_oxidation_states[cation] = [1]
            elif cation in self.cation_plus_two:
                possible_oxidation_states[cation] = [2]
            else:
                # Start from all oxidation states in shannon_radii
                states = [int(val) for val in self.shannon_radii[cation].keys() if int(val) > 0]
                
                # Apply dictionary filter if it exists
                if cation in self.disallowed_oxidation_dict:
                    states = [s for s in states if s not in self.disallowed_oxidation_dict[cation]]

                possible_oxidation_states[cation] = states

        for anion in self.anion_symbols:
            possible_oxidation_states[anion] = [self.anion_oxidation_states[anion]]

        return possible_oxidation_states

    def _get_element_oxidation_combos(self) -> Dict[str, np.ndarray]:
        """Prepare oxidation combination data for all elements."""
        return {
            el: np.array(list(combinations_with_replacement(
                self.possible_oxidation_states[el],
                self.symbols.count(el)
            )))
            for el in self.elements
        }

    def _apply_filters(
            self,
            combo_indices: np.ndarray,
            element_oxidation_combos: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply all filtering steps after charge neutrality.

        Parameters
        ----------
        combo_indices
            Array of shape (n_combos, n_elements) with indices into element_oxidation_combos
        element_oxidation_combos
            Dict mapping element -> array of oxidation combinations

        Returns
        -------
        Filtered combo_indices array
        """
        if len(combo_indices) == 0:
            return combo_indices

        # Step 2: Filter by preferred +3 cations
        if self.cation_plus_three:
            pref_diffs = np.array([
                np.mean([np.abs(3 - np.mean(element_oxidation_combos[el][idx]))
                         if el in self.cation_plus_three else 0.0
                         for el, idx in zip(self.elements, combo)])
                for combo in combo_indices
            ])
            min_pref = np.min(pref_diffs)
            combo_indices = combo_indices[pref_diffs == min_pref]
            if self.oxidation_print:
                print(f"  [VECTORIZED] Stage 2 - Min pref deviation: {min_pref:.6f}, combos remaining: {len(combo_indices)}")
        else:
            if self.oxidation_print:
                print(f"  [VECTORIZED] Stage 2 - Skipped (no preferred +3 cations)")

        # Step 3: Filter by maximum oxidation difference within a combo
        max_diffs = np.array([
            np.max([np.max(element_oxidation_combos[el][idx]) - np.min(element_oxidation_combos[el][idx])
                    for el, idx in zip(self.elements, combo)])
            for combo in combo_indices
        ])
        min_max_diff = np.min(max_diffs)
        if min_max_diff > self.max_oxidation_diff and self.strict:
            raise ValueError(f'No combos with oxidation difference below {self.max_oxidation_diff}')
        combo_indices = combo_indices[max_diffs == min_max_diff]
        if self.oxidation_print:
            print(f"  [VECTORIZED] Stage 3 - Min max diff: {min_max_diff:.6f}, combos remaining: {len(combo_indices)}")

        # Step 4: Filter by electronegativity preference
        min_ox_map = {el: min(self.possible_oxidation_states[el]) for el in self.elements}
        span_map = {el: max(1, max(self.possible_oxidation_states[el]) - min_ox_map[el]) for el in self.elements}
        ox_chis = np.array([
            np.sum([((np.mean(element_oxidation_combos[el][idx]) - min_ox_map[el]) / span_map[el]) *
                    self.electronegativities[el]
                    for el, idx in zip(self.elements, combo)])
            for combo in combo_indices
        ])
        min_ox_chi = np.min(ox_chis)
        combo_indices = combo_indices[ox_chis == min_ox_chi]
        if self.oxidation_print:
            print(f"  [VECTORIZED] Stage 4 - Min ox-chi: {min_ox_chi:.6f}, combos remaining: {len(combo_indices)}")

        # Step 5: Filter by lowest mean oxidation difference
        mean_diffs = np.array([
            np.mean([np.mean(np.abs(element_oxidation_combos[el][idx] - np.mean(element_oxidation_combos[el][idx])))
                     for el, idx in zip(self.elements, combo)])
            for combo in combo_indices
        ])
        min_mean_diff = np.min(mean_diffs)
        combo_indices = combo_indices[mean_diffs == min_mean_diff]
        if self.oxidation_print:
            print(f"  [VECTORIZED] Stage 5 - Min mean diff: {min_mean_diff:.6f}, combos remaining: {len(combo_indices)}")

        # Step 6: Filter by largest A/B radii difference
        rad_diffs = np.array([
            np.sum([self.get_shannon_radius(el, ox) * self._site_radii_sign(el)
                    for el, idx in zip(self.elements, combo)
                    for ox in element_oxidation_combos[el][idx]])
            for combo in combo_indices
        ])
        max_rad_diff = np.max(rad_diffs)
        combo_indices = combo_indices[rad_diffs == max_rad_diff]
        if self.oxidation_print:
            print(f"  [VECTORIZED] Stage 6 - Max rad diff: {max_rad_diff:.6f}, combos remaining: {len(combo_indices)}")

        return combo_indices

    def _combo_to_result(
            self,
            combo: np.ndarray,
            element_oxidation_combos: Dict[str, np.ndarray]
    ) -> Dict[str, List[int]]:
        """Convert a combo index array to the result dictionary."""
        return {el: element_oxidation_combos[el][idx].tolist()
                for el, idx in zip(self.elements, combo)}

    def _oxidation_state_iterative(self) -> Dict[str, List[int]]:
        """
        Original iterative method for finding oxidation states.
        Iterative (slow but memory-efficient) method for finding oxidation states.
        """
        if self.oxidation_print:
            print("  [ITERATIVE] Starting oxidation state determination")
        element_oxidation_combos = {element: list(
            combinations_with_replacement(
                self.possible_oxidation_states[element],
                self.symbols.count(element)))
                for element in self.elements}
        all_index_combos = list(product(*[list(range(len(
            element_oxidation_combos[element])))
            for element in self.elements]))
        if self.oxidation_print:
            print(f"  [ITERATIVE] Total combinations: {len(all_index_combos)}")

        # find combinations with lowest charge
        all_charges = np.absolute(np.sum([[
            np.sum(element_oxidation_combos[element][i])
            for element, i in zip(self.elements, combo)]
            for combo in all_index_combos], axis=1))
        if min(all_charges) > 0:
            message = 'No charge balanced combination was found'
            if self.strict:
                raise ValueError(message)
            else:
                warnings.warn(message)
        min_charge_combos = [all_index_combos[i] for i in
                            np.where(all_charges == min(all_charges))[0]]
        if self.oxidation_print:
            print(f"  [ITERATIVE] Stage 1 - Min charge: {min(all_charges)}, combos remaining: {len(min_charge_combos)}")

        # let those elements that prefer be 3+
        if len(set(self.cation_plus_three).intersection(
                set(element_oxidation_combos.keys()))) > 0:
            all_pref_diffs = np.mean([[np.absolute(
                3 - np.mean(element_oxidation_combos[element][i]))
                if element in self.cation_plus_three else 0.0
                for element, i in zip(self.elements, combo)]
                for combo in min_charge_combos], axis=1)
            min_pref_diff_combos = [
                min_charge_combos[i] for i in
                np.where(all_pref_diffs == min(all_pref_diffs))[0]]
            if self.oxidation_print:
                print(f"  [ITERATIVE] Stage 2 - Min pref deviation: {min(all_pref_diffs):.6f}, combos remaining: {len(min_pref_diff_combos)}")
        else:
            min_pref_diff_combos = min_charge_combos
            if self.oxidation_print:
                print(f"  [ITERATIVE] Stage 2 - Skipped (no preferred +3 cations)")

        # restrict the maximum oxidation difference
        all_max_diffs = np.max([[
            np.max(element_oxidation_combos[element][i]) -
            np.min(element_oxidation_combos[element][i])
            for element, i in zip(self.elements, combo)]
            for combo in min_pref_diff_combos], axis=1)
        if min(all_max_diffs) > self.max_oxidation_diff:
            message = f'There are no combinations with an oxidation'\
                f' difference below {self.max_oxidation_diff}'
            if self.strict:
                raise ValueError(message)
            else:
                warnings.warn(message)
        min_max_diff_combos = [
            min_pref_diff_combos[i] for i in
            np.where(all_max_diffs == min(all_max_diffs))[0]]
        if self.oxidation_print:
            print(f"  [ITERATIVE] Stage 3 - Min max diff: {min(all_max_diffs):.6f}, combos remaining: {len(min_max_diff_combos)}")

        # find best combination based on the electronegativities
        min_oxidation = {element: min(self.possible_oxidation_states[element])
                        for element in self.elements}
        oxidation_span = {element: max(self.possible_oxidation_states[element])
                        - min_oxidation[element] for element in
                        self.elements}
        all_ox_chis = np.sum([[
            (np.mean(element_oxidation_combos[element][i])
            - min_oxidation[element]) / max(oxidation_span[element], 1) *
            self.electronegativities[element]
            for element, i in zip(self.elements, combo)]
            for combo in min_max_diff_combos], axis=1)
        min_ox_chi_combos = [
            min_max_diff_combos[i] for i in
            np.where(all_ox_chis == min(all_ox_chis))[0]]
        if self.oxidation_print:
            print(f"  [ITERATIVE] Stage 4 - Min ox-chi: {min(all_ox_chis):.6f}, combos remaining: {len(min_ox_chi_combos)}")

        # find combination that gives lowest mean oxidation difference
        all_mean_diffs = np.mean([[np.mean(np.absolute(
            element_oxidation_combos[element][i] -
            np.mean(element_oxidation_combos[element][i])))
            for element, i in zip(self.elements, combo)]
            for combo in min_ox_chi_combos], axis=1)
        min_mean_diff_combos = [
            min_ox_chi_combos[i] for i in
            np.where(all_mean_diffs == min(all_mean_diffs))[0]]
        if self.oxidation_print:
            print(f"  [ITERATIVE] Stage 5 - Min mean diff: {min(all_mean_diffs):.6f}, combos remaining: {len(min_mean_diff_combos)}")

        # find combination with the largest difference between A and B radii
        all_rad_diffs = np.sum([[
            self.get_shannon_radius(element, oxidation) *
            self._site_radii_sign(element)
            for element, i in zip(self.elements, combo)
            for oxidation in element_oxidation_combos[element][i]]
            for combo in min_mean_diff_combos], axis=1)
        max_rad_diff_combos = [
            min_mean_diff_combos[i] for i in
            np.where(all_rad_diffs == max(all_rad_diffs))[0]]
        if self.oxidation_print:
            print(f"  [ITERATIVE] Stage 6 - Max rad diff: {max(all_rad_diffs):.6f}, combos remaining: {len(max_rad_diff_combos)}")

        result = {element: element_oxidation_combos[element][i]
                for element, i in zip(self.elements, max_rad_diff_combos[0])}
        if self.oxidation_print:
            print(f"  [ITERATIVE] Final result: {dict(sorted(result.items()))}")
        return result

    def _oxidation_state_vectorized(self) -> Dict[str, List[int]]:
        """
        Vectorized (faster than iterative but memory-intensive) method for finding oxidation states.
        """
        if self.oxidation_print:
            print("  [VECTORIZED] Starting oxidation state determination")
        element_oxidation_combos = self._get_element_oxidation_combos()
        element_combo_sums = {
            el: np.sum(element_oxidation_combos[el], axis=1)
            for el in self.elements
        }

        ranges = [np.arange(len(element_oxidation_combos[el])) for el in self.elements]
        n_rows = np.prod([len(r) for r in ranges]) if ranges else 0
        dtype = np.uint16 if n_rows <= np.iinfo(np.uint16).max else np.uint32
        if self.oxidation_print:
            print(f"  [VECTORIZED] Total combinations: {n_rows}")

        mesh = np.meshgrid(*ranges, indexing='ij')
        combo_indices = np.stack([m.ravel() for m in mesh], axis=-1).astype(dtype)

        # Step 1: Filter by charge neutrality
        charges = np.sum([element_combo_sums[el][combo_indices[:, i]]
                          for i, el in enumerate(self.elements)], axis=0)
        abs_charges = np.abs(charges)
        min_charge = np.min(abs_charges)
        combo_indices = combo_indices[abs_charges == min_charge]
        if self.oxidation_print:
            print(f"  [VECTORIZED] Stage 1 - Min charge: {min_charge}, combos remaining: {len(combo_indices)}")

        if min_charge > 0:
            message = 'No charge-balanced combination was found'
            if self.strict:
                raise ValueError(message)
            else:
                warnings.warn(message)

        if len(combo_indices) == 0:
            raise ValueError("No valid combination was found after filtering by charge neutrality.")

        # Apply remaining filters
        combo_indices = self._apply_filters(combo_indices, element_oxidation_combos)
        if self.oxidation_print:
            print(f"  [VECTORIZED] After all filters, combos remaining: {len(combo_indices)}")

        if len(combo_indices) == 0:
            raise ValueError("No valid combination was found after all filtering steps.")

        result = self._combo_to_result(combo_indices[0], element_oxidation_combos)
        if self.oxidation_print:
            print(f"  [VECTORIZED] Final result: {dict(sorted(result.items()))}")
        return result

    def _oxidation_state_ilp(self) -> Dict[str, List[int]]:
        """
        Integer Linear Programming method using lexicographic optimization.

        Applies the same 6-stage filtering as iterative/vectorized methods:
        1. Minimize |total charge|
        2. Minimize mean preference deviation from +3
        3. Minimize max oxidation difference
        4. Minimize electronegativity-weighted score
        5. Minimize mean absolute deviation
        6. Maximize A/B radii difference

        Each stage solves an ILP and fixes the optimal value as a constraint
        for subsequent stages. Fastest and least memory-intensive method.
        Has been validated against iterative method.
        """
        element_oxidation_combos = self._get_element_oxidation_combos()
        n_elements = len(self.elements)

        # Precompute coefficients for each (element, combo) pair
        n_combos = {el: len(element_oxidation_combos[el]) for el in self.elements}

        # Build variable index mapping: x[offsets[el] + i] = 1 if element el uses combo i
        offsets = {}
        idx = 0
        for el in self.elements:
            offsets[el] = idx
            idx += n_combos[el]
        n_binary = idx

        # Precompute all coefficient arrays
        # 1. Charge sum for each combo
        charge_sums = {el: np.sum(element_oxidation_combos[el], axis=1).astype(float)
                       for el in self.elements}

        # 2. Mean oxidation for each combo
        combo_means = {el: np.mean(element_oxidation_combos[el], axis=1)
                       for el in self.elements}

        # 3. Max-min oxidation diff for each combo
        ox_diffs = {el: (np.max(element_oxidation_combos[el], axis=1) -
                         np.min(element_oxidation_combos[el], axis=1)).astype(float)
                    for el in self.elements}

        # 4. Electronegativity score for each combo
        min_ox = {el: min(self.possible_oxidation_states[el]) for el in self.elements}
        ox_span = {el: max(1, max(self.possible_oxidation_states[el]) - min_ox[el])
                   for el in self.elements}
        electro_scores = {el: ((combo_means[el] - min_ox[el]) / ox_span[el]) *
                              self.electronegativities[el]
                          for el in self.elements}

        # 5. Mean abs deviation for each combo
        mean_abs_devs = {}
        for el in self.elements:
            combos = element_oxidation_combos[el]
            means = np.mean(combos, axis=1, keepdims=True)
            mean_abs_devs[el] = np.mean(np.abs(combos - means), axis=1)

        # 6. Radii contribution for each combo
        rad_contribs = {}
        for el in self.elements:
            sign = self._site_radii_sign(el)
            contribs = []
            for combo in element_oxidation_combos[el]:
                total = sum(self.get_shannon_radius(el, int(ox)) * sign for ox in combo)
                contribs.append(total)
            rad_contribs[el] = np.array(contribs)

        # Helper: build coefficient vector from per-element dict
        def build_coeffs(coeff_dict):
            c = np.zeros(n_binary)
            for el in self.elements:
                for i, val in enumerate(coeff_dict[el]):
                    c[offsets[el] + i] = val
            return c

        charge_coeffs = build_coeffs(charge_sums)
        ox_diff_coeffs = build_coeffs(ox_diffs)
        electro_coeffs = build_coeffs(electro_scores)
        mean_dev_coeffs = build_coeffs(mean_abs_devs)
        rad_coeffs = build_coeffs(rad_contribs)

        # Build preference deviation coefficients (only for cation_plus_three elements)
        pref_dev_coeffs = np.zeros(n_binary)
        for el in self.elements:
            if el in self.cation_plus_three:
                for i, mean_val in enumerate(combo_means[el]):
                    pref_dev_coeffs[offsets[el] + i] = abs(3 - mean_val)

        # Base constraint: exactly one combo per element
        A_base = []
        for el in self.elements:
            row = np.zeros(n_binary)
            for i in range(n_combos[el]):
                row[offsets[el] + i] = 1.0
            A_base.append(row)
        A_base = np.array(A_base)
        b_base = np.ones(n_elements)

        # Track accumulated constraints
        eq_constraints = [(A_base, b_base)]  # List of (A, b) for equality constraints
        ub_constraints = []  # List of (A, b) for inequality constraints (A @ x <= b)

        def solve_ilp(obj_coeffs, minimize=True, n_aux=0, aux_lb=None, aux_ub=None,
                      extra_eq=None, extra_ub=None):
            """Solve ILP with current constraints plus optional extras."""
            n_total = n_binary + n_aux

            # Build equality constraint matrix
            A_eq_parts = []
            b_eq_parts = []
            for A, b in eq_constraints:
                A_ext = np.hstack([A, np.zeros((A.shape[0], n_aux))]) if n_aux > 0 else A
                A_eq_parts.append(A_ext)
                b_eq_parts.append(b)
            if extra_eq is not None:
                A_eq_parts.append(extra_eq[0])
                b_eq_parts.append(extra_eq[1])

            A_eq = np.vstack(A_eq_parts) if A_eq_parts else None
            b_eq = np.concatenate(b_eq_parts) if b_eq_parts else None

            # Build inequality constraint matrix
            A_ub_parts = []
            b_ub_parts = []
            for A, b in ub_constraints:
                A_ext = np.hstack([A, np.zeros((A.shape[0], n_aux))]) if n_aux > 0 else A
                A_ub_parts.append(A_ext)
                b_ub_parts.append(b)
            if extra_ub is not None:
                A_ub_parts.append(extra_ub[0])
                b_ub_parts.append(extra_ub[1])

            A_ub = np.vstack(A_ub_parts) if A_ub_parts else None
            b_ub = np.concatenate(b_ub_parts) if b_ub_parts else None

            # Build constraints list
            all_constraints = []
            if A_eq is not None:
                all_constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
            if A_ub is not None:
                all_constraints.append(LinearConstraint(A_ub, -np.inf * np.ones(len(b_ub)), b_ub))

            # Bounds
            lb = np.concatenate([np.zeros(n_binary), aux_lb]) if n_aux > 0 else np.zeros(n_binary)
            ub = np.concatenate([np.ones(n_binary), aux_ub]) if n_aux > 0 else np.ones(n_binary)
            bounds = Bounds(lb, ub)

            # Integrality (binary vars are integer, aux vars are continuous)
            integrality = np.concatenate([np.ones(n_binary), np.zeros(n_aux)]) if n_aux > 0 else np.ones(n_binary)

            # Objective
            c = np.zeros(n_total)
            c[:len(obj_coeffs)] = obj_coeffs if minimize else -obj_coeffs

            result = milp(c, constraints=all_constraints, bounds=bounds, integrality=integrality)
            return result

        def extract_combo(x):
            """Extract combo indices from solution vector."""
            combo = []
            for el in self.elements:
                for i in range(n_combos[el]):
                    if round(x[offsets[el] + i]) == 1:
                        combo.append(i)
                        break
            return tuple(combo) if len(combo) == len(self.elements) else None

        # Tolerance for floating point comparisons
        tol = 1e-6

        if self.oxidation_print:
            print("  [ILP] Starting oxidation state determination")
            total_combos = np.prod([n_combos[el] for el in self.elements])
            print(f"  [ILP] Total possible combinations: {total_combos}, ILP variables: {n_binary}")

        # Stage 1: Minimize |total charge|
        # Use auxiliary variable z: charge <= z, -charge <= z, minimize z
        A_aux = np.zeros((2, n_binary + 1))
        A_aux[0, :n_binary] = charge_coeffs
        A_aux[0, n_binary] = -1
        A_aux[1, :n_binary] = -charge_coeffs
        A_aux[1, n_binary] = -1
        extra_ub_1 = (A_aux, np.zeros(2))

        obj_1 = np.zeros(n_binary + 1)
        obj_1[n_binary] = 1  # minimize z

        result = solve_ilp(obj_1, minimize=True, n_aux=1,
                          aux_lb=np.array([0.0]), aux_ub=np.array([1e6]),
                          extra_ub=extra_ub_1)

        if not result.success:
            message = 'No charge-balanced combination was found'
            if self.strict:
                raise ValueError(message)
            warnings.warn(message)
            return {el: list(element_oxidation_combos[el][0]) for el in self.elements}

        min_abs_charge = round(result.x[n_binary])
        if self.oxidation_print:
            print(f"  [ILP] Stage 1 - Min charge: {min_abs_charge}")
        if min_abs_charge > 0:
            message = f'No perfectly charge-balanced combination found (best: ±{min_abs_charge})'
            if self.strict:
                raise ValueError(message)
            warnings.warn(message)

        # Fix charge constraint: -min_abs_charge <= charge <= min_abs_charge
        ub_constraints.append((np.array([charge_coeffs, -charge_coeffs]),
                               np.array([min_abs_charge + tol, min_abs_charge + tol])))

        # Stage 2: Minimize mean preference deviation from +3
        has_pref_cations = any(el in self.cation_plus_three for el in self.elements)
        if has_pref_cations:
            result = solve_ilp(pref_dev_coeffs, minimize=True)
            if result.success:
                opt_pref = np.dot(pref_dev_coeffs, np.round(result.x[:n_binary]))
                if self.oxidation_print:
                    print(f"  [ILP] Stage 2 - Min pref deviation: {opt_pref:.6f}")
                # Fix: pref_dev <= opt_pref
                ub_constraints.append((np.array([pref_dev_coeffs]), np.array([opt_pref + tol])))
            else:
                if self.oxidation_print:
                    print(f"  [ILP] Stage 2 - Failed to solve")
        else:
            if self.oxidation_print:
                print(f"  [ILP] Stage 2 - Skipped (no preferred +3 cations)")

        # Stage 3: Minimize max oxidation difference
        # For each element: ox_diff[el] * x[el] <= M, minimize M
        A_max = np.zeros((n_elements, n_binary + 1))
        for i, el in enumerate(self.elements):
            for j in range(n_combos[el]):
                A_max[i, offsets[el] + j] = ox_diffs[el][j]
            A_max[i, n_binary] = -1  # -M
        extra_ub_3 = (A_max, np.zeros(n_elements))

        obj_3 = np.zeros(n_binary + 1)
        obj_3[n_binary] = 1  # minimize M

        result = solve_ilp(obj_3, minimize=True, n_aux=1,
                          aux_lb=np.array([0.0]), aux_ub=np.array([100.0]),
                          extra_ub=extra_ub_3)

        if result.success:
            opt_max_diff = result.x[n_binary]
            if self.oxidation_print:
                print(f"  [ILP] Stage 3 - Min max diff: {opt_max_diff:.6f}")
            # Fix: for each element, ox_diff <= opt_max_diff
            ub_constraints.append((A_max[:, :n_binary], np.full(n_elements, opt_max_diff + tol)))

            if opt_max_diff > self.max_oxidation_diff:
                message = f'No combinations with oxidation difference below {self.max_oxidation_diff}'
                if self.strict:
                    raise ValueError(message)
                warnings.warn(message)
        else:
            if self.oxidation_print:
                print(f"  [ILP] Stage 3 - Failed to solve")

        # Stage 4: Minimize electronegativity-weighted score
        result = solve_ilp(electro_coeffs, minimize=True)
        if result.success:
            opt_electro = np.dot(electro_coeffs, np.round(result.x[:n_binary]))
            if self.oxidation_print:
                print(f"  [ILP] Stage 4 - Min ox-chi: {opt_electro:.6f}")
            ub_constraints.append((np.array([electro_coeffs]), np.array([opt_electro + tol])))
        else:
            if self.oxidation_print:
                print(f"  [ILP] Stage 4 - Failed to solve")

        # Stage 5: Minimize mean absolute deviation
        result = solve_ilp(mean_dev_coeffs, minimize=True)
        if result.success:
            opt_mean_dev = np.dot(mean_dev_coeffs, np.round(result.x[:n_binary]))
            if self.oxidation_print:
                print(f"  [ILP] Stage 5 - Min mean diff: {opt_mean_dev:.6f}")
            ub_constraints.append((np.array([mean_dev_coeffs]), np.array([opt_mean_dev + tol])))
        else:
            if self.oxidation_print:
                print(f"  [ILP] Stage 5 - Failed to solve")

        # Stage 6: Maximize radii difference (minimize negative)
        result = solve_ilp(rad_coeffs, minimize=False)

        if not result.success:
            raise ValueError("ILP failed to find valid oxidation state combination")

        opt_rad = np.dot(rad_coeffs, np.round(result.x[:n_binary]))
        if self.oxidation_print:
            print(f"  [ILP] Stage 6 - Max rad diff: {opt_rad:.6f}")

        combo = extract_combo(result.x)
        if combo is None:
            raise ValueError("Failed to extract valid combination from ILP solution")

        result_dict = {el: element_oxidation_combos[el][combo[i]].tolist()
                       for i, el in enumerate(self.elements)}
        if self.oxidation_print:
            print(f"  [ILP] Final result: {dict(sorted(result_dict.items()))}")
        return result_dict

    def _estimate_memory_gb(self) -> float:
        """Estimate peak memory usage for vectorized approach in GB."""
        element_oxidation_combos = self._get_element_oxidation_combos()
        ranges = [np.arange(len(element_oxidation_combos[el])) for el in self.elements]
        n_rows = np.prod([len(r) for r in ranges]) if ranges else 0
        dtype = np.uint16 if n_rows <= np.iinfo(np.uint16).max else np.uint32
        if n_rows > 0:
            bytes_per_element = np.dtype(dtype).itemsize
            return (n_rows * len(ranges) * bytes_per_element * (len(ranges) + 2)) / (1024**3)
        return 0

    @cached_property
    def oxidation_state(self) -> Dict[str, List[int]]:
        """
        Determine the "best" oxidation states for all elements based on a multi-step
        filtering process.

        The method used depends on self.oxidation_method:
        - 'auto': Choose based on estimated memory usage
        - 'vectorized': Fast but memory-intensive
        - 'iterative': Slow but memory-efficient
        - 'ilp': Integer Linear Programming (ILP), fastest and least memory-intensive, 
            others are kept for testing, ref.
        """
        if self.oxidation_print:
            print(f'{"-"*20}\n{self.structure.get_chemical_formula()}')

        method = self.oxidation_method

        if method == 'auto':
            # ILP is generally faster and more memory efficient than vectorized
            method = 'ilp'

        if method == 'vectorized':
            return self._oxidation_state_vectorized()
        elif method == 'iterative':
            return self._oxidation_state_iterative()
        elif method == 'ilp':
            return self._oxidation_state_ilp()
        else:
            raise ValueError(f"Unknown oxidation_method: {method}. "
                           f"Choose from 'auto', 'vectorized', 'iterative', or 'ilp'.")

    @cached_property
    def abx_oxidation(self):
        """average oxidation on the A, B and X sublattice based on the "best"
        oxidation state"""

        return {key: np.mean([oxidation for element in elements for oxidation
                              in self.oxidation_state[element]])
                for key, elements in self.abx_elements.items()}

    @cached_property
    def abx_radii(self):
        """average radii on the A, B and X sublattice based on the "best"
        oxidation state"""

        return {key: np.mean([
                    self.get_shannon_radius(element, oxidation) for element in
                    elements for oxidation in self.oxidation_state[element]])
                for key, elements in self.abx_elements.items()}

    def _site_radii_sign(self, element: str) -> int:
        """Return the sign for radii-based tiebreaking in oxidation state
        determination (Stage 6).

        In perovskites, A-site cations are larger than B-site cations,
        so we maximize sum(r_A) - sum(r_B): A gets +1, B gets -1.

        Override in subclasses where the size relationship is reversed.
        """
        if element in self.a_elements:
            return 1
        elif element in self.b_elements:
            return -1
        return 0

    def _ideal_coordination(self, element: str) -> int:
        """Return the ideal coordination number for an element based on its
        sublattice. Override in subclasses for different crystal structures."""
        if element in self.a_elements:
            return 12
        elif element in self.b_elements:
            return 6
        elif element in self.x_elements:
            return 6 if element == 'N' else 4
        raise ValueError(f'{element} not found in any sublattice')

    def get_shannon_radius(self,
                           element: str,
                           oxidation: int) -> float:
        """Get radius of a given element with a specific oxidation state

        Parameters
        ----------
        element
            chemical symbol
        oxidation
            oxidation number
        """

        coordination_radii = self.shannon_radii[element][oxidation]
        possible_coordinations = list(coordination_radii.keys())
        ideal_coordination = self._ideal_coordination(element)
        coordination_diffs = [
            abs(coordination - ideal_coordination) for coordination
            in possible_coordinations]
        min_index = np.argmin(coordination_diffs)
        coordination = possible_coordinations[min_index]
        return coordination_radii[coordination]['only_spin']

    @cached_property
    def goldschmidt_tolerance(self) -> float:
        """Calculate the Goldschmidt tolerance factor, which can be used to
        judge if a given perovskite structure can be synthesised or not"""

        return ((self.abx_radii['a'] + self.abx_radii['x']) /
                (self.abx_radii['b'] + self.abx_radii['x']) / np.sqrt(2))

    @cached_property
    def bartel_tolerance(self) -> float:
        """Calculate the Bartel tolerance factor, which can be used to judge if
        a given perovskite structure can be synthesised or not"""

        return (self.abx_radii['x'] / self.abx_radii['b'] -
                self.abx_oxidation['a'] * (
                    self.abx_oxidation['a'] - self.abx_radii['a'] /
                    self.abx_radii['b'] /
                    np.log(self.abx_radii['a'] / self.abx_radii['b'])))

    def filtered_goldschmidt(self,
                             minimum: float = 0.825,
                             maximum: float = 1.059) -> bool:
        """Determine if the Goldschmidt tolerance factor falls within the
        specified range"""
        return ((self.goldschmidt_tolerance > minimum) and
                (self.goldschmidt_tolerance < maximum))

    def filtered_bartel(self, maximum: float = 4.18) -> bool:
        """Determine if the Bartel tolerance factor is lower than the specified
        upper limit"""
        return self.bartel_tolerance < maximum

    @cached_property
    def total_charge(self) -> bool:
        """Calculate the total charge"""
        return sum([sum(oxidations) for oxidations in
                    self.oxidation_state.values()])

    @property
    def charge_neutrality(self) -> bool:
        """Determine if the perovskite is charge neutral"""
        return self.total_charge == 0

    @cached_property
    def calibrated_bartel_classifier(self):
        """Calibrated classifier to yield Bartel tolerance probabilities"""
        return get_calibrated_bartel_classifier()

    def bartel_probability(self,
                           classifier: CalibratedClassifierCV = None):
        """Calculate the probability that the estimate of`` the Bartel
        tolerance is accurate

        Parameters
        ----------
        classifier
            calibrated classifier based on the Bartel tolerance
        """
        if classifier is None:
            classifier = get_calibrated_bartel_classifier()
        return classifier.predict_proba([[self.bartel_tolerance]])[0][1]

    def delta_charge_neutrality(
            self,
            deltas: List[float] = [0.0, 5.0]) -> bool:
        """Check charge neutrality is achievable for a perovskite of the type
        ABO3, which determines if it is stable or not at the specified vacancy
        concentrations

        Parameters
        ----------
        deltas
            vacancy concentrations, i.e. delta in ABO_(3-delta)
        """
        # check charge neutrality
        neutralities = []
        for delta in deltas:
            vacancies = int(delta * len(self.structure) / 5)
            if vacancies == 0:
                neutralities.append(self.charge_neutrality)
            else:
                for combo in np.unique(list(combinations(
                        self.anion_symbols, vacancies)), axis=0):
                    structure = self.structure.copy()
                    for symbol in combo:
                        del structure[structure.symbols.index(symbol)]
                    neutralities.append(PerovskiteTolerance(
                        structure,
                        possible_a_elements=self.a_elements,
                        possible_b_elements=self.b_elements,
                        possible_x_elements=self.x_elements,
                        max_oxidation_diff=self.max_oxidation_diff,
                        strict=self.strict).charge_neutrality)

        return all(neutralities)

    def entropy(self,
                stirling: bool = True,
                anion_vacancies: bool = True) -> float:
        """Calculate an estimate of the configurational entropy by counting
        the number of possible microstates for the specified structure

        Parameters
        ----------
        stirling
            use the Stirling approximation; otherwise the result only
            approaches the macroscopic value for an infitely large supercell
        anion_vacancies
            account for the number of anion vacancies
        """

        total_entropy = 0.0
        for key, elements in self.abx_elements.items():
            natoms = len(self.abx_symbols[key])
            sublattice_entropy = 0.0
            if key == 'x' and anion_vacancies:
                vacancies = len(self.abx_symbols['a']) * 3 - natoms
                natoms += vacancies
                if stirling:
                    sublattice_entropy -= (0.0 if vacancies == 0 else vacancies
                                           * np.log(vacancies / natoms))
                else:
                    sublattice_entropy -= np.log(float(math.factorial(
                        vacancies)))
            if not stirling:
                sublattice_entropy += np.log(float(math.factorial(natoms)))
            for element in elements:
                for oxidation in set(self.oxidation_state[element]):
                    count = self.oxidation_state[element].count(oxidation)
                    if stirling:
                        sublattice_entropy -= (0.0 if count == 0 else count *
                                               np.log(count / natoms))
                    else:
                        sublattice_entropy -= np.log(float(math.factorial(
                            count)))
            total_entropy += kB * sublattice_entropy

        return total_entropy


class SpinelTolerance(PerovskiteTolerance):
    def __init__(
            self,
            atoms_row: Union[Atoms, AtomsRow],
            possible_a_elements: List[str] = None,
            possible_b_elements: List[str] = None,
            possible_x_elements: List[str] = None,
            max_oxidation_diff: int = 1,
            strict: bool = False,
            reduce: bool = False,
            oxidation_method: str = 'ilp',
            oxidation_print: bool = False,
            site_tags: List[str] = None
    ):

        # Set your custom defaults here
        if possible_a_elements is None:
            possible_a_elements = ['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca'] # custom A-site
        if possible_b_elements is None:
            possible_b_elements = ['Al', 'Cr', 'Ga', 'Ce', 'Ti', 'Fe', 'Mn', 'Zr', 'V'] # custom B-site
        if possible_x_elements is None:
            possible_x_elements = ['O']         # custom X-site

        # Call the parent class __init__ with these values
        super().__init__(
            atoms_row,
            possible_a_elements=possible_a_elements,
            possible_b_elements=possible_b_elements,
            possible_x_elements=possible_x_elements,
            max_oxidation_diff=max_oxidation_diff,
            strict=strict,
            reduce=reduce,
            oxidation_method=oxidation_method,
            oxidation_print=oxidation_print
        )
        
        self._pseudo_to_real = {}
        self.disallowed_oxidation_dict = {}

        # Split elements on both sublattices into site-specific pseudo-elements
        mixed = set(possible_a_elements) & set(possible_b_elements)
        has_mixed = bool(mixed & set(self.elements))
        if has_mixed:
            if site_tags is None:
                raise ValueError(
                    f"Elements {mixed & set(self.elements)} appear in both "
                    f"possible_a_elements and possible_b_elements. "
                    f"site_tags must be provided to assign atoms to sites.")
            self._apply_mixed_site_split(site_tags, mixed)

    def _apply_mixed_site_split(self, site_tags, mixed_elements):
        """Split mixed-site elements into site-specific pseudo-elements.

        Elements that can occupy both A and B sites are renamed to
        '{El}_A' and '{El}_B' so downstream code (oxidation states, entropy,
        radii) treats them as independent species on each sublattice.

        Parameters
        ----------
        site_tags : list of str
            Per-atom site label ('a', 'b', or 'x'), same length as self.symbols.
        mixed_elements : set of str
            Elements present in both possible_a_elements and possible_b_elements.
        """
        if len(site_tags) != len(self.symbols):
            raise ValueError(
                f"site_tags length ({len(site_tags)}) does not match "
                f"number of atoms ({len(self.symbols)})")

        new_symbols = []
        for sym, tag in zip(self.symbols, site_tags):
            if sym in mixed_elements:
                pseudo = f'{sym}_{tag.upper()}'
                new_symbols.append(pseudo)
                self._pseudo_to_real[pseudo] = sym
            else:
                new_symbols.append(sym)

        self.symbols = new_symbols
        self.elements = list(set(new_symbols))

        # Rebuild sublattice element lists from site tags
        a_set, b_set, x_set = set(), set(), set()
        for sym, tag in zip(new_symbols, site_tags):
            if tag == 'a':
                a_set.add(sym)
            elif tag == 'b':
                b_set.add(sym)
            else:
                x_set.add(sym)

        self.a_elements = [e for e in self.elements if e in a_set]
        self.b_elements = [e for e in self.elements if e in b_set]
        self.x_elements = [e for e in self.elements if e in x_set]

        # Copy Shannon radii for pseudo-elements
        for pseudo, real in self._pseudo_to_real.items():
            if pseudo not in self.shannon_radii:
                self.shannon_radii[pseudo] = self.shannon_radii[real]

        # Copy disallowed oxidation states for pseudo-elements
        for pseudo, real in self._pseudo_to_real.items():
            if real in self.disallowed_oxidation_dict:
                self.disallowed_oxidation_dict[pseudo] = \
                    self.disallowed_oxidation_dict[real]

    def _expand_with_pseudos(self, element_list):
        """Expand an element list to include any active pseudo-element variants."""
        expanded = list(element_list)
        for pseudo, real in self._pseudo_to_real.items():
            if real in element_list and pseudo in self.elements:
                expanded.append(pseudo)
        return expanded

    @property
    def cation_plus_one(self):
        return self._expand_with_pseudos(super().cation_plus_one)

    @property
    def cation_plus_two(self):
        return self._expand_with_pseudos(super().cation_plus_two)

    @property
    def cation_plus_three(self):
        return self._expand_with_pseudos(super().cation_plus_three)

    def _site_radii_sign(self, element: str) -> int:
        """In spinels, B-site (octahedral) cations are larger than A-site
        (tetrahedral) cations, so we maximize sum(r_B) - sum(r_A):
        A gets -1, B gets +1 (reversed from perovskite)."""
        if element in self.a_elements:
            return -1
        elif element in self.b_elements:
            return 1
        return 0

    def _ideal_coordination(self, element: str) -> int:
        """Spinel coordination: A-site tetrahedral (4), B-site octahedral (6),
        X-site (4)."""
        if element in self.a_elements:
            return 4
        elif element in self.b_elements:
            return 6
        elif element in self.x_elements:
            return 4
        raise ValueError(f'{element} not found in any sublattice')

    @cached_property
    def electronegativities(self):
        real_needed = {self._pseudo_to_real.get(el, el) for el in self.elements}
        real_eng = {}
        dir_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(dir_path, 'data/electronegativities.csv')) as f:
            for line in f:
                line = line.split(',')
                if line[0] in real_needed:
                    real_eng[line[0]] = float(line[1][:-1])
        eng = {}
        for el in self.elements:
            real = self._pseudo_to_real.get(el, el)
            if real in real_eng:
                eng[el] = real_eng[real]
        return eng

    @cached_property
    def spinel_tolerance(self) -> float:
        a = self.abx_radii['a']
        b = self.abx_radii['b']
        x = self.abx_radii['x']
        tau = (np.sqrt(3) / 2) * (b + x) / (a + x)
        print(f'b={b}, x={x}, a={a}')
        print(f'Spinel Tolerance: {tau}')
        return tau
    
    @cached_property
    def charge_inversion(self) -> float:
        """Charge-based inversion degree for the spinel composition.

        Convention: ox <= +2 is charge-native A, ox >= +3 is charge-native B.
        An atom is inverted when its oxidation state does not match its site:
        high-charge (>= +3) on A-site or low-charge (<= +2) on B-site.

        Requires site_tags so that every element is assigned to exactly one
        sublattice (via pseudo-elements for mixed-site species).

        Returns the fraction of inverted cations in [0, 1].
        """
        inverted = 0
        total = 0
        for elem in self.a_elements:
            for ox in self.oxidation_state[elem]:
                total += 1
                if ox >= 3:
                    inverted += 1
        for elem in self.b_elements:
            for ox in self.oxidation_state[elem]:
                total += 1
                if ox <= 2:
                    inverted += 1
        if total == 0:
            return 0.0
        return inverted / total

    def entropy(self,
                    stirling: bool = True,
                    anion_vacancies: bool = True) -> float:
            # ... (documentation remains the same)

            total_entropy = 0.0
            for key, elements in self.abx_elements.items():
                natoms = len(self.abx_symbols[key])
                sublattice_entropy = 0.0
                if key == 'x' and anion_vacancies:
                    
                    # Determine the smaller cation sublattice count (A-site basis)
                    # This accounts for structures where 'a' or 'b' might hold the
                    # smallest count representing one formula unit, N_FU.
                    n_a = len(self.abx_symbols['a'])
                    n_b = len(self.abx_symbols['b'])
                    
                    # The count of the smallest sublattice (should be N_FU)
                    smallest_cation_count = min(n_a, n_b)

                    # Spinel (AB2X4) has 4 anion sites for every 1 formula unit.
                    vacancies = smallest_cation_count * 4 - natoms 
                    
                    natoms += vacancies
                    
                    if stirling:
                        if vacancies > 0: 
                            sublattice_entropy -= vacancies * np.log(vacancies / natoms)
                    else:
                        sublattice_entropy -= np.log(float(math.factorial(
                                    vacancies)))
                
                if not stirling:
                    sublattice_entropy += np.log(float(math.factorial(natoms)))
                
                for element in elements:
                    for oxidation in set(self.oxidation_state[element]):
                        count = self.oxidation_state[element].count(oxidation)
                        if stirling:
                            if count > 0: 
                                sublattice_entropy -= count * np.log(count / natoms)
                        else:
                            sublattice_entropy -= np.log(float(math.factorial(
                                    count)))
                
                total_entropy += kB * sublattice_entropy

            return total_entropy

def read_shannon_radii(
        from_raw: bool = False,
        write: bool = False
        ) -> Dict[str, Dict[int, Dict[int, Dict[str, float]]]]:
    """Read Shannon radii from preprocessed file or process the raw data

    Parameters
    ----------
    from_raw
        process raw data
    write
        write processed data to file
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    fjson = os.path.join(dir_path, 'data/Shannon_radii_dict.json')
    if os.path.exists(fjson) and not from_raw:
        with open(fjson) as f:
            shannon_dict = json.load(f)
            for k1 in shannon_dict:
                k2s = list(shannon_dict[k1].keys())
                for k2 in k2s:
                    k3s = list(shannon_dict[k1][k2].keys())
                    shannon_dict[k1][int(k2)] = shannon_dict[k1][k2]
                    del shannon_dict[k1][k2]
                    for k3 in k3s:
                        shannon_dict[k1][int(k2)][int(k3)] =\
                            shannon_dict[k1][int(k2)][k3]
                        del shannon_dict[k1][int(k2)][k3]
        return shannon_dict

    df = pd.read_csv('Shannon_Effective_Ionic_Radii.csv')

    df = df.rename(columns={'OX. State': 'ox',
                            'Coord. #': 'coord',
                            'Crystal Radius': 'rcryst',
                            'Ionic Radius': 'rion',
                            'Spin State': 'spin'})

    df['spin'] = [spin if spin in ['HS', 'LS'] else 'only_spin' for spin in
                  df.spin.values]

    def get_el(row):
        ION = row['ION']
        if ' ' in ION:
            return ION.split(' ')[0]
        elif '+' in ION:
            return ION.split('+')[0]
        elif '-' in ION:
            return ION.split('-')[0]

    df['el'] = df.apply(lambda row: get_el(row), axis=1)

    el_to_ox = {}
    for el in df.el.values:
        el_to_ox[el] = list(set(df.ox.get((df['el'] == el)).tolist()))

    shannon_dict = {}
    for el in el_to_ox:
        oxs = el_to_ox[el]
        ox_to_coord = {}
        for ox in oxs:
            coords = df.coord.get((df['el'] == el) & (df['ox'] == ox)).tolist()
            ox_to_coord[ox] = coords
            coord_to_spin = {}
            for coord in ox_to_coord[ox]:
                spin = df.spin.get((df['el'] == el) & (df['ox'] == ox) &
                                   (df['coord'] == coord)).tolist()
                coord_to_spin[coord] = spin
                spin_to_rad = {}
                for spin in coord_to_spin[coord]:
                    rad = df.rion.get((df['el'] == el) & (df['ox'] == ox) &
                                      (df['coord'] == coord) &
                                      (df['spin'] == spin)).tolist()[0]
                    spin_to_rad[spin] = rad
                    coord_to_spin[coord] = spin_to_rad
                    ox_to_coord[ox] = coord_to_spin
        shannon_dict[el] = ox_to_coord

    spin_els = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
    starting_d = [4, 5, 6, 7, 8, 9]
    d_dict = dict(zip(spin_els, starting_d))
    for el in spin_els:
        for ox in shannon_dict[el].keys():
            for coord in shannon_dict[el][ox].keys():
                if len(shannon_dict[el][ox][coord].keys()) > 1:
                    num_d = d_dict[el] + 2 - ox
                    if num_d in [4, 5, 6, 7]:
                        shannon_dict[el][ox][coord]['only_spin'] =\
                            shannon_dict[el][ox][coord]['HS']
                    else:
                        shannon_dict[el][ox][coord]['only_spin'] =\
                            shannon_dict[el][ox][coord]['LS']
                elif 'HS' in shannon_dict[el][ox][coord].keys():
                    shannon_dict[el][ox][coord]['only_spin'] =\
                        shannon_dict[el][ox][coord]['HS']
                elif 'LS' in shannon_dict[el][ox][coord].keys():
                    shannon_dict[el][ox][coord]['only_spin'] =\
                        shannon_dict[el][ox][coord]['LS']

    if write:
        with open(fjson, 'w') as f:
            json.dump(shannon_dict, f)


def get_calibrated_bartel_classifier() -> CalibratedClassifierCV:
    """Obtain a calibrated classifier to yield Bartel tolerance probabilities
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    f_clf = os.path.join(dir_path, 'data/save_clf.p')
    if not os.path.exists(f_clf):

        # read experimental data
        df = pd.read_csv(os.path.join(dir_path, 'data/TableS1.csv'))
        bartels = []
        for _, row in df.iterrows():
            structure = Atoms(row.ABX3)
            perovskite_tolerance = PerovskiteTolerance(
                structure,
                possible_a_elements=[row.A],
                possible_b_elements=[row.B],
                possible_x_elements=[row.X])
            bartels.append(perovskite_tolerance.bartel_tolerance)
        df['tau'] = bartels

        # calibrate classifier
        X, y = df['tau'].values.reshape(-1, 1), df['exp_label'].values
        clf = CalibratedClassifierCV(cv=3)
        clf.fit(X, y)

        # save classifier
        pickle.dump(clf, open(f_clf, 'wb'))

    else:

        # read calibrated classifier from fil
        clf = pickle.load(open(f_clf, 'rb'))

    return clf


def calculate_tolerances_wang2022(
        atoms_row: Union[Atoms, AtomsRow]) -> Dict[str, float]:
    """Calculate the Bartel and Goldschmidt tolerance factors, which can be
    used to judge if a given perovskite structure can be synthesised or not,
    using data from Wang et al (2022)

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    """

    # define A, B, and X symbols in ABX3
    abx_symbols = {'a': ['Ca', 'K', 'Y', 'Ba', 'La', 'Sm', 'Sr'],
                   'b': ['Co', 'Cu', 'Mn', 'Mg', 'Ni', 'Ti', 'Fe'],
                   'x': ['O', 'XO']}

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)
    all_symbols = structure.get_chemical_symbols()
    symbols = list(set(all_symbols))

    sum_valences = {}
    sum_radii = {}
    counts = {}
    abx_counts = {key: 0 for key in abx_symbols.keys()}
    for symbol in symbols:

        # count atoms
        counts[symbol] = all_symbols.count(symbol)
        for key in abx_symbols.keys():
            if symbol in abx_symbols[key]:
                abx_counts[key] += counts[symbol]

        # find possible combinations of ionic valences
        combos = list(combinations_with_replacement(
            range(len(valences_Wang2022[symbol])), counts[symbol]))
        sum_valences[symbol] = [sum([valences_Wang2022[symbol][i] for i in
                                     combo]) for combo in combos]
        sum_radii[symbol] = [sum([radii_Wang2022[symbol][i] for i in
                                  combo]) for combo in combos]

    # find combinations with lowest charge
    combos = list(product(*[list(range(len(sum_valences[symbol])))
                            for symbol in symbols]))
    abs_charges = np.absolute(np.sum([[sum_valences[symbol][i] for symbol, i in
                                       zip(symbols, combo)]
                                      for combo in combos], axis=1))
    min_combos = [combos[i] for i in
                  np.where(abs_charges == min(abs_charges))[0]]

    # find combination that gives lowest Goldschmidt and Bartel tolerances
    goldschmidts = []
    bartels = []
    for combo in min_combos:
        abx_radii = {key: sum([sum_radii[symbol][i] for symbol, i in
                               zip(symbols, combo) if symbol in
                               abx_symbols[key]]) / abx_counts[key] for key in
                     abx_symbols.keys()}
        abx_valences = {key: sum([sum_valences[symbol][i] for symbol, i in
                                  zip(symbols, combo) if symbol in
                                  abx_symbols[key]]) / abx_counts[key] for key
                        in abx_symbols.keys()}
        goldschmidts.append((abx_radii['a'] + abx_radii['x']) /
                            (abx_radii['b'] + abx_radii['x']) / np.sqrt(2))
        bartels.append(abx_radii['x'] / abx_radii['b'] - abx_valences['a'] *
                       (abx_valences['a'] - abx_radii['a'] / abx_radii['b'] /
                        np.log(abx_radii['a'] / abx_radii['b'])))
    goldschmidts = np.asarray(goldschmidts)
    return {
        'goldschmidt': goldschmidts[
            np.argmin(np.absolute(goldschmidts - 1.0))],
        'bartel': min(bartels)}


def calculate_goldschmidt_tolerance_wang2022(
        atoms_row: Union[Atoms, AtomsRow]) -> float:
    """Calculate the Goldschmidt tolerance factor, which can be used to judge
    if a given perovskite structure can be synthesised or not, using data from
    Wang et al (2022)

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    """

    return calculate_tolerances_wang2022(atoms_row)['goldschmidt']


def calculate_bartel_tolerance_wang2022(
        atoms_row: Union[Atoms, AtomsRow]) -> float:
    """Calculate the Bartel tolerance factor, which can be used to judge if a
    given perovskite structure can be synthesised or not, using data from Wang
    et al (2022)

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    """

    return calculate_tolerances_wang2022(atoms_row)['bartel']


def calculate_bartel_tolerance_analytical_wang2022(
        atoms_row: Union[Atoms, AtomsRow],
        deltas: List[float] = None) -> float:
    """Calculate the Bartel tolerance factor for a perovskite of the type
    Sr(x)A(1-x)Fe(y)B(1-y)O(3-delta), which can be used to judge if it can be
    synthesised or not; adapted from Wang et al (2022)

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    deltas
        list of vacancy concentrations; the default is `[0]`
    """

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # check symbols
    symbols = structure.get_chemical_symbols()
    unique_symbols = set(symbols)
    sr_replacements = ['Ca', 'K', 'Y', 'Ba', 'La', 'Sm']
    fe_replacements = ['Co', 'Cu', 'Mn', 'Mg', 'Ni', 'Ti']
    possible_symbols = set(['Sr', 'Fe', 'O'] + sr_replacements +
                           fe_replacements)
    if (len(unique_symbols) > 5 or not
            unique_symbols.issubset(possible_symbols)):
        raise Exception('The structure must have the form'
                        ' Sr(x)A(1-x)Fe(y)B(1-y)O(3-delta)')
    symbol_A = (list(unique_symbols.intersection(sr_replacements))[0] if
                len(unique_symbols.intersection(sr_replacements)) > 0 else
                None)
    symbol_B = (list(unique_symbols.intersection(fe_replacements))[0] if
                len(unique_symbols.intersection(fe_replacements)) > 0 else
                None)

    # properties related to Sr, Fe, O, and vacancies
    rO = radii_Wang2022['O'][0]
    rX = radii_Wang2022['XO'][0]
    rSr = radii_Wang2022['Sr'][0]
    nSr = valences_Wang2022['Sr'][0]
    rFe_max = max(radii_Wang2022['Fe'])
    rFe_min = min(radii_Wang2022['Fe'])
    nFe_max = max(valences_Wang2022['Fe'])
    nFe_min = min(valences_Wang2022['Fe'])

    # average valence and radii of A and Sr
    if symbol_A is not None:
        x = symbols.count('Sr') / (symbols.count(symbol_A) +
                                   symbols.count('Sr'))
        nA = valences_Wang2022[symbol_A][0]
        rA = radii_Wang2022[symbol_A][0]
        nSrA = (nSr * x + nA * (1 - x))
        rSrA = (rSr * x + rA * (1 - x))
    else:
        nSrA = nSr
        rSrA = rSr

    # constant related to radii of B
    if symbol_B is not None:
        y = symbols.count('Fe') / (symbols.count(symbol_B) +
                                   symbols.count('Fe'))
        nB_max = max(valences_Wang2022[symbol_B])
        nB_min = min(valences_Wang2022[symbol_B])
        rB_max = max(radii_Wang2022[symbol_B])
        rB_min = min(radii_Wang2022[symbol_B])
    else:
        y = 1

    if deltas is None:
        deltas = [0]
    taus = []
    for delta in deltas:

        # average radii of oxygen and vacancies
        rOX = rO * (3 - delta) / 3 + rX * delta / 3

        # function to minimize
        def tau_fun(nB):

            # average radius of B
            if y < 1:
                if nB_min == nB_max:
                    rB = rB_min
                else:
                    P = (nB_max - nB) / (nB_max - nB_min)
                    rB = P * rB_max + (1 - P) * rB_min

            # average radius of Fe
            if y > 0:
                nFe = (6 - 2 * delta - nSrA) / y - nB * (1 - y) / y
                rFe = (nFe_max - nFe) * rFe_max + (nFe - nFe_min) * rFe_min

            # average radius of Fe and B
            if y == 1:
                rFeB = rFe
            elif y == 0:
                rFeB = rB
            else:
                rFeB = (rFe * y + rB * (1 - y))

            # tolerance factor
            rSrA_FeB = rSrA / rFeB
            tau = rOX / rFeB - nSrA * (nSrA - rSrA_FeB / np.log(rSrA_FeB))

            return tau

        if y == 1:
            tau = tau_fun(0)
        else:
            if nB_min == nB_max:
                tau = tau_fun(nB_min)
            else:
                solution = minimize(
                    tau_fun, x0=np.array((nB_max + nB_min) / 2),
                    bounds=((nB_min, nB_max),))
                if solution.success:
                    tau = solution.fun
                else:
                    warnings.warn(f'Failed to calculate Bartel tolerance for'
                                  f' {structure.get_chemical_formula()}'
                                  f' through minimization, trying scanning'
                                  f' approach')
                    tau = min(tau_fun(np.linspace(nB_min, nB_max, 100000)))

        taus.append(float(tau))

    return min(taus)


def check_charge_neutrality_wang2022(
        atoms_row: Union[Atoms, AtomsRow],
        min_delta: float = 0.0,
        max_delta: float = 0.5) -> bool:
    """Check charge neutrality is achievable for a perovskite of the type ABO3,
     which determines if it is stable or not. Note that the only allowed
    elements on the A and B sites are ['Ca', 'K', 'Y', 'Ba', 'La', 'Sm', 'Sr']
    and ['Co', 'Cu', 'Mn', 'Mg', 'Ni', 'Ti', 'Fe'], respectively; adapted from
    Wang et al (2022)

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    min_delta
        minimum vacancy concentration
    max_delta
        maximum vacancy concentration
    """

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # estimate atomic valences
    nO = valences_Wang2022['O'][0]
    ab_symbols = {'a': ['Ca', 'K', 'Y', 'Ba', 'La', 'Sm', 'Sr'],
                  'b': ['Co', 'Cu', 'Mn', 'Mg', 'Ni', 'Ti', 'Fe']}
    ab_valences = {'a': [], 'b': []}
    for symbol in structure.get_chemical_symbols():
        if symbol == 'O':
            continue
        valence = None
        for key, symbols in ab_symbols.items():
            if symbol in symbols:
                valence = valences_Wang2022[symbol]
                break
        if valence is None:
            raise Exception(f'The structure contains an element ({symbol})'
                            f' with unknown valence')
        ab_valences[key].append(valence)

    # check charge neutrality
    min_charge = sum([np.mean([min(val) for val in vals]) for vals in
                      ab_valences.values()]) + nO * (3 - min_delta)
    max_charge = sum([np.mean([max(val) for val in vals]) for vals in
                      ab_valences.values()]) + nO * (3 - max_delta)
    return (min_charge <= 0 and max_charge >= 0)


def get_phase_stability(
        atoms_row: Union[Atoms, AtomsRow],
        api_key: str = None,
        per_atom: bool = False,
        attempts: int = 2,
        user_potcar_settings: dict = None) -> Dict[str, float]:
    """Determine the energy above the convex hull as well as the decomposition
    and formation energies by constructing a phase diagram using data from the
    Materials Project database

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    api_key
        Materials Project API key
    per_atom
        return energies per atom
    attempts
        number of attempts at retrieving data from the Materials Project
    user_potcar_settings
        override POTCAR symbols, e.g. ``{"W": "W_sv"}``
    """

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # create Materials Project entry
    entry = get_computed_entry(structure,
                               user_potcar_settings=user_potcar_settings)

    # create rester instance
    mprester = MPRester(api_key=api_key)

    # retrieve structures from the Materials Project
    for i in range(attempts):
        try:
            entries = mprester.get_entries_in_chemsys(
                elements=list(set(structure.symbols)))
            break
        except Exception:
            entries = None
            warnings.warn(''.join([
                f'Attempt ({i}/{attempts}) at retrieving entries from the'
                f'Materials Project failed because of', '\n',
                traceback.format_exc()]))
        if entries is None:
            raise ValueError(f'Failed to retrieve entries for'
                             f' {structure.get_chemical_formula()} from the Materials Project')

    # construct phase diagram
    pd = PhaseDiagram([entry] + entries)

    # return the phase separation and formation energy as well as the energy
    # above the hull
    phase_stability = {
        'e_above_hull': pd.get_e_above_hull(entry),
        'phase_separation_energy': pd.get_phase_separation_energy(entry),
        'form_energy': pd.get_form_energy_per_atom(entry)}
    if not per_atom:
        phase_stability = {k: v * len(structure) for k, v in
                           phase_stability.items()}
    return phase_stability


def update_phase_stability_dbs(
        filenames: Union[str, List[str]],
        api_key: str = None,
        per_atom: bool = False,
        attempts: int = 2,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        asynchronus: bool = False,
        processes: int = 1,
        connect_each_row: bool = False,
        use_lock_file: bool = True,
        user_potcar_settings: dict = None) -> None:

    """Add the energy above the convex hull as well as the decomposition and
    formation energies to the selected rows in the specified databases by
    constructing phase diagrams using data from the Materials Project database

    Parameters
    ----------
    filenames
        path(s) to database file(s); the calculations will be parallelised over
        multiple files if provided, and rows otherwise
    api_key
        Materials Project API key
    per_atom
        return energies per atom
    attempts
        number of attempts at retrieving data from the Materials Project
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    user_potcar_settings
        override POTCAR symbols, e.g. ``{"W": "W_sv"}``
    """

    fun_kwargs = {'api_key': api_key,
                  'per_atom': per_atom,
                  'attempts': attempts,
                  'keyword_filters': keyword_filters,
                  'exclude_filtered': exclude_filtered,
                  'property_calculators': property_calculators,
                  'connect_each_row': connect_each_row,
                  'use_lock_file': use_lock_file,
                  'user_potcar_settings': user_potcar_settings}

    with (tempfile.TemporaryDirectory() if (isinstance(filenames, str) and
                                            '.gz' in filenames) else
            nullcontext()) as tmpdir:

        # decompress file if necessary
        if isinstance(filenames, str) and '.gz' in filenames:
            tmpnames = os.path.join(tmpdir, 'database.db')
            if os.path.isfile(filenames):
                decompress_db(filenames, tmpnames, delete=False)
        else:
            tmpnames = filenames

        if processes == 1:
            _update_phase_stability_dbs_mp(dict(filenames=tmpnames,
                                                selection=selection,
                                                **fun_kwargs))
        else:

            kwargs_list = []
            if isinstance(filenames, str):
                row_ids = get_row_ids(tmpnames, selection)
                processes = min(len(row_ids), processes)
                for row_id in row_ids:
                    kwargs = dict(
                        filenames=tmpnames,
                        selection=row_id,
                        **fun_kwargs)
                    kwargs_list.append(kwargs)
            else:
                processes = min(len(filenames), processes)
                for filename in tmpnames:
                    kwargs = dict(
                        filenames=[filename],
                        selection=selection,
                        **fun_kwargs)
                    kwargs_list.append(kwargs)

            Pool = multiprocessing.Pool
            with Pool(processes) as pool:
                if asynchronus:
                    results = pool.map_async(
                        _update_phase_stability_dbs_mp,
                        kwargs_list
                    )
                    results.get()
                else:
                    pool.map(
                        _update_phase_stability_dbs_mp,
                        kwargs_list
                    )

        if isinstance(filenames, str) and '.gz' in filenames:
            # Zip database and delete the existing file
            compress_db(tmpnames, filenames, delete=True)


def _update_phase_stability_dbs_mp(kwargs) -> None:
    """Parallelisable version of _update_phase_stability_dbs"""
    try:
        _update_phase_stability_dbs(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to add phase stabilities to the databases'
            f' {kwargs["filenames"]} because of', '\n',
            traceback.format_exc()]))


def _update_phase_stability_dbs(
        filenames: Union[str, List[str]],
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        api_key: str = None,
        per_atom: bool = False,
        attempts: int = 2,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        connect_each_row: bool = False,
        use_lock_file: bool = True,
        user_potcar_settings: dict = None) -> None:
    """Add the energy above the convex hull as well as the decomposition and
    formation energies to the selected rows in the specified databases by
    constructing phase diagrams using data from the Materials Project database

    Parameters
    ----------
    filenames
        path(s) to database file(s); the calculations will be parallelised over
        multiple files if provided, and rows otherwise
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    api_key
        Materials Project API key
        AtomsRow or Atoms object as input and return a scalar or dictionary
    per_atom
        return energies per atom
    attempts
        number of attempts at retrieving data from the Materials Project
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    fun_kwargs = {
        'selection': selection,
        'api_key': api_key,
        'per_atom': per_atom,
        'attempts': attempts,
        'keyword_filters': keyword_filters,
        'exclude_filtered': exclude_filtered,
        'property_calculators': property_calculators,
        'connect_each_row': connect_each_row,
        'use_lock_file': use_lock_file,
        'user_potcar_settings': user_potcar_settings
    }
    for filename in filenames:
        apply_function_db(filename=filename,
                          function=_update_phase_stability_db_mp,
                          fun_kwargs=fun_kwargs)


def _update_phase_stability_db_mp(kwargs) -> None:
    """Parallelisable version of _update_phase_stability_db"""
    try:
        _update_phase_stability_db(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to add phase stabilities to the database'
            f' {kwargs["filename"]} because of', '\n',
            traceback.format_exc()]))


def _update_phase_stability_db(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        api_key: str = None,
        per_atom: bool = False,
        attempts: int = 2,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        connect_each_row: bool = False,
        use_lock_file: bool = True,
        user_potcar_settings: dict = None) -> None:
    """Add the energy above the convex hull as well as the decomposition and
    formation energies to the selected rows in the specified database by
    constructing phase diagrams using data from the Materials Project database

    Parameters
    ----------
    filename
        path to ASE database
    row_id
        ID of row to update
    api_key
        Materials Project API key
        AtomsRow or Atoms object as input and return a scalar or dictionary
    per_atom
        return energies per atom
    attempts
        number of attempts at retrieving data from the Materials Project
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    user_potcar_settings
        override POTCAR symbols, e.g. ``{"W": "W_sv"}``
    """

    # setup calculators
    property_calculators = {
        'phase_stability': lambda x: get_phase_stability(x, api_key, per_atom,
                                                         attempts,
                                                         user_potcar_settings),
        **({} if property_calculators is None else property_calculators)}

    # get row ids
    row_ids = get_row_ids(filename, selection)

    with (nullcontext() if connect_each_row else
          connect(filename, use_lock_file=use_lock_file)) as database:

        for row_id in row_ids:

            with (nullcontext() if not connect_each_row else
                  connect(filename, use_lock_file=use_lock_file)) as db:

                # update phase stability
                if db is None:
                    db = database
                _add_structure_db(database=db,
                                  row_id=row_id,
                                  keyword_filters=keyword_filters,
                                  exclude_filtered=exclude_filtered,
                                  property_calculators=property_calculators)
