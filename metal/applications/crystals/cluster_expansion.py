from collections.abc import Iterable
from itertools import combinations, product
import numpy as np
import os
from random import sample, choice, choices, SystemRandom
import traceback
from typing import Dict, List, Union, Tuple, Callable
import warnings

from ase import Atoms
from ase.geometry import get_distances
from ase.units import kB
import pandas as pd

from icet import ClusterSpace, StructureContainer, ClusterExpansion
from icet.tools import enumerate_structures
from icet.tools.structure_generation import (generate_sqs,
                                             generate_sqs_from_supercells,
                                             generate_sqs_by_enumeration,
                                             occupy_structure_randomly,
                                             _validate_concentrations)
from icet.tools.ground_state_finder import GroundStateFinder
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from mchammer import DataContainer
from trainstation import CrossValidationEstimator, Optimizer
from trainstation.tools import ScatterData

from .tools import _add_structure_db_mp
from ...utils import apply_function_db, run_function_parallel

#### For reproducibility of structure generation for build_training_database.py and possibly other scripts
#
# import random
# 
# SEED = 3
# np.random.seed(SEED)
# random.seed(SEED)


def build_db_all_compositions(
        primitive: Atoms,
        filename: str,
        chemical_symbols: Union[List[str], List[List[str]]],
        dimensions: Union[int, List[int]] = 3,
        max_elements: Union[int, Dict[str, int]] = 2,
        n_structures: int = None,
        replace: bool = False,
        symprec: float = 1e-5,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None) -> None:
    """Create a database containing all possible compositions, with random
    atomic configurations, based on a specific primitive structure and list of
    chemical symbols for each site

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    dimensions
        dimensions of the supercell
    max_elements
        integer or dictionary indicating the maximum number of distinct
        elements per active sublattice
    n_structures
        number of structures to select, before filtering; the default is all
    replace
        treat the list of chemical symbols as possible replacements of the
        elements on the provided primitive structure
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or
        dictionary; note that it can be used to overwrite the ``tag``
        associated with each row, which is used to determine the uniqueness
    """

    # generate structures
    structures = generate_all_compositions(
            primitive=primitive,
            chemical_symbols=chemical_symbols,
            dimensions=dimensions,
            max_elements=max_elements,
            n_structures=n_structures,
            replace=replace,
            symprec=symprec)

    # add structures to database
    fun_kwargs = [
        dict(structure=structure,
             tag=structure.get_chemical_formula(),
             vacancies=vacancies,
             keyword_filters=keyword_filters,
             exclude_filtered=exclude_filtered,
             property_calculators=property_calculators)
        for structure in structures]
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def generate_all_compositions(
        primitive: Atoms,
        chemical_symbols: Union[List[str], List[List[str]]],
        dimensions: Union[int, List[int]] = 3,
        max_elements: Union[int, Dict[str, int]] = 2,
        n_structures: int = None,
        replace: bool = False,
        symprec: float = 1e-5) -> List[Atoms]:
    """Generate a list of structures corresponding to all possible
    compositions, with random atomic configurations, based on a specific
    primitive structure and list of chemical symbols for each site

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    dimensions
        dimensions of the supercell
    max_elements
        integer or dictionary indicating the maximum number of distinct
        elements per active sublattice
    n_structures
        number of structures to select, before filtering; the default is all
    replace
        treat the list of chemical symbols as possible replacements of the
        elements on the provided primitive structure
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    """

    # create cluster space
    cs = ClusterSpace(primitive, cutoffs=[1.0],
                      chemical_symbols=chemical_symbols, symprec=symprec)

    # set maximum number of elements per sublattice
    supercell = primitive.repeat(dimensions)
    sl_sites_species = {sl.symbol: (len(sl.indices), sl.chemical_symbols) for
                        sl in cs.get_sublattices(supercell) if
                        len(sl.chemical_symbols) > 1}
    if isinstance(max_elements, int):
        max_elements = {sl_symbol: max_elements for sl_symbol in
                        sl_sites_species.keys()}
    elif isinstance(max_elements, dict):
        warnings.warn('\n'.join([
            'When providing a dictionary with the maximum number of elements'
            ' the keys must match the sublattices of the cluster space:',
            str(cs)]))
        if not(set(max_elements.keys()) == set(sl_sites_species.keys())):
            raise ValueError(f'The sublattice symbols in the dictionary with'
                             f' the maximum number of elements'
                             f' ({list(max_elements.keys())}) and the cluster'
                             f' space ({list(sl_sites_species.keys())}) do not'
                             f' match')
    else:
        raise ValueError(f'The maximum number of elements should be provided'
                         f' as a single integer or a dictionary thereof not'
                         f' {type(max_elements)}')

    # generate possible sublattice concentrations
    all_sl_concs = []
    for sl_symbol, (sites, chemical_symbols) in sl_sites_species.items():
        sl_max = max_elements[sl_symbol]
        counts = [i for i in range(sites + 1)]
        count_combos = np.array(list(product(counts, repeat=sl_max)))
        sum_mask = np.where(np.sum(count_combos, axis=1) == sites)
        conc_combos = count_combos[sum_mask] / sites
        if replace:
            fixed_symbols = list(set(
                supercell.symbols).intersection(set(chemical_symbols)))
            replace_symbols = list(set(chemical_symbols) - set(fixed_symbols))
            if len(replace_symbols) == 0:
                sym_combos = [fixed_symbols]
            else:
                sym_combos = [
                    list(combo) + fixed_symbols for combo in
                    combinations(replace_symbols,
                                 max(sl_max - len(fixed_symbols), 0))]
        else:
            sym_combos = np.array(combinations(chemical_symbols, sl_max))
        all_sl_concs.append([
            {sym: (concs[syms.index(sym)] if sym in syms else 0.0) for sym in
             chemical_symbols} for concs in conc_combos for syms in sym_combos
        ])

    # generate possible structures
    structures = {}
    for sl_conc_combos in product(*all_sl_concs):
        structure = supercell.copy()
        sl_concs = {sl_sym: sl_conc for sl_sym, sl_conc in
                    zip(sl_sites_species.keys(), sl_conc_combos)}
        occupy_structure_randomly(structure, cs, sl_concs)
        formula = structure.get_chemical_formula()
        structures[formula] = structure
    structures = list(structures.values())

    if n_structures is not None:
        return sample(structures, n_structures)
    return structures


def build_db_enumerate(
        primitive: Atoms,
        filename: str,
        chemical_symbols: Union[List[str], List[List[str]]],
        concentration_restrictions: Dict[str, float],
        sizes: Union[List[int], range],
        n_structures: int = None,
        symprec: float = 1e-5,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None) -> None:
    """Create a database with special quasi-random structures based on a
    specific primitive structure and concentration restrictions

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    concentration_restrictions
        allowed concentration range for one or more element in
        `chemical_symbols`, e.g., ``{'Au': (0, 0.2)}`` will only
        enumerate structures in which the Au content is between 0 and
        20 %; here, concentration is always defined as the number of
        atoms of the specified kind divided by the number of *all* atoms.
    sizes
        range of supercell sizes, i.e. number of primitive cells, (included in
        enumeration)
    n_structures
        number of structures to select, before filtering; the default is all
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or dictionary
    """

    if append and n_structures is not None:
        warnings.warn('When adding randomly selected enumerated structures'
                      ' to an existing database, there may be duplicates')

    # generate structures
    structures = enumerate_structures(
            structure=primitive, sizes=sizes,
            chemical_symbols=chemical_symbols,
            concentration_restrictions=concentration_restrictions,
            symprec=symprec)
    if n_structures is not None:
        structures = sample(list(structures), n_structures)

    # add structures to database
    fun_kwargs = [
        dict(structure=structure,
             vacancies=vacancies,
             keyword_filters=keyword_filters,
             exclude_filtered=exclude_filtered,
             property_calculators=property_calculators)
        for structure in structures]
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def build_db_random_configurations(
        primitive: Atoms,
        filename: str,
        chemical_symbols: Union[List[str], List[List[str]]],
        sublattice_concentrations_list: List[Dict[str, Dict[str, float]]],
        dimensions: Union[int, List[int]] = 3,
        n_structures: int = 1,
        symprec: float = 1e-5,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None) -> None:
    """Create a database with randomly generated configurations based on a
    specific primitive structure and sublattice concentrations

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    sublattice_concentrations_list
        list of dictionaries with concentration of each species in the target
        structure, per sublattice (for example {'Au': 0.5, 'Pd': 0.5} for a
        single sublattice Au-Pd structure, or {'A': {'Au': 0.5, 'Pd': 0.5},
        'B': {'H': 0.25, 'X': 0.75}} for a system with two sublattices. The
        symbols defining sublattices ('A', 'B' etc) can be found by printing
        the cluster_space
    dimensions
        dimensions of the supercell
    n_structures
        number of structures to generate, for each concentration, before
        filtering
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or dictionary
    """

    # create dummy cluster space
    cs = ClusterSpace(primitive, cutoffs=[1.0],
                      chemical_symbols=chemical_symbols, symprec=symprec)
    supercell = primitive.repeat(dimensions)

    # generate structures
    fun_kwargs = []
    for sublattice_concentrations in sublattice_concentrations_list:
        for i in range(n_structures):
            if (i + 1) % 10 == 0:
                print('Generating structure', i + 1, 'of', n_structures, end='\r')
            structure = supercell.copy()
            occupy_structure_randomly(structure, cs, sublattice_concentrations)
            tag = (None if n_structures > 1 else
                   structure.get_chemical_formula())
            fun_kwargs.append(
                dict(structure=structure,
                     vacancies=vacancies,
                     tag=tag,
                     keyword_filters=keyword_filters,
                     exclude_filtered=exclude_filtered,
                     property_calculators=property_calculators))

    # add structures to database
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def build_sqs_db(
        primitive: Atoms,
        filename: str,
        cutoffs: List[float],
        chemical_symbols: Union[List[str], List[List[str]]],
        sublattice_concentrations_list: List[Dict[str, Dict[str, float]]],
        dimensions: Union[int, List[int]] = None,
        max_size: int = 4,
        enumeration: bool = False,
        n_steps: int = None,
        symprec: float = 1e-5,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        overwrite: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        asynchronus: bool = False,
        processes: int = 1) -> None:
    """Create a database with randomly generated configurations based on a
    specific primitive structure and sublattice concentrations

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    cutoffs
        cutoff radii per order that define the cluster space

        Cutoffs are specified in units of Angstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. ``cutoffs=[7.0, 4.5]`` thus implies
        that all pairs distanced 7 A or less will be included,
        as well as all triplets among which the longest distance is no
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    sublattice_concentrations_list
        list of dictionaries with concentration of each species in the target
        structure, per sublattice (for example {'Au': 0.5, 'Pd': 0.5} for a
        single sublattice Au-Pd structure, or {'A': {'Au': 0.5, 'Pd': 0.5},
        'B': {'H': 0.25, 'X': 0.75}} for a system with two sublattices. The
        symbols defining sublattices ('A', 'B' etc) can be found by printing
        the cluster_space
    dimensions
        dimensions of the supercell; is superseded by ``max_size``
    max_size
        search among all supercells obtained by repeating the primitive cell,
        in a single direction, between one and ``max_size`` number of times;
        supersedes ``dimensions``
    enumeration
        find SQS via enumeration; ignored if ``dimensions`` is specified
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    overwrite
        overwrite exisiting entries if the formulas match
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or
        dictionary; note that it can be used to overwrite the ``tag``
        associated with each row, which is used to determine the uniqueness
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    """

    # generate SQS structures
    fun_kwargs = [
        dict(primitive=primitive,
             cutoffs=cutoffs,
             chemical_symbols=chemical_symbols,
             dimensions=dimensions,
             max_size=max_size,
             symprec=symprec,
             n_steps=n_steps,
             enumeration=enumeration,
             sublattice_concentrations=sublattice_concentrations)
        for sublattice_concentrations in sublattice_concentrations_list]
    structures = run_function_parallel(function=_generate_sqs_wrapper_mp,
                                       fun_kwargs=fun_kwargs,
                                       asynchronus=asynchronus,
                                       processes=processes)

    # add SQS structures to database
    fun_kwargs = [
        dict(structure=structure,
             tag=structure.get_chemical_formula(),
             vacancies=vacancies,
             overwrite=overwrite,
             keyword_filters=keyword_filters,
             exclude_filtered=exclude_filtered,
             property_calculators=property_calculators)
        for structure in structures]
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def _generate_sqs_wrapper_mp(kwargs):
    """Parallelisable version of generate_sqs"""
    try:
        structure = generate_sqs_wrapper(**kwargs)
    except Exception:
        warnings.warn(''.join([
            'Failed to generate SQS structure with concentrations', '\n',
            kwargs["sublattice_concentrations"], f'based on '
            f'{kwargs["primitive"].get_chemical_formula()} because of',
            '\n', traceback.format_exc()]))
        return
    return structure


def generate_sqs_wrapper(
        primitive: Atoms,
        cutoffs: List[float],
        chemical_symbols: Union[List[str], List[List[str]]],
        sublattice_concentrations: Dict[str, Dict[str, float]],
        dimensions: Union[int, List[int]] = None,
        max_size: int = 4,
        n_steps: int = None,
        symprec: float = 1e-5,
        enumeration: bool = False) -> None:
    """Generate a special quasi-random structure based on a specific primitive
    structure and sublattice concentrations

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    cutoffs
        cutoff radii per order that define the cluster space

        Cutoffs are specified in units of Angstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. ``cutoffs=[7.0, 4.5]`` thus implies
        that all pairs distanced 7 A or less will be included,
        as well as all triplets among which the longest distance is no
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    sublattice_concentrations
        concentration of each species in the target structure, per sublattice
        (for example {'Au': 0.5, 'Pd': 0.5} for a single sublattice Au-Pd
        structure, or {'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25,
        'X': 0.75}} for a system with two sublattices. The symbols defining
        sublattices ('A', 'B' etc) can be found by printing the cluster_space
    dimensions
        dimensions of the supercell; is superseded by ``max_size``
    max_size
        search among all supercells obtained by repeating the primitive cell,
        in a single direction, between one and ``max_size`` number of times;
        supersedes ``dimensions``
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    enumeration
        find SQS via enumeration; ignored if ``dimensions`` is specified
    """

    # create cluster space
    cs = ClusterSpace(primitive, cutoffs=cutoffs,
                      chemical_symbols=chemical_symbols, symprec=symprec)
    sqs_kwargs = {}
    if n_steps is not None:
        sqs_kwargs['n_steps'] = n_steps

    # find SQS
    if _check_concentrations(sublattice_concentrations, cs):
        if dimensions is not None:
            structure = generate_sqs_from_supercells(
                cluster_space=cs,
                supercells=[primitive.repeat(dimensions)],
                target_concentrations=sublattice_concentrations,
                **sqs_kwargs)
        else:
            if enumeration:
                structure = generate_sqs_by_enumeration(
                    cluster_space=cs,
                    max_size=max_size,
                    target_concentrations=sublattice_concentrations,
                    include_smaller_cells=True)
            else:
                structure = generate_sqs(
                    cluster_space=cs,
                    max_size=max_size,
                    target_concentrations=sublattice_concentrations,
                    include_smaller_cells=True)
    else:

        warnings.warn('No swaps are possible given the specified'
                      ' concentrations {sublattice_concentrations}, a'
                      ' simple supercell of the primitive structure will'
                      ' therefore be created')

        if dimensions is None:
            dims = list([(size, 1, 1) for size in range(1, max_size + 1)])
        else:
            dims = [dimensions]

        for dim in dims:
            structure = primitive.repeat(dim)
            try:
                occupy_structure_randomly(structure, cs,
                                          sublattice_concentrations)
                break
            except ValueError:
                pass

            structure = None

        if structure is None:
            raise Exception('Failed to generate a supercell that matches the'
                            ' specificed concentrions')

    return structure


def _check_concentrations(concentrations: dict,
                          cluster_space: ClusterSpace) -> bool:
    """
    Checks that the concentrations are valide against a cluster space (raises
    `ValueError` if they do not match) and that swaps are possible one or more
    of the sublattices.

    Parameters
    ----------
    concentrations
        concentration specification
    cluster_space
        cluster space to check against
    """
    concentrations = _validate_concentrations(cluster_space=cluster_space,
                                              concentrations=concentrations)

    # check that swaps are possible on at least one sublattice
    for sl_symbol, conc_dict in concentrations.items():
        swap_possible = len(np.nonzero(list(conc_dict.values()))[0]) > 1
        if swap_possible:
            return True
    return False


def build_db_random_concentrations(
        primitive: Atoms,
        filename: str,
        chemical_symbols: Union[List[str], List[List[str]]],
        dimensions: Union[int, List[int]] = 3,
        fixed_symbols: List[str] = None,
        min_total: int = None,
        max_total: int = None,
        min_sublattice: dict = None,
        max_sublattice: dict = None,
        deviation_limit: int = None,
        n_structures: int = 100,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        overwrite: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None) -> None:
    """Create a database of structures with random configurations and
       concentrations based on a specific primitive structure and set of
       chemical symbols

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    dimensions
        dimensions of the supercell
    fixed_symbols
        symbols for chemical elements in the primitive that should not be
        replaced
    min_total
        minimum total number of distinct elements in each structure
    max_total
        maximum total number of distinct elements in each structure
    min_sublattice
        minimum number of distinct elements on each sublattice
    max_sublattice
        maximum number of distinct elements on each sublattice
    deviation_limit
        maximum deviation in the number of atoms, of a given species, per
        sublattice
    n_structures
        number of structures to generate, before filtering
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    overwrite
        overwrite exisiting entries if the formulas match
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or
        dictionary; note that it can be used to overwrite the ``tag``
        associated with each row, which is used to determine the uniqueness
    """

    # create cluster space
    cluster_space = ClusterSpace(primitive, cutoffs=[1.0],
                                 chemical_symbols=chemical_symbols)
    supercell = primitive.repeat(dimensions)
    sublattices = cluster_space.get_sublattices(supercell)

    # determine possible combinations
    combos = None
    if not all([lim is None for lim in [min_sublattice, max_sublattice]]):
        combos = []
        for sl in sublattices:
            if min_sublattice is not None:
                min_sl = min_sublattice.get(sl.symbol, 1)
            else:
                min_sl = 1
            if max_sublattice is not None:
                max_sl = max_sublattice.get(sl.symbol,
                                            len(sl.chemical_symbols))
            else:
                max_sl = len(sl.chemical_symbols)
            combos.append(list(range(min_sl, max_sl)))
        combos = np.array(list(product(*combos)))
    if not all([lim is None for lim in [min_total, max_total]]):
        if combos is None:
            combos = np.array(list(product(*[
                list(range(1, len(sl.chemical_symbols))) for sl in
                sublattices])))
        if min_total is not None:
            combos = combos[np.where(np.sum(combos, axis=1) >= min_total)]
        if max_total is not None:
            combos = combos[np.where(np.sum(combos, axis=1) <= max_total)]

    # generate structures
    fun_kwargs = []
    for _ in range(n_structures):
        structure = supercell.copy()
        if combos is not None:
            combo = choice(combos)
        for s, sl in enumerate(sublattices):
            if combos is not None:
                possible_symbols = sample(sl.chemical_symbols, k=combo[s])
            else:
                possible_symbols = sl.chemical_symbols
            indices = [
                index for index in sl.indices if (
                    fixed_symbols is None or structure.symbols[index] not in
                    fixed_symbols)]
            if len(indices) == 0:
                continue
            if (len(possible_symbols) == 1 or deviation_limit is None or
                    sl.symbol not in deviation_limit):
                for index in indices:
                    structure.symbols[index] = choice(possible_symbols)
            else:
                counts = np.array(list(product(
                    list(range(len(indices))), repeat=len(possible_symbols))))
                counts = counts[np.where(np.sum(counts, axis=1) ==
                                         len(indices))]
                max_diff = np.max(np.abs(np.atleast_2d([
                    counts[:, i] - counts[:, j] for i in
                    range(counts.shape[1] - 1) for j in
                    range(i + 1, counts.shape[1])])), axis=0)
                counts = counts[np.where(
                    max_diff <= deviation_limit[sl.symbol])]
                for count, symbol in zip(
                        choice(counts), sample(possible_symbols,
                                               k=len(possible_symbols))):
                    symbol_indices = sample(indices, k=count)
                    indices = list(set(indices) - set(symbol_indices))
                    for index in symbol_indices:
                        structure.symbols[index] = symbol
        tag = structure.get_chemical_formula()
        fun_kwargs.append(
            dict(structure=structure,
                 overwrite=overwrite,
                 vacancies=vacancies,
                 tag=tag,
                 keyword_filters=keyword_filters,
                 exclude_filtered=exclude_filtered,
                 property_calculators=property_calculators))

    # add structures to database
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def create_ce_structure_container(
        primitive: Atoms,
        cutoffs: List[float],
        structures: List[Atoms],
        chemical_symbols: Union[List[str], List[List[str]]],
        symprec: float = 1e-5,
        filename: str = None,
        property_list: List[Dict[str, float]] = None,
        minimum_distance: Dict[str, float] = None) -> StructureContainer:
    """
    Create a structure container for all structures based on the cluster space
    created created from the provided primitive structure and cutoffs

    Parameters
    ----------
    primitive : ase.Atoms
        primitive structure
    cutoffs : list of float
        cutoff radii per order that define the cluster space. Cutoffs are
        specified in units of Angstrom and refer to the longest distance
        between two atoms in the cluster. The first element refers to pairs,
        the second to triplets, the third to quadruplets, and so on.
        cutoffs=[7.0, 4.5] thus implies that all pairs distanced 7 A or less
        will be included, as well as all triplets among which the longest
        distance is no longer than 4.5 A.
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    structures : ase.Atoms
        the atomic structures to be added
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    filename
        name of file to which to write the output. If left unspecified no file
        is written
    property_list
        list of scalar properties. If properties are not specified the
        structure objects will be checked for an attached ASE calculator object
        with a calculated potential energy
    minimum_distance
        dictionary defining the minimum distance, for each sublattice, below
        which all orbits should be removed
    """

    if property_list is not None:
        if len(property_list) != len(structures):
            raise Exception(f'The number of structures ({len(structures)})'
                            f' and properties ({len(property_list)}) do not'
                            f' match')
    else:
        property_list = [None] * len(structures)

    # create cluster space
    cs = ClusterSpace(primitive, cutoffs=cutoffs,
                      chemical_symbols=chemical_symbols, symprec=symprec)
    if minimum_distance is not None:
        remove_orbits(cs, minimum_distance)

    # initiate structure container
    sc = StructureContainer(cluster_space=cs)

    # add structures
    for structure, properties in zip(structures, property_list):
        sc.add_structure(structure, properties=properties)

    # write structure container to file
    if filename is not None:
        sc.write(filename)

    return sc


def remove_orbits(
        cluster_space: ClusterSpace,
        minimum_distance: Dict[str, float]) -> None:
    """Remove orbits in which the minimum distance for the atoms on a specific
    sublattice is lower than the specified limit.

    Parameters
    ----------
    cluster_space
        cluster space
    minimum_distance
        dictionary with the minimum distance for each sublattice
    """

    orbits_to_delete = []
    primitive = cluster_space.primitive_structure
    sublattices = cluster_space.get_sublattices(primitive)
    for orbit_index in range(len(cluster_space.orbit_list)):
        representative_cluster = cluster_space.orbit_list.get_orbit(
            orbit_index).representative_cluster
        orbit_sublattices = [sublattices[
            sublattices.get_sublattice_index_from_site_index(ls.index)].symbol
                             for ls in representative_cluster.lattice_sites]
        positions = representative_cluster.positions
        remove = False
        for subl, mind in minimum_distance.items():
            indices = [i for i, s in enumerate(orbit_sublattices) if s == subl]
            if len(indices) < 2:
                continue
            if remove:
                break
            for i in range(len(indices)):
                if remove:
                    break
                for j in range(i + 1, len(indices)):
                    dist = get_distances(positions[i], positions[j])[1][0]
                    if dist < mind:
                        orbits_to_delete.append(orbit_index)
                        remove = True
                        break

    # update merge/prune history
    cluster_space._prune_orbit_list(orbits_to_delete)


def train_multiple_ces(structure_container: Union[str, StructureContainer],
                       directory: str = None,
                       common_kwargs: Dict = None,
                       additional_kwargs: List[Dict] = None,
                       common_tag: str = None,
                       additional_tags: List[str] = None,
                       write_ces: bool = True,
                       write_summaries: bool = True,
                       include_summary: bool = True,
                       cross_validation: bool = True) -> None:
    """Train multiple cluster expansion, and perform cross validations if
    requested, based on the provided dictionaries with keywords arguments

    Parameters
    ----------
    structure_container
        StructureContainer object or path to structure container file
        the folder that contains the structure container file
    directory
        path to directory where output files should be stored, the default is
        the folder that contains the structure container file
    common_kwargs
        dictionary with keyword arguments related to the fitting that shall be
        used for all cases
    additional_kwargs
        list of dictionaries with additional keyword arguments
    common_tag
        tag used to name the output files, the default is to extract it
        from the name of the structure container file
    additional_tags
        tags that should be added to name of the individual output files
    write_ces
        write the CEs to file
    write_summaries
        write the summaries to file
    include_summary
        include summary in the output
    cross_validation
        perform cross validation
    """

    if not write_ces and not write_summaries:
        raise Exception('Aborting since neither CEs nor summaries would be'
                        ' written')

    if isinstance(structure_container, str):
        structure_container = StructureContainer.read(structure_container)
        if common_tag is None:
            common_tag = os.path.basename(
                structure_container).replace('.sc', '')
        if directory is None:
            directory = os.path.dirname(structure_container)
    elif isinstance(structure_container, StructureContainer):
        if common_tag is None:
            common_tag = 'cluster_expansion'
        if directory is None:
            directory = os.getcwd()
    else:
        raise Exception(f'structure_container must be a string or'
                        f' StructurContainer object, not'
                        f' {type(structure_container)}')

    # update kwargs
    default_kwargs = dict(fit_method='rfe')
    if common_kwargs is not None:
        default_kwargs.update(common_kwargs)
    if additional_kwargs is None:
        fun_kwargs = [default_kwargs]
        tags = [common_tag]
    else:
        if additional_tags is None:
            tags = [common_tag] * len(additional_kwargs)
        else:
            if len(additional_kwargs) != len(additional_tags):
                raise Exception(f'The number of argument sets'
                                f' ({len(additional_kwargs)}) and tags'
                                f' ({len(additional_tags)}) do not match')
            tags = ['_'.join([common_tag, tag]) for tag in additional_tags]

        fun_kwargs = [{**kw, **default_kwargs} for kw in additional_kwargs]

    # train CEs
    for tag, kwargs in zip(tags, fun_kwargs):

        train_ce(structure_container=structure_container,
                 directory=directory,
                 tag=tag,
                 write=True,
                 include_ce=write_ces,
                 include_summary=write_summaries,
                 cross_validation=cross_validation)


def train_ce(structure_container: Union[str, StructureContainer],
             directory: str = None,
             tag: str = None,
             write: bool = True,
             include_ce: bool = True,
             include_summary: bool = True,
             cross_validation: bool = True,
             append: bool = True,
             **kwargs) -> Union[ClusterExpansion,
                                Tuple[ClusterExpansion, Dict]]:
    """Train a cluster expansion, and perform a cross validation if requested,
    based on the provides dictionaries with keywords arguments

    Parameters
    ----------
    structure_container
        StructureContainer object or path to structure container file
    directory
        path to directory where output files should be stored, the default is
        the folder that contains the structure container file, if provided, or
        the current working directory
    tag
        tag that shall be used to name the output files, the default is to use
        the name of the structure container file, if provided, or
        cluster_expansion/cluster_expansion_summary
    write
        write the CE, and summary if included, to file
    include_ce
        include CE in the output
    include_summary
        include summary in the output
    cross_validation
        perform cross validation
    append
        append the summary if the corresponding file already exists, this also
        means that only scalar properties are included
    """

    # collect data from structure container
    if isinstance(structure_container, str):
        if tag is None:
            tag = os.path.basename(structure_container).replace(
                '.sc', '').replace('structure_container', 'cluster_expansion')
        if directory is None:
            directory = os.path.realpath(os.path.dirname(structure_container))
        structure_container = StructureContainer.read(structure_container)
    elif isinstance(structure_container, StructureContainer):
        if tag is None:
            tag = 'cluster_expansion'
        if directory is None:
            directory = os.getcwd()
    else:
        raise Exception(f'structure_container must be a string or'
                        f' StructurContainer object, not'
                        f' {type(structure_container)}')
    cluster_space = structure_container.cluster_space.copy()
    cvs, props = structure_container.get_fit_data(key='energy')
    avg = np.average(props)
    props -= avg
    fit_data = (cvs, props)

    # train CE
    sysrand = SystemRandom()

    seed = sysrand.randint(0, 2*32-1)
    if cross_validation:
        # set up cross-validation-estimator
        opt = CrossValidationEstimator(fit_data=fit_data, seed=seed,
                                       **kwargs)
    else:
        # set up optimizer
        opt = Optimizer(fit_data=fit_data, seed=seed, **kwargs)

    # train cluster expansion
    if cross_validation:
        opt.validate()
    opt.train()
    if include_ce:
        parameters = opt.parameters
        parameters[0] += avg
        cluster_expansion = ClusterExpansion(cluster_space, parameters)

    # prepare summary
    if include_summary:
        summary = opt.summary
        for c, cutoff in enumerate(cluster_space.cutoffs):
            summary[f'cutoff_{c + 2}'] = cutoff
        for order, number_of_orbits in\
                cluster_space.number_of_orbits_by_order.items():
            summary[f'norbits_{order}'] = number_of_orbits

    if write:

        os.makedirs(directory, exist_ok=True)
        if include_ce:
            cluster_expansion.write(f'{directory}/{tag}.ce')

        if include_summary:
            save_dataframe(summary, f'{directory}/{tag}_summary.csv', append)

    if include_ce:
        if include_summary:
            return cluster_expansion, summary
        return cluster_expansion
    if include_summary:
        return summary


def save_dataframe(
        summary: Dict,
        filename: str,
        append=True) -> None:
    """Write a dataframe to csv, appending new data (in dict-format) to an
    existing file if requested.

    Parameters
    ----------
    summary
        summary in the form of a dictionary
    filename
        path to .csv file
    append
        append the summary if the corresponding file already exists, this also
        means that only scalar properties are included
    """
    summary = {k: v for k, v in summary.items() if not
               (isinstance(v, (ScatterData, Iterable)) and not
                isinstance(v, str))}
    if append and os.path.exists(filename):
        df = pd.read_csv(filename, sep='\t', index_col=0)
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    df.to_csv(filename, sep='\t')


def run_multiple_mc(
        cluster_expansion: Union[str, ClusterExpansion],
        primitive: Atoms,
        sublattice_concentrations_list: Dict[str, Dict[str, float]],
        directories: Union[str, List[str]] = None,
        tags: Union[str, List[str]] = None,
        dimensions: Union[int, List[int]] = 2,
        temperatures: List[float] = range(2000, -1, -100),
        mc_cycles: int = 32000,
        observation_interval: int = 1,
        write_period: int = 1800,
        max_steps: int = None,
        asynchronus: bool = True,
        processes: int = 1) -> None:
    """
    Sample CE at all specified temperatures and write the results to data
    container files

    Parameters
    ----------
    cluster_expansion
        path to cluster expansion file
    primitive
        primitive structure
    sublattice_concentrations_list
        list of dictionaries with concentration of each species in the target
        structure, per sublattice (for example {'Au': 0.5, 'Pd': 0.5} for a
        single sublattice Au-Pd structure, or {'A': {'Au': 0.5, 'Pd': 0.5},
        'B': {'H': 0.25, 'X': 0.75}} for a system with two sublattices. The
        symbols defining sublattices ('A', 'B' etc) can be found by printing
        the cluster_space
    directories
        path or list of paths to directories where output files for each of the
        specified concentrations should be stored; the default is to use to
        create folders based on the chemical formulas
    tags
        tag or list of tags used for naming the data container files, to which
        `_T` followed by the temperature is added; the default is to determine
        the tag from the cluster expansion file, if one has been provided
    dimensions
        dimensions of the supercell
    temperatures
        list of temperatures
    mc_cycles : int
        number of MC cycles, each of which constists of as many steps as there
        are atoms in the supercell
    write_period
        period in units of seconds at which the data container is written to
        file
    observation_interval
        interval, measured in cycles, at which data is written to the
        data container, including the current occupation vector of the atomic
        configuration
    max_steps
        maximum number of MC steps to perform
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    """

    if directories is None:
        directories = [None] * len(sublattice_concentrations_list)
    else:
        if isinstance(directories, str):
            directories = [directories] * len(sublattice_concentrations_list)
        else:
            if len(directories) != len(sublattice_concentrations_list):
                raise Exception(f'The number of directories'
                                f' ({len(directories)}) and sublattice'
                                f' concentrations'
                                f' ({len(sublattice_concentrations_list)}) do'
                                f' not match')
    if tags is None:
        tags = [None] * len(sublattice_concentrations_list)
    else:
        if isinstance(tags, str):
            tags = [tags] * len(sublattice_concentrations_list)
        else:
            if len(tags) != len(sublattice_concentrations_list):
                raise Exception(f'The number of tags'
                                f' ({len(tags)}) and sublattice concentrations'
                                f' ({len(sublattice_concentrations_list)}) do'
                                f' not match')

    if processes > 1:

        from multiprocessing import Pool

        # Collect list of keyword arguments
        fun_kwargs = [
            dict(cluster_expansion=cluster_expansion,
                 primitive=primitive,
                 sublattice_concentrations=sublattice_concentrations,
                 directory=directory,
                 tag=tag,
                 dimensions=dimensions,
                 temperatures=temperatures,
                 mc_cycles=mc_cycles,
                 observation_interval=observation_interval,
                 write_period=write_period,
                 max_steps=max_steps)
            for tag, directory, sublattice_concentrations in
            zip(tags, directories, sublattice_concentrations_list)]

        with Pool(processes) as pool:
            if asynchronus:
                results = pool.map_async(_run_mc_mp, fun_kwargs)
                results.get()
            else:
                pool.map(_run_mc_mp, fun_kwargs)

    else:

        for tag, directory, sublattice_concentrations in zip(
                tags, directories, sublattice_concentrations_list):
            run_mc(
                cluster_expansion=cluster_expansion, primitive=primitive,
                sublattice_concentrations=sublattice_concentrations,
                directory=directory, tag=tag, temperatures=temperatures,
                mc_cycles=mc_cycles, observation_interval=observation_interval,
                write_period=write_period, max_steps=max_steps)


def _run_mc_mp(kwargs):
    """Parallelisable version of run_mc"""
    try:
        run_mc(**kwargs)
    except Exception:
        warnings.warn(''.join([
            'Failed to run MC simulations for structure with concentrations',
            '\n', kwargs["sublattice_concentrations"], f'based on '
            f'{kwargs["primitive"].get_chemical_formula()} because of',
            '\n', traceback.format_exc()]))


def run_mc(
        cluster_expansion: Union[str, ClusterExpansion],
        primitive: Atoms,
        sublattice_concentrations: Dict[str, Dict[str, float]],
        directory: str = None,
        tag: str = None,
        dimensions: Union[int, List[int]] = 2,
        temperatures: List[float] = range(2000, -1, -100),
        mc_cycles: int = 32000,
        observation_interval: int = 1,
        write_period: int = 1800,
        max_steps: int = None) -> None:
    """
    Sample CE at all specified temperatures and write the results to data
    container files

    Parameters
    ----------
    cluster_expansion
        ClusterExpansion instance or path to cluster expansion file
    primitive
        primitive structure
    sublattice_concentrations
        concentration of each species in the target structure, per sublattice
        (for example {'Au': 0.5, 'Pd': 0.5} for a single sublattice Au-Pd
        structure, or {'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25,
        'X': 0.75}} for a system with two sublattices. The symbols defining
        sublattices ('A', 'B' etc) can be found by printing the cluster_space
    directory
        path to directory where output files should be stored; the default is
        to use to create a folder based on the chemical formula
    tag
        tag used for naming the data container files, to which `_T` followed by
        the temperature is added; the default is to determine the tag from the
        cluster expansion file, if one has been provided
    dimensions
        dimensions of the supercell
    temperatures
        list of temperatures
    mc_cycles : int
        number of MC cycles, each of which constists of as many steps as there
        are atoms in the supercell
    write_period
        period in units of seconds at which the data container is written to
        file
    observation_interval
        interval, measured in cycles, at which data is written to the
        data container, including the current occupation vector of the atomic
        configuration
    max_steps
        maximum number of MC steps to perform
    """
    # read cluster expansion
    if isinstance(cluster_expansion, str):
        if tag is None:
            tag = os.path.basename(cluster_expansion).replace(
                '.ce', '').replace('cluster_expansion', '')
        cluster_expansion = ClusterExpansion.read(cluster_expansion)
    elif not isinstance(cluster_expansion, ClusterExpansion):
        raise Exception(f'cluster_expansion must be a string or'
                        f' ClusterExpansion object, not'
                        f' {type(cluster_expansion)}')
    cluster_space = cluster_expansion.get_cluster_space_copy()

    # generate initial structure
    structure = primitive.repeat(dimensions)
    occupy_structure_randomly(structure, cluster_space,
                              sublattice_concentrations)
    if directory is None:
        directory = structure.get_chemical_formula()
    calculator = ClusterExpansionCalculator(structure, cluster_expansion)

    # run simulation
    observation_interval *= len(structure)
    if tag is None or tag == '':
        tag = ''
    else:
        tag += '_'
    os.makedirs(directory, exist_ok=True)
    for T in temperatures:

        # check if the calculations are already finished
        dc_file = f'{directory}/{tag}T{T}.dc'
        number_of_trial_steps = len(structure) * mc_cycles
        if max_steps is not None:
            if os.path.isfile(dc_file):
                data_container = DataContainer.read(dc_file)
                if data_container.last_state['last_step'] == max_steps:
                    structure = data_container.get_trajectory()[-1]
                    continue
                else:
                    number_of_trial_steps =\
                        max_steps - data_container.last_state['last_step']

        mc = CanonicalEnsemble(
            structure=structure, calculator=calculator, temperature=T,
            dc_filename=dc_file,
            ensemble_data_write_interval=observation_interval,
            trajectory_write_interval=observation_interval,
            data_container_write_period=write_period)
        mc.run(number_of_trial_steps)
        structure = mc.configuration.structure


def build_gs_db(
        filename: str,
        ce_file: str,
        structure: Atoms,
        species_count_list: List[Dict[str, float]] = None,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        overwrite: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        asynchronus: bool = False,
        processes: int = 1,
        threads: int = 0) -> Atoms:
    """Build a database with ground states obtained with the provided cluster
    expansion based on the primitive structure and species counts; only works
    for systems with at most two species per active sublattice

    Parameters
    ----------
    filename
        path to database where the GS structures shall be stored
    ce_file
        path to cluster expansion file
    structure
        atomic configuration
    species_count_list
        list of dictionaries with count for one of the species on each active
        sublattice. If no count is provided for a sublattice, the concentration
        is allowed to vary
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    overwrite
        overwrite exisiting entries if the formulas match
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or
        dictionary; note that it can be used to overwrite the ``tag``
        associated with each row, which is used to determine the uniqueness
    asynchronus
        Run the processes asynchronously
    processes
        Number of processes to start; multiprocessing is only used if there are
        at least two
    threads
        number of threads to be used when solving the problem, given that a
        positive integer has been provided. If set to 0 the solver default
        configuration is used while -1 corresponds to using the environment
        variable `OMP_NUM_THREADS`, if set, and otherwise all available
        processing cores.

    """

    # generate GS structures
    fun_kwargs = [
        dict(cluster_expansion=ce_file,
             structure=structure,
             threads=threads,
             species_count=species_count)
        for species_count in species_count_list]
    structures = run_function_parallel(function=_find_ground_state_mp,
                                       fun_kwargs=fun_kwargs,
                                       asynchronus=asynchronus,
                                       processes=processes)

    # add GS structures to database
    fun_kwargs = [
        dict(structure=structure,
             tag=structure.get_chemical_formula(),
             vacancies=vacancies,
             overwrite=overwrite,
             keyword_filters=keyword_filters,
             exclude_filtered=exclude_filtered,
             property_calculators=property_calculators)
        for structure in structures]
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def _find_ground_state_mp(kwargs):
    """Parallelisable version of find_ground_state"""
    try:
        ground_state = find_ground_state(**kwargs)
    except Exception:
        warnings.warn(''.join([
            'Failed to find ground state with species count',
            '\n', kwargs["species_count"], f'based on '
            f'{kwargs["structure"].get_chemical_formula()} because of',
            '\n', traceback.format_exc()]))
        return
    return ground_state


def find_ground_state(
        cluster_expansion: Union[str, ClusterExpansion],
        structure: Atoms,
        species_count: Dict[str, float] = None,
        threads: int = 0) -> Atoms:
    """
    Find the ground state based on the provided cluster expansion and
    structure; only works for systems with at most two species per active
    sublattice

    Parameters
    ----------
    cluster_expansion
        ClusterExpansion instance or path to cluster expansion file
    structure
        atomic configuration
    species_count : dict
        dictionary with count for one of the species on each active sublattice.
        If no count is provided for a sublattice, the concentration is allowed
        to vary
    threads
        number of threads to be used when solving the problem, given that a
        positive integer has been provided. If set to 0 the solver default
        configuration is used while -1 corresponds to using the environment
        variable `OMP_NUM_THREADS`, if set, and otherwise all available
        processing cores.
    """
    # read cluster expansion
    if isinstance(cluster_expansion, str):
        cluster_expansion = ClusterExpansion.read(cluster_expansion)
    elif isinstance(cluster_expansion, ClusterExpansion):
        pass
    else:
        raise Exception(f'cluster_expansion must be a string or'
                        f' ClusterExpansion object, not'
                        f' {type(cluster_expansion)}')

    # set up the ground states finder.
    gsf = GroundStateFinder(cluster_expansion, structure)

    # find the ground state
    if threads == -1:
        threads = int(os.getenv('OMP_NUM_THREADS', default=-1))
    ground_state = gsf.get_ground_state(species_count, threads=threads)

    return ground_state


def collect_mc_data(
        dc_files: List[str],
        output_file: str,
        append: bool = False,
        asynchronus: bool = False,
        processes: int = 1) -> None:
    """
    Read data from data container files and collect the mean values and
    standard deviations into a .csv file

    Parameters
    ----------
    dc_files
         list of paths to data container files
    output_file
         path to the output .csv file
    append
         append data to existing file
    asynchronus
        Run the processes asynchronously
    processes
        Number of processes to start; multiprocessing is only used if there are
        at least two
    """

    # create output directory
    dirname = os.path.dirname(output_file)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

    # collect data from MC simulations
    fun_kwargs = [dict(dc_file=dc_file) for dc_file in dc_files]
    data = run_function_parallel(function=_read_dc_file_mp,
                                 fun_kwargs=fun_kwargs,
                                 asynchronus=asynchronus,
                                 processes=processes)

    # compile data sets
    df = pd.DataFrame(list(data))
    df['heat_capacity'] = df['potential_std'] ** 2 / kB / df.temperature ** 2

    if output_file is not None:
        # write .csv file
        if append:
            try:
                dfs = [df, pd.read_csv(output_file, delimiter='\t')]
            except Exception:
                raise Exception(f"Failed to read data from the file"
                                f" {output_file} because of"
                                f" {traceback.format_exc()}")
            df = pd.concat(dfs, ignore_index=True)
        df.to_csv(output_file, sep='\t')

    return df


def _read_dc_file_mp(kwargs):
    """Parallelisable version of read_dc_file"""
    try:
        data_row = read_dc_file(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to read data container file {kwargs["dc_file"]} because'
            f' of', '\n', traceback.format_exc()]))
        return
    return data_row


def read_dc_file(
        dc_file: str) -> Dict:
    """
    Read data from a data container file and collect the mean values and
    standard deviations

    Parameters
    ----------
    dc_file
         path to data container file
    """

    dc = DataContainer.read(dc_file)
    data_row = dc.ensemble_parameters
    data_row['dc_file'] = dc_file
    n_atoms = data_row['n_atoms']

    no_error = ['acceptance_ratio']
    not_scalar = ['occupations']
    equilibration = 2000 * n_atoms
    write_interval = dc._last_state['last_step'] / len(dc.data)
    max_lags = [
        int(n_atoms / write_interval * 10 ** i) for i in range(1, 4)] + [None]

    for max_lag in max_lags:
        increase_max_lag = False
        for observable in dc.observables:
            if observable in not_scalar:
                continue
            stats = dc.analyze_data(observable, start=equilibration,
                                    max_lag=max_lag)
            if stats['correlation_length'] is None:
                increase_max_lag = True
                break
            else:
                data_row[f"{observable}_avg"] = stats['mean']
                if observable not in no_error:
                    data_row[f"{observable}_err"] = stats['error_estimate']
                    data_row[f"{observable}_std"] = stats['std']
        if not increase_max_lag:
            break

    return data_row
