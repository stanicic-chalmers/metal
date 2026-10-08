from collections.abc import Iterable
import copy
from math import ceil, log10
import numbers
import os
import pickle
import tempfile
import tarfile
from typing import Dict, List, Union, Callable

from ase.io import read as ase_read
from ase.io import write as ase_write
from ase import Atoms

from icet import ClusterSpace
from icet.tools.structure_generation import occupy_structure_randomly

from ..space import BaseSpace

from .cluster_expansion import (build_db_random_configurations,
                                build_db_random_concentrations,
                                build_db_enumerate, build_sqs_db,
                                build_db_all_compositions)
from .tools import (get_sublattice_concentrations,
                    _check_filter, expand_chemical_symbols)


class CrystalSpace(BaseSpace):
    """This class is used for handling a configurational space for crystalline
    materialss

    Parameters
    ----------
    structure_list : List[ase.Atoms]
        list of atomic structures
    chemical_symbols_list : Union[List[List[str]],
                                  List[List[List[str]]]]
        list of lists of chemical symbols, each of which corresponds to one of
        the provided structures and where each symbol must map to an element of
        the periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of list of chemical symbols is provided for a given
        structure then the outer list must be the same length as the
        `structures[j]` object and ``chemical_symbols[j][i]`` will correspond
        to the allowed species on lattice site ``i``.
    cutoffs_list : List[List[float]]
        optional list of cutoff radii per order that define the cluster spaces
        associated with each of the provided structures; if unspecified, dummy
        cluster spaces will be created based on a cutoff of `[1.0]`, which will
        give poor results when, e.g., creating SQS structures.

        Cutoffs are specified in units of Angstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. ``cutoffs_list=[[7.0, 4.5]]`` thus implies
        that all pairs distanced 7 A or less will be included,
        as well as all triplets among which the longest distance is no
    symprec_list
        optional list of tolerances that are imposed when using spglib to
        analyze the symmetry of structures based on those provided in the list
        of structures; the default is to use 1e-5
    filters_list : List[List[Dict[str, Union[int, str, float]]]]
        list of list of keyword filters assoctiated with the list of structures
        in the form of dictionaries with the following keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
    db_path : str
        Path to directory where databases are stored; defaults to the current
        working directory
    """

    def __init__(self,
                 structure_list: List[Atoms],
                 chemical_symbols_list: Union[
                     List[List[str]], List[List[List[str]]]],
                 cutoffs_list: List[List[float]] = None,
                 filters_list: List[List[Dict[
                     str, Union[int, str, float]]]] = None,
                 symprec_list: List[float] = None,
                 db_path: str = None) -> None:

        self._set_structure_list(structure_list)
        self._set_cutoffs_list(cutoffs_list)
        self._set_filters_list(filters_list)
        self._set_symprec_list(symprec_list)
        self._input_chemical_symbols_list = copy.deepcopy(
            chemical_symbols_list)
        self.chemical_symbols_list = self._get_chemical_symbols_list()
        self.cluster_space_list = self._get_cluster_spaces()
        if db_path is None:
            self.db_path = os.getcwd()
        else:
            self.db_path = db_path

    def _set_structure_list(self, structure_list: List[Atoms]) -> None:
        """Sets the provided structure list as a property after sanity checks

        Parameters
        ----------
        structure_list : List[ase.Atoms]
            list of atomic structures
        """
        for structure in structure_list:
            if not isinstance(structure, Atoms):
                raise TypeError('Input structures must be ASE Atoms'
                                f' objects, not type {type(structure)}.')
            if not all(structure.pbc):
                raise ValueError('Input structures must have periodic Boundary'
                                 ' conditions.')
        self.structure_list = structure_list.copy()
        self.nstructures = len(self.structure_list)

    def _set_cutoffs_list(self, cutoffs_list: List[List[float]]) -> None:
        """Sets the provided cutoffs list as a property after sanity checks

        Parameters
        ----------
        cutoffs_list
            list of lists of cutoffs
        """
        if cutoffs_list is not None:
            if not len(cutoffs_list) == self.nstructures:
                raise TypeError(f'The number of cutoffs {len(cutoffs_list)}'
                                f' and input structures ({self.nstructures})'
                                f' do not match')
            for cutoffs in cutoffs_list:
                if not (isinstance(cutoffs, Iterable) and
                        all([isinstance(c, numbers.Number) for c in cutoffs])):
                    raise TypeError('The cutoffs list should contain lists of'
                                    ' floating point numbers')
            self.cutoffs_list = cutoffs_list.copy()
        else:
            self.cutoffs_list = None

    def _set_filters_list(self,
                          filters_list: List[List[Dict[
                              str, Union[int, str, float]]]]) -> None:
        """Sets the provided filters list as a property after sanity checks

        Parameters
        ----------
        filters_list
            list of lists of keyword filters
        """
        if filters_list is not None:
            if not len(filters_list) == self.nstructures:
                raise TypeError(f'The number of filters {len(filters_list)}'
                                f' and input structures ({self.nstructures})'
                                f' do not match')
            for filters in filters_list:
                for filter in filters:
                    _check_filter(filter)
            self.filters_list = filters_list.copy()
        else:
            self.filters_list = None

    def _set_symprec_list(self, symprec_list: List[float]) -> None:
        """Sets the provided cutoffs list as a property after sanity checks

        Parameters
        ----------
        symprec_list
            list of tolerances
        """
        if symprec_list is not None:
            if not len(symprec_list) == self.nstructures:
                raise TypeError(f'The number of tolerances {len(symprec_list)}'
                                f' and input structures ({self.nstructures})'
                                f' do not match')
            if not (isinstance(symprec_list, Iterable) and
                    all([isinstance(s, numbers.Number) for s in symprec_list])
                    ):
                raise TypeError('The symprec list should only contain'
                                ' floating point numbers')
            self.symprec_list = symprec_list.copy()
        else:
            self.symprec_list = [1e-5] * self.nstructures

    def _get_chemical_symbols_list(self) -> List[List[List[str]]]:
        """Returns chemical symbols list using input structure list and
        chemical symbols list. Carries out multiple sanity checks. Adapted
        from icet.ClusterSpace (version 2.2)."""

        if len(self._input_chemical_symbols_list) != self.nstructures:
            raise ValueError(
                'The lists of chemical symbols and structures must have the'
                ' same length.\n'
                f' len(chemical_symbols_list) ='
                f' {len(self._input_chemical_symbols_list)},'
                f' len(structure_list)= {self.nstructures}')
        chemical_symbols_list = []
        for j, (input_chemical_symbols, structure) in enumerate(zip(
                self._input_chemical_symbols_list, self.structure_list)):

            try:
                chemical_symbols = expand_chemical_symbols(
                    input_chemical_symbols, structure)
            except Exception as e:
                raise Exception('Error found in chemical_symbols_list[{j}]'
                                ) from e

            chemical_symbols_list.append(chemical_symbols)

        return chemical_symbols_list

    def _get_cluster_spaces(self) -> List[ClusterSpace]:
        """Returns a list of `icet.ClusterSpace` objects, one for each of the
        structures those associated with this space."""

        cluster_spaces = []
        for structure_index in range(self.nstructures):
            structure = self.structure_list[structure_index]
            chemical_symbols = self.chemical_symbols_list[structure_index]
            if self.cutoffs_list is not None:
                cutoffs = self.cutoffs_list[structure_index]
            else:
                cutoffs = [1.0]

            # initiate cluster space
            cluster_spaces.append(ClusterSpace(
                structure=structure,
                cutoffs=cutoffs,
                chemical_symbols=chemical_symbols))

        return cluster_spaces

    def __repr__(self) -> str:
        """Representation."""
        representation = ', '.join([
            type(self).__name__,
            'structure_list:\n  ' +
            '; '.join([s.__repr__() for s in self.structure_list]),
            'chemical_symbols_list:\n  ' +
            '; '.join([c.__repr__() for c in self.chemical_symbols_list]),
            'cutoffs_list:' +
            f' {self.cutoffs_list}' if self.cutoffs_list is None else
            '\n  ' + '; '.join([c.__repr__() for c in self.cutoffs_list]),
            'filters_list:' +
            f' {self.filters_list}' if self.filters_list is None else
            '\n  ' + '; '.join([f.__repr__() for f in self.filters_list]),
            f'db_path: {self.db_path}'])
        return representation

    def __str__(self) -> str:
        """String representation."""
        representation = '\n'.join([
            type(self).__name__,
            '-' * len(type(self).__name__),
            'structure_list:\n    ' +
            '\n    '.join([s.__repr__() for s in self.structure_list]),
            'chemical_symbols_list:\n    ' +
            '\n    '.join([c.__repr__() for c in self.chemical_symbols_list]),
            'cutoffs_list:' +
            (f' {self.cutoffs_list}' if self.cutoffs_list is None else '\n    '
             + '\n    '.join([c.__repr__() for c in self.cutoffs_list])),
            'filters_list:' +
            (f' {self.filters_list}' if self.filters_list is None else '\n    '
             + '\n    '.join([f.__repr__() for f in self.filters_list])),
            'db_path:\n   ' + self.db_path])
        return representation

    def map(self,
            concentrations: Dict[str, float],
            structure_index: int,
            dimensions: Union[int, List[int]] = 1) -> Atoms:
        """
        Map the concentrations by randomly occupying a supercell constructed
        from the specified structure, among those associated with this space

        Parameters
        ----------
        concentrations
            dictionary with chemical symbols as keys and concentrations,
            relative to all sites in the unit cell, as values
        structure_index
            index for the structure in the list of structures
        dimensions
            dimensions of the supercell
        """

        # convert concentrations to sublattice concentration
        target_concentrations = self.get_sublattice_concentrations(
            concentrations=concentrations,
            structure_index=structure_index,
            dimensions=dimensions)

        # occupy structure randomly
        structure = self.structure_list[structure_index].repeat(dimensions)
        cluster_space = self.cluster_space_list[structure_index]
        occupy_structure_randomly(structure=structure,
                                  cluster_space=cluster_space,
                                  target_concentrations=target_concentrations)

        return structure

    def get_sublattice_concentrations(
            self,
            concentrations: Dict[str, float],
            structure_index: int,
            dimensions: Union[int, List[int]] = 1
            ) -> Dict[str, Dict[str, float]]:
        """
        Convert the overall concentrations into a concentration per sublattice
        that best matches the number of available sites

        Parameters
        ----------
        concentrations
            dictionary with chemical symbols as keys and concentrations,
            relative to all sites in the unit cell, as values
        structure_index
            index for the structure in the list of structures
        dimensions
            dimensions of the supercell
        """

        # calculate concentrations on each sublattice
        structure = self.structure_list[structure_index].repeat(dimensions)
        cluster_space = self.cluster_space_list[structure_index]
        sublattice_concentrations = get_sublattice_concentrations(
            concentrations=concentrations,
            structure=structure,
            cluster_space=cluster_space)

        return sublattice_concentrations

    def build_db_all_compositions(
            self,
            structure_index: int,
            dimensions: Union[int, List[int]] = 3,
            max_elements: Union[int, Dict[str, int]] = 2,
            n_structures: int = None,
            replace: bool = False,
            vacancies: List[str] = None,
            db_path: str = None,
            filename: str = None,
            append: bool = False,
            delete_existing: bool = False,
            compress: bool = False,
            apply_filter: bool = False,
            exclude_filtered: bool = False,
            property_calculators: Dict[str, Callable] = None) -> None:
        """Create a database containing all possible compositions, with random
        atomic configurations, based on a specific primitive structure and list
        of chemical symbols for each site

        Parameters
        ----------
        structure_index
            index for the structure in the list of structures
        dimensions
            dimensions of the supercell
        max_elements
            integer or dictionary indicating the maximum number of distinct
            elements per sublattice
        n_structures
            number of structures to select, before filtering; the default is
            all
        replace
            treat the list of chemical symbols as possible replacements of the
            elements on the provided primitive structure
        vacancies
            list of symbols that correspond to vacancies and should therefore
            be removed from the generated structures
        db_path
            path to directory where databases are stored; the default is to use
            the value associated with the instance
        filename
            path to database file; the default is `training.db` in the database
            directory
        append
            append structures if the output database already exists, otherwise
            an exception is raised unless ``delete_existing`` is True
        delete_existing
            delete existing database if ``append`` is not True
        compress
            if the database should be compressed; the default is to decide this
            from the filename
        apply_filter
            if the filters associated with the instance should be applied when
            building the database
        exclude_filtered
            exclude the structures that do not pass through the filter; the
            default is to only add the boolean keyword ``filtered`` to the row
        property_calculators
            dictionary with keywords that should be added to the row and
            functions that are used calculate to calculate the value, which
            AtomsRow or Atoms object as input and return a scalar or
            dictionary; note that it can be used to overwrite the ``tag``
            associated with each row, which is used to determine the uniqueness
        """
        # parameters from instance
        if db_path is None:
            db_path = self.db_path
        os.makedirs(db_path, exist_ok=True)
        primitive = self.structure_list[structure_index]
        chemical_symbols = self.chemical_symbols_list[structure_index]
        symprec = self.symprec_list[structure_index]
        if apply_filter:
            if self.filters_list is None:
                raise Exception('There are no filters associated with this'
                                ' crystal space, use the set_filters function')
            keyword_filters = self.filters_list[structure_index]
        else:
            keyword_filters = None

        if filename is None:
            filename = os.path.join(db_path, 'all_replacements.db' if replace
                                    else 'all_compositions.db')

        # build database with all compositions
        build_db_all_compositions(
            primitive=primitive,
            filename=filename,
            chemical_symbols=chemical_symbols,
            dimensions=dimensions,
            max_elements=max_elements,
            n_structures=n_structures,
            replace=replace,
            symprec=symprec,
            vacancies=vacancies,
            append=append,
            delete_existing=delete_existing,
            compress=compress,
            keyword_filters=keyword_filters,
            exclude_filtered=exclude_filtered,
            property_calculators=property_calculators)

    def build_db_enumerate(
            self,
            structure_index: int,
            concentration_restrictions: Dict[str, float],
            sizes: Union[List[int], range],
            n_structures: int = None,
            vacancies: List[str] = None,
            append: bool = False,
            delete_existing: bool = False,
            db_path: str = None,
            filename: str = None,
            compress: bool = False,
            apply_filter: bool = False,
            exclude_filtered: bool = False,
            property_calculators: Dict[str, Callable] = None) -> None:
        '''Build a database with enumerated configurations based on a specific
        primitive structure and set of concentration restrictions

        Parameters
        ----------
        structure_index
            index for the structure in the list of structures
        concentration_restrictions
            allowed concentration range for one or more element in
            `chemical_symbols`, e.g., ``{'Au': (0, 0.2)}`` will only
            enumerate structures in which the Au content is between 0 and
            20 %; here, concentration is always defined as the number of
            atoms of the specified kind divided by the number of *all* atoms.
        sizes
            range of supercell sizes, i.e. number of primitive cells, (included
            in enumeration)
        n_structures
            number of structures to select, before filtering; the default is
            all
        db_path
            path to directory where databases are stored; the default is to use
            the value associated with the instance
        filename
            path to database file; the default is `training.db` in the database
            directory
        vacancies
            list of symbols that correspond to vacancies and should therefore
            be removed from the generated structures
        append
            append structures if the output database already exists, otherwise
            an exception is raised unless ``delete_existing`` is True
        delete_existing
            delete existing database if ``append`` is not True
        compress
            if the database should be compressed; the default is to decide this
            from the filename
        apply_filter
            if the filters associated with the instance should be applied when
            building the database
        exclude_filtered
            exclude the structures that do not pass through the filter; the
            default is to only add the boolean keyword ``filtered`` to the row
        property_calculators
            dictionary with keywords that should be added to the row and
            functions that are used calculate to calculate the value, which
            must take an AtomsRow or Atoms object as input and return a scalar
            or dictionary
        '''
        # parameters from instance
        if db_path is None:
            db_path = self.db_path
        os.makedirs(db_path, exist_ok=True)
        primitive = self.structure_list[structure_index]
        chemical_symbols = self.chemical_symbols_list[structure_index]
        symprec = self.symprec_list[structure_index]
        if apply_filter:
            if self.filters_list is None:
                raise Exception('There are no filters associated with this'
                                ' crystal space, use the set_filters function')
            keyword_filters = self.filters_list[structure_index]
        else:
            keyword_filters = None

        if filename is None:
            filename = os.path.join(db_path, 'enumerated_structures.db')

        # build enumerated database
        build_db_enumerate(
            primitive=primitive,
            filename=filename,
            chemical_symbols=chemical_symbols,
            concentration_restrictions=concentration_restrictions,
            sizes=sizes,
            n_structures=n_structures,
            symprec=symprec,
            vacancies=vacancies,
            append=append,
            delete_existing=delete_existing,
            compress=compress,
            keyword_filters=keyword_filters,
            exclude_filtered=exclude_filtered,
            property_calculators=property_calculators)

    def build_db_random_configurations(
            self,
            structure_index: int,
            concentrations_list: List[Dict[str, float]],
            dimensions: Union[int, List[int]] = 3,
            n_structures: int = 1,
            vacancies: List[str] = None,
            append: bool = False,
            delete_existing: bool = False,
            db_path: str = None,
            filename: str = None,
            compress: bool = False,
            apply_filter: bool = False,
            exclude_filtered: bool = False,
            property_calculators: Dict[str, Callable] = None) -> None:
        '''Build a database with randomly generated configurations based on a
        specific primitive structure and target composition

        Parameters
        ----------
        structure_index
            index for the structure in the list of structures
        concentrations_list
            list of dictionaries with chemical symbols as keys and
            concentrations, relative to all sites in the unit cell, as values
        dimensions
            dimensions of the supercell
        n_structures
            number of structures to generate, for each concentration, before
            filtering
        db_path
            path to directory where databases are stored; the default is to use
            the value associated with the instance
        vacancies
            list of symbols that correspond to vacancies and should therefore
            be removed from the generated structures
        append
            append structures if the output database already exists, otherwise
            an exception is raised unless ``delete_existing`` is True
        delete_existing
            delete existing database if ``append`` is not True
        compress
            if the database should be compressed; the default is to decide this
            from the filename
        filename
            path to database file; the default is `training.db` in the database
            directory
        apply_filter
            if the filters associated with the instance should be applied when
            building the database
        exclude_filtered
            exclude the structures that do not pass through the filter; the
            default is to only add the boolean keyword ``filtered`` to the row
        property_calculators
            dictionary with keywords that should be added to the row and
            functions that are used calculate to calculate the value, which
            must take an AtomsRow or Atoms object as input and return a scalar
            or dictionary
        '''

        # parameters from instance
        if db_path is None:
            db_path = self.db_path
        os.makedirs(db_path, exist_ok=True)
        sublattice_concentrations_list = [
            self.get_sublattice_concentrations(
                concentrations=concentrations,
                structure_index=structure_index,
                dimensions=dimensions)
            for concentrations in concentrations_list]
        primitive = self.structure_list[structure_index]
        chemical_symbols = self.chemical_symbols_list[structure_index]
        symprec = self.symprec_list[structure_index]
        if apply_filter:
            if self.filters_list is None:
                raise Exception('There are no filters associated with this'
                                ' crystal space, use the set_filters function')
            keyword_filters = self.filters_list[structure_index]
        else:
            keyword_filters = None

        if filename is None:
            filename = os.path.join(db_path, 'random_configurations.db')

        # build random database
        build_db_random_configurations(
            primitive=primitive,
            filename=filename,
            chemical_symbols=chemical_symbols,
            sublattice_concentrations_list=sublattice_concentrations_list,
            dimensions=dimensions,
            n_structures=n_structures,
            symprec=symprec,
            vacancies=vacancies,
            append=append,
            delete_existing=delete_existing,
            compress=compress,
            keyword_filters=keyword_filters,
            exclude_filtered=exclude_filtered,
            property_calculators=property_calculators)

    def build_db_random_concentrations(
            self,
            structure_index: int,
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
            db_path: str = None,
            filename: str = None,
            compress: bool = False,
            apply_filter: bool = False,
            exclude_filtered: bool = False,
            property_calculators: Dict[str, Callable] = None) -> None:
        '''Build a database with randomly generated configurations and
        concentrations based on a specific primitive structure

        Parameters
        ----------
        structure_index
            index for the structure in the list of structures
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
            number of structures to generate
        db_path
            path to directory where databases are stored; the default is to use
            the value associated with the instance
        filters
            function that is used for filtering structures based on, e.g., the
            concentrations
        vacancies
            list of symbols that correspond to vacancies and should therefore
            be removed from the generated structures
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
        filename
            path to database file; the default is `training.db` in the database
            directory
        apply_filter
            if the filters associated with the instance should be applied when
            building the database
        exclude_filtered
            exclude the structures that do not pass through the filter; the
            default is to only add the boolean keyword ``filtered`` to the row
        property_calculators
            dictionary with keywords that should be added to the row and
            functions that are used calculate to calculate the value, which
            AtomsRow or Atoms object as input and return a scalar or
            dictionary; note that it can be used to overwrite the ``tag``
            associated with each row, which is used to determine the uniqueness
        '''

        # parameters from instance
        if db_path is None:
            db_path = self.db_path
        os.makedirs(db_path, exist_ok=True)
        if apply_filter:
            if self.filters_list is None:
                raise Exception('There are no filters associated with this'
                                ' crystal space, use the set_filters function')
            keyword_filters = self.filters_list[structure_index]
        else:
            keyword_filters = None

        if filename is None:
            filename = os.path.join(db_path, 'random_concentrations.db')

        # build random database
        build_db_random_concentrations(
            primitive=self.structure_list[structure_index],
            filename=filename,
            chemical_symbols=self.chemical_symbols_list[structure_index],
            dimensions=dimensions,
            fixed_symbols=fixed_symbols,
            min_total=min_total,
            max_total=max_total,
            min_sublattice=min_sublattice,
            max_sublattice=max_sublattice,
            deviation_limit=deviation_limit,
            n_structures=n_structures,
            vacancies=vacancies,
            append=append,
            overwrite=overwrite,
            delete_existing=delete_existing,
            compress=compress,
            keyword_filters=keyword_filters,
            exclude_filtered=exclude_filtered,
            property_calculators=property_calculators)

    def build_sqs_db(
            self,
            structure_index: int,
            concentration_list: Dict[str, float],
            dimensions: Union[int, List[int]] = None,
            max_size: int = 4,
            n_structures: int = 100,
            vacancies: List[str] = None,
            enumeration: bool = False,
            append: bool = False,
            delete_existing: bool = False,
            overwrite: bool = False,
            db_path: str = None,
            filename: str = None,
            compress: bool = False,
            apply_filter: bool = False,
            exclude_filtered: bool = False,
            property_calculators: Dict[str, Callable] = None,
            asynchronus: bool = False,
            processes: int = 1) -> None:
        '''Build a database with special quasirandom structures based on a
        specific primitive structure and list of concentrations

        Parameters
        ----------
        structure_index
            index for the structure in the list of structures
        concentration_list
            list of dictionaries with chemical symbols as keys and
            concentrations, relative to all sites in the unit cell, as values
        dimensions
            dimensions of the supercell; is superseded by ``max_size``
        max_size
            search among all supercells obtained by repeating the primitive
            cell, in a single direction, between one and ``max_size`` number of
            times; supersedes ``dimensions``
        n_structures
            number of structures to generate
        vacancies
            list of symbols that correspond to vacancies and should therefore
            be removed from the generated structures
        enumeration
            find SQS via enumeration; ignored if ``dimensions`` is specified
        append
            append structures if the output database already exists, otherwise
            an exception is raised unless ``delete_existing`` is True
        delete_existing
            delete existing database if ``append`` is not True
        overwrite
            overwrite exisiting entries if the formulas match
        db_path
            path to directory where databases are stored; the default is to use
            the value associated with the instance
        compress
            if the database should be compressed; the default is to decide this
            from the filename
        filename
            path to database file; the default is `training.db` in the database
            directory
        apply_filter
            filter structures
        exclude_filtered
            exclude the structures that do not pass through the filter; the
            default is to only add the boolean keyword ``filtered`` to the row
        property_calculators
            dictionary with keywords that should be added to the row and
            functions that are used calculate to calculate the value, which
            AtomsRow or Atoms object as input and return a scalar or
            dictionary; note that it can be used to overwrite the ``tag``
            associated with each row, which is used to determine the uniqueness
        asynchronous
            run the processes asynchronously, ignored if sequential
        processes
            number of processes to start; if the number is less than two these
            are run sequentially
        '''

        # parameters from instance
        if db_path is None:
            db_path = self.db_path
        os.makedirs(db_path, exist_ok=True)
        sublattice_concentrations_list = [
            self.get_sublattice_concentrations(
                concentrations=concentrations,
                structure_index=structure_index,
                dimensions=((max_size, 1, 1) if dimensions is None else
                            dimensions))
            for concentrations in concentration_list]
        primitive = self.structure_list[structure_index]
        chemical_symbols = self.chemical_symbols_list[structure_index]
        cutoffs = self.cutoffs_list[structure_index]
        symprec = self.symprec_list[structure_index]
        if apply_filter:
            if self.filters_list is None:
                raise Exception('There are no filters associated with this'
                                ' crystal space, use the set_filters function')
            keyword_filters = self.filters_list[structure_index]
        else:
            keyword_filters = None

        if filename is None:
            filename = os.path.join(db_path, 'sqs_structures.db')

        # build SQS database
        build_sqs_db(
            primitive=primitive,
            filename=filename,
            chemical_symbols=chemical_symbols,
            cutoffs=cutoffs,
            sublattice_concentrations_list=sublattice_concentrations_list,
            dimensions=dimensions,
            max_size=max_size,
            n_structures=n_structures,
            enumeration=enumeration,
            symprec=symprec,
            vacancies=vacancies,
            append=append,
            delete_existing=delete_existing,
            overwrite=overwrite,
            compress=compress,
            keyword_filters=keyword_filters,
            exclude_filtered=exclude_filtered,
            property_calculators=property_calculators,
            asynchronus=asynchronus,
            processes=processes)

    def write(self, filename: str) -> None:
        """
        Saves crystal space to a file.

        Parameters
        ---------
        filename
            name of file to which to write
        """

        with tarfile.open(name=filename, mode='w') as tar_file:

            # write items
            items = dict(
                db_path=self.db_path,
                cutoffs_list=self.cutoffs_list,
                filters_list=self.filters_list,
                chemical_symbols_list=self._input_chemical_symbols_list)
            temp_file = tempfile.TemporaryFile()
            pickle.dump(items, temp_file)
            temp_file.seek(0)
            tar_info = tar_file.gettarinfo(arcname='items', fileobj=temp_file)
            tar_file.addfile(tar_info, temp_file)
            temp_file.close()

            # write structures
            width = ceil(log10(len(self.structure_list)))
            for s, structure in enumerate(self.structure_list):
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.close()
                ase_write(temp_file.name, structure, format='json')
                with open(temp_file.name, 'rb') as tt:
                    tar_info = tar_file.gettarinfo(
                            arcname=f'structure{s:0>{width}}', fileobj=tt)
                    tar_file.addfile(tar_info, tt)
                os.remove(temp_file.name)

    @staticmethod
    def read(filename: str):
        """
        Reads crystal space from filename.

        Parameters
        ---------
        filename
            name of file from which to read crystal space
        """
        if isinstance(filename, str):
            tar_file = tarfile.open(mode='r', name=filename)
        else:
            tar_file = tarfile.open(mode='r', fileobj=filename)

        # read items
        items = pickle.load(tar_file.extractfile('items'))

        # read structures and filters
        structure_list = []
        for name in sorted(tar_file.getnames()):
            if 'structure' not in name:
                continue
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(tar_file.extractfile(name).read())
            temp_file.close()
            structure_list.append(ase_read(temp_file.name, format='json'))
            os.remove(temp_file.name)

        tar_file.close()

        cs = CrystalSpace(db_path=items['db_path'],
                          structure_list=structure_list,
                          chemical_symbols_list=items['chemical_symbols_list'],
                          cutoffs_list=items['cutoffs_list'],
                          filters_list=items['filters_list'])

        return cs
