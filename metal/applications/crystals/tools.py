from collections.abc import Iterable, Mapping
from contextlib import nullcontext
import copy
from flatten_dict import flatten
from itertools import product
import matplotlib.pyplot as plt
import numbers
import numpy as np
import operator
import os
import pandas as pd
import traceback
from typing import Callable, Dict, List, Tuple, Union
import warnings

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db.core import connect, Database
from ase.db.row import AtomsRow, atoms2dict
from ase.io import read, write
import ase.units as u

from icet import ClusterSpace
from icet.tools.geometry import ase_atoms_to_spglib_cell

from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from scipy.interpolate import InterpolatedUnivariateSpline

from spglib import get_spacegroup_type, get_symmetry_dataset

from ...utils import apply_function_db, run_function_parallel


def rational_approx(x, maxden=None) -> Tuple[int, int]:
    """
    Finds a rational approximation to a given real number. Based on
    algorithm by David Eppstein / UC Irvine / 8 Aug 1993

    Parameters:
        x: floating point instance to approximate
        maxden: maximum denominator allowed
    """
    if abs(x - round(x)) < 1e-9:
        return (int(round(x)), 1)
    startx = x
    m = np.eye(2, dtype=int)
    if maxden is None:
        maxden = 100

    # First check for factors of 10
    while x < 0.1:
        x *= 10
    x10factor = int(round(x/startx))

    # loop finding terms until demom gets too big
    ai = int(x)
    while (m[1, 0] * ai + m[1, 1] <= maxden):
        t = m[0, 0] * ai + m[0, 1]
        m[0, 1] = m[0, 0]
        m[0, 0] = t
        t = m[1, 0] * ai + m[1, 1]
        m[1, 1] = m[1, 0]
        m[1, 0] = t
        if m[1, 0] > 0.0 and x-float(ai) == 0.0:
            break
        x = 1.0/(x - float(ai))
        ai = int(x)
    # now remaining x is between 0 and 1/ai
    # approx as either 0 or 1/m where m is max that will fit in maxden
    # first try 0
    num0 = m[0, 0]
    den0 = m[1, 0] * x10factor
    error0 = startx - float(num0) / float(den0)
    if m[0, 0] == 0:
        error1 = np.inf
    else:
        # other possibility
        ai = (maxden - m[1, 1]) // m[0, 0]
        m[0, 0] = m[0, 0] * ai + m[0, 1]
        m[1, 0] = m[1, 0] * ai + m[1, 1]
        num1 = m[0, 0]
        den1 = m[1, 0] * x10factor
        error1 = startx - float(num1) / float(den1)

    if abs(error1) >= abs(error0):
        if abs(error0/startx) > 1e-9:
            if maxden < 20000:
                return rational_approx(startx, maxden*10)
            warnings.warn(f'Large error {startx} =/= {num0}/{den0}')
        return (num0, den0)
    else:
        if abs(error1)/startx > 1e-9:
            if maxden < 20000:
                return rational_approx(startx, maxden*10)
            warnings.warn(f'Large error {startx} =/= {num1}/{den1}')
        return (num1, den1)


def get_mu_oxygen_ref_raw_data(
        only_dh: bool = False) -> (List[float], List[float]):
    """ Get raw JANAF data for O2 (https://janaf.nist.gov/tables/O-029.html)

    Parameters
    ----------
    only_dh
        do not include the entropy contribution
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir_path, "data/O2_ref.txt")
    data = np.loadtxt(path)
    temperatures = data[:, 0]      # [K]
    S0 = 1.e-3 * data[:, 2]  # [kJ/mol/K]
    dH = data[:, 4]        # [kJ/mol]

    # Eq. 22 in Reuter Scheffler PRB (2001) 65 035406
    mu_ref = 0.5 * (dH - dH[0] - (0.0 if only_dh else temperatures * S0)
                    ) * u.kJ / u.mol

    return temperatures, mu_ref


def get_mu_oxygen_ref(
        temperatures: List[float],
        add_u_dft: bool = True,
        retrieve_dft: bool = False,
        only_dh: bool = False,
        show_plot: bool = False,
        print_data: bool = False) -> List[float]:
    """ Get the oxygen chemical potential at the specified temperature by
    interpolating, or extrapolating, the JANAF data for O2
    (https://janaf.nist.gov/tables/O-029.html)

    Parameters
    ----------
    temperatures
        temperatures in kelvin
    add_u_dft
        add the 0 K internal energy retrieved from the Materials Project
    retrieve_dft
        retrieve the most recent 0 K internal energy from the Materials
        Project instead of using data for the lowest energy structure in
        February 2023
    only_dh
        do not include the entropy contribution
    show_plot
        show a plot of the spline fit
    print_data
        print the chemical potential calculated from the raw NIST-JANAF data
    """

    # get raw data
    temperatures_raw, mu_raw = get_mu_oxygen_ref_raw_data(only_dh)

    # add internal energy from Materials Project
    if add_u_dft:
        if retrieve_dft:
            atoms = _retrieve_oxygen_molecule()
        else:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(dir_path, "data/O2.json")
            atoms = read(path)
        mu_raw += atoms.get_potential_energy() / len(atoms)

    if print_data:
        for x, y in zip(temperatures, mu_raw):
            print('%3i K    %6.3f eV' % (x, y))

    # extrapolate data
    spl = InterpolatedUnivariateSpline(temperatures_raw, mu_raw, k=3)

    if show_plot:
        fig = plt.figure(facecolor=(1, 1, 1), figsize=(12, 7))
        ax = fig.add_subplot(111)

        mu_fit = spl(temperatures)

        T1 = 1173  # 950C in Air reactor
        T2 = 1273  # 1050C in Fuel reactor
        y1 = spl(T1)
        y2 = spl(T2)

        ax.plot(temperatures, mu_raw, '.', linewidth=3.0, markersize=15,
                color='#0571b0', label='tabulated data')
        ax.plot(temperatures, mu_fit, '--', linewidth=2.0, markersize=15,
                color='#92c5de', label='fitted')
        ax.plot(T1, y1, 's', markersize=10, color='#ef8a62',
                label=r'T=900$^\circ C$')
        ax.plot(T2, y2, 's', markersize=10, color='#b2182b',
                label=r'T=950$^\circ C$')

        ax.set_ylabel(r'$' + (r'{\Delta}H' if only_dh else r'\mu')
                      + r'_{O}(temperatures, p_0)$ [eV]', fontsize=26)
        ax.set_xlabel('T [K]', fontsize=26)
        ax.tick_params(labelsize=16, direction='in')
        ax.legend(loc='best', numpoints=1, fontsize=18)
        ax.set_xlim([0, 2000])
        ax.set_ylim([-2.5, 0.1])

        fig.savefig(('deltah' if only_dh else 'mu') + '0_plot.pdf',
                    format='pdf', transparent=True)

    return spl(temperatures)


def _retrieve_oxygen_molecule(filename: str = None) -> Atoms:
    """Retrieve the lowest energy structure for a oxygen molecule from the
    Materials Project database and include information regaridng the energy
    per atom

    Parameters
    ----------
    filename
        path to output file; by default no file is written
    """

    # retrieve O2 structures
    formula = 'O2'
    fields = ['structure', 'formation_energy_per_atom', 'energy_per_atom']
    mprester = MPRester()
    try:
        summary = mprester.get_summary(dict(formula=formula), fields=fields)
    except Exception:
        try:
            summary = mprester.summary_search(formula=formula, fields=fields)
        except Exception:
            summary = mprester.summary.search(formula=formula, fields=fields)
    data = [{'structure': doc['structure'] if isinstance(doc, dict) else
             doc.structure,
             'formation_energy_per_atom': doc['formation_energy_per_atom']
             if isinstance(doc, dict) else doc.formation_energy_per_atom,
             'energy_per_atom': doc['energy_per_atom']
             if isinstance(doc, dict) else doc.energy_per_atom}
            for doc in summary]

    # find most stable structure
    df = pd.DataFrame.from_records(data)
    row = df.loc[df.formation_energy_per_atom ==
                 df.formation_energy_per_atom.min()
                 ].iloc[0]

    # create Atoms object and add energy
    atoms = AseAtomsAdaptor.get_atoms(row.structure)
    atoms.calc = SinglePointCalculator(
        atoms, energy=row.energy_per_atom * len(atoms))

    # write Atoms to file
    if filename is not None:
        write(filename, atoms)

    return atoms


def get_mu_oxygen(
        temperatures: List[float],
        oxygen_pressures: List[float],
        logarithmic: bool = False,
        add_u_dft: bool = True,
        retrieve_dft: bool = False) -> List[float]:
    """
    Interpolates values on the oxygen chemical potential from JANAF table O-029
    and uses a formula from Eq. 22 in Reuter Scheffler PRB (2001) 65 035406.

    Parameters
    ----------
    temperatures
        temperatures in kelvin
    oxygen_pressures
        normalised partial pressure of oxygen, relative to 1 bar
    logarithmic
        if oxygen pressures correspond to log values
    add_u_dft
        add the 0 K internal energy retrieved from the Materials Project
    retrieve_dft
        retrieve the most recent 0 K internal energy from the Materials
        Project instead of using data for the lowest energy structure in
        February 2023
    """
    temperatures, oxygen_pressures = (np.asarray(temperatures),
                                      np.asarray(oxygen_pressures))
    mu_oxygen_ref = get_mu_oxygen_ref(temperatures=temperatures,
                                      add_u_dft=add_u_dft,
                                      retrieve_dft=retrieve_dft)
    if not logarithmic:
        log_oxygen_pressures = np.log(oxygen_pressures)
    else:
        log_oxygen_pressures = oxygen_pressures
    mu_oxygen = (mu_oxygen_ref + 0.5 * u.kB * temperatures *
                 log_oxygen_pressures)
    return mu_oxygen


def get_oxygen_pressures(
        temperatures: List[float],
        mu_oxygen: List[float],
        logarithmic: bool = False,
        add_u_dft: bool = True,
        retrieve_dft: bool = False) -> List[float]:
    """
    Get normalised oxygen partial pressure, relative to 1 bar, by interpolating
    oxygen chemical potentia values from JANAF table O-029 and uses a formula
    from Eq. 22 in Reuter Scheffler PRB (2001) 65 035406.

    Parameters
    ----------
    temperatures
        temperatures in kelvin
    mu_oxygen
        oxygen chemical potential in eV per oxygen atom
    logarithmic
        if log values should be returned
    add_u_dft
        add the 0 K internal energy retrieved from the Materials Project to
        mu_oxygen values from JANAF
    retrieve_dft
        retrieve the most recent 0 K internal energy from the Materials
        Project instead of using data for the lowest energy structure in
        February 2023
    """
    temperatures, mu_oxygen = np.asarray(temperatures), np.asarray(mu_oxygen)
    mu_oxygen_ref = get_mu_oxygen_ref(temperatures=temperatures,
                                      add_u_dft=add_u_dft,
                                      retrieve_dft=retrieve_dft)
    log_oxygen_pressures = (mu_oxygen - mu_oxygen_ref
                            ) / 0.5 / u.kB / temperatures
    if logarithmic:
        return log_oxygen_pressures
    else:
        return np.exp(log_oxygen_pressures)

def get_mu_oxygen_fr(temperatures, ratio_co2_co, add_u_dft=True, retrieve_dft=False):
    """mu_O (eV per O atom) in the FR from the CO2/CO ratio.
    CO + 1/2 O2 <-> CO2:  0.5*ln(pO2) = ln(pCO2/pCO) - [ln Kf(CO2) - ln Kf(CO)]"""
    temperatures = np.atleast_1d(temperatures)
    dir_path = os.path.dirname(os.path.abspath(__file__))

    def lnKf(name):
        d = np.loadtxt(os.path.join(dir_path, f"data/{name}_ref.txt"))
        T, lnK = d[:, 0], d[:, 7] * np.log(10)
        ok = np.isfinite(lnK)
        return InterpolatedUnivariateSpline(T[ok], lnK[ok], k=3)(temperatures)

    lnK_rxn = lnKf("CO2") - lnKf("CO")
    log_pO2 = 2.0 * (np.log(ratio_co2_co) - lnK_rxn)
    return get_mu_oxygen(temperatures, log_pO2, logarithmic=True,
                         add_u_dft=add_u_dft, retrieve_dft=retrieve_dft)

def get_mu_oxygen_sr(temperatures, ratio, add_u_dft=True, retrieve_dft=False):
    """mu_O (eV per O atom) in the steam reactor from the H2O/H2 ratio.
    H2 + 1/2 O2 <-> H2O:  0.5*ln(pO2) = ln(pH2O/pH2) - ln Kf(H2O)"""
    temperatures = np.atleast_1d(temperatures)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data = np.loadtxt(os.path.join(dir_path, "data/H2O_ref.txt"))
    T, lnK = data[:, 0], data[:, 7] * np.log(10)
    valid = np.isfinite(lnK)
    lnK_at_T = InterpolatedUnivariateSpline(T[valid], lnK[valid], k=3)(temperatures)

    log_pO2 = 2.0 * (np.log(ratio) - lnK_at_T)   # ln(pO2 / p°), with p° = 1 bar
    return get_mu_oxygen(temperatures, log_pO2, logarithmic=True,
                         add_u_dft=add_u_dft, retrieve_dft=retrieve_dft)

def get_list_of_matching_entries(
        db: Database,
        conditions: str) -> List[int]:
    """
    Get a list of all row ids in the database that matches the conditions

    Parameters
    ----------
        db
            ASE database object
        conditions
            conditions for selecting rows.
    """
    lst = []
    try:
        for entry in db.select(**conditions):
            lst.append(entry.id)
    except Exception:
        pass
    return lst


def get_gcd_num_den(prefactor, external_prefactor):
    """
    Calculate the greatest common denominator given the prefactor and
    external prefactor

    Parameters
    ----------
        prefactor (float): Prefactor
        external_prefactor (list or int): External prefactor in the form of
            either a list with the numerator and denominator or an integer

    Returns:
        num (int): Numerator
        den (int): Denominator
    """

    if type(external_prefactor) == int:
        extnum = external_prefactor
        extden = 1
    else:
        extnum = external_prefactor[0]
        extden = external_prefactor[1]
    gcd = np.gcd(prefactor * extnum, extden)
    num = extnum * prefactor // gcd
    den = extden // gcd
    return num, den


def combine_prefactors(prefactor, external_prefactor=None, latex=False):
    """
    Combine the provided prefactors and return a string with numerator and
    denominator

    Parameters
    ----------
        prefactor (float): Prefactor
        external_prefactor (float or int): External prefactor
        latex (bool): Return a LaTeX string

    Returns:
        num_den (str): String with the numerator and denominator
    """

    if external_prefactor is None:
        if prefactor == 1:
            return ''
        return f'{prefactor}'
    num, den = get_gcd_num_den(prefactor, external_prefactor)
    if num == 1 and den == 1:
        return ''
    elif den == 1:
        return f'{num} '
    elif latex:
        return r'\frac{' + f'{num}' + r'}{' + f'{den}' + r'} '
    return f'({num}/{den})'


def add_structure_db(
        filename: str,
        structure: Atoms = None,
        row_id: int = None,
        overwrite: bool = False,
        data: Dict = None,
        tag: str = None,
        vacancies: List[str] = None,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        use_lock_file: bool = True,
        **kwargs) -> None:
    """
    Adds a structure, associated data, and additional key-value pairs to an ASE
    database

    Parameters
    ----------
    filename
        path to database where the structures should be added
    structure
        atomic configuration
    row_id
        ID of row to update
    data
        additional data
    overwrite
        overwrite existing entries if the tags match; ignored of ``row_id`` has
        been specified
    tag
        tag associated with the entry; used to check for uniqueness if it
        is not left unspecified
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the structure
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
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    with connect(filename, use_lock_file=use_lock_file) as db:
        _add_structure_db(
            database=db,
            structure=structure,
            row_id=row_id,
            overwrite=overwrite,
            data=data,
            tag=tag,
            vacancies=vacancies,
            keyword_filters=keyword_filters,
            exclude_filtered=exclude_filtered,
            property_calculators=property_calculators,
            **kwargs)


def _add_structure_db(
        database: Database,
        structure: Atoms = None,
        row_id: int = None,
        overwrite: bool = False,
        data: Dict = None,
        tag: str = None,
        vacancies: List[str] = None,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        **kwargs) -> None:
    """
    Adds a structure, associated data, and additional key-value pairs to an ASE
    database

    Parameters
    ----------
    database
        ASE Database object to which the structure should be added
    structure
        atomic configuration; must be provided unless a `row_id` has been
        specified
    row_id
        ID of row to update
    data
        additional data
    overwrite
        overwrite existing entries if the tags match; ignored of ``row_id`` has
        been specified
    tag
        tag associated with the entry; used to check for uniqueness if it
        is not left unspecified
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the structure
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

    # Check if entry exists
    if row_id is None:
        if tag is not None:
            matches = get_list_of_matching_entries(database, {'tag': tag})
            if len(matches) > 0:
                if overwrite:
                    database.delete(list(set(matches)))
                else:
                    warnings.warn(f'Entry with tag "{tag}" already'
                                  f' exists')
                    return
        key_value_pairs = dict(**kwargs)
        if structure is None:
            raise ValueError('A structure or row ID needs to be provided')
    else:
        try:
            row = database.get(row_id)
            if data is not None:
                data = {**row.data, **data}
            else:
                data = row.data
            key_value_pairs = {**row.key_value_pairs, **kwargs}
            if structure is None:
                structure = row.toatoms()
        except Exception:
            warnings.warn(f'There is no entry with id {row_id}')
            return

    # remove vacancies
    if vacancies:
        del structure[[atom.index for atom in structure if atom.symbol in
                       vacancies]]

    # create AtomsRow object
    atoms_dict = atoms2dict(structure)
    atoms_dict['data'] = data
    atoms_dict['key_value_pairs'] = key_value_pairs
    if tag is not None:
        atoms_dict['key_value_pairs']['tag'] = tag

    # calculate properties
    if property_calculators is not None:
        row = AtomsRow(atoms_dict)
        for key, prop_calc in property_calculators.items():
            prop = prop_calc(row)
            if isinstance(prop, dict):
                atoms_dict['key_value_pairs'].update(prop)
            else:
                atoms_dict['key_value_pairs'][key] = prop

    # apply filters
    if keyword_filters is not None:
        row = AtomsRow(atoms_dict)
        filtered = apply_multiple_filters(row, keyword_filters)
        if filtered and exclude_filtered:
            warnings.warn('The structure has been filtered')
            if row_id is not None:
                del database[row_id]
            return
        atoms_dict['key_value_pairs']['filtered'] = filtered

    # add new entry
    row = AtomsRow(atoms_dict)
    try:
        data = dict(row.data)
    except Exception:
        data = row._data
    if row_id is None:
        database.write(row.toatoms(), data=data, **row.key_value_pairs)
    else:
        database.update(row_id, atoms=row.toatoms(), data=data,
                        **row.key_value_pairs)


def _add_structure_db_mp(kwargs):
    """Parallelisable version of add_structure_db"""
    try:
        add_structure_db(**kwargs)
    except Exception:
        warnings.warn(''.join([
            'Failed to', f' update row {kwargs["row_id"]}' if "row_id" in
            kwargs and kwargs["row_id"] is not None else f' add structure with'
            f' formula "{kwargs["structure"].get_chemical_formula()}"',
            f' to the database {kwargs["filename"]} because of', '\n',
            traceback.format_exc()]))


def get_rows_db(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        selection_filter: Callable = None) -> List[int]:
    """Get a list with the selected rows in the specified database

    Parameters
    ----------
    filename
        path to database
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    selection_filter
        a function that takes as input a row and returns True or False.
    """

    fun_kwargs = dict(selection=selection, selection_filter=selection_filter)
    return apply_function_db(filename=filename,
                             function=_get_rows_db_mp,
                             fun_kwargs=fun_kwargs)


def _get_rows_db(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        selection_filter: Callable = None,
        use_lock_file: bool = True) -> List[int]:
    """Get a list with the selected rows in the specified database

    Parameters
    ----------
    filename
        path to database
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    selection_filter
        a function that takes as input a row and returns True or False.
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    with connect(filename, use_lock_file=use_lock_file) as db:
        if isinstance(selection, list) and all([isinstance(s, int) for s in
                                                selection]):
            rows = [db.get(row_id) for row_id in selection]
            if selection_filter is not None:
                rows = [row for row in rows if selection_filter(row)]
        else:
            rows = list(db.select(selection, filter=selection_filter))
    return rows


def _get_rows_db_mp(kwargs):
    """Parallelisable version of _get_rows_db"""
    try:
        rows = _get_rows_db(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to retrieve rows from the database {kwargs["filename"]}'
            f' because of', '\n', traceback.format_exc()]))
        return
    return rows


def get_structures_db(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        selection_filter: Callable = None,
        include_data: bool = False,
        include_key_value_pairs: bool = False) -> List[Atoms]:
    """Get a list of structures for the selected rows IDs in the specified
    database

    Parameters
    ----------
    filename
        path to database
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    selection_filter
        a function that takes as input a row and returns True or False.
    include_data
        add ``AtomsRow.data`` to ``Atoms.info``
    include_key_value_pairs
        add ``AtomsRow.key_value_pairs`` to ``Atoms.info``
    """

    fun_kwargs = dict(selection=selection,
                      selection_filter=selection_filter,
                      include_data=include_data,
                      include_key_value_pairs=include_key_value_pairs)
    return apply_function_db(filename=filename,
                             function=_get_structures_db_mp,
                             fun_kwargs=fun_kwargs)


def _get_structures_db(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        selection_filter: Callable = None,
        include_data: bool = False,
        include_key_value_pairs: bool = False) -> List[Atoms]:
    """Get a list of structures for the selected rows IDs in the specified
    database

    Parameters
    ----------
    filename
        path to database
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    selection_filter
        a function that takes as input a row and returns True or False.
    include_data
        add ``AtomsRow.data`` to ``Atoms.info``
    include_key_value_pairs
        add ``AtomsRow.key_value_pairs`` to ``Atoms.info``
    """

    for row in _get_rows_db(filename=filename, selection=selection,
                            selection_filter=selection_filter):
        atoms = row.toatoms()
        info = {}
        if include_data:
            info['data'] = row.data
        if include_key_value_pairs:
            info.update(row.key_value_pairs)
        if len(info) > 0:
            atoms.info = info
        yield atoms


def _get_structures_db_mp(kwargs):
    """Parallelisable version of _get_structures_db"""
    try:
        structures = _get_structures_db(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to retrieve structures from the database'
            f' {kwargs["filename"]} because of', '\n',
            traceback.format_exc()]))
        return
    return structures


def get_row_ids(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        selection_filter: Callable = None) -> List[int]:
    """
    Get a list of the IDs associated with the selected rows in the specified
    database

    Parameters
    ----------
    filename
        path to database where the enumerated structures shall be stored
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    selection_filter
        a function that takes as input a row and returns True or False.
    """

    # check if selection is a list of integer ids
    if selection_filter is not None and isinstance(selection, list) and all([
            isinstance(s, int) for s in selection]):
        return selection

    return apply_function_db(filename=filename,
                             function=_get_row_ids_mp,
                             fun_kwargs={'selection': selection,
                                         'selection_filter': selection_filter})


def _get_row_ids(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        selection_filter: Callable = None,
        use_lock_file: bool = True) -> List[int]:
    """
    Get a list of the IDs associated with the selected rows in the specified
    database

    Parameters
    ----------
    filename
        path to database where the enumerated structures shall be stored
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    selection_filter
        a function that takes as input a row and returns True or False.
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    # check if selection is a list of integer ids
    if isinstance(selection, list) and all([isinstance(s, int) for s in
                                            selection]):
        if selection_filter is None:
            row_ids = selection
        else:
            with connect(filename, use_lock_file=use_lock_file) as db:
                [row_id for row_id in row_ids if
                 selection_filter(db.get(row_id))]
    else:
        with connect(filename, use_lock_file=use_lock_file) as db:
            row_ids = [row.id for row in
                       db.select(selection, filter=selection_filter)]
    return row_ids


def _get_row_ids_mp(kwargs):
    """Parallelisable version of _get_row_ids"""
    try:
        row_ids = _get_row_ids(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to retrieve row IDs from the database'
            f' {kwargs["filename"]} because of', '\n',
            traceback.format_exc()]))
        return
    return row_ids


def apply_multiple_filters_db(
        filename: str,
        keyword_filters: List[Dict[str, Union[int, str, float]]],
        exclude_filtered: bool = False,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        connect_each_row: bool = False) -> None:
    """
    Apply filters to the selected rows in an existing, possibly compressed, ASE
    database

    Parameters
    ----------
    filename
        path to ASE database
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
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    """

    fun_kwargs = {
        'keyword_filters': keyword_filters,
        'exclude_filtered': exclude_filtered,
        'selection': selection,
        'connect_each_row': connect_each_row}
    apply_function_db(filename=filename,
                      function=_apply_multiple_filters_db_mp,
                      fun_kwargs=fun_kwargs)


def _apply_multiple_filters_db_mp(kwargs):
    """Parallelisable version of _apply_multiple_filters_db"""
    try:
        _apply_multiple_filters_db(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to apply filters to the database {kwargs["filename"]}'
            f' because of', '\n', traceback.format_exc()]))


def _apply_multiple_filters_db(
        filename: str,
        keyword_filters: List[Dict[str, Union[int, str, float]]],
        exclude_filtered: bool = False,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        connect_each_row: bool = False,
        use_lock_file: bool = True) -> None:
    """
    Apply filters to the selected rows in an existing ASE database

    Parameters
    ----------
    filename
        path to ASE database
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
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    with (nullcontext() if connect_each_row else
          connect(filename, use_lock_file=use_lock_file)) as database:

        # get rows
        if connect_each_row:
            rows = get_row_ids(filename, selection)
        else:
            rows = database.select(selection)

        to_delete = []
        for row in rows:
            with (nullcontext() if not connect_each_row else
                  connect(filename, use_lock_file=use_lock_file)) as db:

                # get row
                if db is None:
                    db = database
                if connect_each_row:
                    row = db.get(row)

                # apply keyword filters
                filtered = apply_multiple_filters(row, keyword_filters)
                if filtered and exclude_filtered:
                    to_delete.append(row.id)
                    continue

                db.update(row.id, filtered=filtered)

        # delete filtered rows
        if len(to_delete) > 0 and exclude_filtered:
            with (connect(filename, use_lock_file=use_lock_file) if
                  connect_each_row else database) as db:
                db.delete(to_delete)


def apply_multiple_filters(
        atoms_row: Union[Atoms, AtomsRow],
        keyword_filters: List[Dict[str, Union[int, str, float]]]) -> None:
    """
    Apply multiple filters to ASE Atoms or AtomsRow object

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    """

    filtered = False
    for keyword_filter in keyword_filters:

        try:
            filtered = _apply_filter(atoms_row, keyword_filter)
        except Exception:
            warnings.warn('\n'.join(['Could not apply the filter',
                                     keyword_filter, 'because of ',
                                     traceback.format_exc()]))
            filtered = False

        if filtered:
            break

    return filtered


def _apply_filter(
        atoms_row: Union[Atoms, AtomsRow],
        keyword_filter: Dict[str, Union[int, str, float]]) -> None:
    """
    Apply filter to ASE Atoms or AtomsRow object

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    keyword_filter
        keyword filter in the form of a dictionary with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    """

    _check_filter(keyword_filter)
    key = keyword_filter['key']
    if isinstance(atoms_row, Atoms):
        left = atoms_row.info.get(key, None)
        if left is None:
            raise Exception(f'The keyword {key} cannot be found in the'
                            f' Atoms.info dictionary')
    elif isinstance(atoms_row, AtomsRow):
        left = atoms_row.get(key, None)
        if left is None:
            left = atoms_row.data.get(key, None)
        if left is None:
            raise Exception(f'The keyword {key} is neither associated with the'
                            f' AtomsRow object nor found in the AtomsRow.data'
                            f' dictionary')
    else:
        raise Exception(f'An Atoms or AtomsRow object should be supplied not'
                        f' {type(atoms_row)}')

    return get_truth(left=left,
                     right=keyword_filter['value'],
                     symbol=keyword_filter['symbol'])


def _check_filter(
        keyword_filter: Dict[str, Union[int, str, float]]) -> None:
    """Sanity check of keyword filter

    Parameters
    ----------
    keyword_filter
        keyword filter in the form of a dictionary with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    """

    symbols = ['<', '<=', '=', '>=', '>', '!=', '~']
    keys = ['key', 'symbol', 'value']
    if not (isinstance(keyword_filter, dict) and set(keys) ==
            set(keyword_filter.keys())):
        raise Exception(' '.join(['The keyword filter must be a dictionary'
                                  ' with the keys', print_list(keys)]))
    if not isinstance(keyword_filter['key'], str):
        raise Exception(f'The "key" in the keyword filter must be a string not'
                        f' {type(keyword_filter["key"])}')
    if keyword_filter['symbol'] not in symbols:
        raise Exception(' '.join(['The "symbol" in the keyword filter must be'
                                  ' one of the following', print_list(symbols),
                                  'not', str(type(keyword_filter["symbol"]))]))


def print_list(
        values: List,
        separator: str = ',',
        final_separator: str = ', and',
        apostrophes: int = 1,
        format_spec: str = None) -> str:
    """Convert the value to the same type as the reference

    Parameters
    ----------
    value
        value to be converted
    separator
        marker for separating the entries in the list
    final_separator
        strin that separates the last two entries
    apostrophes
        number of apostrophes around each entry, e.g. 1.0 (0), '1.0' (1) or
        "1.0" (2)
    format_spec
        format specifier with the form [[fill]align][sign]["z"]["#"]["0"]
        [width][grouping_option]["." precision][type]
    """

    if not isinstance(values, Iterable):
        raise Exception(f'The values should be a sequence, not a'
                        f' {type(values)}')

    qm = ('' if apostrophes is None else '"' if apostrophes == 2 else
          "'" * apostrophes)
    str_values = [qm + ('{:{}}'.format(v, format_spec) if format_spec is not
                        None else str(v)) + qm for v in values]
    if final_separator is not None:
        last_value = str_values.pop()
        str_values[-1] += f'{final_separator} {last_value}'

    return f"{separator} ".join(str_values)


def same_type(
        value: Union[str, int, float],
        reference: Union[str, int, float]) -> Union[str, int, float]:
    '''
    Convert the value to the same type as the reference

    Parameters
    ----------
    value
        value to be converted
    reference
        reference representing the type to which the value should be converted
    '''
    if isinstance(reference, (float, np.floating)):
        return float(value)
    elif isinstance(reference, numbers.Integral):
        return int(value)
    elif isinstance(reference, str):
        return str(value)
    else:
        raise Exception('The reference must be a float, integer or string')


def get_truth(
        left: Union[str, int, float],
        right: Union[str, int, float],
        symbol: str,
        convert: bool = False) -> bool:
    '''
    Evaluate the truth of the expression defined by right hand and left hand
    values together with the comparison symbol

    Parameters
    ----------
    left
        value on the left hand side of the (in)equality
    right
        value on the right hand side of the (in)equality
    symbol
        symbol representing the comparison operator, i.e. '<', '<=', '=', '>=',
        '>', '!=' or '~', which corresponds to ``numpy.isclose()``
    convert
        convert the right hand side value so that the type matches the left
        hand side
    '''
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq,
           '!=': operator.ne}
    if convert:
        right = same_type(right, left)
    if symbol == '~':
        return np.isclose(left, right)
    return ops[symbol](left, right)


# Help function to update dict of dicts
def update_dict(old_dict, new_dict):
    """
    Updates stacked dictionaries, from
    https://stackoverflow.com/a/3233356
    """
    if not isinstance(new_dict, dict):
        return
    for k, v in new_dict.items():
        if isinstance(v, Mapping):
            old_dict[k] = update_dict(
                old_dict.get(k, {}),
                v,
            )
        else:
            old_dict[k] = v
    return old_dict


def determine_supercell_dimensions(primitive, min_size):
    """
    Determine the dimension of the supercell which size is closest to
    specified the minimum number of atoms or lattice parameter

    Parameters
    ----------
    primitive : ase.Atoms
        primitive structure in the form of an ase.Atoms object
    min_size : int
        minimum size of the supercell specified either as an integer,
        corresponding to the number of atoms, or a float, which is
        interpreted as a lower limit for the lattice parameters
    """

    dimensions = np.ones(3).astype(int)
    minind = np.argmin(primitive.cell.lengths())
    natoms = len(primitive)
    lengths = primitive.cell.lengths()
    while True:
        if isinstance(min_size, numbers.Integral):
            if natoms >= min_size:
                return dimensions
        elif isinstance(min_size, (float, np.floating)):
            if min(lengths) >= min_size:
                return dimensions
        else:
            raise Exception(f"The minimum size must be either an int or"
                            f" float not a {type(min_size)}.")
        dimensions = dimensions + np.array([
            1 if i == minind else 0 for i in range(3)])
        supercell = primitive.repeat(dimensions)
        natoms = len(supercell)
        lengths = supercell.cell.lengths()
        minind = np.argmin(lengths)


def structure_to_atoms(structure):
    """Convert a qmpy.Structure to an ase.Atoms object
    Example::
        >>> import qmpy.io as io
        >>> structure = io.read('POSCAR')
        >>> atoms = structure_to_atoms(structure)
    """
    atoms = Atoms(
        symbols=structure.atom_types,
        cell=structure.cell,
        scaled_positions=structure.coords,
        magmoms=structure.magmoms,
    )
    return atoms


def find_matching_atom_counts(
        natoms: int,
        target_concentrations: Dict[str, float]) -> Dict[str, int]:
    """Find the atom counts that best matches target concentration
    given the total number of atoms per unit cell. This is achieved
    by reformulating the problem as an optimization that involves a
    number of binary variables, which determine if the discrete number
    of atoms should be rounded up or down.

    Parameters
    ----------
    natoms
        total number of atoms in the unit cell
    target_concentrations
        dictionary with chemical symbols as keys and concentrations,
        relative to all sites in the unit cell, as values
    """

    concs = np.asarray(list(target_concentrations.values()))
    min_counts = (concs * natoms).astype(int)

    # determine all rounding possibilities, i.e. by adding 1 or 0 to the
    # atom counts
    round_combos = np.array(list(product(range(2), repeat=len(concs))))
    count_combos = min_counts + round_combos
    count_sums = np.sum(count_combos, axis=1)

    # the sum of the counts must be the total number of atoms
    good_combos = np.where(count_sums == natoms)
    good_counts = count_combos[good_combos]

    # find the atom counts that best matches the target concentrations
    error_combos = np.sum((good_counts - natoms * concs) ** 2, axis=1
                          ) / len(concs)
    best_ind = np.argmin(error_combos)
    best_counts = {elem: int(count) for elem, count in zip(
        target_concentrations.keys(), good_counts[best_ind])}

    return best_counts


def find_matching_concentrations(
        natoms: int,
        target_concentrations: Dict[str, float]) -> Dict[str, float]:
    """Find the concentrations that best matches target given the total number
    of atoms per unit cell. This is achieved by reformulating the problem as an
    optimization that involves a number of binary variables, which determine if
    the discrete number of atoms should be rounded up or down.

    Parameters
    ----------
    natoms
        total number of atoms in the unit cell
    target_concentrations
        dictionary with chemical symbols as keys and concentrations,
        relative to all sites in the unit cell, as values
    """
    best_counts = find_matching_atom_counts(natoms, target_concentrations)
    best_concs = {elem: count / natoms for elem, count in best_counts.items()}

    return best_concs


def get_sublattice_concentrations(
        concentrations: Dict[str, float],
        structure: Atoms,
        cluster_space: ClusterSpace = None,
        sublattice_map: Dict[str, str] = None,
        atoms_per_sublattice: Dict[str, int] = None,
        return_counts: bool = False
        ) -> Dict[str, Dict[str, float]]:
    """
    Convert the overall concentrations into a concentration per sublattice

    Parameters
    ----------
    concentrations
        dictionary with chemical symbols as keys and concentrations,
        relative to all sites in the unit cell, as values
    structure
        atomic configuration
    cluster_space
        cluster space, which must be provided if no sublattice map has been
        specified
    sublattice_map
        dictionary with symbols defining a specific sublattice as keys and
        chemical symbols as values, which must be provided if no cluster
        space has been specified
    atoms_per_sublattice
        dictionary with symbols defining a specific sublattice and values
        corresponding to the number of sites, which must be provided if no
        cluster space has been specified
    return_counts
        return atom counts instead of concentrations
    """

    if sublattice_map is None:
        if cluster_space is None:
            raise ValueError('Cluster space or sublattice map must be'
                             ' provided')
        sublattice_map = get_sublattice_map(structure, cluster_space)
    all_symbols = [symbol for symbols in sublattice_map.values() for symbol in
                   symbols]
    for symbol in all_symbols:
        if all_symbols.count(symbol) > 1:
            raise ValueError(f'The element {symbol} is found on multiple'
                             f' sublattices')
    for symbol in concentrations.keys():
        if all([symbol not in symbols for symbols in sublattice_map.values()]):
            raise ValueError(f'The element {symbol} is not found on any'
                             f' sublattice')

    if atoms_per_sublattice is None:
        if cluster_space is None:
            raise ValueError('Cluster space or sublattice counts must be'
                             ' provided')
        atoms_per_sublattice = get_atoms_per_sublattice(structure,
                                                        cluster_space)

    # count total number of atoms
    n_atoms_tot = len(structure)

    # calculate the best matching concentrations on each sublattice
    sublattice_concentrations = {}
    for sl_symbol, chemical_symbols in sublattice_map.items():
        target_concentrations = {
            symbol: (concentrations.get(symbol, 0.0) * n_atoms_tot /
                     atoms_per_sublattice[sl_symbol])
            for symbol in chemical_symbols
        }
        if return_counts:
            sublattice_concentrations[
                sl_symbol] = find_matching_atom_counts(
                    atoms_per_sublattice[sl_symbol], target_concentrations)
        else:
            sublattice_concentrations[
                sl_symbol] = find_matching_concentrations(
                    atoms_per_sublattice[sl_symbol], target_concentrations)

    return sublattice_concentrations


def get_sublattice_map(
        structure: Atoms,
        cluster_space: ClusterSpace
        ) -> Dict[str, str]:
    """Obtain a dictionary with the chemical symbols associated with each
    sublattice

    Parameters
    ----------
    structure
        atomic configuration
    cluster_space
        cluster space
    """

    return {sl.symbol: sl.chemical_symbols for sl in
            cluster_space.get_sublattices(structure)}


def get_atoms_per_sublattice(
        structure: Atoms,
        cluster_space: ClusterSpace
        ) -> Dict[str, int]:
    """Obtain a dictionary with the number of atoms on each sublattice.

    Parameters
    ----------
    structure
        atomic configuration
    cluster_space
        cluster space
    """

    atoms_per_sublattice = {sl.symbol: len(sl.indices) for sl in
                            cluster_space.get_sublattices(structure)}

    return atoms_per_sublattice


def expand_chemical_symbols(chemical_symbols: Union[List[str],
                                                    List[List[str]]],
                            structure: Atoms) -> List[List[str]]:
    """Expands the chemical symbols into a list for each site in the
       structure and chemical symbols. Carries out multiple sanity input
       checks. Adapted from icet.ClusterSpace (version 2.2).

    Parameters
    ----------
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    structure
        atomic configuration
    """

    # setup chemical symbols as List[List[str]]
    if all(isinstance(i, str) for i in chemical_symbols):
        expanded_chemical_symbols = [chemical_symbols] * len(structure)
    # also accept tuples and other iterables but not, e.g.,
    # List[List, str] (need to check for str explicitly here because
    # str is an Iterable)
    elif not all(isinstance(i, Iterable) and not isinstance(i, str)
                 for i in chemical_symbols):
        raise TypeError(f'chemical_symbols must be List[str] or'
                        f' List[List[str]], not '
                        f'{type(chemical_symbols)}')
    elif len(chemical_symbols) != len(structure):
        raise ValueError(
            'chemical_symbols must have the same length as structure.\n'
            f' len(chemical_symbols) = {len(chemical_symbols)},'
            f' len(structure) = {len(structure)}')
    else:
        expanded_chemical_symbols = copy.deepcopy(chemical_symbols)

    for i, symbols in enumerate(expanded_chemical_symbols):
        if len(symbols) != len(set(symbols)):
            raise ValueError(
                f'Found duplicates of allowed chemical symbols on site'
                f' {i}. Allowed species on site {i} is {symbols}')

    if len([tuple(sorted(s)) for s in chemical_symbols if len(s) > 1
            ]) == 0:
        raise ValueError('No active sites found')

    return expanded_chemical_symbols


def build_prototype_db(
        filename: str,
        formulas: List[str],
        properties: List[str] = None,
        space_groups: List[Union[int, str]] = None,
        crystal_systems: List[str] = None,
        make_ideal: bool = False,
        api_key: str = None,
        overwrite: bool = False,
        append: bool = False,
        delete_existing: bool = False,
        compress: bool = False,
        asynchronus: bool = False,
        processes: int = 1) -> None:
    '''
    Create a database with prototype structures retrieved from the Materials
    Project

    Parameters
    ----------
    filename
        path to database where the structures shall be stored
    formulas
        list of chemical formulas
    space_groups
        list of space groups either in the form of numbers or strings
    crystal_systems
        list of crystal systems, e.g. 'hexagonal'
    properties
        list of properties to include
    make_ideal
        find ideal structures for the given symmetry
    api_key
        Materials Project API key
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
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    '''

    # handle optional selection criteria
    fun_kwargs = [
        dict(formula=formula,
             properties=properties,
             make_ideal=make_ideal,
             api_key=api_key)
        for formula in formulas]
    if space_groups is not None:
        if len(space_groups) != len(formulas):
            raise Exception(f"The number of space groups ({len(space_groups)})"
                            f" does not match the number or formulas"
                            f" ({len(formulas)})")
        fun_kwargs = [dict(space_group=space_group, **kwargs) for kwargs,
                      space_group in zip(fun_kwargs, space_groups)]
    if crystal_systems is not None:
        if len(crystal_systems) != len(formulas):
            raise Exception(f"The number of crystal systems"
                            f" ({len(crystal_systems)}) does not match the"
                            f" number or formulas ({len(formulas)})")
        fun_kwargs = [dict(crystal_system=crystal_system, **kwargs) for kwargs,
                      crystal_system in zip(fun_kwargs, crystal_systems)]

    # retrieve prototype structures
    structures = run_function_parallel(function=_retrieve_prototype_mp,
                                       fun_kwargs=fun_kwargs,
                                       asynchronus=asynchronus,
                                       processes=processes)

    # add prototype structures to database
    fun_kwargs = [
        dict(structure=structure,
             data=structure.info,
             tag=structure.info['material_id'],
             overwrite=overwrite)
        for structure in structures]
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def _retrieve_prototype_mp(kwargs):
    """Parallelisable version of retrieve_prototype"""
    try:
        prototype = retrieve_prototype(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to retrieve prototype with formula {kwargs["formula"]}'
            f'from the Materials Project database because of', '\n',
            traceback.format_exc()]))
        return
    return prototype


def retrieve_prototype(
        formula: str,
        properties: List[str] = None,
        space_group: Union[int, str] = None,
        crystal_system: str = None,
        make_ideal: bool = False,
        api_key: str = None) -> None:
    '''
    Retrieve prototype structures from the Materials Project

    Parameters
    ----------
    formula
        chemical formula
    space_group
        space group either in the form of a number or string; cannot be
        specified together with crystal system
    crystal_system
        crystal system, e.g. 'hexagonal'; cannot be specified together with
        space group
    properties
        list of properties to include
    make_ideal
        find ideal structures for the given symmetry
    api_key
        Materials Project API key
    '''

    # create rester instance
    mprester = MPRester(api_key=api_key)

    # handle optional selection criteria
    skip_properties = ['structure']
    properties = list(set(properties + ['formula_pretty', 'structure',
                                        'material_id', 'spacegroup',
                                        'formation_energy_per_atom',
                                        'symmetry']))

    # retrieve structures from the Materials Project
    try:
        data = mprester.get_summary(dict(formula=formula),
                                    fields=properties)
    except Exception:
        try:
            data = mprester.summary_search(formula=formula,
                                           fields=properties)
        except Exception:
            data = mprester.summary.search(formula=formula,
                                           fields=properties)

    # flatten the data so that the properties/fields acts as columns in the
    # DataFrame
    flattened = [{
        k: v for k, v in flatten(doc if isinstance(doc, dict) else
                                 doc.dict(), reducer="underscore").items()
        if k != "fields_not_requested"} for doc in data]

    # from_records is also used to get the correct columns in DataFrame,
    # both this and "flattened" is currently necessary (2023-05-15)
    df = pd.DataFrame.from_records(flattened)
    if len(df) == 0:
        raise Exception(f"Unable to retrieve data for {formula}")

    # apply selection criteria
    if space_group is not None:
        if isinstance(space_group, int):
            df = df.loc[df.spacegroup_number == str(space_group)]
        elif isinstance(space_group, str):
            df = df.loc[df.spacegroup_symbol == space_group]
    if crystal_system is not None:
        df = df.loc[df.symmetry_crystal_system == crystal_system]
    row = df.loc[df.formation_energy_per_atom ==
                 df.formation_energy_per_atom.min()].iloc[0]

    # create atoms object
    prototype = AseAtomsAdaptor.get_atoms(row.structure)
    info = row.to_dict()
    if make_ideal:
        info['ideal'] = True
        hall_number = get_hall_numbers(
            international_short=info['symmetry_symbol'])[0]
        prototype, symprec = get_ideal_structure(
            structure=prototype,
            hall_number=hall_number,
            return_symprec=True)
        info['symprec'] = symprec
    for key in skip_properties:
        del info[key]
    prototype.info = info

    return prototype


def get_ideal_structure(
        structure: Atoms,
        hall_number: int,
        initial_symprec: float = 1e-5,
        max_symprec: float = 10,
        step_factor: float = 10,
        return_symprec: bool = True) -> Union[Atoms, Tuple[Atoms, float]]:
    """

    Parameters
    ----------
    structure
        atomic configuration
    hall_number
        Hall number representation of space group (1-530)
    initial_symprec
        initial guess of tolerance, which is imposed when analyzing the
        symmetry using spglib, required to match the specified space group
    max_symprec
        maxium value on the tolerance to test in the attempt to match the
        specified space group
    step_factor
        factor by which the tolerance is increased in the attempt to match the
        specified space group
    return_symprec
        include the final tolerance in the output
    """

    # convert structure to spglib cell
    spglib_cell = ase_atoms_to_spglib_cell(structure)

    # get matching symmetry dataset
    dataset = None
    symprec = initial_symprec
    while True:
        dataset = get_symmetry_dataset(spglib_cell, symprec=symprec,
                                       hall_number=hall_number)
        if dataset is not None:
            break

        symprec *= step_factor
        if symprec >= max_symprec:
            warnings.warn(f'Unable to match Hall number {hall_number} for'
                          f' symprec < {max_symprec}')
            break

    if dataset is None:
        ideal_structure = None
    else:
        lattice = dataset['std_lattice']
        numbers = dataset['std_types']
        scaled_positions = [np.round(pos, 12) for pos in
                            dataset['std_positions']]
        ideal_structure = Atoms(scaled_positions=scaled_positions,
                                numbers=numbers, cell=lattice,
                                pbc=structure.pbc)
        ideal_structure.wrap()

    if return_symprec:
        return ideal_structure, symprec
    return ideal_structure


def get_hall_numbers(
        number: int = None,
        international_short: str = None,
        international_full: str = None,
        international: str = None,
        schoenflies: str = None,
        hall_symbol: str = None,
        pointgroup_schoenflies: str = None,
        pointgroup_international: str = None,
        arithmetic_crystal_class_number: int = None,
        arithmetic_crystal_class_symbol: str = None) -> int:
    """

    Parameters
    ----------
    number
        space group number (1-230), e.g. 221
    international_short
        Hermann-Mauguin space group with short symbols and without blank
        spaces, e.g. ``Pm-3m``
    international_full
        Hermann-Mauguin space group with full symbols and blank spaces, e.g.
        ``P 4/m -3 2/m``
    international
        Hermann-Mauguin space group with short symbols and blank spaces, e.g.
        ``P m -3 m``
    schoenflies
        Schoenflies point group symbol, e.g. ``Oh^1``
    hall_symbol
        Hall point group symbol, e.g. ``-P 4 2 3``
    pointgroup_schoenflies
        Schoenflies lattice point group symbol, e.g. ``Oh``
    pointgroup_international
        Hermann-Mauguin (geometric) point group with short symbols, e.g.
        ``m-3m``
    arithmetic_crystal_class_number
        arithmetic crystal class number (1-73), e.g. 71
    arithmetic_crystal_class_symbol
        arithmetic crystal class symbol, e.g. ``m-3mP``
    """
    args = locals()
    none_tuples = [(k, v) for k, v in args.items() if v is not None]
    if len(none_tuples) != 1:
        raise Exception(f'Exactly one argument should be provided not'
                        f' {len(none_tuples)}')
    key, value = none_tuples[0]
    return [hall_number for hall_number in range(1, 531) if
            get_spacegroup_type(hall_number)[key] == value]


def _atoms_row_to_structure(
        atoms_row: Union[Atoms, AtomsRow],
        ) -> Atoms:
    """Perform sanity check of Atoms or AtomsRow object and return an Atoms
    object

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    """

    if isinstance(atoms_row, Atoms):
        structure = atoms_row
    elif isinstance(atoms_row, AtomsRow):
        structure = atoms_row.toatoms()
        structure.info = atoms_row.key_value_pairs
        try:
            structure.info['data'] = atoms_row.data
        except Exception:
            pass
    else:
        raise Exception(f'An Atoms or AtomsRow object should be supplied not'
                        f' {type(atoms_row)}')

    return structure


def get_average_abundance(atoms_row: Union[Atoms, AtomsRow]) -> float:
    '''Estimate the average abundance for a structure based on data retrieved
    from `chemnetbase.com <https://hbcp.chemnetbase.com/faces/documents/14_10/
    14_10_0001.xhtml>`_

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # read data from file
    elements, abundances = np.genfromtxt(
        os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'data/abundances.csv'), delimiter='\t',
        skip_header=1, usecols=(2, 5), dtype=str, unpack=True)

    # calculate average abundance
    structure_abundance = []
    for symbol in set(structure.symbols):
        try:
            abundance = float(abundances[np.where(elements == symbol)][0])
        except Exception:
            warnings.warn(f'No abundance available for {symbol}, assuming 0')
            abundance = 0.0
        structure_abundance.append(abundance * structure.symbols.count(symbol))
    return np.sum(structure_abundance) / len(structure)


def get_price(atoms_row: Union[Atoms, AtomsRow]) -> float:
    '''Estimate the price for a structure based on data retrieved from
    `www.leonland.de <http://www.leonland.de/elements_by_price/en/list>`_

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # read data from file
    elements, prices = np.genfromtxt(
        os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'data/symbolprice.csv'),
        delimiter=',', dtype=str, unpack=True)

    # calculate price
    structure_price = []
    for symbol in set(structure.symbols):
        try:
            price = float(prices[np.where(elements == symbol)][0])
        except ValueError:
            warnings.warn(f'No market price available for {symbol}')
            return None
        structure_price.append(price * structure.symbols.count(symbol))
    return np.sum(structure_price) / len(structure)
