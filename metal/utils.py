from contextlib import nullcontext
import gzip
import os
import shutil
import tempfile
from typing import Callable, Dict, List, Union


def decompress_db(
        file_in: str,
        file_out: str = None,
        delete: bool = False) -> None:
    """Decompress a database file

    Parameters:
        file_in
            path to compressed database file
        file_out
            path to decompressed database file, is otherwise obtained by
            removing the 'gz' suffix from `file_in`
        delete
            if the `file_in` file should be deleted
    """

    if '.gz' not in file_in:
        raise Exception('The input file is not compressed.')
    if file_out is None:
        file_out = file_in.replace('.gz', '')
    with gzip.open(file_in, 'rb') as f_in:
        with open(file_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if delete:
        os.remove(file_in)


def compress_db(
        file_in: str,
        file_out: str = None,
        delete: bool = False) -> None:
    """
    Compress a database file

    Parameters:
        file_in
            path to uncompressed database file
        file_out
            path to compressed database file, is otherwise obtained by removing
            the 'gz' suffix from `file_in`
        delete
            if the `file_in` file should be deleted
    """

    if '.gz' in file_in:
        raise Exception('The input file is already compressed.')
    if file_out is None:
        file_out = f"{file_in}.gz"
    with open(file_in, 'rb') as f_in:
        with gzip.open(file_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if delete:
        os.remove(file_in)


def apply_function_db(
        function: Callable,
        filename: str,
        fun_kwargs: Union[Dict, List[Dict]],
        append: bool = True,
        delete_existing: bool = False,
        dbpath: str = None,
        compress: bool = False,
        asynchronus: bool = False,
        processes: int = 1) -> None:
    '''Apply the supplied function to the specified, possibly compressed,
    database

    Parameters
    ----------
    function
        function that should be applied to the database, which must accept a
        dictionary with input arguments as input
    filename
        path to database file
    fun_kwargs
        dictionary, or list thereof, with keyword arguments, each of which
        should represent an input argument that is accepted by the specified
        function; the keyword ``filename``, i.e. the path to the database file,
        will be added to each of the dictionaries
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    compress
        if the database should be compressed; the default is to decide this
        from the filename
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    '''

    # handle filename
    if os.path.isfile(filename):
        if not append:
            if delete_existing:
                os.remove(filename)
            else:
                raise Exception(f'The database {filename} already exists')
    dirname = os.path.dirname(filename)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    if compress and '.gz' not in filename:
        filename = f"{filename}.gz"

    # handle kwargs
    orig_type = type(fun_kwargs.copy())
    if fun_kwargs is not None:
        fun_kwargs = _process_fun_kwargs(fun_kwargs)
    else:
        fun_kwargs = [{}]

    with (tempfile.TemporaryDirectory() if '.gz' in filename else
          nullcontext()) as tmpdir:

        # decompress file if necessary
        if '.gz' in filename:
            tmpname = os.path.join(tmpdir, 'database.db')
            if os.path.isfile(filename):
                decompress_db(filename, tmpname, delete=False)
        else:
            tmpname = filename

        # apply function, in parallel
        fun_kwargs = [dict(filename=tmpname, **kwargs) for kwargs in
                      fun_kwargs]
        results = run_function_parallel(
            function=function,
            fun_kwargs=fun_kwargs,
            asynchronus=asynchronus,
            processes=processes)

        if '.gz' in filename:
            # Zip database and delete the existing file
            compress_db(tmpname, filename, delete=True)

    if orig_type == dict and len(results) == 1:
        return results[0]
    return results


def run_function_parallel(
        function: Callable,
        fun_kwargs: Union[Dict, List[Dict]],
        asynchronus: bool = False,
        processes: int = 1) -> None:
    '''Run the supplied function in parallel

    Parameters
    ----------
    function
        function that should be applied to the database, which must accept a
        dictionary with input arguments as input
    fun_kwargs
        dictionary, or list thereof, with keyword arguments, each of which
        should represent an input argument that is accepted by the specified
        function
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    '''

    # process list of kwargs:
    orig_type = type(fun_kwargs.copy())
    fun_kwargs = _process_fun_kwargs(fun_kwargs)
    processes = min(processes, len(fun_kwargs))

    if processes > 1:

        from multiprocessing import Pool

        with Pool(processes) as pool:
            if asynchronus:
                results = pool.map_async(function, fun_kwargs)
                results = list(results.get())
            else:
                results = list(pool.map(function, fun_kwargs))

    else:

        results = [function(kwargs) for kwargs in fun_kwargs]
        if orig_type == dict and len(results) == 1:
            results = results[0]

    return results


def _process_fun_kwargs(fun_kwargs: Union[Dict, List[Dict]]) -> List[Dict]:
    '''Process the (list of) keyword arguments

    Parameters
    ----------
    fun_kwargs
        dictionary, or list thereof, with keyword arguments
    '''

    # process list of kwargs:
    if isinstance(fun_kwargs, dict):
        fun_kwargs = [fun_kwargs]
    else:
        if not (isinstance(fun_kwargs, (list, tuple)) and
                all([isinstance(fun_kwarg, dict) for fun_kwarg in fun_kwargs])
                ):
            raise Exception('The "fun_kwargs" must be a single dictionary or a'
                            ' list of dictionaries')

    return fun_kwargs
