from __future__ import annotations

from calorine.calculators import CPUNEP, GPUNEP

from .base_optimizer import BaseStructOptimizer


class NEPOptimizer(BaseStructOptimizer):
    """The NEP optimizer is a class for structural relaxation using a
    neuroevolution potential (NEP) trained with GPUMD via calorine. (Adapted
    from `matgl.ext.ase.Relaxer`)

    Parameters
    ----------
    model : str
        path to file in ``nep.txt`` format with model parameters.
    label : str
        label for this calclator
    use_gpu : bool
        if GPU should be used. Since this means writing files that are used as
        input to the `gpumd` executable, it is likely to be slow if many
        calculations are to be performed.
    directory : str
        directory to run GPUMD in if ``use_gpu`` is True. If None, a temporary
        directory will be created and removed once the calculations are
        finished. If specified, the directory will not be deleted. In the
        latter case, it is advisable to do no more than one calculation with
        this calculator (unless you know exactly what you are doing).
    command : str
        command to run GPUMD with if `use_gpu` is True
    debug : bool
        flag to toggle debug mode if ``use_gpu`` is False; which means printing
        GPUMD output
    optimizer : str
        the optimization algorithm
    relax_cell : bool
        whether to relax the lattice cell
    """

    def __init__(self,
                 model: str,
                 label: str | None = None,
                 use_gpu: bool = False,
                 directory: str | None = None,
                 command: str = 'gpumd',
                 debug: bool = False,
                 optimizer: str = 'FIRE',
                 relax_cell: bool = True):

        super().__init__(optimizer=optimizer, relax_cell=relax_cell)

        # set calculator
        if use_gpu:
            self.calculator = GPUNEP(
                model_filename=model,
                directory=directory,
                label='GPUNEP' if label is None else label,
                command=command)
        else:
            self.calculator = CPUNEP(
                model_filename=model,
                label='CPUNEP' if label is None else label,
                debug=debug)
