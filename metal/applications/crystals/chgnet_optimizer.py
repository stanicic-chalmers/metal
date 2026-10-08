from __future__ import annotations

import os
from typing import Literal

from ase.optimize.optimize import Optimizer

from chgnet.model.model import CHGNet, module_dir
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import OPTIMIZERS, CHGNetCalculator

import torch

from .torch_tools import cuda_devices_sorted_by_free_mem


class NonBlockCalculator(CHGNetCalculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = ("energy", "forces", "stress", "magmoms")

    def __init__(
        self,
        model: CHGNet | None = None,
        use_device: str | None = None,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> None:
        """Provide a CHGNet instance to calculate various atomic properties using ASE.

        Args:
            model (CHGNet): instance of a chgnet model. If set to None,
                the pretrained CHGNet is loaded.
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            **kwargs: Passed to the Calculator parent class.
        """
        super(CHGNetCalculator, self).__init__(**kwargs)

        # Determine the device to use
        if use_device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = use_device or ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cuda":
                self.device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"

        # Move the model to the specified device
        if model is not None:
            self.model = model.to(self.device, non_blocking=True)
        else:
            self.model = NonBlockCHGNet.load(use_device=self.device, verbose=False)
        self.model.graph_converter.set_isolated_atom_response(on_isolated_atoms)
        self.stress_weight = stress_weight
        print(f"CHGNet will run on {self.device}")


class CHGNetOptimizer(StructOptimizer):
    """Wrapper class for structural relaxation."""

    def __init__(
        self,
        model: CHGNet | CHGNetCalculator | None = None,
        optimizer_class: Optimizer | str | None = "FIRE",
        use_device: str | None = None,
        stress_weight: float = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
        """
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class

        if isinstance(model, CHGNetCalculator):
            self.calculator = model
        else:
            self.calculator = NonBlockCalculator(
                model=model,
                stress_weight=stress_weight,
                use_device=use_device,
                on_isolated_atoms=on_isolated_atoms,
            )


class NonBlockCHGNet(CHGNet):
    """Crystal Hamiltonian Graph neural Network
    A model that takes in a crystal graph and output energy, force, magmom, stress."""

    @classmethod
    def load(
        cls,
        model_name="0.3.0",
        use_device: str | None = None,
        verbose: bool = True,
    ) -> CHGNet:
        """Load pretrained CHGNet model.

        Args:
            model_name (str, optional):
                Default = "0.3.0".
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            verbose (bool): whether to print model device information
                Default = True
        Raises:
            ValueError: On unknown model_name.
        """
        checkpoint_path = {
            "0.3.0": "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
            "0.2.0": "../pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        }.get(model_name)

        if checkpoint_path is None:
            raise ValueError(f"Unknown {model_name=}")

        model = cls.from_file(
            os.path.join(module_dir, checkpoint_path),
            # mlp_out_bias=True is set for backward compatible behavior but in rare
            # cases causes unphysical jumps in bonding energy. see
            # https://github.com/CederGroupHub/chgnet/issues/79
            mlp_out_bias=model_name == "0.2.0",
            version=model_name,
        )

        # Determine the device to use
        if use_device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = use_device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda":
                device = f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"

        # Move the model to the specified device
        model = model.to(device, non_blocking=True)
        if verbose:
            print(f"CHGNet will run on {device}")
        return model
