"""KOL package with lazy public imports for metadata-only tooling."""

__all__ = ["phasor_dft_at_f0", "phasors_for_channels", "OperatorBankConfig", "KnownOperatorBank", "KOLAugmentedDataset"]


def __getattr__(name):
    if name in {"phasor_dft_at_f0", "phasors_for_channels"}:
        from . import phasor_utils
        return getattr(phasor_utils, name)
    if name in {"OperatorBankConfig", "KnownOperatorBank"}:
        from . import operator_bank
        return getattr(operator_bank, name)
    if name == "KOLAugmentedDataset":
        from .kol_dataset import KOLAugmentedDataset
        return KOLAugmentedDataset
    raise AttributeError(name)
