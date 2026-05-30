"""Neural-network training and inference utilities.

Submodules expose their own symbols; this file intentionally re-exports nothing
so that touching ``ai.nn`` does not pull torch/lightning until the caller asks
for a specific submodule (``from ai.nn.dataset import RaceDataset`` etc.).
"""
