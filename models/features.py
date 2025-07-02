from dataclasses import dataclass

@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int = 1

@dataclass
class DenseFeat:
    name: str
    dimension: int = 1

