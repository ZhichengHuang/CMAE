from .base import BaseModel
from .mae import MAE
from .classification import Classification,ClassificationMixup
from .cmae import CMAE

__all__ = [
'BaseModel', 'MAE', 'Classification', 'CMAE','ClassificationMixup'
]