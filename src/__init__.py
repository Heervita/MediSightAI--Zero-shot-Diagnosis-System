# src/__init__.py

# Expose the main classes to the outside world
from .model import MedicalModel
from .rag import RAGEngine
from .utils import process_image

# This allows you to do:
# from src import MedicalModel, RAGEngine
# Instead of:
# from src.model import MedicalModel