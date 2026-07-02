"""Rend le package src importable lors de la collecte des tests (avant tout import)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
