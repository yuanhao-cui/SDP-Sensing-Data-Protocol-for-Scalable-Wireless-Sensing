"""Pytest configuration for wsdp tests."""
import sys
import os

# Add src directory to Python path
src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
