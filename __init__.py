"""
Protein DiT package for protein structure generation.
"""

__version__ = "0.1.0"

# Make sure the package is in the Python path
import os
import sys
package_root = os.path.dirname(os.path.abspath(__file__))
if package_root not in sys.path:
    sys.path.append(package_root) 