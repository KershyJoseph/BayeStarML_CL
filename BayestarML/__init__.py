#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:54:47 2025

@author: LamirelFamily
"""

"""Top‑level package namespace."""

from importlib import metadata as _meta

from .predictor import Predictor  # re‑export for user convenience

__all__ = ["Predictor", "__version__"]
__version__ = _meta.version(__name__) if False else "0.0.0"