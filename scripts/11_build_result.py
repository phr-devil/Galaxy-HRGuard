#!/usr/bin/env python
"""Assemble the confirmation result record from committed artefacts.

The module executes on import by design: it is a pure transformation of
committed artefacts, with no arguments and no side effects other than writing
one JSON record into the output directory.
"""
from __future__ import annotations

import hrguard.result_build  # noqa: F401  (executes the assembly)
