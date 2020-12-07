"""
Miscellaneous utilities for gmvc file processing
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__file__)


def nearest(items, pivot, direction=None):
    """Return the element in items that is closest in value to pivot"""
    if direction == 'after':
        items = [i for i in items if i > pivot]
        if not items:
            return
        return min(items, key=lambda x: abs(x - pivot))
    elif direction == 'before':
        items = [i for i in items if i > pivot]
        if not items:
            return
        return min(items, key=lambda x: abs(x - pivot))

    return min(items, key=lambda x: abs(x - pivot))


