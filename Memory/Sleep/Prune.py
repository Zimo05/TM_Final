"""Deprecated compatibility surface for removed Sleep pruning.

Deep Sleep no longer deletes episodic-memory rows and no longer performs the
legacy low-mass leaf prune. Complete-refinement structural deletion lives in
``Sleep.TopologyPrune``. Routing mass remains a diagnostic statistic, so its
updater is re-exported temporarily for old imports.
"""

from Sleep.TopologyPrune import update_leaf_mass

__all__ = ["update_leaf_mass"]
