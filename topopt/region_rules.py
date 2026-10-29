"""The size rules of a load or solid region, in one place (fine pass, #318).

`topopt check` (`cli.check_data`) and the part builder (`regions_builder`) both warn about a region that is small or thin. Both import
these numbers. The fine pass (`topopt/fine_pass.py`) reads them too: it asks for the voxel at which a region has `REGION_ACROSS`
voxels across. The module imports nothing from the package, so the builder stays light.
"""
from __future__ import annotations

MIN_REGION_VOXELS = 30    # a region with fewer voxels is small: a warning level
MIN_REGION_THICK = 1.9    # the largest distance (in voxels) from a voxel of the region to the outside: a region below it is thin
# What `MIN_REGION_THICK` gives in voxels across. `check` shows `2 * thick - 1` as the voxels across, so 1.9 gives 2.8 voxels across
# (not 3: the rule is a little under 3 voxels).
REGION_ACROSS = 2.0 * MIN_REGION_THICK - 1.0
