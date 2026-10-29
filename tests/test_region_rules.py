"""The size rules of a load or solid region are in one module that `cli` and `regions_builder` both import (fine pass, #318)."""
from topopt import cli, region_rules, regions_builder


def test_cli_and_builder_use_the_numbers_of_region_rules():
    assert cli.MIN_REGION_VOXELS == region_rules.MIN_REGION_VOXELS
    assert regions_builder.MIN_REGION_VOXELS == region_rules.MIN_REGION_VOXELS == 30
    assert cli.MIN_REGION_THICK == regions_builder.MIN_REGION_THICK == region_rules.MIN_REGION_THICK == 1.9


def test_the_voxels_across_of_the_thickness_rule_are_2_point_8():
    # `check` shows 2 * thick - 1 as the voxels across: the rule 1.9 is 2.8 voxels across, a little under 3
    assert abs(region_rules.REGION_ACROSS - 2.8) < 1e-12


def test_region_rules_imports_nothing_of_the_package():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(region_rules.__file__).read_text())
    imported = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(getattr(n, "module", None) == "__future__" for n in imported)
