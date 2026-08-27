"""
Guard the library's Python 3.9 floor.

The diffusion-policy container's conda env is ``python=3.9`` (numpy 1.24) and imports this library;
the ROS node imports it under 3.12. There is no 3.9 interpreter on a development laptop, so the
floor cannot be checked by running the tests -- and the realistic way to break it is an ``X | None``
annotation, which raises ``TypeError`` at import time on 3.9 and is invisible on 3.12.

``from __future__ import annotations`` makes every annotation a string, which defuses exactly that
class of break. This asserts each module carries it, and that none uses 3.10+ syntax outside an
annotation.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'polyumi_inference'
MODULES = sorted(PACKAGE.rglob('*.py'))


def test_the_package_was_found():
    """A rglob that silently matched nothing would make every test below vacuously pass."""
    assert len(MODULES) >= 8


@pytest.mark.parametrize('path', MODULES, ids=lambda p: p.name)
def test_module_defers_annotations(path):
    """Every module must carry `from __future__ import annotations`."""
    tree = ast.parse(path.read_text())
    futures = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == '__future__'
        for alias in node.names
    }
    if not tree.body or (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)):
        pytest.skip('docstring only')
    assert 'annotations' in futures, (
        f'{path.name} is missing `from __future__ import annotations`. Without it a `X | None` '
        "annotation raises TypeError at import under the container's Python 3.9."
    )


@pytest.mark.parametrize('path', MODULES, ids=lambda p: p.name)
def test_module_has_no_match_statement(path):
    """`match` is 3.10+ and is a syntax error on 3.9, which deferred annotations cannot help with."""
    tree = ast.parse(path.read_text())
    assert not any(isinstance(node, getattr(ast, 'Match', ())) for node in ast.walk(tree)), (
        f'{path.name} uses a match statement, which does not parse on Python 3.9'
    )
