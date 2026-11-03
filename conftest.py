"""Root pytest configuration.

Redirects the ``tmp_path`` factory base directory to a local ``.tmp/``
folder inside the project root.  This avoids the Windows permission error
that arises when pytest tries to use the system temp directory at
``C:\\Users\\<user>\\AppData\\Local\\Temp\\pytest-of-<user>``, which can
become inaccessible after a process runs as a different privilege level.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Redirect tmp_path base to project-local .tmp/ to avoid Windows PermissionError
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent
_LOCAL_TMP = _PROJECT_ROOT / ".tmp"


@pytest.fixture(scope="session")
def tmp_path_factory(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> pytest.TempPathFactory:
    """Override the tmp_path factory to use a project-local base directory.

    On Windows the default base (``%TEMP%\\pytest-of-<user>``) can become
    permission-denied after mixed-privilege runs.  Pointing at ``.tmp/``
    inside the workspace keeps all test artefacts local and avoids the
    OS-level access error entirely.
    """
    _LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    tmp_path_factory._basetemp = _LOCAL_TMP  # noqa: SLF001 – intentional override
    return tmp_path_factory
