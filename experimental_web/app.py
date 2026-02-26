from __future__ import annotations

import argparse
import os
import sys

from nicegui import ui

from experimental_web.core.app_init import init_app
from experimental_web.core.secret import get_storage_secret
from experimental_web.logging_setup import setup_logging

# register routes by importing pages
from experimental_web.pages import home, experiment  # noqa: F401


def _parse_debug_flag() -> int | None:
    """Parse --debug [LEVEL] without breaking NiceGUI/uvicorn args."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--debug',
        nargs='?',
        const=-1,  # --debug without number => all
        default=None,
        type=int,
        help='Debug verbosity: (no arg)=all, 0=ERROR,1=WARNING,2=INFO,3=DEBUG,4=TRACE',
    )
    args, unknown = parser.parse_known_args(sys.argv[1:])
    # Keep other args intact (if any).
    sys.argv = [sys.argv[0]] + unknown
    return args.debug


debug_level = _parse_debug_flag()
if debug_level is None:
    os.environ.pop('EXPERIMENTAL_WEB_DEBUG', None)
else:
    os.environ['EXPERIMENTAL_WEB_DEBUG'] = str(int(debug_level))

setup_logging(debug_level)

init_app()

ui.run(
    title='Experiment Manager',
    reload=False,
    storage_secret=get_storage_secret(),
)
