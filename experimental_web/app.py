from nicegui import ui

from experimental_web.core.app_init import init_app
from experimental_web.core.secret import get_storage_secret

# register routes by importing pages
from experimental_web.pages import home, experiment  # noqa: F401

init_app()

ui.run(
    title='Experiment Manager',
    reload=False,
    storage_secret=get_storage_secret(),
)
