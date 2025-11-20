# app/test_page.py

from nicegui import ui
import asyncio


# ---------------- TAB CLASSES ---------------- #

class OverviewTab:
    def __init__(self, state: dict, propagate):
        self.state = state
        self.propagate = propagate  # sync function defined in MainPage

    @ui.refreshable
    def show(self):
        with ui.column():
            ui.label('Overview Tab').classes('text-xl font-bold mb-2')

            ui.label(f"Overview computations: {self.state['overview_runs']}")
            ui.label(f"Processing computations: {self.state['processing_runs']}")
            ui.label(f"Plot computations: {self.state['plot_runs']}")
            ui.label(f"Settings computations: {self.state['settings_runs']}")

            ui.separator()
            ui.label(
                f"Last computation triggered by: "
                f"{self.state['last_triggered'] or 'none yet'}"
            )

            ui.separator()
            ui.button(
                'Run OVERVIEW computation (2 s)',
                on_click=self.run,
            )

    async def run(self):
        # simulate heavy load
        await asyncio.sleep(2)

        # update shared state
        self.state['overview_runs'] += 1
        self.state['last_triggered'] = 'Overview'

        # refresh this tab
        self.show.refresh()

        # refresh following tabs (2, 3, 4)
        self.propagate(from_tab=1)


class ProcessingTab:
    def __init__(self, state: dict, propagate):
        self.state = state
        self.propagate = propagate

    @ui.refreshable
    def show(self):
        with ui.column():
            ui.label('Processing Tab').classes('text-xl font-bold mb-2')

            ui.label(f"Overview computations: {self.state['overview_runs']}")
            ui.label(f"Processing computations: {self.state['processing_runs']}")
            ui.label(f"Plot computations: {self.state['plot_runs']}")
            ui.label(f"Settings computations: {self.state['settings_runs']}")

            ui.separator()
            ui.label(
                f"Last computation triggered by: "
                f"{self.state['last_triggered'] or 'none yet'}"
            )

            ui.separator()
            ui.button(
                'Run PROCESSING computation (2 s)',
                on_click=self.run,
            )

    async def run(self):
        await asyncio.sleep(2)

        self.state['processing_runs'] += 1
        self.state['last_triggered'] = 'Processing'

        self.show.refresh()

        # refresh following tabs (3, 4), but not 1
        self.propagate(from_tab=2)


class PlotTab:
    def __init__(self, state: dict, propagate):
        self.state = state
        self.propagate = propagate

    @ui.refreshable
    def show(self):
        with ui.column():
            ui.label('Plot Tab').classes('text-xl font-bold mb-2')

            ui.label(f"Overview computations: {self.state['overview_runs']}")
            ui.label(f"Processing computations: {self.state['processing_runs']}")
            ui.label(f"Plot computations: {self.state['plot_runs']}")
            ui.label(f"Settings computations: {self.state['settings_runs']}")

            ui.separator()
            ui.label(
                f"Last computation triggered by: "
                f"{self.state['last_triggered'] or 'none yet'}"
            )

            ui.separator()
            ui.button(
                'Run PLOT computation (2 s)',
                on_click=self.run,
            )

    async def run(self):
        await asyncio.sleep(2)

        self.state['plot_runs'] += 1
        self.state['last_triggered'] = 'Plot'

        self.show.refresh()

        # refresh following tab (4 only)
        self.propagate(from_tab=3)


class SettingsTab:
    def __init__(self, state: dict, propagate):
        self.state = state
        self.propagate = propagate

    @ui.refreshable
    def show(self):
        with ui.column():
            ui.label('Settings Tab').classes('text-xl font-bold mb-2')

            ui.label(f"Overview computations: {self.state['overview_runs']}")
            ui.label(f"Processing computations: {self.state['processing_runs']}")
            ui.label(f"Plot computations: {self.state['plot_runs']}")
            ui.label(f"Settings computations: {self.state['settings_runs']}")

            ui.separator()
            ui.label(
                f"Last computation triggered by: "
                f"{self.state['last_triggered'] or 'none yet'}"
            )

            ui.separator()
            ui.button(
                'Run SETTINGS computation (2 s)',
                on_click=self.run,
            )

    async def run(self):
        await asyncio.sleep(2)

        self.state['settings_runs'] += 1
        self.state['last_triggered'] = 'Settings'

        self.show.refresh()
        # no further cascade


# ---------------- MAIN PAGE CLASS ---------------- #

class MainPage:

    def __init__(self):
        # Shared application state across all tabs
        self.state = {
            'overview_runs': 0,
            'processing_runs': 0,
            'plot_runs': 0,
            'settings_runs': 0,
            'last_triggered': None,
        }

        # create tabs
        self.overview = OverviewTab(self.state, self.propagate)
        self.processing = ProcessingTab(self.state, self.propagate)
        self.plot = PlotTab(self.state, self.propagate)
        self.settings = SettingsTab(self.state, self.propagate)

        # build UI
        self.build_ui()

    def build_ui(self):
        ui.label('Class-based NiceGUI app with cascading tab refresh') \
            .classes('text-2xl font-bold my-4')

        tabs = ui.tabs().classes('mb-4')
        with tabs:
            tab1 = ui.tab('Overview')
            tab2 = ui.tab('Processing')
            tab3 = ui.tab('Plot')
            tab4 = ui.tab('Settings')

        with ui.tab_panels(tabs, value=tab1):
            with ui.tab_panel(tab1):
                self.overview.show()

            with ui.tab_panel(tab2):
                self.processing.show()

            with ui.tab_panel(tab3):
                self.plot.show()

            with ui.tab_panel(tab4):
                self.settings.show()

    def propagate(self, from_tab: int):
        """Refresh following tabs depending on where the action started."""
        if from_tab <= 1:
            self.processing.show.refresh()
        if from_tab <= 2:
            self.plot.show.refresh()
        if from_tab <= 3:
            self.settings.show.refresh()


# ---------------- ENTRY POINT ---------------- #

@ui.page('/')
def main():
    # NOTE: do NOT call this yourself anywhere else.
    MainPage()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
