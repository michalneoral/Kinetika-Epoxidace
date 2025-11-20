from abc import ABC, abstractmethod
from nicegui import events, ui


class AbstractTab(ABC):
    def __init__(self, theme=None, propagate=None, tab_number=None, debug=False):
        super().__init__()
        self.theme = theme
        self.refreshable_elements = []
        self._configs = {}
        self.propagate = propagate
        self.tab_number = tab_number

        self._refresh_scheduled = False
        self._allow_propagate = True

        self.debug_propagation_counter = 0
        self.debug = debug

    # @ui.refreshable
    # @abstractmethod
    # def __call__(self, *args, **kwargs):
    #     raise NotImplementedError

    @ui.refreshable
    @abstractmethod
    def show(self, *args, **kwargs):
        raise NotImplementedError

    def propagation_disabled(self):
        self._allow_propagate = False

    def propagation_enabled(self):
        self._allow_propagate = True

    def propagate_changes(self, text=None):
        if self.propagate is not None and self._allow_propagate:
            self.debug_propagation_counter += 1
            if self.debug:
                print(f' -- Propagating changes from tab: {self.tab_number} with {self.debug_propagation_counter}')
                if text is not None:
                    print(f' -- Propagating changes from method {text}')
            self.propagate(self.tab_number)

    def add_refreshable_element(self, refreshable_element):
        self.refreshable_elements.append(refreshable_element)

    def set_refreshable_elements(self, refreshable_elements):
        self.refreshable_elements = refreshable_elements

    def call_refreshable_elements(self):
        if self.debug:
            print(f'Calling refreshable elements from tab: {self.tab_number}')
        for refreshable_elements in self.refreshable_elements:
            refreshable_elements.refresh()

    def _do_refresh(self):
        self._refresh_scheduled = False
        if self.debug:
            print(f' -- Refreshing tab: {self.tab_number}')
        self.show.refresh()
        # if you have nested refreshables, you could also call them here

    def refresh(self, coalesce: bool = True):
        """Schedule a refresh; if coalesce=True, multiple calls collapse into one."""
        if not coalesce:
            self._do_refresh()
            return

        if self._refresh_scheduled:
            return  # there is already a refresh scheduled

        self._refresh_scheduled = True
        # run at next iteration of the event loop; multiple calls before then are ignored
        ui.timer(0, self._do_refresh, once=True)

    def get_configs(self):
        return self._configs

    def set_configs(self, configs):
        self._configs = configs

    def add_config(self, name, config):
        self._configs[name] = config


