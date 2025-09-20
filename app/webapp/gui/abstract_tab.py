from abc import ABC, abstractmethod
from nicegui import events, ui


class AbstractTab(ABC):
    def __init__(self, theme=None):
        super().__init__()
        self.theme = theme
        self.refreshable_elements = []
        self._configs = {}

    @ui.refreshable
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def add_refreshable_element(self, refreshable_element):
        self.refreshable_elements.append(refreshable_element)

    def set_refreshable_elements(self, refreshable_elements):
        self.refreshable_elements = refreshable_elements

    def call_refreshable_elements(self):
        for refreshable_elements in self.refreshable_elements:
            refreshable_elements.refresh()

    def refresh(self):
        self.__call__.refresh()
        # self.call_refreshable_elements()

    def get_configs(self):
        return self._configs

    def set_configs(self, configs):
        self._configs = configs

    def add_config(self, name, config):
        self._configs[name] = config


