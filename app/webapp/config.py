from pathlib import Path

from nicegui import binding, ui
from typing import Literal, get_args, List, get_type_hints
from enum import Enum
# from attrs import define, evolve, asdict, Factory

import yaml
from dataclasses import field, dataclass, asdict, is_dataclass
from abc import ABC, abstractmethod
from webapp.utils import StrEnum
from webapp.kinetic_model import InitConditions
from platformdirs import user_data_dir, user_documents_path
from webapp.processing import get_possible_models

from webapp.utils.logging import _LOGGER

default_colors = ["#FF0000", "#000000", "#5B0687", "#777777", "#0000FF", "#00FF00", "#00FFFF"]
default_colors = [d.lower() for d in default_colors]


def _get_converter(config_class):
    def convert(data):
        if isinstance(data, config_class):
            return data
        return config_class(**data)

    return convert


@binding.bindable_dataclass
class ABCconfig(ABC):
    # @classmethod
    # def get_choices(cls, name):
    #     return list(get_args(cls.__annotations__[name]))
    @classmethod
    def get_choices(cls, name):
        typ = get_type_hints(cls)[name]

        # Case 1: Enum field
        if isinstance(typ, type) and issubclass(typ, Enum):
            return [e.value for e in typ]

        # Case 2: Literal field
        literal_args = get_args(typ)
        if literal_args:
            return list(literal_args)

        raise TypeError(f"{name} must be annotated with an Enum or Literal")

    @classmethod
    def get_description(cls, name, choice=None):
        typ = get_type_hints(cls)[name]
        if choice is not None:
            return typ.get_description(choice)
        if isinstance(typ, type) and issubclass(typ, StrEnum):
            return {k: typ.get_description(k) for k in cls.get_choices(name)}
        # if isinstance(typ, type) and issubclass(typ, Enum):
        #     return None
        return None


@binding.bindable_dataclass
class GuiConfig(ABC):
    dark: Literal[True, False, None] = True


@binding.bindable_dataclass
class TablePickConfig(ABCconfig):
    row_start: int = 3
    row_end: int = 16
    col_start: int = 2
    col_end: int = 15


@binding.bindable_dataclass
class TablePickConfigFAME(TablePickConfig):
    row_start: int = 3
    row_end: int = 16
    col_start: int = 2
    col_end: int = 15


@binding.bindable_dataclass
class TablePickConfigEPO(TablePickConfig):
    row_start: int = 28
    row_end: int = 37
    col_start: int = 2
    col_end: int = 15


@binding.bindable_dataclass
class ProcessingConfig(ABC):
    initialization: InitConditions = InitConditions.TIME_SHIFT
    t_shift: float = 6.0
    optim_time_shift: bool = False
    # models_to_compute: List[str] = field(default_factory=lambda: get_possible_models())
    models_to_compute: List[str] = field(default_factory=lambda: ['C18:1_simplified'])

@binding.bindable_dataclass
class SingleCurveConfig(ABCconfig):
    linestyle: Literal["solid", "dashed", "dashdot", "dotted", "None"] = "solid"
    marker: Literal["s", "o", "v", "^", "D", "x", "+", "*", ".", "None"] = "s"
    color: str | None = None
    linewidth: float = 1.5
    markersize: float = 6.0
    label: str | None = None
    additional_text_enabled: bool = False
    additional_text_x: float = 0.5
    additional_text_y: float = 0.5
    additional_text_size: int = 14
    additional_text_text: str = 'Empty'

@binding.bindable_dataclass
class GridConfig(ABCconfig):
    visible: bool = False
    axis: Literal["both", "x", "y"] = "both"
    which: Literal["both", "major", "minor"] = "major"
    color: str | None = '#0f0f0f80'


@binding.bindable_dataclass
class AdvancedGraphConfig(ABCconfig):
    show_title: bool = True
    title: str = ''
    legend_mode: Literal["components_only", "both", "single", "None"] = "components_only"
    fig_width: int = 6
    fig_height: int = 6
    xlabel: str = 'čas t [min]'
    xlim_mode: Literal["default", "auto", "manual", "all_data"] = "default"
    xlim_min: float = 0.0
    xlim_max: float | None = None
    ylabel: str = 'koncentrace [-]'
    ylim_mode: Literal["default", "manual"] = "default"
    ylim_min: float | None = None
    ylim_max: float | None = None
    curve_styles: List[SingleCurveConfig] = field(default_factory=list)
    grid_config: GridConfig = field(default_factory=lambda: GridConfig(which='major', visible=True))
    grid_config_minor: GridConfig = field(default_factory=lambda: GridConfig(which='minor',
                                                                             visible=False,
                                                                             color='#0f0f0f40'))


@dataclass
class ColorConfig:
    primary: str = "#98194e"
    secondary: str = "#3da345"


def from_dict(cls, d: dict):
    """Recursively convert nested dicts into dataclass instances."""
    if not is_dataclass(cls):
        return d

    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        value = d.get(f.name)
        if value is not None:
            if is_dataclass(f.type) and isinstance(value, dict):
                value = from_dict(f.type, value)
        else:
            value = f.default_factory() if f.default_factory is not None else None
        kwargs[f.name] = value
    return cls(**kwargs)


@binding.bindable_dataclass
class Config(ABCconfig):
    gui: GuiConfig = field(default_factory=GuiConfig)

    # def __init__(self):
    #     super().__init__()
    #     self.app_name = "Computation_App_UPCE_FCHT"
    #     self.sub_app_name = "Kinetika"
    #     self.default_config_name = 'recent_config.yml'
    #
    #     self.load()

    def update(self, config):
        """Update this config in-place with values from another config or dict."""
        if isinstance(config, dict):
            config = from_dict(type(self), config)

        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(config, f))

    def save(self, url: Path | None = None):
        data_dir = Path(user_data_dir(self.app_name, self.sub_app_name))
        config_dir = data_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        url = url if url is not None else (config_dir / self.default_config_name)

        with open(url, 'w') as f:
            yaml.dump(asdict(self), f, sort_keys=False)
        _LOGGER.info(f"Config saved to {url}")

    def load(self, url: Path | None = None):
        data_dir = Path(user_data_dir(self.app_name, self.sub_app_name))
        config_dir = data_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        default_config_path = config_dir / self.default_config_name

        url = url if url is not None else default_config_path

        _LOGGER.info(f"Loading config from {url}")
        if url.exists():
            try:
                with open(url) as config_file:
                    new_config = yaml.safe_load(config_file) or {}
                    self.update(new_config)
            except Exception as e:
                _LOGGER.error(f"Failed to load config from {url}: {e}")
        else:
            _LOGGER.warning(f"Config file {url} not found. Using default values.")
            self.save(url)
