from nicegui import events, ui, run
from typing import List
import os
import pandas as pd

from app.webapp.core import extract_df, extract_df_dict, validate_table
from app.webapp.gui.styled_elements.custom_color_picker import CustomColorPicker, ColorPickerButton
from app.webapp.gui.utils.context import disable
from app.webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from app.webapp.processing import TableProcessor
from app.webapp.kinetic_model import InitConditions
import logging
from io import BytesIO
from app.webapp.gui.styled_elements.label import StyledLabel
from app.webapp.gui.utils.tables import sanitize_df_for_table
from app.webapp.gui.utils.plots import update_plot_image, export_all_figures_as_zip
from app.webapp.gui.utils.latex import LatexLabel, typeset_latex
from app.webapp.advanced_plotting import AdvancedPlotter
from app.webapp.config import SingleCurveConfig, AdvancedGraphConfig, default_colors, GridConfig
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from app.webapp.gui.abstract_tab import AbstractTab
import asyncio
import httpx


class GraphTab(AbstractTab):
    def __init__(self, processor: TableProcessor = None, executor: ThreadPoolExecutor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.executor = executor
        self.params = {
            'expansion': True,
        }

        self.plotters = {name: AdvancedPlotter(name, model) for name, model in self.processor.k_models.items()}
        self.plotter_config = {}
        self.plots = {}
        self.plot_enabled = False

    @ui.refreshable
    def __call__(self, *args, **kwargs):
        with ui.column().classes('justify-center w-full'):
            if not self.processor.has_data() or len(self.processor.k_models) == 0:
                with ui.row().classes('justify-center w-full'):
                    StyledLabel('Data pro grafy ještě nejsou načteny', status='warning')
                    self.plotter_config = {}
                    return
            self.plotters = {name: AdvancedPlotter(name, model) for name, model in self.processor.k_models.items()}
            with ui.row().classes('justify-center w-full'):
                ui.label('Grafy').classes('text-lg font-semibold')
            with ui.column().classes('justify-center w-full'):
                self.create_plot_settings()
                self.disable_plot()
                for name, plotter in self.plotters.items():
                    with ui.row().classes('w-full flex-wrap gap-4 justify-center'):
                        # LEFT SIDE: Image
                        with ui.column().classes('flex-1 min-w-[300px] items-end'):
                            image_element = ui.image().classes('max-w-xl')
                            self.plots[plotter.name] = image_element

                        # RIGHT SIDE: Expandable controls
                        with ui.column().classes('flex-1 min-w-[300px] items-start'):
                            with ui.expansion('⚙️ Nastavení grafu', value=False):
                                self.create_plot_controls(name)
                    ui.separator()

                self.enable_plot()
                for name, plotter in self.plotters.items():
                    self.plot_graph(plotter)

        self.save_button()

    async def save_all_handler(self, button: ui.button, save_format: str = 'png') -> None:
        # with disable(button):
        with ui.dialog() as dialog, ui.card():
            ui.label('⏳ Exportuji grafy...')
            ui.spinner()
            dialog.open()

            await export_all_figures_as_zip(self.plotters, self.plotter_config, save_format=save_format)

            dialog.close()
        ui.notify('✅ Grafy připraveny ke stažení')

    # def save_button(self):
    #     with ui.page_sticky(x_offset=18, y_offset=18):
    #         ui.button(icon='save',
    #                   on_click=lambda e: self.save_all_handler(e.sender),
    #                   ).props('fab color=secondary')

    def save_button(self):
        with ui.page_sticky(x_offset=18, y_offset=18):
            with ui.element('q-fab').props(
                    'icon=save color=secondary direction="up" vertical-actions-align="right"').classes('z-top') as el:
                ui.element('q-fab-action').props('icon=image color=blue-5 label="png"') \
                    .on('click', lambda: self.save_all_handler(el, 'png'))
                ui.element('q-fab-action').props('icon=description color=red-5 label="pdf"') \
                    .on('click', lambda: self.save_all_handler(el, 'pdf'))
                ui.element('q-fab-action').props('icon=photo_album color=purple-5 label="svg"') \
                    .on('click', lambda: self.save_all_handler(el, 'svg'))

    def create_plot_controls(self, name):
        all_controls = []
        c_config = self.plotter_config[name]

        if hasattr(self.processor, "k_models"):
            model = self.processor.k_models[name]
            with ui.row().classes('justify-center w-full'):
                ui.markdown(f"#### Výsledky pro {name}")
            if model.k_fit is not None:
                # latex_eq = r",\quad ".join([f"k_{{{i + 1}}} = {k:.5f}" for i, k in enumerate(model.k_fit)])
                latex_eq = r"$$\begin{aligned}" + "\n" + "\n".join(
                    [r"k_{" + f"{idx + 1}" + r"}" + f" = {k['latex']} &= {k['value']:.8f} \\\\"
                     for idx, k in
                     enumerate(model.get_constants_with_names())]
                ) + r"\end{aligned}$$"
                LatexLabel(rf"""{latex_eq}""")
                for idx, k in enumerate(model.get_constants_with_names()):
                    latex_eq_pure = ("k_{" + f"{idx + 1}" + "}" + f" = {k['latex']} = {k['value']:.8f}" +
                                     " \\mathrm{min}^{-1}")
                    ui.label(latex_eq_pure)

        with ui.row().classes('justify-start w-full'):
            ui.markdown('#### Hlavní nastavení grafu:')

        with ui.row().classes('justify-start w-full'):
            # SHOW TITLE
            show_title = ui.toggle({True: "Zobrazit nadpis", False: "Bez nadpisu"},
                                   on_change=lambda e, c_name=name: self.handle_change(e, c_name), value=True)
            show_title.bind_value(c_config, 'show_title')
            all_controls.append(show_title)

            # TITLE
            title = ui.input(f'Nadpis grafu {name}', on_change=lambda e, c_name=name: self.handle_change(e, c_name))
            title.classes('w-64')
            title.bind_value(c_config, 'title')
            title.bind_enabled_from(show_title, 'value')
            all_controls.append(title)

            # LEGEND MODE
            legend_mode = ui.select(AdvancedGraphConfig.get_choices('legend_mode'),
                                    label='Nastavení legendy',
                                    value=c_config.legend_mode,
                                    on_change=lambda e, c_name=name: self.handle_change(e, c_name))
            legend_mode.bind_value(c_config, 'legend_mode')
            all_controls.append(legend_mode)

        with ui.row().classes('justify-start w-full'):
            fig_width = ui.number('Šířka figure (inch)',
                                  value=c_config.fig_width,
                                  min=1, max=20, step=0.5,
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name))
            fig_width.classes('w-32')
            fig_width.bind_value(c_config, 'fig_width')
            all_controls.append(fig_width)

            fig_height = ui.number('Výška figure (inch)',
                                   value=c_config.fig_height,
                                   min=1, max=20, step=0.5,
                                   on_change=lambda e, c_name=name: self.handle_change(e, c_name))
            fig_height.classes('w-32')
            fig_height.bind_value(c_config, 'fig_height')
            all_controls.append(fig_height)

        with ui.row().classes('justify-start w-full'):
            ui.markdown('#### Grid - Mřížka')
        with ui.row().classes('justify-start w-full'):
            picker = ColorPickerButton(icon='colorize',
                                       color=c_config.grid_config.color,
                                       color_type='hexa',
                                       on_pick=lambda e, c_name=name: self.handle_change(e, c_name))
            picker.bind_color(c_config.grid_config, 'color')
            all_controls.append(picker)

            show_grid = ui.toggle({True: "Hlavní mřížka", False: "Bez mřížky"},
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name), value=True)
            show_grid.bind_value(c_config.grid_config, 'visible')
            all_controls.append(show_grid)

            grid_axis = ui.select(label='Osy hlavní mřížky',
                                  options=GridConfig.get_choices('axis'),
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                  value=c_config.grid_config.axis)
            grid_axis.classes('w-64')
            grid_axis.bind_value(c_config.grid_config, 'axis')
            all_controls.append(grid_axis)

        with ui.row().classes('justify-start w-full'):
            picker = ColorPickerButton(icon='colorize',
                                       color=c_config.grid_config_minor.color,
                                       color_type='hexa',
                                       on_pick=lambda e, c_name=name: self.handle_change(e, c_name))
            picker.bind_color(c_config.grid_config_minor, 'color')
            all_controls.append(picker)
            show_grid = ui.toggle({True: "Vedlejší mřížka", False: "Bez mřížky"},
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name), value=True)
            show_grid.bind_value(c_config.grid_config_minor, 'visible')
            all_controls.append(show_grid)

            grid_axis = ui.select(label='Osy vedlejší mřížky',
                                  options=GridConfig.get_choices('axis'),
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                  value=c_config.grid_config_minor.axis)
            grid_axis.classes('w-64')
            grid_axis.bind_value(c_config.grid_config_minor, 'axis')
            all_controls.append(grid_axis)

        with ui.row().classes('justify-start w-full'):
            ui.markdown('#### Osy:')

        with ui.row().classes('justify-start w-full'):
            label = ui.input(f'Popisek osy-x',
                             on_change=lambda e, c_name=name: self.handle_change(e, c_name))
            label.bind_value(c_config, 'xlabel')
            all_controls.append(label)

            xlim_mode = ui.select(label='Osa X - nastavení limitace',
                                  options=AdvancedGraphConfig.get_choices('xlim_mode'),
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                  value=c_config.xlim_mode)
            xlim_mode.classes('w-64')
            xlim_mode.bind_value(c_config, 'xlim_mode')
            all_controls.append(xlim_mode)

            xlim_min = ui.number('Min osa-X', value=c_config.xlim_min,
                                 min=0,
                                 on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                 )
            xlim_min.bind_value(c_config, 'xlim_min')
            xlim_min.bind_enabled_from(xlim_mode, 'value', backward=lambda mode: mode == 'manual')
            all_controls.append(xlim_min)

            xlim_max = ui.number('Max osa-X', value=c_config.xlim_max,
                                 min=0,
                                 on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                 )
            xlim_max.bind_value(c_config, 'xlim_max')
            xlim_max.bind_enabled_from(xlim_mode, 'value', backward=lambda mode: mode == 'manual')
            all_controls.append(xlim_max)

        with ui.row().classes('justify-start w-full'):
            label = ui.input(f'Popisek osy-y',
                             on_change=lambda e, c_name=name: self.handle_change(e, c_name))
            label.bind_value(c_config, 'ylabel')
            all_controls.append(label)

            ylim_mode = ui.select(label='Osa Y - nastavení limitace',
                                  options=AdvancedGraphConfig.get_choices('ylim_mode'),
                                  on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                  value=c_config.ylim_mode)
            ylim_mode.classes('w-64')
            ylim_mode.bind_value(c_config, 'ylim_mode')
            all_controls.append(ylim_mode)

            ylim_min = ui.number('Min osa-Y', value=c_config.ylim_min,
                                 min=0,
                                 on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                 )
            ylim_min.bind_value(c_config, 'ylim_min')
            ylim_min.bind_enabled_from(ylim_mode, 'value', backward=lambda mode: mode == 'manual')
            all_controls.append(ylim_min)

            ylim_max = ui.number('Max osa-Y', value=c_config.ylim_max,
                                 min=0,
                                 on_change=lambda e, c_name=name: self.handle_change(e, c_name),
                                 )
            ylim_max.bind_value(c_config, 'ylim_max')
            ylim_max.bind_enabled_from(ylim_mode, 'value', backward=lambda mode: mode == 'manual')
            all_controls.append(ylim_max)

        with ui.row().classes('justify-start w-full'):
            ui.markdown('#### Jednotlivé křivky:')

        with ui.column().classes('justify-start w-full'):
            for i in range(len(c_config.curve_styles)):
                c_curve_config = c_config.curve_styles[i]
                with ui.row().classes('justify-start w-full'):
                    # COLOR PICKER
                    picker = ColorPickerButton(icon='colorize',
                                               color=c_curve_config.color,
                                               color_type='rgba',
                                               on_pick=lambda e, c_name=name: self.handle_change(e, c_name))
                    picker.bind_color(c_curve_config, 'color')
                    all_controls.append(picker)

                    # LABEL
                    label = ui.input(f'Popisek',
                                     on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    label.bind_value(c_curve_config, 'label')
                    all_controls.append(label)

                    # LINE STYLE
                    linestyle = ui.select(options=SingleCurveConfig.get_choices('linestyle'),
                                          label="Styl křivky", value=c_curve_config.linestyle,
                                          on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    linestyle.classes('w-32')
                    linestyle.bind_value(c_curve_config, 'linestyle')
                    all_controls.append(linestyle)

                    # LINE WIDTH
                    linewidth = ui.number('Tloušťka čáry', value=c_curve_config.linewidth,
                                          min=0.1, max=None, step=0.1,
                                          on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    linewidth.bind_value(c_curve_config, 'linewidth')
                    all_controls.append(linewidth)

                    # MARKER
                    marker = ui.select(options=SingleCurveConfig.get_choices('marker'),
                                       label="Styl značky", value=c_curve_config.marker,
                                       on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    marker.classes('w-32')
                    marker.bind_value(c_curve_config, 'marker')
                    all_controls.append(marker)

                    # MARKER SIZE
                    markersize = ui.number('Velikost značku', value=c_curve_config.markersize,
                                           min=0.1, max=None, step=0.1,
                                           on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    markersize.bind_value(c_curve_config, 'markersize')
                    all_controls.append(markersize)

                ## PŘÍDAVNÝ TEXT
                with ui.row().classes('justify-start w-full'):
                    show_add_text = ui.toggle({True: "Přídavný text", False: "Bez textu"},
                                           on_change=lambda e, c_name=name: self.handle_change(e, c_name), value=c_curve_config.additional_text_enabled)
                    show_add_text.bind_value(c_curve_config, 'additional_text_enabled')
                    all_controls.append(show_add_text)

                    # LABEL
                    add_text_text = ui.input(f'Přídavný popisek',
                                     on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    add_text_text.bind_value(c_curve_config, 'additional_text_text')
                    all_controls.append(add_text_text)

                    add_text_x = ui.number('Pozice x', value=c_curve_config.additional_text_x,
                                          min=0.0, max=None, step=1,
                                          on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    add_text_x.bind_value(c_curve_config, 'additional_text_x')
                    all_controls.append(add_text_x)

                    add_text_y = ui.number('Pozice y', value=c_curve_config.additional_text_y,
                                           min=0.0, max=None, step=0.025,
                                           on_change=lambda e, c_name=name: self.handle_change(e, c_name))
                    add_text_y.bind_value(c_curve_config, 'additional_text_y')
                    all_controls.append(add_text_y)

        # c_input.on('click', lambda: ui.notify('You clicked the button B.'))
        # c_input.on('focus', lambda: ui.notify('You focus the button B.'))
        # c_input.on('blur', lambda: ui.notify('You blur (unfocus) the button B.'))

    def handle_color_change(self, e, name, button):
        print('handle color change')
        button.classes(f'!bg-[{e.color}]')
        print(self.plotter_config[name])
        self.handle_change(e, name)

    def handle_change(self, e, name):
        if self.plot_enabled:
            plotter = self.plotters[name]
            self.plot_graph(plotter)

    def enable_plot(self):
        self.plot_enabled = True

    def disable_plot(self):
        self.plot_enabled = False

    def plot_graph(self, plotter):
        # print(self.plotter_config[plotter.name])

        image_element = self.plots[plotter.name]
        fig = plotter.plot(
            **asdict(self.plotter_config[plotter.name]),
            ui=True
        )
        update_plot_image(fig, image_element)

    def create_plot_settings(self):
        if not self.processor.has_data():
            return
        for name, plotter in self.plotters.items():
            n_curves = plotter.get_n_curves()
            if name in self.plotter_config:
                continue
            curve_styles = []
            for i in range(n_curves):
                curve_styles.append(SingleCurveConfig(color=default_colors[i % len(default_colors)],
                                                      label=plotter.data['column_names'][i]
                                                      ))
            plotter_config = AdvancedGraphConfig(title=name,
                                                 curve_styles=curve_styles)
            self.plotter_config[name] = plotter_config

        # print(self.plotter_config)
