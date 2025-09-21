from dataclasses import asdict

from nicegui import events, ui, run
from typing import List
import os
import pandas as pd

from app.webapp.config import ProcessingConfig
from app.webapp.core import extract_df, extract_df_dict, validate_table
from app.webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from app.webapp.processing import TableProcessor, get_possible_models
from app.webapp.kinetic_model import InitConditions
import logging
from io import BytesIO
from app.webapp.gui.styled_elements.label import StyledLabel
from app.webapp.gui.utils.tables import sanitize_df_for_table
from app.webapp.gui.utils.plots import update_plot_image
from app.webapp.gui.utils.latex import LatexLabel, typeset_latex
from app.webapp.gui.abstract_tab import AbstractTab

from concurrent.futures import ThreadPoolExecutor
import asyncio


class ProcessingTab(AbstractTab):
    def __init__(self, processor: TableProcessor = None, executor: ThreadPoolExecutor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.executor = executor
        self.params_gui = {'expansion': True}
        self.params = ProcessingConfig()

    @ui.refreshable
    def __call__(self, *args, **kwargs):
        with ui.column().classes('justify-center w-full'):
            if not self.processor.has_data():
                with ui.row().classes('justify-center w-full'):
                    StyledLabel('Tabulky ještě nejsou načteny', status='warning')
                    return
        with ui.row().classes('justify-center w-full'):
            ui.label('Tabulky').classes('text-lg font-semibold')
        with ui.row().classes('justify-center w-full'):
            StyledLabel('Zobrazené grafy jsou pouze pro validaci výsledků. Grafy pro export jsou na další záložce.',
                        status='info')
        self.display_processing_options()
        self.display_processing()

    def recompute_button(self):
        # TODO: disable button
        # with disable(button):
        button = ui.button('Spočítat/přepočítat',
                              icon='🔁',
                           on_click=lambda e: self.display_processing.refresh())
        button.props('unelevated color=primary')
        # button.style('font-size: 1.2rem; padding: 0.6rem 1.2rem; min-width: 160px;')

    @ui.refreshable
    def display_processing_options(self):
        with ui.row().classes('justify-center w-full'):
            self.recompute_button()
        with ui.row().classes('justify-center w-full'):
            with ui.expansion('Nastavení parametrů pro výpočet', icon='⚙️'
                              ).classes('w-full rounded-lg shadow-sm p-4 bg-neutral-100 dark:bg-neutral-800'
                                        ).bind_value(self.params_gui, 'expansion'):

                with ui.row().classes('justify-center w-full'):
                    with ui.column().tooltip('Vyberte jen modely, které chcete spočítat. Více modelů snižuje rychlost výpočtu'):
                        ui.label('Modely, které se budou počítat:')
                        for c_item in get_possible_models():
                            ui.checkbox(c_item, value=c_item in self.params.models_to_compute,
                                        on_change=lambda e, item=c_item: self.on_toggle(e, item))

                    with ui.column():
                        options = list(InitConditions)
                        labels = [opt.name for opt in options]
                        if self.params.initialization is None:
                            self.params.initialization = labels[1]
                        help_text = "### Nastavení volby\n\n"
                        help_text += "\n".join([f"- **{opt.name.upper()}**: {opt.description}" for opt in options])
                        help_text += "\nDoporučeno: Použít **TIME_SHIFT**"
                        with ui.select(options=labels,
                                  with_input=False,
                                  label="Zvolte způsob inicializace:",
                                  # on_change=lambda e: self.display_processing.refresh(),
                                  ) as select:
                            select.bind_value(self.params, 'initialization').classes('w-80')
                            with ui.tooltip():
                                ui.markdown(help_text)

                        ui.checkbox('Time shift automaticky',
                                    value=self.params.optim_time_shift,
                                    # on_change=lambda e: self.display_processing.refresh()
                                    ).bind_value(self.params, 'optim_time_shift')
                        tmp_param = {'manual': not self.params.optim_time_shift}
                        with ui.column().bind_visibility_from(tmp_param, 'manual'):
                            ui.number(label=r"Manuální nastavení t_0",
                                      min=0.01,
                                      max=20.0,
                                      precision=2,
                                      # on_change=lambda e: self.display_processing.refresh(),
                                      value=self.params.t_shift,
                                      ).bind_value(self.params, 't_shift').classes('w-80')


    def on_toggle(self, e, item):
        print(self.params.models_to_compute)
        print(e)
        if e.value:
            if item not in self.params.models_to_compute:
                self.params.models_to_compute.append(item)
        else:
            self.params.models_to_compute.remove(item)
        print(self.params.models_to_compute)

    def compute_all_kinetics(self, params_dict):
        self.processor.compute_all_kinetics(**params_dict)

    @ui.refreshable
    async def display_processing(self):
        spinner = ui.spinner(size='lg', color='primary')
        spinner.visible = True
        # Pass parameters to compute using run_in_executor
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.compute_all_kinetics,
            asdict(self.params)
        )
        ui.notify("DONE")
        spinner.visible = False

        with ui.column().classes('justify-center w-full'):
            if hasattr(self.processor, "k_models"):
                for name, model in self.processor.k_models.items():
                    with ui.row().classes('justify-center w-full'):
                        ui.markdown(f"#### Výsledky pro {name}")

                    with ui.row().classes('justify-center w-full'):
                        if model.k_fit is not None:
                            # Simulate and plot
                            try:
                                fig = model.plot_debug(ui=True, legend_mode='components_only')
                                image_element = ui.image().classes('justify-center max-w-xl')
                                update_plot_image(fig, image_element)
                            except Exception as e:
                                import traceback
                                StyledLabel("Chyba při vykreslování grafu:", 'error')
                                ui.label(traceback.format_exc())

                            # latex_eq = r",\quad ".join([f"k_{{{i + 1}}} = {k:.5f}" for i, k in enumerate(model.k_fit)])
                            latex_eq = r"$$\begin{aligned}" + "\n" + "\n".join(
                                [r"k_{" + f"{idx + 1}" + r"}" + f" = {k['latex']} &= {k['value']:.8f} \\\\" for idx, k in
                                 enumerate(model.get_constants_with_names())]
                            ) + r"\end{aligned}$$"
                            print(latex_eq)
                            LatexLabel(rf"""{latex_eq}""")

                            ui.separator()
                        else:
                            StyledLabel("Model nebyl fitován.", status='warning')
            else:
                StyledLabel("Kinetické modely nebyly spočítány.", status='warning')

        self.call_refreshable_elements()
