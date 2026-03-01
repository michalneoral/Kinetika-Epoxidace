from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from nicegui import ui

from abstract_tab import AbstractTab
from utils import safe_float
from ui_helpers import info, warning

EXEC = ThreadPoolExecutor(max_workers=1)


class ProcessingTab(AbstractTab):
    name = "processing"

    def __init__(self, orchestrator: "Kinetika") -> None:
        super().__init__(orchestrator)
        self.models = ["Model_A", "Model_B", "Model_C"]
        self.models_to_compute = set(self.models)

        self.init_mode = "Auto"
        self.optim_time_shift = True
        self.t_shift = 1.0

        self.settings_open = True
        self._running = False

    def is_accessible(self) -> bool:
        return self.orchestrator.processor.has_data()

    async def load_from_db(self) -> None:
        db = self.orchestrator.db
        self.models_to_compute = set(db.get_path(("processing", "models_to_compute"), self.models))
        self.init_mode = db.get_path(("processing", "init_mode"), "Auto")
        self.optim_time_shift = bool(db.get_path(("processing", "optim_time_shift"), True))
        self.t_shift = float(db.get_path(("processing", "t_shift"), 1.0))
        self.settings_open = bool(db.get_path(("processing", "settings_open"), True))

    @ui.refreshable
    def show(self) -> None:
        ui.markdown("## Processing (Krok 3) – Výpočet kinetik")

        if not self.orchestrator.processor.has_data():
            warning("Nejdřív nahraj data v Input tabu.")
            return

        with ui.expansion("Nastavení výpočtu", value=self.settings_open).classes("w-full") as exp:
            exp.on("update:model-value", lambda e: self._set("settings_open", bool(e.value)))

            ui.markdown("### Modely k výpočtu")
            with ui.row().classes("gap-6"):
                for m in self.models:
                    ui.checkbox(
                        m,
                        value=(m in self.models_to_compute),
                        on_change=lambda e, model=m: self._toggle_model(model, bool(e.value)),
                    )

            ui.separator()
            ui.markdown("### Inicializace")
            ui.select(
                options=["Auto", "FromData", "ManualDummy"],
                value=self.init_mode,
                label="InitConditions (dummy enum)",
                on_change=lambda e: self._set("init_mode", e.value),
            ).classes("w-72")

            ui.separator()
            ui.checkbox(
                "Optimalizovat time shift",
                value=self.optim_time_shift,
                on_change=lambda e: self._set("optim_time_shift", bool(e.value)),
            )

            if not self.optim_time_shift:
                ui.number(
                    "t_shift (manuální)",
                    value=self.t_shift,
                    min=0.01,
                    max=20.0,
                    step=0.01,
                    on_change=lambda e: self._set("t_shift", safe_float(e.value, 1.0)),
                ).classes("w-72")

        ui.separator()

        btn = ui.button("Spočítat / Přepočítat", on_click=self._compute)
        if self._running:
            btn.disable()
            ui.spinner(size="lg")
            ui.label("Probíhá výpočet… (dummy)").classes("text-sm")

        ui.separator()
        self._render_results()

    def _render_results(self) -> None:
        kin = self.orchestrator.processor.kinetics
        if not kin:
            info("Zatím nejsou výsledky. Klikni na Spočítat.")
            return

        ui.markdown("### Výsledky (dummy)")
        for name, res in kin.items():
            with ui.expansion(f"{name} – parametry a debug", value=False).classes("w-full"):
                ui.markdown("**Fitované konstanty (dummy):**")
                ui.code("\n".join([f"{k}: {v}" for k, v in res.params.items()]), language="text")
                ui.markdown("**Debug data (head):**")
                ui.table.from_pandas(res.debug_df.head(20)).classes("w-full")

    async def _compute(self) -> None:
        if self._running:
            return
        self._running = True
        self.refresh()

        models = sorted(self.models_to_compute)
        init_mode = self.init_mode
        optim = self.optim_time_shift
        t_shift = self.t_shift

        self.orchestrator.db.log_step(
            "processing",
            "compute_clicked",
            {"models": models, "init_mode": init_mode, "optim_time_shift": optim, "t_shift": t_shift},
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            EXEC,
            lambda: self.orchestrator.processor.compute_all_kinetics(
                models_to_compute=models,
                init_mode=init_mode,
                optim_time_shift=optim,
                t_shift=t_shift,
            ),
        )

        self._running = False
        self.refresh()
        self.propagate_changes(from_tab=3)

    def _set(self, key: str, value) -> None:
        setattr(self, key, value)
        self.orchestrator.db.set_path(("processing", key), value)
        self.orchestrator.db.log_step("processing", "setting_changed", {"key": key, "value": value})
        self.refresh()

    def _toggle_model(self, model: str, checked: bool) -> None:
        if checked:
            self.models_to_compute.add(model)
        else:
            self.models_to_compute.discard(model)
        self.orchestrator.db.set_path(("processing", "models_to_compute"), sorted(self.models_to_compute))
        self.orchestrator.db.log_step("processing", "model_toggled", {"model": model, "enabled": checked})
        self.refresh()
