from __future__ import annotations

import io
import zipfile
from typing import Dict

import matplotlib.pyplot as plt
from nicegui import ui

from abstract_tab import AbstractTab
from utils import safe_float
from ui_helpers import info, warning, success

class GraphTab(AbstractTab):
    name = "graphs"

    def __init__(self, orchestrator: "Kinetika") -> None:
        super().__init__(orchestrator)
        self._auto_plot = True
        self._img_cache: Dict[str, bytes] = {}

    def is_accessible(self) -> bool:
        # Přístupné až když existují spočítané modely
        return bool(self.orchestrator.processor.kinetics)

    async def load_from_db(self) -> None:
        # per-model graf config se načítá lazy až když jsou modely
        return

    @ui.refreshable
    def show(self) -> None:
        ui.markdown("## Graphs (Krok 4) – Finální grafy + nastavení")

        kin = self.orchestrator.processor.kinetics
        if not kin:
            warning("Nejsou spočítané modely. Nejdřív Processing → Spočítat.")
            # reset graf konfigurací, když nic není
            self.orchestrator.db.set_path(("graphs", "configs"), {})
            return

        ui.markdown("Dummy grafy (matplotlib). Konfig se ukládá do SQLite při každé změně.")

        with ui.row().classes("w-full gap-6"):
            with ui.column().classes("w-2/3"):
                for model_name in kin.keys():
                    img = self._render_plot(model_name)
                    ui.markdown(f"### {model_name}")
                    ui.image(img).classes("w-full")
            with ui.column().classes("w-1/3"):
                ui.markdown("### Nastavení grafů (per-model)")
                for model_name in kin.keys():
                    self._model_controls(model_name)

                ui.separator()
                ui.markdown("### Export")
                ui.button("Exportovat všechny grafy do ZIP (png)", on_click=self._export_zip).props("outline")

    def _default_cfg(self) -> dict:
        return {
            "show_title": True,
            "title": "Dummy plot",
            "fig_width": 7.0,
            "fig_height": 4.0,
            "xlabel": "t",
            "ylabel": "y",
        }

    def _get_cfg(self, model: str) -> dict:
        configs = self.orchestrator.db.get_path(("graphs", "configs"), {}) or {}
        cfg = configs.get(model)
        if not isinstance(cfg, dict):
            cfg = self._default_cfg()
            configs[model] = cfg
            self.orchestrator.db.set_path(("graphs", "configs"), configs)
        return cfg

    def _set_cfg(self, model: str, key: str, value) -> None:
        configs = self.orchestrator.db.get_path(("graphs", "configs"), {}) or {}
        if model not in configs or not isinstance(configs[model], dict):
            configs[model] = self._default_cfg()
        configs[model][key] = value
        self.orchestrator.db.set_path(("graphs", "configs"), configs)
        self.orchestrator.db.log_step("graphs", "graph_setting_changed", {"model": model, "key": key, "value": value})
        if self._auto_plot:
            self.refresh()

    def _model_controls(self, model: str) -> None:
        cfg = self._get_cfg(model)

        with ui.expansion(f"{model} – Nastavení grafu", value=False).classes("w-full"):
            ui.checkbox(
                "show_title",
                value=bool(cfg.get("show_title", True)),
                on_change=lambda e, m=model: self._set_cfg(m, "show_title", bool(e.value)),
            )
            ui.input(
                "title",
                value=str(cfg.get("title", "")),
                on_change=lambda e, m=model: self._set_cfg(m, "title", str(e.value)),
            ).props("dense")

            ui.number(
                "fig_width",
                value=float(cfg.get("fig_width", 7.0)),
                min=3.0,
                max=20.0,
                step=0.1,
                on_change=lambda e, m=model: self._set_cfg(m, "fig_width", safe_float(e.value, 7.0)),
            ).props("dense")

            ui.number(
                "fig_height",
                value=float(cfg.get("fig_height", 4.0)),
                min=2.0,
                max=20.0,
                step=0.1,
                on_change=lambda e, m=model: self._set_cfg(m, "fig_height", safe_float(e.value, 4.0)),
            ).props("dense")

            ui.input(
                "xlabel",
                value=str(cfg.get("xlabel", "t")),
                on_change=lambda e, m=model: self._set_cfg(m, "xlabel", str(e.value)),
            ).props("dense")

            ui.input(
                "ylabel",
                value=str(cfg.get("ylabel", "y")),
                on_change=lambda e, m=model: self._set_cfg(m, "ylabel", str(e.value)),
            ).props("dense")

    def _render_plot(self, model: str) -> bytes:
        cfg = self._get_cfg(model)
        res = self.orchestrator.processor.kinetics[model]

        w = float(cfg.get("fig_width", 7.0))
        h = float(cfg.get("fig_height", 4.0))

        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111)

        ax.plot(res.debug_df["t"], res.debug_df["y_obs"], label="obs")
        ax.plot(res.debug_df["t"], res.debug_df["y_fit"], label="fit")

        ax.set_xlabel(str(cfg.get("xlabel", "t")))
        ax.set_ylabel(str(cfg.get("ylabel", "y")))
        if bool(cfg.get("show_title", True)):
            ax.set_title(str(cfg.get("title", model)))

        ax.legend()

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)

        b = buf.getvalue()
        self._img_cache[model] = b
        return b

    def _export_zip(self) -> None:
        kin = self.orchestrator.processor.kinetics
        if not kin:
            ui.notify("Není co exportovat.", type="warning")
            return

        # zajistíme aktuální rendery
        for m in kin.keys():
            self._render_plot(m)

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for model, img in self._img_cache.items():
                zf.writestr(f"{model}.png", img)

        data = zbuf.getvalue()
        ui.download(data, filename="graphs_export.zip")
        self.orchestrator.db.log_step("graphs", "export_zip", {"models": list(kin.keys()), "bytes": len(data)})
