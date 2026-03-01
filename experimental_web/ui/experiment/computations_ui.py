from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sqlite3

import pandas as pd
from nicegui import ui

from experimental_web.core.paths import DB_PATH
from experimental_web.data.repositories import ExperimentComputationRepository, ProcessedTablesRepository
from experimental_web.domain.processing_config import ProcessingConfig
from experimental_web.ui.experiment.compute_builder.draganddrop import DragState, Column, Card
from experimental_web.ui.experiment.compute_builder.graph import (
    GraphState,
    GraphController,
    create_graph_widget,
    build_compact_mermaid,
    normalize_graph_state,
)
from experimental_web.ui.experiment.compute_builder.ode_generator import generate_odes_model
from experimental_web.logging_setup import get_logger
from experimental_web.ui.instrumentation import wrap_ui_handler
from experimental_web.core.state import get_state
from experimental_web.ui.utils.staleness import compute_staleness
from experimental_web.ui.utils.tooltips import attach_tooltip


log = get_logger(__name__)


@dataclass(frozen=True)
class HeadItem:
    key: str
    title: str


def computations_block(experiment_id: int, params: Optional[ProcessingConfig] = None) -> None:
    """Zobrazení uložených výpočtů + dialog pro přidání/upravení (ukládá se do DB pro experiment)."""

    params = params or ProcessingConfig()

    # Session state is used to trigger lightweight refreshes when other tabs
    # (or background computations) update persisted data.
    st = get_state()

    repo = ExperimentComputationRepository(DB_PATH)
    tables_repo = ProcessedTablesRepository(DB_PATH)

    ui.add_head_html('''
<style>
  .comp-editor-root { background: #f5f5f5; color: #111; }
  .comp-editor-header { background: white; color: #111; }
  .body--dark .comp-editor-root { background: #121212; color: #fff; }
  .body--dark .comp-editor-header { background: #1d1d1d; color: #fff; }
  .body--dark .comp-editor-root .text-grey-8 { color: #cfcfcf !important; }
  .body--dark .comp-editor-root .text-grey-7 { color: #bdbdbd !important; }
</style>
''')

    delete_dialog = ui.dialog()
    dialog = ui.dialog().props("maximized")
    ode_dialog = ui.dialog()

    def load_available_tables() -> Dict[str, pd.DataFrame]:
        raw = tables_repo.load_latest_tables(experiment_id)
        out: Dict[str, pd.DataFrame] = {}
        for name, entry in raw.items():
            try:
                # entry může být tuple(df_json, text_md) nebo dict {df_json, text_md}
                df_json = entry.get("df_json") if isinstance(entry, dict) else entry[0]
                out[name] = pd.read_json(pd.io.common.StringIO(df_json), orient="split")
            except Exception:
                continue
        return out

    def auto_name(n: int) -> str:
        return f"Výpočet {n}"

    def _normalize_name_key(s: str) -> str:
        """Normalize a user-visible name for uniqueness checks.

        SQLite UNIQUE may be case-sensitive depending on collation. For UX we treat names
        case-insensitively and trim whitespace.
        """

        return (s or "").strip().casefold()

    def _make_unique_variant(base: str, existing_names: set[str]) -> str:
        """Return a non-colliding name by appending a numeric suffix if needed."""

        base = (base or "").strip()
        if not base:
            base = "Výpočet"

        existing_keys = {_normalize_name_key(x) for x in existing_names}
        if _normalize_name_key(base) not in existing_keys:
            return base

        i = 2
        while True:
            candidate = f"{base} ({i})"
            if _normalize_name_key(candidate) not in existing_keys:
                return candidate
            i += 1

    def _make_unique_auto_name(existing_names: set[str]) -> str:
        existing_keys = {_normalize_name_key(x) for x in existing_names}
        i = 1
        while True:
            candidate = auto_name(i)
            if _normalize_name_key(candidate) not in existing_keys:
                return candidate
            i += 1

    def make_default_graph_state(nodes: List[str]) -> GraphState:
        return GraphState(nodes=list(nodes))

    def sync_graph_state_with_used_columns(st: GraphState, nodes: List[str]) -> None:
        new_nodes = list(nodes)
        new_set = set(new_nodes)
        st.nodes = new_nodes
        st.edge_enabled = {(a, b): v for (a, b), v in st.edge_enabled.items() if a in new_set and b in new_set}
        # Remove stale "backward" edges created by node reordering (they are not representable in the UI
        # but would still be saved and used for ODE generation, leading to extra constants).
        normalize_graph_state(st)

    @ui.refreshable
    def render_saved() -> None:
        """Render list of saved computations.

        Important: must stay refreshable so that "Změněno" badges disappear immediately
        after a new run is computed (without requiring a full page reload).
        """
        items = repo.list_for_experiment(experiment_id)
        _run, stale = compute_staleness(experiment_id)

        with ui.column().classes("w-full gap-2"):
            lbl_saved = ui.label("Uložené výpočty/grafy").classes("text-h6")
            attach_tooltip(
                lbl_saved,
                "Uložené výpočty",
                """Seznam uložených konfigurovatelných výpočtů (ODE grafů).

Tady je můžete upravit, smazat nebo rychle otestovat fit (tlačítko *Spočítat*).
V hlavním výpočtu v záložce Zpracování se pak počítají společně s prebuilt modely.""",
            )
            if not items:
                ui.label("Zatím nic uloženého.").classes("text-grey-7")
                return

            for idx, s in enumerate(items):
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.row().classes('items-center gap-2'):
                            ui.label(s.name).classes("text-bold")
                            # Staleness indicator: definition/settings changed since last run
                            try:
                                if stale.custom_changed.get(s.name, False):
                                    bchg = ui.badge('Změněno', color='orange').props('outline')
                                    attach_tooltip(
                                        bchg,
                                        'Změněno',
                                        'Definice tohoto výpočtu nebo související nastavení se změnily od posledního běhu.\n\nPřepočítáním v záložce Zpracování se badge ztratí.',
                                    )
                            except Exception:
                                pass

                        with ui.row().classes("items-center gap-2"):
                            btn_edit = ui.button(
                                "Upravit",
                                on_click=wrap_ui_handler(
                                    'computations.edit.click',
                                    lambda i=idx: open_calc_dialog(edit_index=i),
                                    level=20,
                                    data=lambda i=idx, s=s: {'index': i, 'id': s.id, 'name': s.name},
                                ),
                            ).props("flat")
                            attach_tooltip(btn_edit, "Upravit", """Otevře editor grafu a výběru sloupců.

Použijte, když chcete změnit použité veličiny nebo přepojit hrany (hlavní/kontrolní).""")

                            btn_fit = ui.button(
                                "Spočítat",
                                on_click=wrap_ui_handler(
                                    'computations.fit.click',
                                    lambda i=idx: open_fit_dialog(i),
                                    level=20,
                                    data=lambda i=idx, s=s: {'index': i, 'id': s.id, 'name': s.name},
                                ),
                            ).props("flat")
                            attach_tooltip(btn_fit, "Spočítat (náhled)", """Rychle spočítá fit jen pro tento graf a zobrazí náhled výsledku.

Neukládá to do hlavních výsledků – ty vznikají tlačítkem *Spočítat / přepočítat* nahoře v Zpracování.""")

                            btn_delete = ui.button(
                                "Smazat",
                                on_click=wrap_ui_handler(
                                    'computations.delete.click',
                                    lambda i=idx: open_delete_dialog(i),
                                    level=20,
                                    data=lambda i=idx, s=s: {'index': i, 'id': s.id, 'name': s.name},
                                ),
                            ).props("flat color=negative")
                            attach_tooltip(btn_delete, "Smazat", """Trvale odstraní uložený výpočet/graf z databáze.

Nemá vliv na nahraný Excel ani na tabulky; jen na definici výpočtu.""")

                    lbl_table = ui.label(f"Tabulka: {s.table_name}").classes("text-grey-8")
                    attach_tooltip(lbl_table, "Zdrojová tabulka", """Tabulka (z cache zpracovaných tabulek), ze které se berou data pro tento graf.

Pokud tabulka chybí, nejdřív ji vygenerujte v záložce Tabulky/Zpracování.""")

                    mermaid = build_compact_mermaid(s.graph_state)
                    mer = ui.mermaid(
                        mermaid,
                        config={"securityLevel": "loose", "flowchart": {"nodeSpacing": 40, "rankSpacing": 40}},
                    ).classes("w-full")
                    attach_tooltip(
                        mer,
                        "Náhled grafu",
                        """Zhuštěný náhled výpočetního grafu (uzly = veličiny, hrany = vazby).

Klikem na hrany v editoru můžete přepínat: vypnuto → hlavní → kontrolní.""",
                    )

    def open_delete_dialog(index: int) -> None:
        item = repo.list_for_experiment(experiment_id)[index]
        delete_dialog.clear()
        with delete_dialog, ui.card().classes("w-96"):
            ui.label("Smazat výpočet?").classes("text-h6")
            ui.label(f"„{item.name}“").classes("text-grey-8")
            ui.separator()
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Zrušit", on_click=delete_dialog.close).props("flat")
                ui.button("Smazat", on_click=lambda: delete_confirm(item.id)).props("color=negative")
        delete_dialog.open()

    def delete_confirm(cid: int) -> None:
        log.info('[UI] computations.delete_confirm: id=%s', cid)
        repo.delete(cid)
        # Any change in computations affects graphs/results staleness and graph regeneration.
        try:
            st = get_state()
            st.graphs_version = st.graphs_version + 1
        except Exception:
            pass
        delete_dialog.close()
        render_saved.refresh()
        ui.notify("Smazáno.", type="positive")

    def open_ode_dialog() -> None:
        log.info('[UI] computations.open_ode_dialog')
        items = repo.list_for_experiment(experiment_id)
        ode_dialog.clear()
        with ode_dialog, ui.card().classes("w-[1000px] max-w-[95vw]"):
            ui.label("ODE rovnice pro všechny výpočty").classes("text-h6")
            ui.separator()

            if not items:
                ui.label("Zatím nejsou uložené žádné výpočty.").classes("text-grey-7")
            else:
                for s in items:
                    ui.label(f"{s.name} (tabulka: {s.table_name})").classes("text-subtitle1")
                    # Show ODEs for BOTH computations:
                    # - main computation: mode=1 (persisted on save as s.ode_text)
                    # - control computation: mode=2 (generated on demand)
                    main_text = s.ode_text or generate_odes_model(s.graph_state, include_modes=(1,)).ode_text

                    try:
                        edge_modes = dict(getattr(s.graph_state, 'edge_modes', {}) or {})
                        has_control = any(int(m) == 2 for m in edge_modes.values())
                    except Exception:
                        has_control = False

                    ui.label("Hlavní výpočet").classes("text-caption text-grey-7")
                    ui.textarea(value=main_text).props("readonly autogrow").classes("w-full font-mono text-xs")

                    ui.label("Kontrolní výpočet").classes("text-caption text-grey-7")
                    if has_control:
                        ctrl_text = generate_odes_model(s.graph_state, include_modes=(2,)).ode_text
                        ui.textarea(value=ctrl_text).props("readonly autogrow").classes("w-full font-mono text-xs")
                    else:
                        ui.textarea(value="# (žádné kontrolní hrany)\n").props("readonly autogrow").classes(
                            "w-full font-mono text-xs"
                        )
                    ui.separator()

            with ui.row().classes("w-full justify-end"):
                ui.button("Zavřít", on_click=ode_dialog.close).props("flat")

        ode_dialog.open()

    def open_fit_dialog(index: int) -> None:
        """Fit selected graph-defined ODEs to the selected table and show a plot."""

        item = repo.list_for_experiment(experiment_id)[index]
        log.info('[UI] computations.open_fit_dialog: id=%s name=%s table=%s used=%s', item.id, item.name, item.table_name, len(item.used_heads or []))
        if not item.used_heads:
            ui.notify("Nejsou vybrána použitá data.", type="warning")
            return

        dlg = ui.dialog().props("maximized")
        with dlg, ui.card().classes("w-full"):
            ui.label(f"Výpočet: {item.name}").classes("text-h6")
            ui.label(f"Tabulka: {item.table_name}").classes("text-caption")
            status = ui.label("Připravuji výpočet…").classes("text-body2")
            spinner = ui.spinner(size="lg")
            result_container = ui.column().classes("w-full")
            ui.separator()
            with ui.row().classes("w-full justify-end"):
                ui.button("Zavřít", on_click=dlg.close).props("outline")

        async def run_fit() -> None:
            from nicegui import run
            import pandas as pd

            from experimental_web.data.repositories import ProcessedTablesRepository
            from experimental_web.domain.graph_model_fit import fit_graph_ode_model_split

            try:
                tables_repo = ProcessedTablesRepository(DB_PATH)
                raw = tables_repo.load_latest_tables(experiment_id)
                if item.table_name not in raw:
                    raise ValueError(
                        f"Tabulka '{item.table_name}' nebyla nalezena v posledních zpracovaných tabulkách. "
                        "Otevři záložku 'Tabulky' a ujisti se, že jsou spočítané."
                    )
                # load_latest_tables() může vracet buď dict {df_json,text_md},
                # nebo tuple (df_json, text_md). Ošetříme obě varianty.
                table_entry = raw[item.table_name]
                if isinstance(table_entry, dict):
                    df_json = table_entry.get("df_json")
                else:
                    df_json = table_entry[0]
                # `pd.read_json` with a literal json string is deprecated; wrap in StringIO.
                df = pd.read_json(pd.io.common.StringIO(df_json), orient="split")
                for c in item.used_heads:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                status.text = "Optimalizuji parametry…"
                log.info(
                    '[UI] computations.fit.start: name=%s initialization=%s t_shift=%s optim_time_shift=%s',
                    item.name,
                    params.initialization.name,
                    float(params.t_shift),
                    bool(params.optim_time_shift),
                )
                res = await run.cpu_bound(
                    fit_graph_ode_model_split,
                    df,
                    item.used_heads,
                    item.graph_state,
                    initialization=params.initialization.name,
                    t_shift=float(params.t_shift),
                    optim_time_shift=bool(params.optim_time_shift),
                )

                spinner.set_visibility(False)
                status.text = "Hotovo."
                result_container.clear()
                with result_container:
                    with ui.card().classes("w-full"):
                        ui.label("Nalezené parametry:").classes("text-subtitle1")
                        for name, val in zip(res.param_names, res.optimal_params):
                            ui.label(f"{name} = {val:.6g}").classes("text-body2")
                    ui.image(f"data:image/png;base64,{res.plot_png_base64}").classes("w-full")
            except Exception as e:
                log.exception('computations.fit.error: %s', e)
                spinner.set_visibility(False)
                status.text = f"Chyba: {e}"
                ui.notify(f"Chyba při výpočtu: {e}", type="negative")

        dlg.open()
        ui.timer(0.05, run_fit, once=True)

    def open_calc_dialog(edit_index: Optional[int] = None) -> None:
        edit_mode = edit_index is not None
        items = repo.list_for_experiment(experiment_id)
        existing = items[edit_index] if edit_mode else None

        tables = load_available_tables()
        if not tables:
            ui.notify("Nejdřív spočítejte tabulky v záložce 'zpracování' (cache do DB).", type="warning")
            return

        state = {"loaded_table": None, "used": [], "unused": []}
        graph_state: GraphState = GraphState()
        graph_controller: Optional[GraphController] = None

        def compute_used_unused_for_table(tname: str) -> Tuple[List[HeadItem], List[HeadItem]]:
            heads = list(tables[tname].columns)
            items_by_key = {h: HeadItem(key=h, title=h) for h in heads}

            if existing and existing.table_name == tname:
                used_keys = [k for k in existing.used_heads if k in items_by_key]
                used_items = [items_by_key[k] for k in used_keys]
                unused_items = [items_by_key[h] for h in heads if h not in set(used_keys)]
                return used_items, unused_items

            default_used = [items_by_key[h] for h in heads if str(h).lower().startswith('zastoupení')]
            default_unused = [items_by_key[h] for h in heads if h not in {x.key for x in default_used}]
            return default_used, default_unused

        dialog.clear()
        with dialog, ui.column().classes("comp-editor-root w-screen h-screen"):
            with ui.row().classes("comp-editor-header w-full items-center justify-between p-4 shadow"):
                dlg_title = ui.label("Upravit výpočet / graf" if edit_mode else "Nový výpočet / graf").classes("text-h6")
                attach_tooltip(dlg_title, "Editor výpočtu", """Tady definujete konfigurovatelný výpočet: vyberete tabulku, sloupce a nakreslíte graf.

Hrany lze přepínat: vypnuto → hlavní → kontrolní.""")
                ui.button("Zavřít", on_click=wrap_ui_handler('computations.dialog.close', dialog.close, level=20)).props("flat").classes("text-grey-8")

            default_table = existing.table_name if existing and existing.table_name in tables else list(tables.keys())[0]
            default_name = existing.name if existing else ""

            with ui.row().classes("w-full p-6 gap-4 items-end"):
                table_select = ui.select(list(tables.keys()), label="Vyber tabulku", value=default_table).classes("w-72")
                name_input = ui.input(label="Název výpočtu / grafu (volitelné)", value=default_name).classes("w-96")
                btn_continue = ui.button("Pokračovat").classes("bg-primary text-white")
                btn_reload = ui.button("Přenačíst").classes("bg-primary text-white")
                btn_reload.set_visibility(False)
                attach_tooltip(table_select, "Tabulka", """Zvolte, ze které zpracované tabulky se budou brát data pro fit.

Tabulky vznikají v záložce Tabulky/Zpracování a ukládají se do DB cache.""")
                attach_tooltip(name_input, "Název", """Volitelný název výpočtu. Pokud ho nevyplníte, doplní se automaticky.""")
                attach_tooltip(btn_continue, "Pokračovat", """Načte sloupce z vybrané tabulky a otevře drag&drop výběr + editor grafu.""")
                attach_tooltip(btn_reload, "Přenačíst", """Znovu načte sloupce pro právě vybranou tabulku.

Použijte, když jste změnil tabulku nebo se tabulky mezitím přepočítaly.""")

            lbl_cols = ui.label("Sloupce (drag & drop nebo dvojklik)").classes("text-subtitle1 px-6")
            attach_tooltip(lbl_cols, "Výběr veličin", """Přesunujte sloupce mezi *Použitá data* a *Nepoužitá data*.

Pořadí v *Použitá data* určuje pořadí uzlů v grafu (číslování 1..N je jen UI).""")

            with ui.row().classes("w-full px-6 items-center gap-6"):
                chosen_table_label = ui.label("").classes("text-grey-8")
                chosen_name_label = ui.label("").classes("text-grey-8")

            dnd_wrapper = ui.column().classes("w-full px-6 gap-4")
            dnd_wrapper.set_visibility(False)

            with dnd_wrapper:
                dnd_area = ui.row().classes("w-full gap-6 items-start")

            ui.separator().classes("mx-6")
            lbl_graph = ui.label("Graf (klik na hrany: vypnuto → hlavní → kontrolní)").classes("text-subtitle1 px-6")
            attach_tooltip(lbl_graph, "Editor grafu", """Klikem na hranu přepínáte režim: vypnuto → hlavní (mode=1) → kontrolní (mode=2).

Kontrolní hrany se počítají separátně a při vykreslení se mergeují podle pravidel priority.""")

            graph_wrapper = ui.column().classes("w-full px-6 gap-2")
            graph_wrapper.set_visibility(False)

            with graph_wrapper:
                graph_container = ui.column().classes("w-full gap-2")

            with ui.row().classes("w-full px-6 justify-end items-center mt-2"):
                btn_save = ui.button("Uložit změny" if edit_mode else "Uložit").classes("bg-primary text-white")
                attach_tooltip(btn_save, "Uložit", """Uloží definici výpočtu do databáze.

Po uložení se výpočet objeví v seznamu a bude zahrnut do hlavního běhu v Zpracování.""")
                # NOTE: Tlačítko "Náhled" je dočasně vypnuté – v preview režimu se volalo
                # jméno fit_graph_ode_model, které v aktuální verzi není definované.
                # Pokud bude potřeba "Náhled" vrátit, odkomentujte následující řádky
                # a opravte implementaci preview_fit.
                # btn_preview = ui.button("Náhled", icon="show_chart").props("outline")
                # attach_tooltip(
                #     btn_preview,
                #     "Náhled",
                #     "Rychle spočítá fit a zobrazí graf ještě před uložením (ověření).\\n\\n"
                #     "Použije aktuální nastavení inicializace/t_shift z ProcessingConfig.",
                # )
                btn_cancel = ui.button("Zrušit", on_click=dialog.close).props("flat")
                attach_tooltip(btn_cancel, "Zrušit", """Zavře editor bez uložení změn.""")

            def update_buttons() -> None:
                selected = table_select.value
                loaded = state["loaded_table"]
                if loaded is None:
                    btn_continue.set_visibility(not edit_mode)
                    btn_reload.set_visibility(False)
                else:
                    btn_continue.set_visibility(False)
                    btn_reload.set_visibility(selected != loaded)

            def ensure_graph_widget_exists() -> None:
                nonlocal graph_controller
                if graph_controller is not None:
                    return
                graph_container.clear()
                with graph_container:
                    graph_controller = create_graph_widget(graph_state, key=f"graph_in_dialog_{experiment_id}")

            def refresh_graph() -> None:
                ensure_graph_widget_exists()
                if graph_controller is not None:
                    graph_controller.refresh()

            def render_dnd() -> None:
                tname = state["loaded_table"]
                cname = (name_input.value or "").strip() or "—"
                chosen_table_label.set_text(f"Tabulka: {tname}")
                chosen_name_label.set_text(f"Název: {cname}")

                drag_state = DragState()

                def apply_drop(item: HeadItem, source_col: str, target_col: str, target_index: Optional[int]) -> None:
                    # remove from both lists
                    state["unused"] = [x for x in state["unused"] if x.key != item.key]
                    state["used"] = [x for x in state["used"] if x.key != item.key]

                    target_list = state["used"] if target_col == "Použitá data" else state["unused"]
                    if target_index is None or target_index >= len(target_list):
                        target_list.append(item)
                    else:
                        target_list.insert(target_index, item)

                    used_names = [x.key for x in state["used"]]
                    sync_graph_state_with_used_columns(graph_state, used_names)
                    refresh_graph()
                    render_dnd()  # re-render to reflect ordering

                def mover(card: Card) -> None:
                    src = card.column_name
                    dst = "Nepoužitá data" if src == "Použitá data" else "Použitá data"
                    apply_drop(card.item, src, dst, None)

                dnd_area.clear()
                with dnd_area:
                    with ui.column().classes("gap-2"):
                        ui.label("Použitá data").classes("text-bold")
                        left = Column("Použitá data", drag_state, on_drop=apply_drop, mover=mover)
                        for idx, it in enumerate(state["used"]):
                            left.accept_card(it, idx)

                    with ui.column().classes("gap-2"):
                        ui.label("Nepoužitá data").classes("text-bold")
                        right = Column("Nepoužitá data", drag_state, on_drop=apply_drop, mover=mover)
                        for idx, it in enumerate(state["unused"]):
                            right.accept_card(it, idx)
            def load_columns_for_selected_table() -> None:
                nonlocal graph_state, graph_controller

                tname = table_select.value
                cname = (name_input.value or "").strip()
                log.info('[UI] computations.load_columns: table=%s edit_mode=%s', tname, edit_mode)
                if not cname:
                    cname = auto_name(len(items) + 1)
                    name_input.value = cname

                used_items, unused_items = compute_used_unused_for_table(tname)
                state["loaded_table"] = tname
                state["used"] = used_items
                state["unused"] = unused_items

                used_names = [x.key for x in state["used"]]

                if existing and existing.table_name == tname:
                    graph_state = GraphState(nodes=list(existing.graph_state.nodes), edge_enabled=dict(existing.graph_state.edge_enabled))
                    sync_graph_state_with_used_columns(graph_state, used_names)
                else:
                    graph_state = make_default_graph_state(used_names)

                graph_controller = None
                ensure_graph_widget_exists()
                refresh_graph()

                dnd_wrapper.set_visibility(True)
                graph_wrapper.set_visibility(True)
                render_dnd()
                update_buttons()

            def save_and_close() -> None:
                if state["loaded_table"] is None:
                    ui.notify("Nejdřív načti sloupce (Pokračovat).", type="warning")
                    return

                # Always check uniqueness of the computation name within the experiment.
                current_items = repo.list_for_experiment(experiment_id)
                existing_names = {
                    it.name
                    for it in current_items
                    if not (edit_mode and existing is not None and it.id == existing.id)
                }

                raw_name = (name_input.value or "").strip()
                if edit_mode and existing is not None:
                    # In edit mode: empty name means keep the original.
                    name = raw_name or existing.name
                else:
                    # New item: empty name => pick the first available auto-name.
                    name = raw_name or _make_unique_auto_name(existing_names)

                # Enforce uniqueness (case-insensitive) before hitting the DB UNIQUE constraint.
                if _normalize_name_key(name) in {_normalize_name_key(x) for x in existing_names}:
                    suggested = _make_unique_variant(name, existing_names)
                    name_input.value = suggested
                    ui.notify(
                        f"Název '{name}' už v tomto experimentu existuje. Navrhuji '{suggested}'.",
                        type="warning",
                    )
                    return
                table_name = state["loaded_table"]
                used_heads = [x.key for x in state["used"]]

                # summarize graph for debugging
                try:
                    modes = graph_state.edge_modes
                    enabled = [e for e, m in modes.items() if int(m) != 0]
                    main = sum(1 for _, m in modes.items() if int(m) == 1)
                    control = sum(1 for _, m in modes.items() if int(m) == 2)
                except Exception:
                    enabled, main, control = [], 0, 0
                log.info(
                    '[UI] computations.save: edit=%s name=%s table=%s used=%s edges_enabled=%s (main=%s control=%s)',
                    edit_mode,
                    name,
                    table_name,
                    len(used_heads),
                    len(enabled),
                    main,
                    control,
                )

                sync_graph_state_with_used_columns(graph_state, used_heads)

                try:
                    if edit_mode:
                        repo.update(existing.id, name, table_name, used_heads, graph_state)
                    else:
                        repo.insert(experiment_id, name, table_name, used_heads, graph_state)
                except sqlite3.IntegrityError:
                    # Defensive: in case another entry was created concurrently.
                    ui.notify(
                        f"Název '{name}' už v tomto experimentu existuje. Zvolte prosím jiný název.",
                        type="warning",
                    )
                    return

                # Bump graphs version so the graphs tab can refresh immediately.
                try:
                    st.graphs_version = st.graphs_version + 1
                except Exception:
                    pass

                render_saved.refresh()
                dialog.close()

            def preview_fit() -> None:
                """Spočítá fit a zobrazí graf ještě před uložením (rychlé ověření)."""
                try:
                    if state["loaded_table"] is None:
                        ui.notify("Nejdřív načti sloupce (Pokračovat).", type="warning")
                        return

                    table_name = state["loaded_table"]
                    used_heads = [x.key for x in state.get("used", [])]
                    if not used_heads:
                        ui.notify("Vyber alespoň jednu veličinu (Used).", type="warning")
                        return

                    # tabulky se mohly změnit (např. po zpracování) – kdyby chyběla, obnovíme
                    if table_name not in tables:
                        tables.update(load_available_tables())
                    if table_name not in tables:
                        ui.notify(f"Tabulka '{table_name}' není k dispozici", type="warning")
                        return

                    sync_graph_state_with_used_columns(graph_state, used_heads)

                    log.info(
                        '[UI] computations.preview_fit: table=%s used=%s initialization=%s t_shift=%s optim_time_shift=%s',
                        table_name,
                        len(used_heads),
                        params.initialization.name,
                        float(params.t_shift),
                        bool(params.optim_time_shift),
                    )

                    from experimental_web.domain.graph_model_fit import fit_graph_ode_model_split

                    ode_model = generate_odes_model(graph_state)
                    res = fit_graph_ode_model(
                        tables[table_name],
                        used_columns=used_heads,
                        ode_text=ode_model.ode_text,
                        state_names=ode_model.state_names,
                        param_names=ode_model.param_names,
                        initialization=params.initialization.name,
                        t_shift=float(params.t_shift),
                        optim_time_shift=bool(params.optim_time_shift),
                    )

                    prev = ui.dialog().props("maximized")
                    with prev, ui.card().classes("w-full"):
                        ui.label("Náhled fitu").classes("text-lg font-bold")
                        ui.markdown(
                            f"**RSS:** {res.rss:.6g}  \\n+**R²:** {res.r2:.6g}  \\n+**Parametry:** {res.params}"
                        )
                        ui.image(f"data:image/png;base64,{res.plot_png_base64}").classes("w-full")
                        ui.button("Zavřít", on_click=prev.close)
                    prev.open()
                except Exception as e:
                    ui.notify(f"Chyba: {e}", type="negative")

            btn_continue.on("click", load_columns_for_selected_table)
            btn_reload.on("click", load_columns_for_selected_table)
            btn_save.on("click", wrap_ui_handler('computations.save.click', save_and_close, level=20))
            # btn_preview.on("click", wrap_ui_handler(\'computations.preview.click\', preview_fit, level=20))

            table_select.on(
                "update:model-value",
                wrap_ui_handler('computations.table_select.change', lambda _: update_buttons(), level=10, data=lambda e: {'value': getattr(e, 'value', None)}),
            )

            if edit_mode:
                load_columns_for_selected_table()
            else:
                update_buttons()

        dialog.open()

    with ui.row().classes("items-center gap-3"):
        btn_add = ui.button(
            "Přidej nový výpočet",
            on_click=wrap_ui_handler('computations.add_new.click', lambda: open_calc_dialog(), level=20),
        ).props("outline")
        attach_tooltip(
            btn_add,
            "Nový výpočet / graf",
            """Vytvoří nový konfigurovatelný výpočet: vyberete tabulku, sloupce a nakreslíte graf (hlavní/kontrolní hrany).

Uložené grafy se pak počítají i v hlavním běhu v záložce Zpracování.""",
        )

        btn_odes = ui.button(
            "Zobraz ODEs",
            on_click=wrap_ui_handler('computations.show_odes.click', open_ode_dialog, level=20),
        ).props("outline")
        attach_tooltip(
            btn_odes,
            "Zobraz ODEs",
            """Zobrazí text ODE rovnic pro všechny uložené grafy.

Ukazuje jak hlavní (mode=1), tak kontrolní (mode=2) ODE.""",
        )

    # Initial render
    render_saved()

    # Keep the list in sync with other parts of the UI.
    # - st.data_version changes when the user changes input data / pick ranges
    # - st.graphs_version changes when a new run is computed (processing tab)
    last_seen_data = st.data_version
    last_seen_graphs = st.graphs_version

    def _tick_refresh_saved() -> None:
        nonlocal last_seen_data, last_seen_graphs
        try:
            if st.data_version != last_seen_data or st.graphs_version != last_seen_graphs:
                last_seen_data = st.data_version
                last_seen_graphs = st.graphs_version
                render_saved.refresh()
        except Exception:
            pass

    ui.timer(1.0, _tick_refresh_saved)
