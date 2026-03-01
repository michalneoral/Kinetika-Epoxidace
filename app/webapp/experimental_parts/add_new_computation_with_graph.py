from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nicegui import ui

from draganddrop import DragState, Column, Card
from nody_v_grafu import GraphState, GraphController, create_graph_widget, build_compact_mermaid
import mydb
import ode_generator

# -------------------------
# Testovací data
# -------------------------
def make_tables() -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)

    c181_cols = [
        "zastoupení C18:1",
        "zastoupení C18:1 EPO",
        "zastoupení hydroxyly",
        "zastoupení C18:1 EPO + hydroxyly",
    ]
    c182_cols = [
        "zastoupení C18:2",
        "zastoupení Σ C18:2 1-EPO",
        "zastoupení C18:2 2-EPO",
        "zastoupení hydroxyly",
        "zastoupení Σ C18:2 EPO + hydroxyly",
    ]

    df_c181 = pd.DataFrame(rng.normal(loc=50, scale=10, size=(12, len(c181_cols))), columns=c181_cols).round(2)
    df_c182 = pd.DataFrame(rng.normal(loc=30, scale=7, size=(10, len(c182_cols))), columns=c182_cols).round(2)

    return {"C18:1": df_c181, "C18:2": df_c182}


tables = make_tables()


@dataclass(frozen=True)
class HeadItem:
    key: str
    title: str


@dataclass
class SavedCalc:
    id: int
    name: str
    table_name: str
    used_heads: List[str]
    graph_state: GraphState


# -------------------------
# In-memory cache (naplní se z DB při startu)
# -------------------------
saved: List[SavedCalc] = []

calc_counter = 0


def auto_name() -> str:
    global calc_counter
    calc_counter += 1
    return f"Výpočet {calc_counter}"


def make_default_graph_state(nodes: List[str]) -> GraphState:
    return GraphState(nodes=list(nodes))


def sync_graph_state_with_used_columns(st: GraphState, nodes: List[str]) -> None:
    """Udrž graf state v souladu s 'použitými' sloupci (uzly řízené drag&drop)."""
    new_nodes = list(nodes)
    new_set = set(new_nodes)
    st.nodes = new_nodes
    st.edge_enabled = {(a, b): v for (a, b), v in st.edge_enabled.items() if a in new_set and b in new_set}


# -------------------------
# UI: seznam uložených
# -------------------------
saved_container = ui.column().classes("w-full gap-2")
delete_dialog = ui.dialog()


def render_saved() -> None:
    saved_container.clear()
    with saved_container:
        ui.label("Uložené výpočty/grafy").classes("text-h6")
        if not saved:
            ui.label("Zatím nic uloženého.").classes("text-grey-7")
            return

        for idx, s in enumerate(saved):
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(s.name).classes("text-bold")

                    with ui.row().classes("items-center gap-2"):
                        ui.button("Upravit", on_click=lambda i=idx: open_calc_dialog(edit_index=i)).props("flat")
                        ui.button("Smazat", on_click=lambda i=idx: open_delete_dialog(i)).props("flat color=negative")

                ui.label(f"Tabulka: {s.table_name}").classes("text-grey-8")

                mermaid = build_compact_mermaid(s.graph_state)
                ui.mermaid(
                    mermaid,
                    config={"securityLevel": "loose", "flowchart": {"nodeSpacing": 40, "rankSpacing": 40}},
                ).classes("w-full")


def open_delete_dialog(index: int) -> None:
    """Potvrzení mazání."""
    item = saved[index]
    delete_dialog.clear()
    with delete_dialog, ui.card().classes("w-96"):
        ui.label("Smazat výpočet?").classes("text-h6")
        ui.label(f"„{item.name}“").classes("text-grey-8")
        ui.separator()
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Zrušit", on_click=delete_dialog.close).props("flat")
            ui.button("Smazat", on_click=lambda: delete_confirm(index)).props("color=negative")

    delete_dialog.open()


def delete_confirm(index: int) -> None:
    item = saved[index]
    mydb.delete_computation(item.id)
    saved.pop(index)
    delete_dialog.close()
    render_saved()
    ui.notify("Smazáno.", type="positive")


# -------------------------
# Dialog (create/edit)
# -------------------------
dialog = ui.dialog().props("maximized")


def open_calc_dialog(edit_index: Optional[int] = None) -> None:
    edit_mode = edit_index is not None
    existing: Optional[SavedCalc] = saved[edit_index] if edit_mode else None

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

        return [items_by_key[h] for h in heads], []

    dialog.clear()
    with dialog, ui.column().classes("w-screen h-screen bg-grey-1"):
        with ui.row().classes("w-full items-center justify-between p-4 bg-white shadow"):
            ui.label("Upravit výpočet / graf" if edit_mode else "Nový výpočet / graf").classes("text-h6")
            ui.button("Zavřít", on_click=dialog.close).props("flat").classes("text-grey-8")

        default_table = existing.table_name if existing and existing.table_name in tables else list(tables.keys())[0]
        default_name = existing.name if existing else ""

        with ui.row().classes("w-full p-6 gap-4 items-end"):
            table_select = ui.select(list(tables.keys()), label="Vyber tabulku", value=default_table).classes("w-72")
            name_input = ui.input(label="Název výpočtu / grafu (volitelné)", value=default_name).classes("w-96")
            btn_continue = ui.button("Pokračovat").classes("bg-primary text-white")
            btn_reload = ui.button("Přenačíst").classes("bg-primary text-white")
            btn_reload.set_visibility(False)

        ui.label("Sloupce (drag & drop nebo dvojklik)").classes("text-subtitle1 px-6")

        with ui.row().classes("w-full px-6 items-center gap-6"):
            chosen_table_label = ui.label("").classes("text-grey-8")
            chosen_name_label = ui.label("").classes("text-grey-8")

        dnd_wrapper = ui.column().classes("w-full px-6 gap-4")
        dnd_wrapper.set_visibility(False)
        dnd_area = ui.row().classes("w-full gap-6 items-start")

        ui.separator().classes("mx-6")
        ui.label("Graf (klik na hrany: vypnuto → hlavní → kontrolní)").classes("text-subtitle1 px-6")

        graph_wrapper = ui.column().classes("w-full px-6 gap-2")
        graph_wrapper.set_visibility(False)
        graph_container = ui.column().classes("w-full gap-2")

        with ui.row().classes("w-full px-6 justify-end items-center mt-2"):
            btn_save = ui.button("Uložit změny" if edit_mode else "Uložit").classes("bg-primary text-white")
            ui.button("Zrušit", on_click=dialog.close).props("flat")

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
                graph_controller = create_graph_widget(graph_state, key="graph_in_dialog")

        def refresh_graph() -> None:
            ensure_graph_widget_exists()
            graph_controller.refresh()

        def render_dnd() -> None:
            tname = state["loaded_table"]
            cname = (name_input.value or "").strip() or "—"
            chosen_table_label.set_text(f"Tabulka: {tname}")
            chosen_name_label.set_text(f"Název: {cname}")

            drag_state = DragState()
            columns_ref = {"Použitá data": None, "Nepoužitá data": None}

            def on_drop(item: HeadItem, target_column_name: str) -> None:
                state["unused"] = [x for x in state["unused"] if x.key != item.key]
                state["used"] = [x for x in state["used"] if x.key != item.key]
                (state["used"] if target_column_name == "Použitá data" else state["unused"]).append(item)

                used_names = [x.key for x in state["used"]]
                sync_graph_state_with_used_columns(graph_state, used_names)
                refresh_graph()

            def mover(card: Card) -> None:
                parent_col = card.parent_slot.parent
                src_name = getattr(parent_col, "name", "")
                dst = columns_ref["Nepoužitá data"] if src_name == "Použitá data" else columns_ref["Použitá data"]
                if dst is None:
                    return
                dst.accept_card(card.item)
                parent_col.remove(card)

            dnd_area.clear()
            with dnd_area:
                with ui.column().classes("gap-2"):
                    ui.label("Použitá data").classes("text-bold")
                    left = Column("Použitá data", drag_state, on_drop=on_drop, mover=mover)
                    columns_ref["Použitá data"] = left
                    for it in state["used"]:
                        with left:
                            Card(it, drag_state, mover=mover)

                with ui.column().classes("gap-2"):
                    ui.label("Nepoužitá data").classes("text-bold")
                    right = Column("Nepoužitá data", drag_state, on_drop=on_drop, mover=mover)
                    columns_ref["Nepoužitá data"] = right
                    for it in state["unused"]:
                        with right:
                            Card(it, drag_state, mover=mover)

        def load_columns_for_selected_table() -> None:
            nonlocal graph_state, graph_controller

            tname = table_select.value
            cname = (name_input.value or "").strip()
            if not cname:
                cname = auto_name()
                name_input.value = cname

            used_items, unused_items = compute_used_unused_for_table(tname)
            state["loaded_table"] = tname
            state["used"] = used_items
            state["unused"] = unused_items

            used_names = [x.key for x in state["used"]]

            if existing and existing.table_name == tname:
                graph_state = GraphState(
                    nodes=list(existing.graph_state.nodes),
                    edge_enabled=dict(existing.graph_state.edge_enabled),
                )
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

            name = (name_input.value or "").strip() or auto_name()
            table_name = state["loaded_table"]
            used_heads = [x.key for x in state["used"]]

            sync_graph_state_with_used_columns(graph_state, used_heads)

            if edit_mode:
                cid = existing.id
                mydb.update_computation(cid, name, table_name, used_heads, graph_state)
                saved[edit_index] = SavedCalc(
                    id=cid,
                    name=name,
                    table_name=table_name,
                    used_heads=used_heads,
                    graph_state=GraphState(
                        nodes=list(graph_state.nodes),
                        edge_enabled=dict(graph_state.edge_enabled),
                    ),
                )
            else:
                cid = mydb.insert_computation(name, table_name, used_heads, graph_state)
                saved.append(
                    SavedCalc(
                        id=cid,
                        name=name,
                        table_name=table_name,
                        used_heads=used_heads,
                        graph_state=GraphState(
                            nodes=list(graph_state.nodes),
                            edge_enabled=dict(graph_state.edge_enabled),
                        ),
                    )
                )

            render_saved()
            dialog.close()

        btn_continue.on("click", load_columns_for_selected_table)
        btn_reload.on("click", load_columns_for_selected_table)
        btn_save.on("click", save_and_close)

        table_select.on("update:model-value", lambda _: update_buttons())

        if edit_mode:
            load_columns_for_selected_table()
        else:
            update_buttons()

    dialog.open()


def open_ode_dialog() -> None:
    ode_dialog.clear()
    with ode_dialog, ui.card().classes("w-[1000px] max-w-[95vw]"):
        ui.label("ODE rovnice pro všechny výpočty").classes("text-h6")
        ui.separator()

        if not saved:
            ui.label("Zatím nejsou uložené žádné výpočty.").classes("text-grey-7")
        else:
            for s in saved:
                ui.label(f"{s.name} (tabulka: {s.table_name})").classes("text-subtitle1")
                text = ode_generator.generate_odes_text(s.graph_state, variable_prefix="y")
                # sanitize=True je povinné v novější NiceGUI pro ui.html, ale my použijeme textarea:
                ui.textarea(value=text).props("readonly autogrow").classes("w-full font-mono text-xs")
                ui.separator()

        with ui.row().classes("w-full justify-end"):
            ui.button("Zavřít", on_click=ode_dialog.close).props("flat")

    ode_dialog.open()


# -------------------------
# App start
# -------------------------
mydb.init_db()
saved = [
    SavedCalc(
        id=x.id,
        name=x.name,
        table_name=x.table_name,
        used_heads=x.used_heads,
        graph_state=x.graph_state,
    )
    for x in mydb.load_all()
]
calc_counter = len(saved)

ui.label("Demo: výpočty / grafy").classes("text-h5")
with ui.row().classes("items-center gap-3"):
    ui.button("Přidej nový výpočet", on_click=lambda: open_calc_dialog()).classes("bg-primary text-white")
    ui.button("Zobraz ODEs", on_click=open_ode_dialog).props("outline")
ui.separator()

# --- Dialog na ODE výpis ---
ode_dialog = ui.dialog()

render_saved()
ui.run(port=8083)
