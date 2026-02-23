from __future__ import annotations

from nicegui import ui

from experimental_web.ui.layout import frame
from experimental_web.core.paths import APP_DIR, DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import ExperimentRepository, MetaRepository


@ui.page("/")
def page_home() -> None:
    exp_repo = ExperimentRepository(DB_PATH)
    meta_repo = MetaRepository(DB_PATH)

    def open_experiment(exp_id: int) -> None:
        exp = exp_repo.get(exp_id)
        if not exp:
            ui.notify("Experiment nenalezen", type="negative")
            return
        exp_repo.touch(exp_id)
        meta_repo.set_last_experiment_id(exp_id)

        st = get_state()
        st.current_experiment_id = exp.id
        st.current_experiment_name = exp.name

        ui.notify(f"Otevřen experiment #{exp.id}: {exp.name}")
        ui.navigate.to("/experiment")

    def new_experiment_dialog() -> None:
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label("Nový experiment").classes("text-h6")
            name_input = ui.input("Název").props("autofocus")

            def do_create() -> None:
                name = (name_input.value or "").strip()
                if not name:
                    ui.notify("Zadej název experimentu", type="warning")
                    return
                if exp_repo.exists_name(name):
                    ui.notify("Experiment s tímto názvem už existuje", type="warning")
                    return

                exp = exp_repo.create(name)
                meta_repo.set_last_experiment_id(exp.id)

                st = get_state()
                st.current_experiment_id = exp.id
                st.current_experiment_name = exp.name

                ui.notify(f"Vytvořen experiment #{exp.id}: {exp.name}")
                dialog.close()
                ui.navigate.to("/experiment")

            with ui.row():
                ui.button("Vytvořit", on_click=do_create)
                ui.button("Zrušit", on_click=dialog.close).props("flat")
        dialog.open()

    def open_last() -> None:
        last_id = meta_repo.get_last_experiment_id()
        if not last_id:
            ui.notify("Žádný poslední experiment není uložen", type="warning")
            return
        open_experiment(last_id)

    def import_placeholder() -> None:
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label("Import experimentu").classes("text-h6")
            ui.label("Placeholder: později sem dáme upload/parsing.")
            ui.button("OK", on_click=dialog.close)
        dialog.open()

    @ui.refreshable
    def experiments_lists() -> None:
        recent = exp_repo.list(limit=3)
        all_exps = exp_repo.list(limit=None)

        def duplicate_dialog(source_id: int, source_name: str) -> None:
            dialog = ui.dialog()
            with dialog, ui.card():
                ui.label("Duplikovat experiment").classes("text-h6")
                name_input = ui.input("Nový název", value=f"{source_name} (kopie)").props("autofocus")

                def do_duplicate() -> None:
                    new_name = (name_input.value or "").strip()
                    if not new_name:
                        ui.notify("Zadej název", type="warning")
                        return
                    if exp_repo.exists_name(new_name):
                        ui.notify("Experiment se stejným názvem už existuje", type="warning")
                        return
                    try:
                        new_exp = exp_repo.duplicate(source_id, new_name)
                    except Exception as e:
                        ui.notify(f"Nelze duplikovat: {e}", type="negative")
                        return

                    meta_repo.set_last_experiment_id(new_exp.id)
                    st = get_state()
                    st.current_experiment_id = new_exp.id
                    st.current_experiment_name = new_exp.name

                    ui.notify(f"Zduplikováno a otevřeno: #{new_exp.id} – {new_exp.name}")
                    dialog.close()
                    ui.navigate.to("/experiment")

                with ui.row():
                    ui.button("Duplikovat", on_click=do_duplicate)
                    ui.button("Zrušit", on_click=dialog.close).props("flat")
            dialog.open()

        def delete_dialog(exp_id: int, exp_name: str) -> None:
            dialog = ui.dialog()
            with dialog, ui.card():
                ui.label("Smazat experiment").classes("text-h6")
                ui.label(f"Opravdu chceš smazat experiment „{exp_name}“?")

                def do_delete() -> None:
                    st = get_state()
                    if st.current_experiment_id == exp_id:
                        st.current_experiment_id = None
                        st.current_experiment_name = ""
                    exp_repo.delete(exp_id, delete_folder=True)
                    ui.notify("Experiment smazán")
                    dialog.close()
                    experiments_lists.refresh()

                with ui.row():
                    ui.button("Smazat", on_click=do_delete).props("color=negative")
                    ui.button("Zrušit", on_click=dialog.close).props("flat")
            dialog.open()

        def experiment_row(exp) -> None:
            # karta je i dvojklikem otevřitelná
            with ui.card().classes("w-full q-pa-sm").on("dblclick", lambda e, exp_id=exp.id: open_experiment(exp_id)):
                with ui.row().classes("w-full items-center no-wrap"):
                    # tlačítka vlevo v button group
                    with ui.button_group().props("unelevated"):
                        ui.button("Otevřít", on_click=lambda exp_id=exp.id: open_experiment(exp_id)).props("color=primary")
                        ui.button("Duplikovat", on_click=lambda: duplicate_dialog(exp.id, exp.name)).props("color=secondary")
                        ui.button("Smazat", on_click=lambda: delete_dialog(exp.id, exp.name)).props("color=negative")

                    with ui.column().classes("q-ml-md gap-0"):
                        ui.label(f"#{exp.id} – {exp.name}").classes("text-subtitle1")
                        ui.label(f"Upraveno: {exp.updated_at}").classes("text-caption")

        # --- render lists (IMPORTANT: outside experiment_row) ---
        ui.label("Poslední experimenty (max 3)").classes("text-subtitle1")
        if not recent:
            ui.label("Zatím žádné experimenty. Klikni na „Nový experiment“.")
        else:
            for exp in recent:
                experiment_row(exp)

        ui.separator()

        ui.label("Všechny experimenty").classes("text-subtitle1")
        if not all_exps:
            ui.label("Zatím žádné experimenty.")
        else:
            for exp in all_exps:
                experiment_row(exp)

    with frame("Správa experimentů"):
        ui.label(f"Data aplikace: {APP_DIR}").classes("text-caption")
        ui.label(f"Databáze: {DB_PATH}").classes("text-caption")

        with ui.row().classes("q-gutter-md"):
            ui.button("Nový experiment", on_click=new_experiment_dialog).props("unelevated")
            ui.button("Otevřít poslední experiment", on_click=open_last).props("unelevated")
            ui.button("Importovat experiment", on_click=import_placeholder).props("unelevated")

        ui.separator()

        experiments_lists()
