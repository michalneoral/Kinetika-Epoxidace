from __future__ import annotations

from nicegui import ui

from experimental_web.ui.layout import frame
from experimental_web.core.paths import APP_DIR, DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import ExperimentRepository, MetaRepository
from experimental_web.logging_setup import get_logger
from experimental_web.ui.instrumentation import wrap_ui_handler
from experimental_web.ui.utils.tooltips import attach_tooltip


log = get_logger(__name__)


@ui.page("/")
def page_home() -> None:
    exp_repo = ExperimentRepository(DB_PATH)
    meta_repo = MetaRepository(DB_PATH)

    def open_experiment(exp_id: int) -> None:
        log.info('[UI] home.open_experiment: id=%s', exp_id)
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
        log.info('[UI] home.new_experiment_dialog.open')
        dialog = ui.dialog()
        with dialog, ui.card():
            title_lbl = ui.label("Nový experiment").classes("text-h6")
            attach_tooltip(
                title_lbl,
                'Vytvoření experimentu',
                'Vytvoří nový záznam experimentu v databázi a nastaví ho jako aktuálně otevřený.',
            )

            name_input = ui.input("Název").props("autofocus")
            attach_tooltip(
                name_input,
                'Název experimentu',
                'Slouží pro rychlé rozlišení projektů. Název musí být unikátní (nelze mít dva se stejným názvem).',
            )

            def do_create() -> None:
                name = (name_input.value or "").strip()
                log.info('[UI] home.create_experiment: name=%s', name)
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
                create_btn = ui.button("Vytvořit", on_click=do_create)
                attach_tooltip(create_btn, 'Vytvořit', 'Vytvoří experiment a rovnou ho otevře v záložce „Experiment“.')

                cancel_btn = ui.button("Zrušit", on_click=dialog.close).props("flat")
                attach_tooltip(cancel_btn, 'Zrušit', 'Zavře dialog bez vytvoření experimentu.')
        dialog.open()

    def open_last() -> None:
        log.info('[UI] home.open_last')
        last_id = meta_repo.get_last_experiment_id()
        if not last_id:
            ui.notify("Žádný poslední experiment není uložen", type="warning")
            return
        open_experiment(last_id)

    def import_placeholder() -> None:
        log.info('[UI] home.import_placeholder')
        dialog = ui.dialog()
        with dialog, ui.card():
            title_lbl = ui.label("Import experimentu").classes("text-h6")
            attach_tooltip(
                title_lbl,
                'Import experimentu',
                'Zatím jen placeholder. Později zde bude import cizích experimentů / exportovaných balíčků.',
            )
            ui.label("Placeholder: později sem dáme upload/parsing.")
            ok_btn = ui.button("OK", on_click=dialog.close)
            attach_tooltip(ok_btn, 'Zavřít', 'Zavře dialog importu.')
        dialog.open()

    @ui.refreshable
    def experiments_lists() -> None:
        recent = exp_repo.list(limit=3)
        all_exps = exp_repo.list(limit=None)

        def duplicate_dialog(source_id: int, source_name: str) -> None:
            dialog = ui.dialog()
            with dialog, ui.card():
                title_lbl = ui.label("Duplikovat experiment").classes("text-h6")
                attach_tooltip(
                    title_lbl,
                    'Duplikace',
                    'Vytvoří kopii vybraného experimentu (včetně souborů a nastavení) pod novým názvem a otevře ji.',
                )

                name_input = ui.input("Nový název", value=f"{source_name} (kopie)").props("autofocus")
                attach_tooltip(
                    name_input,
                    'Nový název',
                    'Zadejte unikátní název pro kopii experimentu.',
                )

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
                    dup_btn = ui.button("Duplikovat", on_click=do_duplicate)
                    attach_tooltip(dup_btn, 'Duplikovat', 'Vytvoří kopii experimentu a otevře ji jako aktuální.')

                    cancel_btn = ui.button("Zrušit", on_click=dialog.close).props("flat")
                    attach_tooltip(cancel_btn, 'Zrušit', 'Zavře dialog bez vytvoření kopie.')
            dialog.open()

        def delete_dialog(exp_id: int, exp_name: str) -> None:
            dialog = ui.dialog()
            with dialog, ui.card():
                title_lbl = ui.label("Smazat experiment").classes("text-h6")
                attach_tooltip(
                    title_lbl,
                    'Smazání experimentu',
                    'Trvale odstraní experiment z databáze a smaže i jeho složku se soubory (pokud existuje).',
                )
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
                    del_btn = ui.button("Smazat", on_click=do_delete).props("color=negative")
                    attach_tooltip(del_btn, 'Smazat', 'Nevratně smaže experiment a aktualizuje seznam.')

                    cancel_btn = ui.button("Zrušit", on_click=dialog.close).props("flat")
                    attach_tooltip(cancel_btn, 'Zrušit', 'Zavře dialog bez smazání experimentu.')
            dialog.open()

        def experiment_row(exp) -> None:
            # karta je i dvojklikem otevřitelná
            card = ui.card().classes("w-full q-pa-sm")
            card.on("dblclick", lambda e, exp_id=exp.id: open_experiment(exp_id))
            attach_tooltip(
                card,
                'Experiment',
                'Dvojklik na kartu experiment otevře. Alternativně použij tlačítka „Otevřít / Duplikovat / Smazat“ vlevo.',
            )

            with card:
                with ui.row().classes("w-full items-center no-wrap"):
                    # tlačítka vlevo v button group
                    with ui.button_group().props("unelevated"):
                        open_btn = ui.button(
                            "Otevřít",
                            on_click=wrap_ui_handler('home.open.click', lambda exp_id=exp.id: open_experiment(exp_id), data=lambda: {'id': exp.id, 'name': exp.name}),
                        ).props("color=primary")
                        attach_tooltip(open_btn, 'Otevřít', 'Nastaví experiment jako aktuální a přejde na stránku „Experiment“.')

                        dup_btn = ui.button(
                            "Duplikovat",
                            on_click=wrap_ui_handler('home.duplicate.click', lambda: duplicate_dialog(exp.id, exp.name), data=lambda: {'id': exp.id, 'name': exp.name}),
                        ).props("color=secondary")
                        attach_tooltip(dup_btn, 'Duplikovat', 'Vytvoří kopii experimentu pod novým názvem.')

                        del_btn = ui.button(
                            "Smazat",
                            on_click=wrap_ui_handler('home.delete.click', lambda: delete_dialog(exp.id, exp.name), data=lambda: {'id': exp.id, 'name': exp.name}),
                        ).props("color=negative")
                        attach_tooltip(del_btn, 'Smazat', 'Otevře potvrzení pro trvalé smazání experimentu.')

                    with ui.column().classes("q-ml-md gap-0"):
                        name_lbl = ui.label(f"#{exp.id} – {exp.name}").classes("text-subtitle1")
                        attach_tooltip(name_lbl, 'Identifikace', 'ID je interní identifikátor v DB; název je uživatelský popis experimentu.')

                        upd_lbl = ui.label(f"Upraveno: {exp.updated_at}").classes("text-caption")
                        attach_tooltip(upd_lbl, 'Poslední změna', 'Čas poslední úpravy experimentu (např. změna dat nebo nastavení).')

        # --- render lists (IMPORTANT: outside experiment_row) ---
        recent_lbl = ui.label("Poslední experimenty (max 3)").classes("text-subtitle1")
        attach_tooltip(recent_lbl, 'Rychlý přístup', 'Zobrazí poslední 3 experimenty podle poslední úpravy.')
        if not recent:
            ui.label("Zatím žádné experimenty. Klikni na „Nový experiment“.")
        else:
            for exp in recent:
                experiment_row(exp)

        ui.separator()

        all_lbl = ui.label("Všechny experimenty").classes("text-subtitle1")
        attach_tooltip(all_lbl, 'Seznam všech projektů', 'Kompletní seznam experimentů uložených v databázi.')
        if not all_exps:
            ui.label("Zatím žádné experimenty.")
        else:
            for exp in all_exps:
                experiment_row(exp)

    with frame("Správa experimentů"):
        app_dir_lbl = ui.label(f"Data aplikace: {APP_DIR}").classes("text-caption")
        attach_tooltip(
            app_dir_lbl,
            'Složka aplikace',
            'Lokální složka, kam aplikace ukládá pomocná data (např. cache, exporty a soubory experimentů).',
        )

        db_lbl = ui.label(f"Databáze: {DB_PATH}").classes("text-caption")
        attach_tooltip(
            db_lbl,
            'Databázový soubor',
            'Cesta k SQLite databázi, kde jsou uložené experimenty, nastavení i metadata aplikace.',
        )

        with ui.row().classes("q-gutter-md"):
            new_btn = ui.button("Nový experiment", on_click=wrap_ui_handler('home.new_experiment.click', new_experiment_dialog)).props("unelevated")
            attach_tooltip(new_btn, 'Nový experiment', 'Otevře dialog pro vytvoření nového experimentu (projektu).')

            open_last_btn = ui.button("Otevřít poslední experiment", on_click=wrap_ui_handler('home.open_last.click', open_last)).props("unelevated")
            attach_tooltip(open_last_btn, 'Otevřít poslední', 'Otevře naposledy otevřený experiment uložený v app_meta.')

            import_btn = ui.button("Importovat experiment", on_click=wrap_ui_handler('home.import.click', import_placeholder)).props("unelevated")
            attach_tooltip(import_btn, 'Import', 'Zatím placeholder – v budoucnu import/export experimentů.')

        ui.separator()

        experiments_lists()
