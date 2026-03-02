from __future__ import annotations

from io import BytesIO
import html
import zipfile

from nicegui import ui

from experimental_web.ui.layout import frame
from experimental_web.core.paths import APP_DIR, DB_PATH
from experimental_web.core.state import get_state
from experimental_web.core.time import parse_datetime_any
from experimental_web.data.repositories import ExperimentRepository, MetaRepository
from experimental_web.data.experiment_transfer import export_all_experiments, export_experiment, import_any
from experimental_web.logging_setup import get_logger
from experimental_web.ui.instrumentation import wrap_ui_handler
from experimental_web.ui.utils.tooltips import attach_tooltip


log = get_logger(__name__)


@ui.page("/")
def page_home() -> None:
    exp_repo = ExperimentRepository(DB_PATH)
    meta_repo = MetaRepository(DB_PATH)

    def _fmt_dt(value: object | None) -> str:
        dt = parse_datetime_any(value)
        if not dt:
            return "–"
        # Keep it short and readable; seconds are rarely useful here.
        return dt.strftime('%Y-%m-%d %H:%M')

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

    def import_experiment_dialog() -> None:
        log.info('[UI] home.import_experiment_dialog.open')

        dialog = ui.dialog()
        with dialog, ui.card().classes('w-[680px] max-w-[92vw]'):
            title_lbl = ui.label('Import experimentu').classes('text-h6')
            attach_tooltip(
                title_lbl,
                'Import experimentu',
                'Nahrajte exportovaný ZIP. Import umí načíst jak jeden experiment (Exportovat), tak i balík všech experimentů (Exportovat vše).',
            )

            info = ui.markdown(
                """- Podporovaný formát: ZIP export z této aplikace\n- Po importu se experiment automaticky otevře"""
            ).classes('text-body2')
            attach_tooltip(info, 'Co import dělá', 'Import načte všechna data z exportu a uloží je jako nový experiment. Pokud už existuje stejný název, přidá se suffix (2), (3), ...')

            upload = ui.upload(
                label='Vyberte export ZIP',
                auto_upload=True,
                max_files=1,
                multiple=False,
            ).props('accept=.zip')
            attach_tooltip(upload, 'Nahrát export', 'Vyberte ZIP vytvořený tlačítkem „Exportovat“ (1 experiment) nebo „Exportovat vše“ (balík experimentů).')

            async def handle_upload(e) -> None:
                try:
                    data = await e.file.read()
                except Exception as ex:
                    ui.notify(f'Nelze načíst soubor: {ex}', type='negative')
                    return

                # Determine how many experiments the archive likely contains (for user feedback)
                expected = None
                try:
                    with zipfile.ZipFile(BytesIO(data), 'r') as zf:
                        names = set(zf.namelist())
                        if 'export.json' in names:
                            expected = 1
                        else:
                            expected = len([n for n in names if n.startswith('experiments/') and n.lower().endswith('.zip')])
                            if expected == 0:
                                expected = None
                except Exception:
                    expected = None
                try:
                    new_ids = import_any(DB_PATH, data)
                except Exception as ex:
                    ui.notify(f'❌ Import selhal: {ex}', type='negative')
                    return

                if len(new_ids) == 1:
                    new_id = int(new_ids[0])
                    ui.notify(f'✅ Import hotov: experiment #{new_id}')
                    dialog.close()
                    open_experiment(new_id)
                else:
                    if expected is not None and expected != len(new_ids):
                        ui.notify(f'⚠️ Import dokončen: naimportováno {len(new_ids)} z {expected} experimentů', type='warning')
                    else:
                        ui.notify(f'✅ Import hotov: naimportováno {len(new_ids)} experimentů')
                    dialog.close()
                    # Otevřeme poslední naimportovaný (typicky nejnovější dle pořadí v zipu)
                    open_experiment(int(new_ids[-1]))

            upload.on_upload(handle_upload)

            with ui.row().classes('justify-end w-full'):
                close_btn = ui.button('Zavřít', on_click=dialog.close).props('flat')
                attach_tooltip(close_btn, 'Zavřít', 'Zavře dialog importu bez změn.')

        dialog.open()

    def export_all() -> None:
        log.info('[UI] home.export_all')
        try:
            zip_bytes, filename = export_all_experiments(DB_PATH, include_folder=True)
        except Exception as ex:
            ui.notify(f'❌ Export selhal: {ex}', type='negative')
            return
        ui.download(zip_bytes, filename=filename)
        ui.notify('📦 Export všech experimentů připraven ke stažení')

    @ui.refreshable
    def experiments_lists() -> None:
        recent = exp_repo.list(limit=3)
        all_exps = exp_repo.list(limit=None)

        # Precompute a few lightweight stats for display on Home.
        exp_ids = [int(e.id) for e in all_exps]
        computations_count: dict[int, int] = {}
        files_count: dict[int, int] = {}
        if exp_ids:
            ph = ','.join(['?'] * len(exp_ids))
            with exp_repo.db.connect() as con:
                for r in con.execute(
                    f"SELECT experiment_id, COUNT(*) AS n FROM experiment_computations WHERE experiment_id IN ({ph}) GROUP BY experiment_id",
                    exp_ids,
                ).fetchall():
                    computations_count[int(r['experiment_id'])] = int(r['n'])
                for r in con.execute(
                    f"SELECT experiment_id, COUNT(*) AS n FROM experiment_files WHERE experiment_id IN ({ph}) GROUP BY experiment_id",
                    exp_ids,
                ).fetchall():
                    files_count[int(r['experiment_id'])] = int(r['n'])

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

        def rename_dialog(exp_id: int, current_name: str) -> None:
            dialog = ui.dialog()
            with dialog, ui.card():
                title_lbl = ui.label('Přejmenovat experiment').classes('text-h6')
                attach_tooltip(
                    title_lbl,
                    'Přejmenování',
                    'Změní název experimentu. Název musí být unikátní v rámci databáze experimentů.',
                )

                name_input = ui.input('Nový název', value=current_name).props('autofocus')
                attach_tooltip(name_input, 'Nový název', 'Zadejte nový unikátní název experimentu.')

                def do_rename() -> None:
                    new_name = (name_input.value or '').strip()
                    if not new_name:
                        ui.notify('Zadej název', type='warning')
                        return
                    if new_name == current_name:
                        dialog.close()
                        return
                    if exp_repo.exists_name(new_name):
                        ui.notify('Experiment se stejným názvem už existuje', type='warning')
                        return
                    try:
                        updated = exp_repo.rename(exp_id, new_name)
                    except Exception as e:
                        ui.notify(f'Nelze přejmenovat: {e}', type='negative')
                        return

                    st = get_state()
                    if st.current_experiment_id == exp_id:
                        st.current_experiment_name = updated.name

                    ui.notify('Přejmenováno')
                    dialog.close()
                    experiments_lists.refresh()

                with ui.row():
                    ok_btn = ui.button('Přejmenovat', on_click=do_rename).props('color=primary')
                    attach_tooltip(ok_btn, 'Přejmenovat', 'Uloží nový název experimentu.')

                    cancel_btn = ui.button('Zrušit', on_click=dialog.close).props('flat')
                    attach_tooltip(cancel_btn, 'Zrušit', 'Zavře dialog bez změny názvu.')

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
                'Dvojklik na kartu experiment otevře. Alternativně použij ikonky vlevo (otevřít, duplikovat, přejmenovat, export, smazat).',
            )

            with card:
                with ui.row().classes("w-full items-center no-wrap"):
                    # tlačítka vlevo v button group
                    with ui.button_group().props("flat"):
                        open_btn = ui.button(
                            icon='folder_open',
                            on_click=wrap_ui_handler('home.open.click', lambda exp_id=exp.id: open_experiment(exp_id), data=lambda: {'id': exp.id, 'name': exp.name}),
                            ).props("color=primary")
                        attach_tooltip(open_btn, 'Otevřít', 'Nastaví experiment jako aktuální a přejde na stránku „Experiment“.')

                        dup_btn = ui.button(
                            icon='content_copy',
                            on_click=wrap_ui_handler('home.duplicate.click', lambda: duplicate_dialog(exp.id, exp.name), data=lambda: {'id': exp.id, 'name': exp.name}),
                            ).props("color=secondary")
                        attach_tooltip(dup_btn, 'Duplikovat', 'Vytvoří kopii experimentu pod novým názvem.')

                        ren_btn = ui.button(
                            icon='drive_file_rename_outline',
                            on_click=wrap_ui_handler('home.rename.click', lambda: rename_dialog(exp.id, exp.name), data=lambda: {'id': exp.id, 'name': exp.name}),
                            ).props('color=accent')
                        attach_tooltip(ren_btn, 'Přejmenovat', 'Změní název experimentu (musí být unikátní).')

                        def do_export() -> None:
                            try:
                                zip_bytes, filename = export_experiment(DB_PATH, int(exp.id), include_folder=True)
                            except Exception as ex:
                                ui.notify(f'❌ Export selhal: {ex}', type='negative')
                                return
                            ui.download(zip_bytes, filename=filename)
                            ui.notify('📦 Export připraven ke stažení')

                        exp_btn = ui.button(
                            icon='download',
                            on_click=wrap_ui_handler('home.export.click', do_export, data=lambda: {'id': exp.id, 'name': exp.name}),
                            ).props('color=info')
                        attach_tooltip(exp_btn, 'Exportovat', 'Stáhne export experimentu (DB data + soubory) jako ZIP.')

                        del_btn = ui.button(
                            icon='delete',
                            on_click=wrap_ui_handler('home.delete.click', lambda: delete_dialog(exp.id, exp.name), data=lambda: {'id': exp.id, 'name': exp.name}),
                            ).props("color=negative")
                        attach_tooltip(del_btn, 'Smazat', 'Otevře potvrzení pro trvalé smazání experimentu.')

                    with ui.column().classes("q-ml-md gap-0"):
                        name_lbl = ui.html(f"#{int(exp.id)} – <b>{html.escape(str(exp.name))}</b>").classes("text-subtitle1")
                        attach_tooltip(name_lbl, 'Identifikace', 'ID je interní identifikátor v DB; název je uživatelský popis experimentu.')

                        dt_lbl = ui.label(
                            f"Vytvořeno: {_fmt_dt(exp.created_at)}  ·  Upraveno: {_fmt_dt(exp.updated_at)}"
                        ).classes("text-caption")
                        attach_tooltip(dt_lbl, 'Časy', 'Vytvořeno = založení experimentu; Upraveno = poslední změna (např. data, výpočty, nastavení).')

                        n_comp = computations_count.get(int(exp.id), 0)
                        n_files = files_count.get(int(exp.id), 0)
                        stats_lbl = ui.label(
                            f"Souborů: {n_files}  ·  Konfigurovaných výpočtů: {n_comp}"
                        ).classes('text-caption')
                        attach_tooltip(stats_lbl, 'Souhrn', 'Rychlý přehled: počet nahraných souborů a počet uložených konfigurovatelných výpočtů (grafů).')

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

        with ui.row().classes("q-gutter-md"):
            new_btn = ui.button("Nový experiment", on_click=wrap_ui_handler('home.new_experiment.click', new_experiment_dialog)).props("unelevated")
            attach_tooltip(new_btn, 'Nový experiment', 'Otevře dialog pro vytvoření nového experimentu (projektu).')

            open_last_btn = ui.button("Otevřít poslední experiment", on_click=wrap_ui_handler('home.open_last.click', open_last)).props("unelevated")
            attach_tooltip(open_last_btn, 'Otevřít poslední', 'Otevře naposledy otevřený experiment uložený v app_meta.')

            import_btn = ui.button(
                "Importovat experiment",
                on_click=wrap_ui_handler('home.import.click', import_experiment_dialog),
            ).props("unelevated")
            attach_tooltip(import_btn, 'Importovat experiment', 'Nahraje exportovaný ZIP a přidá experiment do databáze.')

            export_all_btn = ui.button(
                'Exportovat vše',
                on_click=wrap_ui_handler('home.export_all.click', export_all),
            ).props('unelevated')
            attach_tooltip(
                export_all_btn,
                'Exportovat vše',
                'Vytvoří jeden ZIP obsahující exporty všech experimentů (a globální nastavení). Hodí se pro zálohu nebo přesun mezi počítači.',
            )

        ui.separator()

        experiments_lists()
