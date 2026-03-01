from nicegui import ui, app
from typing import List, Dict, Tuple, Set
import json


def create_app(node_names: List[str]) -> None:
    # stabilní Mermaid IDs (bez diakritiky / mezer)
    node_ids = [f"n{i}" for i in range(len(node_names))]
    name_by_id = dict(zip(node_ids, node_names))

    # stav uzlů (tlačítka nahoře)
    enabled_nodes: Dict[str, bool] = {nid: True for nid in node_ids}

    # stav hran: (from_id, to_id) -> True(enabled)/False(disabled)
    edge_state: Dict[Tuple[str, str], bool] = {}

    # hrany, které už uživatel ručně změnil (nepřepisovat defaultem)
    manual_edges: Set[Tuple[str, str]] = set()

    # aktuální pořadí hran v edit grafu (kvůli klik eventu indexem)
    current_edges: List[Tuple[str, str]] = []

    # mapování hrany -> k_index v edit grafu (k_1..k_m)
    edge_k_index: Dict[Tuple[str, str], int] = {}

    # view režim: uložený mermaid string (kompaktní, jen enabled hrany, se zachovanými k_X)
    saved_view_mermaid: str = "flowchart LR\n"

    # UI prvky
    mode_is_edit = True
    mode_button = None
    buttons_container = None
    mermaid_edit = None
    mermaid_view = None

    # ---------- pomocné funkce pro default ----------
    def compute_default_path_edges(enabled_ids: List[str]) -> Set[Tuple[str, str]]:
        """Default kostra: jen hrany mezi sousedními enabled uzly v aktuálním pořadí."""
        return {(enabled_ids[i], enabled_ids[i + 1]) for i in range(len(enabled_ids) - 1)}

    # ---------- Mermaid build (EDIT) ----------
    def build_mermaid_edit() -> str:
        nonlocal current_edges, edge_k_index

        enabled_ids = [nid for nid in node_ids if enabled_nodes[nid]]
        default_path_edges = compute_default_path_edges(enabled_ids)

        # kompletní DAG i<j
        current_edges = []
        for i in range(len(enabled_ids)):
            for j in range(i + 1, len(enabled_ids)):
                a, b = enabled_ids[i], enabled_ids[j]
                current_edges.append((a, b))

                # pokud hrana není ručně ovládaná, řiď se default kostrou
                if (a, b) not in manual_edges:
                    edge_state[(a, b)] = (a, b) in default_path_edges
                else:
                    edge_state.setdefault((a, b), False)

        # k indexy pro zachování názvů šipek při "Ulož"
        edge_k_index = {edge: idx + 1 for idx, edge in enumerate(current_edges)}

        lines = ["flowchart LR"]

        # uzly
        for nid in enabled_ids:
            label = name_by_id[nid].replace('"', '\\"')
            lines.append(f'{nid}["{label}"]')

        # hrany + labely k_1..k_m
        for edge, k in edge_k_index.items():
            a, b = edge
            lines.append(f"{a} -->|k_{k}| {b}")

        # styly hran (stroke); label dotáhneme JS, aby byl jistě obarven
        for idx, (a, b) in enumerate(current_edges):
            is_on = edge_state.get((a, b), False)
            if is_on:
                lines.append(f"linkStyle {idx} stroke:#16a34a,stroke-width:4px,stroke-opacity:1")
            else:
                lines.append(f"linkStyle {idx} stroke:#94a3b8,stroke-width:2.5px,stroke-opacity:0.12")

        return "\n".join(lines)

    # ---------- Mermaid build (VIEW / SAVED) ----------
    def build_mermaid_view_from_current_state() -> str:
        """
        Vygeneruje kompaktní graf jen z enabled hran, ale zachová názvy k_X z edit grafu
        (tj. používá edge_k_index).
        """
        enabled_ids = [nid for nid in node_ids if enabled_nodes[nid]]

        # uzly, které mají aspoň jednu enabled hranu, případně i osamocené (necháme je tam)
        # (Pokud bys chtěl zobrazit jen uzly, které se účastní enabled hran, řekni.)
        lines = ["flowchart LR"]

        for nid in enabled_ids:
            label = name_by_id[nid].replace('"', '\\"')
            lines.append(f'{nid}["{label}"]')

        # jen enabled hrany, se zachovaným k číslem
        enabled_edges_in_order = [e for e in current_edges if edge_state.get(e, False)]
        for edge in enabled_edges_in_order:
            a, b = edge
            k = edge_k_index.get(edge, None)
            if k is None:
                continue
            lines.append(f"{a} -->|k_{k}| {b}")

        # v view módu můžeme nastavit všechny hrany stejně (zeleně), bez kliků
        for idx in range(len(enabled_edges_in_order)):
            lines.append(f"linkStyle {idx} stroke:#16a34a,stroke-width:4px,stroke-opacity:1")

        return "\n".join(lines)

    # ---------- JS pro EDIT mód: kliky + styl labelů ----------
    def run_wire_and_style_js_edit(client_like=None) -> None:
        enabled_list = [edge_state.get(e, False) for e in current_edges]
        enabled_list_json = json.dumps(enabled_list)

        js = f"""
        (() => {{
          const root = document.getElementById('mermaid_graph_edit');
          if (!root) return;

          const enabled = {enabled_list_json};

          const apply = () => {{
            const svg = root.querySelector('svg');
            if (!svg) return false;

            const edgePaths = Array.from(svg.querySelectorAll('g.edgePath'));
            const edgeLabels = Array.from(svg.querySelectorAll('g.edgeLabel'));

            edgePaths.forEach((g, idx) => {{
              g.style.cursor = 'pointer';
              g.onclick = () => emitEvent('edge_click', idx);
              const on = !!enabled[idx];
              g.style.opacity = on ? '1' : '0.45';
            }});

            edgeLabels.forEach((g, idx) => {{
              g.style.cursor = 'pointer';
              g.onclick = () => emitEvent('edge_click', idx);
              const on = !!enabled[idx];
              g.style.opacity = on ? '1' : '0.45';

              const text = g.querySelector('text');
              if (text) {{
                text.style.fill = on ? '#16a34a' : '#cbd5e1';
                text.style.fontWeight = on ? '600' : '400';
              }}
            }});

            return true;
          }};

          let attempts = 0;
          const timer = setInterval(() => {{
            attempts++;
            if (apply() || attempts > 30) clearInterval(timer);
          }}, 50);
        }})();
        """

        if client_like is None:
            ui.run_javascript(js)
        else:
            client_like.run_javascript(js)

    # ---------- render / refresh ----------
    def render_node_buttons() -> None:
        buttons_container.clear()
        with buttons_container:
            for nid in node_ids:
                is_on = enabled_nodes[nid]
                label = name_by_id[nid]

                def handler(_nid=nid):
                    enabled_nodes[_nid] = not enabled_nodes[_nid]
                    refresh_all()

                if is_on:
                    ui.button(label, on_click=handler).props("color=primary").classes("q-mr-sm q-mb-sm")
                else:
                    ui.button(label, on_click=handler).props("outline color=grey").classes("q-mr-sm q-mb-sm")

    def refresh_edit_graph() -> None:
        mermaid_edit.content = build_mermaid_edit()
        run_wire_and_style_js_edit()

    def refresh_view_graph() -> None:
        mermaid_view.content = saved_view_mermaid

    def refresh_all() -> None:
        # když jsme v edit módu, refreshuje se edit graf (a view zatím neřešíme)
        render_node_buttons()
        if mode_is_edit:
            refresh_edit_graph()

    # ---------- eventy ----------
    def on_edge_click(e) -> None:
        # v view módu by se to nemělo stát (nejsou wired kliky),
        # ale pro jistotu:
        if not mode_is_edit:
            return

        idx = int(e.args)
        if 0 <= idx < len(current_edges):
            key = current_edges[idx]
            manual_edges.add(key)
            edge_state[key] = not edge_state.get(key, False)
            refresh_all()

    def switch_to_view_mode() -> None:
        nonlocal mode_is_edit, saved_view_mermaid

        # nejdřív přepočti edit graf (aby current_edges + k-indexy seděly)
        refresh_edit_graph()

        # uložený kompaktní graf (jen enabled hrany, zachová k_X)
        saved_view_mermaid = build_mermaid_view_from_current_state()
        refresh_view_graph()

        mode_is_edit = False
        mode_button.text = "Uprav"

        # UI přepínání
        buttons_container.visible = False
        mermaid_edit.visible = False
        mermaid_view.visible = True

    def switch_to_edit_mode() -> None:
        nonlocal mode_is_edit
        mode_is_edit = True
        mode_button.text = "Ulož"

        buttons_container.visible = True
        mermaid_edit.visible = True
        mermaid_view.visible = False

        refresh_all()

    def on_mode_button_click() -> None:
        if mode_is_edit:
            switch_to_view_mode()
        else:
            switch_to_edit_mode()

    # ---------- UI ----------
    ui.label("Uprav/Ulož: editovatelný kompletní graf vs. uložený kompaktní graf jen s enabled cestami").classes(
        "text-h5"
    )

    with ui.row().classes("items-center"):
        mode_button = ui.button("Ulož", on_click=on_mode_button_click).props("color=positive")
        ui.label("V edit módu můžeš klikat na uzly i hrany. V uloženém módu je graf jen pro čtení.").classes(
            "text-body2 text-grey-7"
        )

    buttons_container = ui.row().classes("items-center")
    ui.separator()

    mermaid_edit = ui.mermaid(
        build_mermaid_edit(),
        config={
            "securityLevel": "loose",
            # čitelný edit layout (4–8 uzlů)
            "flowchart": {"nodeSpacing": 110, "rankSpacing": 130},
        },
    ).props("id=mermaid_graph_edit").classes("w-full")

    mermaid_view = ui.mermaid(
        saved_view_mermaid,
        config={
            "securityLevel": "loose",
            # kompaktnější uložený layout
            "flowchart": {"nodeSpacing": 70, "rankSpacing": 80},
        },
    ).props("id=mermaid_graph_view").classes("w-full")

    mermaid_view.visible = False  # start v edit módu

    ui.on("edge_click", on_edge_click)

    render_node_buttons()

    @app.on_connect
    def _on_connect(client):
        # kliky + barvy labelů jen pro edit graf
        client.run_javascript("setTimeout(() => {}, 0);")
        run_wire_and_style_js_edit(client)


# ---- POUŽITÍ ----
create_app(["Reaktor", "Míchadlo", "Ohřev", "Chlazení", "Separace", "Výstup"])
ui.run()
