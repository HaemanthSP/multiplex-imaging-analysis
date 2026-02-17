"""Interactive widgets for human-in-the-loop stages (4 & 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


# ---------------------------------------------------------------------------
# Stage 4: Cluster labelling & grouping
# ---------------------------------------------------------------------------


def create_cluster_label_widget(
    cluster_ids: list[str],
    save_callback: Callable[[pd.DataFrame], None] | None = None,
) -> widgets.VBox:
    """Create an interactive widget for labelling clusters and assigning groups.

    Parameters
    ----------
    cluster_ids : list[str]
        List of cluster IDs from Leiden clustering.
    save_callback : callable, optional
        Function called with the resulting DataFrame when Save is clicked.

    Returns
    -------
    widgets.VBox
        Widget to display in a notebook.
    """
    label_inputs = {}
    group_inputs = {}

    header = widgets.HTML("<h3>Cluster Labelling & Grouping</h3>")
    instructions = widgets.HTML(
        "<p>For each cluster, enter a <b>cell-type label</b> "
        "(e.g. 'CD8 T Cell') and a <b>group name</b> "
        "(clusters sharing a group will be pooled together in Stage 5).</p>"
    )

    rows = []
    for cid in sorted(cluster_ids, key=lambda x: int(x) if x.isdigit() else x):
        lbl = widgets.Text(
            value="",
            placeholder="e.g. CD8 T Cell",
            description=f"Cluster {cid}:",
            layout=widgets.Layout(width="350px"),
        )
        grp = widgets.Text(
            value="",
            placeholder="e.g. T Cells",
            description="Group:",
            layout=widgets.Layout(width="350px"),
        )
        label_inputs[cid] = lbl
        group_inputs[cid] = grp
        rows.append(widgets.HBox([lbl, grp]))

    output = widgets.Output()

    def on_save(_):
        with output:
            clear_output()
            data = []
            for cid in cluster_ids:
                label_val = label_inputs[cid].value.strip()
                group_val = group_inputs[cid].value.strip()
                if not label_val:
                    label_val = "Unspecific"
                if not group_val:
                    group_val = label_val  # default group = label
                data.append(
                    {"cluster_id": cid, "label": label_val, "group": group_val}
                )
            df = pd.DataFrame(data)
            display(df)
            if save_callback:
                save_callback(df)
            print("✓ Cluster labels saved!")

    save_btn = widgets.Button(
        description="Save Labels",
        button_style="success",
        icon="check",
    )
    save_btn.on_click(on_save)

    return widgets.VBox([header, instructions, *rows, save_btn, output])


# ---------------------------------------------------------------------------
# Stage 6: Subclustering resolution selection
# ---------------------------------------------------------------------------


def create_resolution_picker(
    group_name: str,
    resolution_options: list[float],
    n_clusters_per_res: dict[float, int],
    default_resolution: float | None = None,
    on_select: Callable[[str, float], None] | None = None,
) -> widgets.VBox:
    """Create a widget for selecting the best subclustering resolution.

    Parameters
    ----------
    group_name : str
        Name of the meta-cluster group.
    resolution_options : list[float]
        Available resolutions.
    n_clusters_per_res : dict[float, int]
        Number of subclusters generated at each resolution.
    default_resolution : float, optional
        Pre-selected resolution.
    on_select : callable, optional
        Callback ``(group_name, selected_resolution)``.

    Returns
    -------
    widgets.VBox
    """
    header = widgets.HTML(f"<h4>Select resolution for: <b>{group_name}</b></h4>")

    options_display = [
        (f"res={r:.2f}  →  {n_clusters_per_res.get(r, '?')} subclusters", r)
        for r in resolution_options
    ]

    dropdown = widgets.Dropdown(
        options=options_display,
        value=default_resolution if default_resolution in resolution_options else resolution_options[0],
        description="Resolution:",
        layout=widgets.Layout(width="400px"),
    )

    output = widgets.Output()

    def on_confirm(_):
        with output:
            clear_output()
            selected = dropdown.value
            print(f"✓ Selected resolution {selected:.2f} for '{group_name}'")
            if on_select:
                on_select(group_name, selected)

    confirm_btn = widgets.Button(
        description="Confirm",
        button_style="primary",
        icon="check",
    )
    confirm_btn.on_click(on_confirm)

    return widgets.VBox([header, dropdown, confirm_btn, output])


def create_batch_resolution_picker(
    group_results: dict[str, dict[float, int]],
    on_save: Callable[[dict[str, float]], None] | None = None,
) -> widgets.VBox:
    """Create a combined widget for selecting resolutions for all groups.

    Parameters
    ----------
    group_results : dict[str, dict[float, int]]
        Mapping of group_name → {resolution: n_subclusters}.
    on_save : callable, optional
        Callback with ``{group_name: selected_resolution}`` dict.

    Returns
    -------
    widgets.VBox
    """
    header = widgets.HTML("<h3>Select Subclustering Resolutions</h3>")
    instructions = widgets.HTML(
        "<p>Review the visual outputs above, then select the best resolution "
        "for each meta-cluster group.</p>"
    )

    dropdowns: dict[str, widgets.Dropdown] = {}
    rows = []

    for group_name, res_map in sorted(group_results.items()):
        options = [
            (f"res={r:.2f} → {n} subclusters", r)
            for r, n in sorted(res_map.items())
        ]
        dd = widgets.Dropdown(
            options=options,
            description=f"{group_name}:",
            layout=widgets.Layout(width="500px"),
            style={"description_width": "150px"},
        )
        dropdowns[group_name] = dd
        rows.append(dd)

    output = widgets.Output()

    def on_save_click(_):
        with output:
            clear_output()
            selections = {name: dd.value for name, dd in dropdowns.items()}
            for name, res in selections.items():
                print(f"  {name}: resolution={res:.2f}")
            if on_save:
                on_save(selections)
            print("\n✓ All resolutions saved!")

    save_btn = widgets.Button(
        description="Save All Selections",
        button_style="success",
        icon="check",
    )
    save_btn.on_click(on_save_click)

    return widgets.VBox([header, instructions, *rows, save_btn, output])
