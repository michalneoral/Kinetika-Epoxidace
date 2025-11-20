import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.lines import Line2D
from typing import Union, Tuple
import matplotlib.colors as mcolors
from app.webapp.utils.colors import convert_color


class AdvancedPlotter:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.data = {}
        self.extract_data_from_model()
        self.axes_values = None

    def extract_data_from_model(self, t_max=2000, time_points=2000):
        # Column names
        self.data['column_names'] = deepcopy(self.model.column_names)

        # Measured values (original)
        self.data['orig_data'] = deepcopy(self.model.original_data)
        self.data['orig_t'] = self.data['orig_data'][:, 0]
        self.data['orig_y'] = self.data['orig_data'][:, 1:]

        # Measured values (adjusted for computation)
        self.data['data_t'] = deepcopy(self.model.time_exp)
        self.data['data_y'] = deepcopy(self.model.y_exp)

        # Simulated values
        sim_t, sim_y = self.model.simulate(t_max=t_max, time_points=time_points)
        self.data['sim_t'] = sim_t
        self.data['sim_y'] = sim_y

        # Simulation results
        self.data['constants'] = deepcopy(self.model.get_constants_with_names())

    def get_n_curves(self):
        return self.data['data_y'].shape[1]

    def plot(self, ui=True, **kwargs):

        # === DEFAULT VALUES ===
        defaults = {
            'title': self.name,
            'figsize': (8, 8),
            'curve_styles': None,
            'legend_mode': 'components_only',
            'show_title': True,
            'xlabel': "Čas",
            'ylabel': "Koncentrace",
            'xlim_mode': 'default',
            'xlim': None,
            'ylim': (0, 1),
            'ylim_mode': 'default',
            'grid_config': {'visible': True},
            'grid_config_minor': {'visible': False},
        }

        # Merge kwargs with defaults
        config = {**defaults, **kwargs}

        # === UNPACK CONFIG ===
        title = config['title']
        figsize = config['figsize']
        if 'fig_width' in config and 'fig_height' in config and config['fig_width'] and config['fig_height']:
            figsize = (config['fig_width'], config['fig_height'])
        curve_styles = config['curve_styles']
        legend_mode = config['legend_mode']
        show_title = config['show_title']
        xlabel = config['xlabel']
        ylabel = config['ylabel']
        x_lim_value = config['xlim']
        if 'xlim_min' in config and config['xlim_min'] is not None and 'xlim_max' in config and config[
            'xlim_max'] is not None:
            x_lim_value = (min(config['xlim_min'], config['xlim_max']), max(config['xlim_min'], config['xlim_max']))
        xlim_mode = config['xlim_mode']
        xlim = self.xlim_mode_selector(x_lim_value, xlim_mode)
        grid_config = config['grid_config']
        grid_config_minor = config['grid_config_minor']

        y_lim_value = config['ylim']
        if 'ylim_min' in config and config['ylim_min'] is not None and 'ylim_max' in config and config[
            'ylim_max'] is not None:
            y_lim_value = (min(config['ylim_min'], config['ylim_max']), max(config['ylim_min'], config['ylim_max']))
        ylim_mode = config['ylim_mode']
        ylim = self.ylim_mode_selector(y_lim_value, ylim_mode)

        # === USAGE EXAMPLE ===
        colors = plt.cm.tab10.colors

        # LOADED DATA
        sim_y = self.data['sim_y']
        sim_t = self.data['sim_t']
        data_t = self.data['data_t']
        data_y = self.data['data_y']
        column_names = self.data['column_names']

        # PLOTTING
        fig, ax = plt.subplots(figsize=figsize)

        num_curves = sim_y.shape[0]

        custom_lines = []  # for combined symbol handles
        for i in range(num_curves):
            style = curve_styles[i] if curve_styles and i < len(curve_styles) else {}
            color = style.get("color", colors[i % len(colors)])
            linestyle = style.get("linestyle", "solid")
            linewidth = style.get("linewidth", 1.5)
            marker = style.get("marker", "o")
            markersize = style.get("markersize", 6)
            label = style.get("label", column_names[i])

            ax.plot(sim_t, sim_y[i], color=color, linestyle=linestyle, linewidth=linewidth)
            ax.plot(data_t, data_y[:, i], linestyle="None", marker=marker,
                    color=color, markersize=markersize)
            custom_lines.append(Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, label=label))

            if style.get('additional_text_enabled', False):
                adtext_x = style.get('additional_text_x', 0.5)
                adtext_y = style.get('additional_text_y', 0.5)
                adtext_text = style.get('additional_text_text', 'bla\n bla')
                adtext_text = adtext_text.replace('\\n', '\n')
                adtext_size = style.get('additional_text_size', 14)
                ax.text(adtext_x, adtext_y, adtext_text, fontsize=adtext_size, color=color)

        # Add legends depending on the mode
        self.add_legend(ax, custom_lines, legend_mode)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        if show_title and title is not None:
            plt.title(title)
        # print(grid_config)
        if grid_config_minor is not None and grid_config_minor.get("visible", True):
            if 'color' in grid_config_minor:
                grid_config_minor['color'], grid_config_minor['alpha'] = convert_color(grid_config_minor['color'],
                                                                                       'rgb-a')
            plt.grid(**grid_config_minor)
            plt.minorticks_on()
        if grid_config is not None and grid_config.get("visible", True):
            if 'color' in grid_config:
                grid_config['color'], grid_config['alpha'] = convert_color(grid_config['color'], 'rgb-a')
            plt.grid(**grid_config)

        # print(grid_config)
        # print(grid_config_minor)

        plt.tight_layout()

        self.save_axes_values(fig, ax)

        if not ui:
            plt.show()
        else:
            return fig

    def add_legend(self, ax, custom_lines, legend_mode):
        if legend_mode == "both":
            first_legend = ax.legend(handles=custom_lines, title="Komponenty", loc="upper right")
            ax.add_artist(first_legend)
            ax.legend(handles=[
                Line2D([0], [0], color='gray', linestyle='-', label='Simulace'),
                Line2D([0], [0], color='gray', linestyle='None', marker='o', label='Naměřené hodnoty')
            ], title="Typ dat", loc="lower right")
        elif legend_mode == "single":
            legend_lines = custom_lines + [
                Line2D([0], [0], color='gray', linestyle='-', label='Simulace'),
                Line2D([0], [0], color='gray', linestyle='None', marker='o', label='Naměřené hodnoty')
            ]
            ax.legend(handles=legend_lines, title=None)
        elif legend_mode == "components_only":
            ax.legend(handles=custom_lines, title=None)
        elif legend_mode == "None":
            pass
        else:
            raise ValueError("legend_mode must be one of 'both', 'single', 'components_only', or None")

    def get_xlim_data_minmax(self):
        return np.min(self.data['data_t']), np.max(self.data['data_t'])

    def xlim_mode_selector(self, xlim, xlim_mode):
        if xlim_mode == 'auto':
            xlim = (np.min(self.data['sim_t']), np.max(self.data['sim_t']))
        elif xlim_mode == 'all_data':
            xlim = (np.min(self.data['data_t']), np.max(self.data['data_t']))
        elif xlim_mode == 'default':
            xlim = (0, 400)
        elif xlim_mode == 'manual':
            xlim = xlim if xlim is not None else (np.min(self.data['data_t']), np.max(self.data['data_t']))
        else:
            xlim = None
        return xlim

    def ylim_mode_selector(self, ylim, ylim_mode):
        if ylim_mode == 'default':
            ylim = (0, 1)
        elif ylim_mode == 'manual':
            ylim = ylim
        else:
            ylim = None
        return ylim

    def rel_to_data(self, rel_x: float, rel_y: float):
        """
        Convert a relative click (0..1 in whole image) into graph data coordinates,
        using only normalized axes bbox (nx0, ny0, nx1, ny1) and xlim/ylim.
        """

        v = self.axes_values

        nx0, ny0 = v["nx0"] / 100, v["ny0"] / 100
        nx1, ny1 = v["nx1"] / 100, v["ny1"] / 100
        xmin, xmax = v["xlim"]
        ymin, ymax = v["ylim"]

        # # ----- 1) Check if click is inside the axes region -----
        # if not (nx0 <= rel_x <= nx1 and ny0 <= rel_y <= ny1):
        #     return None, None  # outside graph area

        # ----- 2) Normalize click inside axes -----
        # relative position inside graph (0..1)
        ax_rel_x = (rel_x - nx0) / (nx1 - nx0)

        # Y coordinate is inverted (browser vs matplotlib)
        ax_rel_y = ((1 - rel_y) - ny0) / (ny1 - ny0)

        # ----- 3) Map to data coordinates -----
        X = xmin + ax_rel_x * (xmax - xmin)
        Y = ymin + ax_rel_y * (ymax - ymin)
        return X, Y

    def save_axes_values(self, fig, ax):
        """
        Extracts all geometric information needed to convert pixel clicks
        (from a rendered PNG) into Matplotlib data coordinates.

        Stores:
            - figure pixel size (fig_w, fig_h)
            - axes bounding box in pixel coordinates (x0,y0,x1,y1)
            - normalized bbox (0..1)
            - data limits (xlim, ylim)
            - DPI
        """
        # --- FIGURE SIZE IN PIXELS ---
        fig_w, fig_h = fig.get_size_inches() * fig.dpi

        # --- AXES BOUNDING BOX (pixel coords) ---
        # get_window_extent returns in display/pixel units
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans)
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1

        # --- NORMALIZED AXIS BBOX (0..1) ---
        nx0 = x0 / fig_w
        ny0 = y0 / fig_h
        nx1 = x1 / fig_w
        ny1 = y1 / fig_h

        # --- DATA LIMITS ---
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Store results
        self.axes_values = {
            "fig_w": float(fig_w),
            "fig_h": float(fig_h),
            "dpi": float(fig.dpi),

            # axes in pixels
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),

            # axes in normalized [0,1]
            "nx0": float(nx0),
            "ny0": float(ny0),
            "nx1": float(nx1),
            "ny1": float(ny1),

            # data ranges
            "xlim": (float(xlim[0]), float(xlim[1])),
            "ylim": (float(ylim[0]), float(ylim[1])),
        }

    def get_axes_values(self):
        return self.axes_values
