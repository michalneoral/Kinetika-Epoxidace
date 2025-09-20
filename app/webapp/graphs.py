from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_kinetic_fit(
    df,
    time_col="time",
    epo_col="zastoupení Σ všech EPO",
    fame_col="zastoupení Σ všech unFAME",
    hydroxyl_col=None,
    xmax=400,
    round_decimals=None,
    figsize=(6, 5),
    title=None,
    legend_labels=("Epoxidy", "FAME"),
    colors=("red", "black"),
    lang="cz"
):
    """
    Plot exponential kinetic fits: EPO formation, FAME decay, and optionally hydroxyls.

    Parameters:
    - df: Pandas DataFrame
    - time_col: time column name
    - epo_col: epoxide column name
    - fame_col: unFAME column name
    - hydroxyl_col: hydroxyls column name (optional)
    - xmax: max x-axis value
    - round_decimals: round inputs to this many decimals
    - figsize: figure size
    - title: optional plot title
    - legend_labels: (epo_label, fame_label)
    - colors: (epo_color, fame_color)
    - lang: "cz" or "en"
    """

    # Clean and filter
    cols = [time_col, epo_col, fame_col]
    if hydroxyl_col:
        cols.append(hydroxyl_col)
    df_kin = df[cols].dropna()
    df_kin = df_kin[(df_kin[epo_col] > 0) & (df_kin[fame_col] > 0)]
    if hydroxyl_col:
        df_kin = df_kin[df_kin[hydroxyl_col] > 0]

    # Round if needed
    def clean(col):
        s = df_kin[col]
        return s.astype(float).round(round_decimals).values if round_decimals else s.astype(float).values

    t = clean(time_col)
    epo = clean(epo_col)
    fame = clean(fame_col)
    if hydroxyl_col:
        hydroxyls = clean(hydroxyl_col)
        total = epo + hydroxyls

    # Model functions
    def model_rise(t, a, k):
        return a * (1 - np.exp(-k * t))

    def model_decay_fixed(t, k):
        return np.exp(-k * t)

    # Fit
    popt_epo, _ = curve_fit(model_rise, t, epo, p0=[max(epo), 0.001])
    a_epo, k_epo = popt_epo

    popt_fame, _ = curve_fit(model_decay_fixed, t, fame, p0=[0.001])
    k_fame = popt_fame[0]

    # Prepare fit curve
    t_fit = np.linspace(0, xmax, 500)
    epo_fit = model_rise(t_fit, a_epo, k_epo)
    fame_fit = model_decay_fixed(t_fit, k_fame)

    # Plot
    plt.figure(figsize=figsize)
    mask = t <= xmax
    plt.plot(t[mask], epo[mask], 's', color=colors[0], label=legend_labels[0])
    plt.plot(t_fit, epo_fit, '-', color=colors[0], linewidth=1.5)

    plt.plot(t[mask], fame[mask], 's', color=colors[1], label=legend_labels[1])
    plt.plot(t_fit, fame_fit, '--', color=colors[1], linewidth=1.5)

    # Optional: hydroxyls and sum
    if hydroxyl_col:
        # Fit hydroxyls
        popt_hyd, _ = curve_fit(model_rise, t, hydroxyls, p0=[max(hydroxyls), 0.001])
        a_hyd, k_hyd = popt_hyd
        hyd_fit = model_rise(t_fit, a_hyd, k_hyd)

        # Fit sum
        popt_sum, _ = curve_fit(model_rise, t, total, p0=[max(total), 0.001])
        a_sum, k_sum = popt_sum
        sum_fit = model_rise(t_fit, a_sum, k_sum)

        # Plot
        plt.plot(t[mask], hydroxyls[mask], 's', color='purple', label="hydroxyly")
        plt.plot(t_fit, hyd_fit, '-', color='purple', linewidth=1.5)

        plt.plot(t[mask], total[mask], '*', color='gray', label="EPO + OH")
        plt.plot(t_fit, sum_fit, '-', color='gray', linewidth=1.5)

        # Annotate hydroxyls + sum
        plt.text(0.55 * xmax, 0.3 * max(total),
                 f"$k_{{\\mathrm{{hydroxyly}}}}$ = {k_hyd:.4f} min$^{{-1}}$",
                 fontsize=12, color='purple')

        plt.text(0.55 * xmax, 0.85 * max(total),
                 f"$k_{{\\mathrm{{EPO+OH}}}}$ = {k_sum:.4f} min$^{{-1}}$",
                 fontsize=12, color='gray')

    # Annotations for EPO and FAME
    plt.text(0.55 * xmax, 0.85 * max(epo),
             f"$k_{{\\mathrm{{EPO}}}}$ = {k_epo:.4f} min$^{{-1}}$",
             fontsize=12, color=colors[0])

    plt.text(0.55 * xmax, 0.15 * max(epo),
             f"$k_{{\\mathrm{{FAME}}}}$ = {k_fame:.4f} min$^{{-1}}$",
             fontsize=12, color=colors[1])

    # Labels
    if lang == "cz":
        plt.xlabel("čas [min]", fontsize=12)
        plt.ylabel("Relativní zastoupení, -", fontsize=12)
    else:
        plt.xlabel("Time [min]", fontsize=12)
        plt.ylabel("Relative concentration", fontsize=12)

    if title:
        plt.title(title, fontsize=13)

    plt.legend()
    plt.grid(False)
    plt.xlim(0, xmax)
    plt.tight_layout()
    plt.show()
