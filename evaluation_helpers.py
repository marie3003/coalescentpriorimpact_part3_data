"""
Helper functions for evaluating the BEAST analyses of the pertussis (France
2010-2024) and SARS-CoV-2 (Alpha variant, UK 2020) data sets under a skygrid
and a constant-size coalescent prior.

Used by the notebooks in the repository root and by the scripts in scripts/.
"""

import copy
import re
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import dendropy
from dendropy.calculate import treecompare
from scipy.stats import gaussian_kde, pearsonr, spearmanr


# =============================================================================
# Shared constants
# =============================================================================

SKYGRID_COLOR = '#80557e'
SKYGRID_SAMPLE_COLOR = '#c9afc8'
CONSTCOAL_COLOR = '#397398'
CONSTCOAL_SAMPLE_COLOR = '#91bdd4'
RED = '#a6444f'
GREY = '#999999'
BLACK = '#222222'

# pertussis ptxP x fim3 subtype combinations used to colour trees
SUBTYPE_COLORS = {
    ('ptxP1', 'fim3-1'): CONSTCOAL_COLOR,
    ('ptxP3', 'fim3-1'): SKYGRID_COLOR,
    ('ptxP3', 'fim3-2'): RED,
}

# trailing "|YYYY-MM-DD" in GISAID-style tip labels (SARS-CoV-2 trees)
_DATE_SUFFIX_RE = re.compile(r"\|(\d{4}-\d{2}-\d{2})$")


# =============================================================================
# General utilities
# =============================================================================

def read_beast_log(log_path):
    """Read a BEAST log file (skipping # comment lines) into a DataFrame."""
    return pd.read_csv(log_path, sep='\t', comment='#')


def _decimal_year_to_datetime(decimal_year):
    """Convert a decimal year (e.g. 2020.997) to a datetime, accounting for leap years."""
    year = int(decimal_year)
    remainder = decimal_year - year
    start = datetime(year, 1, 1)
    next_start = datetime(year + 1, 1, 1)
    # round to the nearest second so float error doesn't land just before midnight
    seconds = round((next_start - start).total_seconds() * remainder)
    return start + timedelta(seconds=seconds)


def _decimal_year(date):
    """Convert a date (decimal-year float, 'YYYY-MM-DD' string, or datetime) to a decimal year."""
    if isinstance(date, (int, float)):
        return float(date)
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    year = date.year
    start = datetime(year, 1, 1)
    next_start = datetime(year + 1, 1, 1)
    return year + (date - start) / (next_start - start)


def hpd_interval(samples, alpha=0.05):
    """Compute the highest posterior density interval."""
    samples = np.sort(samples)
    n = len(samples)
    interval_size = int(np.floor((1 - alpha) * n))
    widths = samples[interval_size:] - samples[:n - interval_size]
    min_idx = np.argmin(widths)
    return samples[min_idx], samples[min_idx + interval_size]


def _hpd_columns(samples):
    """95% HPD (lower, upper) arrays for each column of a (n_samples, n) array."""
    bounds = np.array([hpd_interval(samples[:, i]) for i in range(samples.shape[1])])
    return bounds[:, 0], bounds[:, 1]


def _save_figure(fig, save_path, bbox_inches='tight'):
    """Save a figure at 300 dpi with a transparent background (vector output for .pdf)."""
    fig.savefig(save_path, dpi=300,
                format='pdf' if save_path.endswith('.pdf') else None,
                bbox_inches=bbox_inches, transparent=True)


def _tree_schema(path):
    """Guess the tree file format from its extension (Newick or NEXUS)."""
    return 'newick' if path.endswith(('.nwk', '.newick', '.nw')) else 'nexus'


def _read_tree(path):
    """Read a single tree (NEXUS or Newick), keeping underscores in tip labels."""
    return dendropy.Tree.get(path=path, schema=_tree_schema(path), preserve_underscores=True)


def _isolate_date_lookup(meta_df):
    """
    Map isolate -> decimal_year, with underscores replaced by spaces to match
    the tip labels dendropy produces when reading trees without
    `preserve_underscores`.
    """
    return (
        meta_df.assign(Isolate=meta_df['Isolate'].str.replace('_', ' '))
        .set_index('Isolate')['decimal_year'].to_dict()
    )


def _calendar_ticks(t_min, t_max, maxticks):
    """
    Calendar-date ticks for short time spans (decimal years in, decimal years
    out). Labels are 'Mon YYYY' if all ticks fall on month starts, otherwise
    'YYYY-MM-DD'.
    """
    loc = mdates.AutoDateLocator(minticks=3, maxticks=maxticks)
    tick_dts = [mdates.num2date(v).replace(tzinfo=None)
                for v in loc.tick_values(_decimal_year_to_datetime(t_min),
                                         _decimal_year_to_datetime(t_max))]
    date_fmt = '%b %Y' if all(d.day == 1 for d in tick_dts) else '%Y-%m-%d'
    return [_decimal_year(d) for d in tick_dts], [d.strftime(date_fmt) for d in tick_dts]


# =============================================================================
# Population size trajectories and root heights (BEAST log files)
# =============================================================================

def compute_skygrid_summary(log_df, t_present, end_time, cutoff=24, n_grid_points=48):
    """
    Compute median and 95% HPD for each skygrid interval.

    Grid points are fixed every (cutoff / n_grid_points) years back from
    t_present, covering `cutoff` years in total (the skygrid's piecewise-constant
    period). The last interval extends to end_time, where the population is
    assumed to stay constant. Defaults (cutoff=24, n_grid_points=48) reproduce
    the 0.5-year intervals used for the pertussis example.

    Accepts either skygrid.logPopSize* columns or bPopSizes.*/bGroupSizes.* columns.

    Returns dict with:
      boundaries : n_grid_points + 2 time points (right edge of each interval + left edge of last)
      median, hpd_lower, hpd_upper : arrays of length n_grid_points + 1
      samples : (n_samples, n_grid_points + 1) array of population sizes (linear scale)
    """
    if any(c.startswith('skygrid.logPopSize') for c in log_df.columns):
        pop_cols = sorted(
            [c for c in log_df.columns if c.startswith('skygrid.logPopSize')],
            key=lambda x: int(x.replace('skygrid.logPopSize', ''))
        )
        pop_sizes = np.exp(log_df[pop_cols].values)
    elif any(c.startswith('bPopSizes.') for c in log_df.columns):
        pop_cols = sorted(
            [c for c in log_df.columns if c.startswith('bPopSizes.')],
            key=lambda x: int(x.split('.')[1])
        )
        pop_sizes = log_df[pop_cols].values   # already on linear scale
    else:
        raise ValueError("No recognised population size columns found in log_df.")
    n = pop_sizes.shape[1]

    interval_width = cutoff / n_grid_points

    # boundaries[i] = right edge of interval i, boundaries[n] = left edge of last interval
    boundaries = [t_present - i * interval_width for i in range(n)] + [end_time]
    hpd_lower, hpd_upper = _hpd_columns(pop_sizes)

    return {
        'boundaries': boundaries,
        'median': np.median(pop_sizes, axis=0),
        'hpd_lower': hpd_lower,
        'hpd_upper': hpd_upper,
        'samples': pop_sizes,
    }


def compute_constcoal_summary(log_df):
    """
    Compute median and 95% HPD for the constant coalescent population size.
    Accepts 'constant.popSize', 'bPopSizes.1' or 'bPopSizes' as the population size column.
    """
    for col in ('constant.popSize', 'bPopSizes.1', 'bPopSizes'):
        if col in log_df.columns:
            samples = log_df[col].values
            break
    else:
        raise ValueError("No recognised population size column found in log_df.")
    lower, upper = hpd_interval(samples)
    return {
        'median': np.median(samples),
        'hpd_lower': lower,
        'hpd_upper': upper,
        'samples': samples,
    }


def compute_root_height_summary(skygrid_log, constcoal_log, column='treeModel.rootHeight'):
    """
    Compute median and 95% HPD of a log column (by default the root height)
    for both models.

    Returns a DataFrame indexed by model ('skygrid', 'constcoal') with
    columns median, hpd_lower, hpd_upper.
    """
    rows = {}
    for name, log_df in [('skygrid', skygrid_log), ('constcoal', constcoal_log)]:
        samples = log_df[column].values
        lower, upper = hpd_interval(samples)
        rows[name] = {'median': np.median(samples), 'hpd_lower': lower, 'hpd_upper': upper}
    return pd.DataFrame(rows).T


def fit_exponential_growth_rate(skygrid_summary, start_date, end_date):
    """
    Fit an exponential growth rate to the skygrid population size trajectory
    between `start_date` and `end_date`, separately for each posterior sample
    (log-linear regression of population size vs. time).

    Parameters
    ----------
    skygrid_summary : dict
        Output of compute_skygrid_summary().
    start_date, end_date : float, str, or datetime
        Window to fit within. Accepts a decimal year (e.g. 2020.667), an ISO
        date string ('2020-09-01'), or a datetime.

    Returns
    -------
    dict with:
      rate_median, rate_hpd_lower, rate_hpd_upper : growth rate per year
      rate_samples, intercept_samples : (n_samples,) arrays of per-sample fits
      t_window : the boundary time points (decimal years) used for the fit
    """
    start_yr = _decimal_year(start_date)
    end_yr = _decimal_year(end_date)
    if start_yr >= end_yr:
        raise ValueError(f"start_date ({start_yr}) must be before end_date ({end_yr}).")

    boundaries = np.array(skygrid_summary['boundaries'][:-1])  # one per population-size column
    mask = (boundaries >= start_yr) & (boundaries <= end_yr)
    if mask.sum() < 2:
        raise ValueError(
            f"Fewer than 2 skygrid grid points fall within [{start_yr}, {end_yr}]; "
            f"cannot fit a growth rate. Check n_grid_points/cutoff resolution."
        )

    t = boundaries[mask]
    pop = skygrid_summary['samples'][:, mask]  # (n_samples, n_window_points)

    fits = np.array([np.polyfit(t, np.log(row), 1) for row in pop])  # (n_samples, 2): [rate, intercept]
    rates, intercepts = fits[:, 0], fits[:, 1]

    lo, hi = hpd_interval(rates)
    return {
        'rate_median': float(np.median(rates)),
        'rate_hpd_lower': float(lo),
        'rate_hpd_upper': float(hi),
        'rate_samples': rates,
        'intercept_samples': intercepts,
        't_window': t,
    }


def growth_fit_curve(growth_fit, n_points=100):
    """
    Reconstruct the fitted exponential curve (median + 95% HPD) from the
    output of fit_exponential_growth_rate(), evaluated at `n_points` points
    spanning the fitted window.

    Returns dict with:
      t : (n_points,) decimal-year time points
      median, hpd_lower, hpd_upper : (n_points,) population size arrays
    """
    t_window = growth_fit['t_window']
    t = np.linspace(t_window.min(), t_window.max(), n_points)
    r = growth_fit['rate_samples'][:, None]
    b = growth_fit['intercept_samples'][:, None]
    log_curves = r * t[None, :] + b  # (n_samples, n_points), linear in t
    log_lo, log_hi = _hpd_columns(log_curves)

    return {
        't': t,
        'median': np.exp(np.median(log_curves, axis=0)),
        'hpd_lower': np.exp(log_lo),
        'hpd_upper': np.exp(log_hi),
    }


def _skygrid_step_xy(boundaries, values):
    """Build x, y arrays for a step-function plot from past (left) to present (right)."""
    x, y = [], []
    for i in range(len(values) - 1, -1, -1):  # oldest to most recent so segments connect left→right
        x.extend([boundaries[i + 1], boundaries[i]])
        y.extend([values[i], values[i]])
    return np.array(x), np.array(y)


def _draw_tmrca_marker(ax, y, t_med, color, label, fontsize, t_lo=None, t_hi=None):
    """Dot (median) with optional 95% HPD bar at height `y` (axes fraction), labelled above."""
    xtrans = ax.get_xaxis_transform()  # x in data units, y in axes fraction
    if t_lo is not None and t_hi is not None:
        ax.plot([t_lo, t_hi], [y, y], color=color, linewidth=2,
                transform=xtrans, zorder=5, solid_capstyle='butt')
        ax.plot([t_lo, t_hi], [y, y], linestyle='none', marker='|', markersize=8,
                color=color, transform=xtrans, zorder=5)
    ax.plot([t_med], [y], marker='o', markersize=7, color=color,
            transform=xtrans, zorder=6)
    ax.text(t_med, y + 0.015, label, color=color, ha='center', va='bottom',
            fontsize=fontsize, transform=xtrans, zorder=6)


def plot_pertussis_skygrid(
    skygrid_summary,
    constcoal_summary,
    n_grey_samples=50,
    y_range=None,
    cutoff=None,
    time_unit='years',
    log_scale=True,
    title="",
    x_label=None,
    ax=None,
    figsize=(10, 4),
    save_path=None,
    fontsize=12,
    tick_fontsize=None,
    legend_fontsize=None,
    title_fontsize=None,
    growth_fit=None,
    growth_fit_color='black',
    growth_fit_hpd=True,
    growth_fit_n_points=100,
    root_height_summary=None,
    external_tmrca=None,
):
    """
    Plot skygrid and constant-coalescent population size estimates (used for
    both the pertussis and the SARS-CoV-2 data set).

    Parameters
    ----------
    skygrid_summary : dict
        Output of compute_skygrid_summary().
    constcoal_summary : dict
        Output of compute_constcoal_summary().
    n_grey_samples : int
        Number of posterior sample trajectories to draw per model.
    y_range : tuple or None
        (y_min, y_max) for the y-axis.
    cutoff : float or None
        If set, only show the time range from `cutoff` up to the highest
        (present-day) time point, i.e. everything more recent than `cutoff`.
        Also used as the day-0 reference when `time_unit='days'`.
    time_unit : {'years', 'days', 'dates'}
        If 'days', the x-axis is rescaled to days since `cutoff` (which must
        be set) instead of absolute years, i.e. `cutoff` becomes 0 days.
        If 'dates', the x-axis shows calendar dates, with `cutoff` (a decimal
        year, if set) still clipping the earliest date shown.
    title : str
        Plot title.
    x_label : str or None
        Custom x-axis label. If None, defaults to 'Time (days)', 'Date' or
        'Time (years)' depending on `time_unit`.
    ax : matplotlib Axes or None
        If None, a new figure is created.
    save_path : str or None
        If set, save the figure to this path (use .pdf for vector output).
    growth_fit : dict or None
        Output of fit_exponential_growth_rate(); if set, overlays the fitted
        exponential growth line (median, with 95% HPD band if
        `growth_fit_hpd=True`), restricted to the fitted date window.
    growth_fit_color : str
        Colour for the growth-fit overlay.
    growth_fit_hpd : bool
        If True, shade the 95% HPD band of the fitted growth curve.
    growth_fit_n_points : int
        Number of points to evaluate the fitted curve at.
    root_height_summary : DataFrame or None
        Output of compute_root_height_summary(); if set, marks the tMRCA
        (median dot and 95% HPD bar) of both models at the top of the plot.
        Root heights are converted to times by subtracting them from the most
        recent time point.
    external_tmrca : dict or None
        Optional externally-estimated tMRCA (e.g. from a published paper),
        drawn in the same style below the model tMRCAs. Keys:
          'median' : point estimate (decimal year, 'YYYY-MM-DD' string, or datetime)
          'hpd_lower', 'hpd_upper' : optional 95% HPD bounds, same formats
          'label' : annotation text, default 'External tMRCA'
          'color' : hex colour, default the project's red
    """
    sky = skygrid_summary
    const = constcoal_summary

    if time_unit not in ('years', 'days', 'dates'):
        raise ValueError("time_unit must be 'years', 'days', or 'dates'.")
    if time_unit == 'days' and cutoff is None:
        raise ValueError("cutoff must be set when time_unit='days'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    _tick_fs = tick_fontsize if tick_fontsize is not None else fontsize - 2
    _legend_fs = legend_fontsize if legend_fontsize is not None else fontsize - 2
    _title_fs = title_fontsize if title_fontsize is not None else fontsize + 1

    def _to_axis_units(decimal_years):
        if time_unit == 'days':
            return [(t - cutoff) * 365.25 for t in decimal_years]
        elif time_unit == 'dates':
            return [mdates.date2num(_decimal_year_to_datetime(t)) for t in decimal_years]
        return list(decimal_years)

    boundaries = _to_axis_units(sky['boundaries'])

    # --- posterior samples ---
    x_left, x_right = boundaries[-1], boundaries[0]
    sky_indices = np.random.choice(sky['samples'].shape[0], size=n_grey_samples, replace=False)
    const_indices = np.random.choice(len(const['samples']), size=n_grey_samples, replace=False)
    for sky_idx, const_idx in zip(sky_indices, const_indices):
        x, y = _skygrid_step_xy(boundaries, sky['samples'][sky_idx])
        ax.plot(x, y, color=SKYGRID_SAMPLE_COLOR, linewidth=0.5, alpha=0.6, zorder=1)
        ax.hlines(const['samples'][const_idx], x_left, x_right,
                  colors=CONSTCOAL_SAMPLE_COLOR, linewidth=0.5, alpha=0.6, zorder=1)

    # --- skygrid median and 95% HPD ---
    _, y_lower = _skygrid_step_xy(boundaries, sky['hpd_lower'])
    x_upper, y_upper = _skygrid_step_xy(boundaries, sky['hpd_upper'])
    ax.fill_between(x_upper, y_lower, y_upper,
                    alpha=0.25, color=SKYGRID_COLOR, zorder=2, label='Skygrid 95% HPD')
    x_med, y_med = _skygrid_step_xy(boundaries, sky['median'])
    ax.plot(x_med, y_med, color=SKYGRID_COLOR, linewidth=1.5, zorder=3, label='Skygrid median')

    # --- constant coalescent median and 95% HPD ---
    ax.axhline(const['median'], color=CONSTCOAL_COLOR, linewidth=1.5,
               linestyle='--', zorder=3, label='Const. coal. median')
    ax.axhspan(const['hpd_lower'], const['hpd_upper'],
               alpha=0.2, color=CONSTCOAL_COLOR, zorder=2, label='Const. coal. 95% HPD')

    # --- exponential growth-rate fit ---
    if growth_fit is not None:
        curve = growth_fit_curve(growth_fit, n_points=growth_fit_n_points)
        x_fit = _to_axis_units(curve['t'])
        if growth_fit_hpd:
            ax.fill_between(x_fit, curve['hpd_lower'], curve['hpd_upper'],
                            alpha=0.25, color=growth_fit_color, zorder=3.5,
                            label='Growth-rate fit 95% HPD')
        ax.plot(x_fit, curve['median'], color=growth_fit_color, linewidth=1.5,
                linestyle=':', zorder=4, label='Growth-rate fit')

    # --- tMRCA estimates of both models ---
    if root_height_summary is not None:
        t_present = sky['boundaries'][0]
        for model, color in [('skygrid', SKYGRID_COLOR), ('constcoal', CONSTCOAL_COLOR)]:
            row = root_height_summary.loc[model]
            t_med, t_lo, t_hi = _to_axis_units(
                [t_present - row['median'], t_present - row['hpd_upper'],
                 t_present - row['hpd_lower']])
            _draw_tmrca_marker(ax, 0.84, t_med, color, 'tMRCA', _tick_fs, t_lo, t_hi)

    # --- externally-estimated tMRCA ---
    if external_tmrca is not None:
        t_med = _to_axis_units([_decimal_year(external_tmrca['median'])])[0]
        t_lo = t_hi = None
        if 'hpd_lower' in external_tmrca and 'hpd_upper' in external_tmrca:
            t_lo, t_hi = _to_axis_units([_decimal_year(external_tmrca['hpd_lower']),
                                         _decimal_year(external_tmrca['hpd_upper'])])
        _draw_tmrca_marker(ax, 0.70, t_med, external_tmrca.get('color', RED),
                           external_tmrca.get('label', 'External tMRCA'), _tick_fs, t_lo, t_hi)

    # --- axes ---
    if time_unit == 'days':
        x_min = 0
    elif time_unit == 'dates':
        if cutoff is None:
            x_min = boundaries[-1]
        else:
            x_min = max(mdates.date2num(_decimal_year_to_datetime(cutoff)), boundaries[-1])
    else:
        x_min = boundaries[-1] if cutoff is None else max(cutoff, boundaries[-1])
    ax.set_xlim(x_min, boundaries[0])  # present on the right
    if time_unit == 'dates':
        d_min, d_max = mdates.num2date(x_min), mdates.num2date(boundaries[0])
        n_months = (d_max.year - d_min.year) * 12 + (d_max.month - d_min.month) + 1
        interval = 1 if n_months <= 6 else 2
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%-d. %b'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1))  # unlabelled tick per month
    else:
        ax.ticklabel_format(useOffset=False, axis='x', style='plain')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    if log_scale:
        ax.set_yscale('log')
    if y_range:
        ax.set_ylim(y_range)

    if x_label is None:
        x_label = {'days': 'Time (days)', 'dates': 'Date'}.get(time_unit, 'Time (years)')
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel('Effective Population Size', fontsize=fontsize)
    ax.set_title(title, fontsize=_title_fs)
    ax.tick_params(labelsize=_tick_fs)

    sky_sample_line = Line2D([0], [0], color=SKYGRID_SAMPLE_COLOR, linewidth=1)
    const_sample_line = Line2D([0], [0], color=CONSTCOAL_SAMPLE_COLOR, linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles=[sky_sample_line, const_sample_line] + handles,
                    labels=['Skygrid samples', 'Const. coal. samples'] + labels,
                    loc='lower left' if root_height_summary is not None else 'upper left',
                    fontsize=_legend_fs)
    leg.get_frame().set_alpha(0.4)

    plt.tight_layout()
    if save_path is not None:
        _save_figure(ax.get_figure(), save_path)
    return ax


# =============================================================================
# Summary (MCC) trees
# =============================================================================

def read_summary_tree(tree_path):
    """Read a BEAST treeannotator MCC summary tree with annotations."""
    return dendropy.Tree.get(path=tree_path, schema='nexus')


def _get_ann(node, key):
    """Extract a single annotation value from a dendropy node."""
    for ann in node.annotations:
        if ann.name == key:
            return ann.value
    return None


def _parse_hpd(hpd_val):
    """Parse HPD annotation (list or '{lower,upper}' string) into (lower, upper) floats."""
    if hpd_val is None:
        return None, None
    if isinstance(hpd_val, (list, tuple)):
        return float(hpd_val[0]), float(hpd_val[1])
    s = str(hpd_val).strip('{}').split(',')
    return float(s[0]), float(s[1])


def compute_tree_summary(tree, meta_df=None, bottleneck_year=2020):
    """
    Extract summary statistics from a BEAST treeannotator MCC tree.

    If meta_df is provided (must have 'Isolate' and 'decimal_year' columns),
    tip branch lengths are also split into pre- and post-bottleneck groups.

    Returns a dict with:
      root_height                         : float
      root_height_hpd                     : (lower, upper)
      total_branch_length                 : float
      total_branch_length_hpd             : (lower, upper)
      all_branch_lengths                  : array
      all_branch_lengths_hpd_lower/upper  : arrays
      tip_branch_lengths                  : array
      tip_branch_lengths_hpd_lower/upper  : arrays
      tip_branch_lengths_pre              : array (only if meta_df provided)
      tip_branch_lengths_post             : array (only if meta_df provided)
    """
    root = tree.seed_node
    date_lookup = _isolate_date_lookup(meta_df) if meta_df is not None else {}

    all_lengths, all_lower, all_upper = [], [], []
    tip_lengths, tip_lower, tip_upper = [], [], []
    tip_lengths_pre, tip_lengths_post = [], []

    for node in tree.preorder_node_iter():
        if node is root:
            continue
        length = float(_get_ann(node, 'length_median') or node.edge_length)
        lower, upper = _parse_hpd(_get_ann(node, 'length_95%_HPD'))
        lower = lower if lower is not None else length
        upper = upper if upper is not None else length
        all_lengths.append(length)
        all_lower.append(lower)
        all_upper.append(upper)
        if node.is_leaf():
            tip_lengths.append(length)
            tip_lower.append(lower)
            tip_upper.append(upper)
            date = date_lookup.get(node.taxon.label)
            if date is not None:
                (tip_lengths_pre if date < bottleneck_year else tip_lengths_post).append(length)

    result = {
        'root_height': float(_get_ann(root, 'height')),
        'root_height_hpd': _parse_hpd(_get_ann(root, 'height_95%_HPD')),
        'total_branch_length': np.sum(all_lengths),
        'total_branch_length_hpd': (np.sum(all_lower), np.sum(all_upper)),
        'all_branch_lengths': np.array(all_lengths),
        'all_branch_lengths_hpd_lower': np.array(all_lower),
        'all_branch_lengths_hpd_upper': np.array(all_upper),
        'tip_branch_lengths': np.array(tip_lengths),
        'tip_branch_lengths_hpd_lower': np.array(tip_lower),
        'tip_branch_lengths_hpd_upper': np.array(tip_upper),
    }
    if date_lookup:
        result['tip_branch_lengths_pre'] = np.array(tip_lengths_pre)
        result['tip_branch_lengths_post'] = np.array(tip_lengths_post)
    return result


def print_tree_summary(tree_summary, label="", bottleneck_year=2020):
    """Print key statistics from a compute_tree_summary() dict."""
    ts = tree_summary
    print(f"=== {label} ===")
    print(f"Root height:         {ts['root_height']:.2f} (95% HPD: {ts['root_height_hpd'][0]:.2f}, {ts['root_height_hpd'][1]:.2f})")
    print(f"Total branch length: {ts['total_branch_length']:.2f} (95% HPD: {ts['total_branch_length_hpd'][0]:.2f}, {ts['total_branch_length_hpd'][1]:.2f})")
    tip = ts['tip_branch_lengths']
    tip_lo = ts['tip_branch_lengths_hpd_lower']
    tip_hi = ts['tip_branch_lengths_hpd_upper']
    print(f"Tip branch lengths:  median={np.median(tip):.2f}, mean={np.mean(tip):.2f} (95% HPD bounds: {tip_lo.mean():.2f}, {tip_hi.mean():.2f})")
    if 'tip_branch_lengths_pre' in ts:
        pre, post = ts['tip_branch_lengths_pre'], ts['tip_branch_lengths_post']
        print(f"  pre-{bottleneck_year}  (n={len(pre)}):  median={np.median(pre):.2f}, mean={np.mean(pre):.2f}")
        print(f"  post-{bottleneck_year} (n={len(post)}):  median={np.median(post):.2f}, mean={np.mean(post):.2f}")
    bl = ts['all_branch_lengths']
    bl_lo = ts['all_branch_lengths_hpd_lower']
    bl_hi = ts['all_branch_lengths_hpd_upper']
    print(f"All branch lengths:  median={np.median(bl):.2f}, mean={np.mean(bl):.2f} (95% HPD bounds: {bl_lo.mean():.2f}, {bl_hi.mean():.2f})")
    print()


def subtree_root_height(tree, meta_df, cutoff_year):
    """
    Prune tree to tips sampled at or after cutoff_year and return the root
    height of the resulting subtree.

    Parameters
    ----------
    tree : dendropy.Tree
        MCC summary tree with BEAST treeannotator annotations.
    meta_df : pd.DataFrame
        Must have 'Isolate' and 'decimal_year' columns.
    cutoff_year : float
        Retain only tips with decimal_year >= cutoff_year.

    Returns
    -------
    dict with:
      n_tips        : int    — number of retained tips
      height_median : float  — height_median annotation on the pruned root
      height_hpd    : (lower, upper) — height_95%_HPD of the pruned root
    """
    date_lookup = _isolate_date_lookup(meta_df)
    keep = [
        n.taxon.label for n in tree.leaf_node_iter()
        if date_lookup.get(n.taxon.label, cutoff_year - 1) >= cutoff_year
    ]
    pruned = copy.deepcopy(tree)
    pruned.retain_taxa_with_labels(keep)

    root = pruned.seed_node
    median = float(_get_ann(root, 'height_median') or _get_ann(root, 'height'))
    hpd = _parse_hpd(_get_ann(root, 'height_95%_HPD'))
    return {'n_tips': len(keep), 'height_median': median, 'height_hpd': hpd}


# =============================================================================
# Tree plots: time tree and tanglegram
# =============================================================================

def _fitch_coloring(root, tip_states):
    """
    Two-pass Fitch parsimony. Returns dict node -> assigned state (None = ambiguous).
    """
    fitch_sets = {}

    # downward pass (post-order)
    def _down(n):
        if n.is_leaf():
            fitch_sets[n] = frozenset([tip_states.get(n.taxon.label, 'other')])
        else:
            for c in n.child_nodes():
                _down(c)
            child_sets = [fitch_sets[c] for c in n.child_nodes()]
            inter = frozenset.intersection(*child_sets)
            fitch_sets[n] = inter if inter else frozenset().union(*child_sets)
    _down(root)

    # upward pass (pre-order)
    assigned = {}
    def _up(n, parent_state):
        fs = fitch_sets[n]
        if parent_state is not None and parent_state in fs:
            assigned[n] = parent_state
        elif len(fs) == 1:
            assigned[n] = next(iter(fs))
        else:
            assigned[n] = None   # genuinely ambiguous
        for c in n.child_nodes():
            _up(c, assigned[n])

    root_fs = fitch_sets[root]
    assigned[root] = next(iter(root_fs)) if len(root_fs) == 1 else None
    for c in root.child_nodes():
        _up(c, assigned[root])

    return assigned


def _subtype_node_colors(root, tip_state):
    """Fitch-colour all branches by ptxP x fim3 subtype (ambiguous: grey, 'other': black)."""
    return {n: SUBTYPE_COLORS.get(s, BLACK) if s is not None else GREY
            for n, s in _fitch_coloring(root, tip_state).items()}


def _read_subtype_metadata(metadata_path, ptxp_col, fim3_col):
    """
    Read a pertussis metadata CSV and return three dicts keyed by tip label
    (Isolate with spaces replaced by underscores):
      tip_state : ptxP x fim3 combination, or 'other'
      tip_color : colour from SUBTYPE_COLORS, or black for 'other'
      tip_year  : decimal sampling year (only if a 'decimal_year' column exists)
    """
    tip_state, tip_color, tip_year = {}, {}, {}
    if metadata_path is None:
        return tip_state, tip_color, tip_year
    meta = pd.read_csv(metadata_path)
    meta['_key'] = meta['Isolate'].str.replace(' ', '_')
    for _, row in meta.iterrows():
        key = row['_key']
        combo = (row.get(ptxp_col, None), row.get(fim3_col, None))
        if combo in SUBTYPE_COLORS:
            tip_state[key] = combo
            tip_color[key] = SUBTYPE_COLORS[combo]
        else:
            tip_state[key] = 'other'
            tip_color[key] = BLACK
        if 'decimal_year' in meta.columns and pd.notna(row['decimal_year']):
            tip_year[key] = float(row['decimal_year'])
    return tip_state, tip_color, tip_year


def _subtype_legend_handles(markersize):
    """Legend entries for the ptxP x fim3 subtype colours."""
    return [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=markersize, label=f'{p} + {f}')
        for (p, f), c in SUBTYPE_COLORS.items()
    ] + [Line2D([0], [0], marker='o', color='w', markerfacecolor=BLACK,
                markersize=markersize, label='Other')]


def plot_timetree(
    tree_path,
    metadata_path=None,
    ptxp_col='ptxP',
    fim3_col='fim3',
    figsize=(22, 6),
    most_recent_date=None,
    root_at_top=True,
    ladderize=True,
    color_branches=False,
    branch_lw=1.2,
    tip_size=12,
    legend_fontsize=None,
    legend_anchor=(0.95, 1.0),
    time_axis_label=None,
    time_axis_fontsize=24,
    time_axis_gap=15,
    save_path=None,
    ax=None,
):
    """
    Plot a single BEAST time tree (e.g. the skygrid MCC tree) rotated by 90
    degrees: tips spread along the x axis, time on the y axis, so the figure
    can be wide and flat. Tips are coloured by ptxP x fim3 combination, using
    the same colours as `plot_tanglegram`.

    Parameters
    ----------
    tree_path : str
        NEXUS or Newick tree file. Branch lengths are assumed to be in years.
    metadata_path : str or None
        Optional CSV with an 'Isolate' column plus `ptxp_col`/`fim3_col`
        columns for tip colours, and optionally a 'decimal_year' column used
        to place the tree in calendar time.
    figsize : tuple
        (width, height) in inches of the new figure; ignored if `ax` is given.
    most_recent_date : float, str or None
        Date of the most recent tip (decimal year or 'YYYY-MM-DD'). If None,
        it is taken from the metadata 'decimal_year' column or from a
        trailing "|YYYY-MM-DD" in the tip labels; if neither is available the
        axis shows years before the most recent sample.
    root_at_top : bool
        If True, the root is drawn at the top and time runs downwards;
        otherwise the root is at the bottom and time runs upwards.
    ladderize : bool
        If True, ladderize the tree (smaller clades first) before plotting.
    color_branches : bool
        If True, all branches are Fitch-coloured by ptxP x fim3 combination,
        as in `plot_tanglegram` (ambiguous branches are grey). If False,
        only each tip's terminal branch is coloured, matching its tip dot;
        internal branches are black.
    legend_fontsize : float or None
        Legend font size; defaults to `time_axis_fontsize`.
    legend_anchor : tuple
        (x, y) of the legend's upper-left corner in axes coordinates
        (1.0 = right edge of the tree area; smaller x moves it left).
    time_axis_gap : float
        Gap in points between the time axis and the leftmost tips.
    ax : matplotlib Axes or None
        Axes to draw into; a new figure of size `figsize` is created if None.
    """
    tree = _read_tree(tree_path)
    if ladderize:
        tree.ladderize(ascending=True)
    root = tree.seed_node
    tips = tree.leaf_nodes()

    tip_state, tip_color, tip_year = _read_subtype_metadata(metadata_path, ptxp_col, fim3_col)

    # ---- node times ----------------------------------------------------------
    dist = {}
    for n in root.preorder_iter():
        p = n.parent_node
        dist[n] = 0.0 if p is None else dist[p] + (n.edge_length or 0.0)
    height = max(dist[t] for t in tips)

    mrd = most_recent_date
    if mrd is None:
        years = [tip_year[t.taxon.label] for t in tips if t.taxon.label in tip_year]
        if not years:
            years = [_decimal_year(m.group(1)) for t in tips
                     if (m := _DATE_SUFFIX_RE.search(t.taxon.label))]
        mrd = max(years) if years else None
    mrd = _decimal_year(mrd) if mrd is not None else None
    # calendar time if known, else negative years before the most recent tip
    t_of = {n: (mrd if mrd is not None else 0.0) - height + d for n, d in dist.items()}

    # ---- x positions ---------------------------------------------------------
    x_of = {t: i for i, t in enumerate(tips)}
    for n in root.postorder_iter():
        if not n.is_leaf():
            cx = [x_of[c] for c in n.child_nodes()]
            x_of[n] = (min(cx) + max(cx)) / 2.0

    # node -> colour of the branch leading to it (and of its horizontal bar)
    if color_branches:
        node_color = _subtype_node_colors(root, tip_state)
    else:
        node_color = {t: tip_color.get(t.taxon.label, GREY) for t in tips}

    # ---- draw ----------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    segs, cols = [], []
    for n in root.preorder_iter():
        if n.parent_node is not None:
            # branch from parent's time down to this node's time
            segs.append([(x_of[n], t_of[n.parent_node]), (x_of[n], t_of[n])])
            cols.append(node_color.get(n, BLACK))
        if not n.is_leaf():
            cx = [x_of[c] for c in n.child_nodes()]
            segs.append([(min(cx), t_of[n]), (max(cx), t_of[n])])
            cols.append(node_color.get(n, BLACK))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=branch_lw, zorder=2))

    ax.scatter([x_of[t] for t in tips], [t_of[t] for t in tips], s=tip_size,
               c=[tip_color.get(t.taxon.label, GREY) for t in tips],
               zorder=3, linewidths=0)

    # ---- axes ----------------------------------------------------------------
    t_min, t_max = t_of[root], max(t_of[t] for t in tips)
    pad = 0.02 * (t_max - t_min)
    ax.set_xlim(-1, len(tips))
    ax.set_xticks([])
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)

    if mrd is None:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{-v:g}'))
        default_label = 'Years before most recent sample'
    elif height < 3:
        ticks, labels = _calendar_ticks(t_min, t_max, maxticks=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        default_label = 'Date'
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        default_label = 'Year'
    ax.set_ylim(t_min - pad, t_max + pad)
    if root_at_top:
        ax.invert_yaxis()
    ax.set_ylabel(time_axis_label or default_label, fontsize=time_axis_fontsize + 2)
    ax.tick_params(axis='y', labelsize=time_axis_fontsize, width=1.5, length=7)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_position(('outward', time_axis_gap))

    if metadata_path is not None:
        _legend_fs = legend_fontsize or time_axis_fontsize
        ax.legend(handles=_subtype_legend_handles(0.8 * _legend_fs), loc='upper left',
                  bbox_to_anchor=legend_anchor, fontsize=_legend_fs, frameon=False)

    if save_path is not None:
        _save_figure(fig, save_path)
    return ax


# shades of the project's blue / purple / red / green, interleaved by hue so
# adjacent clusters read as different colours
DEFAULT_CLUSTER_PALETTE = [
    '#a6444f',   # red
    '#c97b83',   # red -> pink
    '#f2c9cd',   # pink
    '#f2c4d8',   # pink -> purple
    '#d991b4',   # purple (mid)
    '#80557e',   # purple
    '#1f4a63',   # blue (extra dark / navy)
    '#397398',   # blue (dark)
    '#7394c2',   # blue (mid)
    '#b5d2f2',   # blue (light)
    '#2f8f8a',   # turquoise (blue -> green bridge)
    '#c5dfc0',   # green (light)
    '#7db87a',   # green (mid)
    '#4e9a8a',   # green (dark)
]


def plot_tanglegram(
    skygrid_tree_path,
    constcoal_tree_path,
    metadata_path=None,
    ptxp_col='ptxP',
    fim3_col='fim3',
    figsize=(18, 14),
    save_path=None,
    tip_fontsize=5,
    show_tip_labels=False,
    color_branches=True,
    connection_alpha=0.35,
    connection_lw=0.6,
    tip_size=5,
    branch_lw=0.8,
    legend_fontsize=None,
    legend_markersize=None,
    sort_left_by_date=False,
    date_lookup=None,
    color_tips_by_date=False,
    date_cmap_colors=("#00181E", "#397398", "#4e9a8a", "#7db87a", "#c5dfc0"),
    colorbar_label='Sampling date',
    cluster_path=None,
    cluster_col='CLUSTER',
    taxa_col='TAXA',
    cluster_palette=None,
    unclustered_color=None,
    show_time_axis=True,
    most_recent_date=None,
    time_axis_label=None,
    time_axis_fontsize=None,
    shared_time_scale=True,
):
    """
    Tanglegram of two trees (skygrid left, constcoal right mirrored), with
    lines connecting the same tip in both trees.

    Tips can be coloured by ptxP x fim3 combination (`metadata_path`), by
    PhyCLIP cluster (`cluster_path`) or by sampling date (`color_tips_by_date`).

    Parameters
    ----------
    color_branches : bool
        If True, branches are Fitch-coloured by tip category (ambiguous
        branches are grey). If False, all branches are black; only connection
        lines, tip dots (and terminal branches for cluster/date colouring)
        are coloured.
    metadata_path : str or None
        Optional CSV with an 'Isolate' column plus `ptxp_col`/`fim3_col`
        columns used to colour tips/branches by subtype combination.
    sort_left_by_date : bool
        If True, order the left (skygrid) tree's tips top-to-bottom by
        sampling date instead of the default tree-traversal order. The
        right (constcoal) tree is still reordered to minimise crossings.
    date_lookup : dict or None
        Optional {tip_label: sortable_date} mapping used for date sorting and
        colouring. If None, the date is parsed from a trailing "|YYYY-MM-DD"
        in the tip label (GISAID-style ids in the SARS-CoV-2 trees).
    color_tips_by_date : bool
        If True, colour each tip dot and its terminal branch on a continuous
        colour scale (`date_cmap_colors`) by sampling date and add a colorbar.
        Applied last, so it overrides other tip colours.
    date_cmap_colors : sequence of str
        Hex colours defining the continuous colormap, from oldest to newest.
    colorbar_label : str
        Label for the date colorbar.
    cluster_path : str or None
        Optional PhyCLIP cluster output file (tab-separated, columns
        `cluster_col` and `taxa_col`). Each tip's dot and terminal branch in
        both trees is coloured by its cluster; tips not in the file are drawn
        in `unclustered_color`. With `color_branches=True`, internal branches
        of the left tree are Fitch-coloured by cluster too.
    cluster_col, taxa_col : str
        Column names in the `cluster_path` file (PhyCLIP's defaults).
    cluster_palette : sequence of str or None
        Hex colours for clusters. Defaults to DEFAULT_CLUSTER_PALETTE.
    unclustered_color : str or None
        Colour for tips not present in the cluster file. Defaults to grey.
    show_time_axis : bool
        If True, draw a time axis below each tree.
    most_recent_date : float, str or None
        Date of the most recent tip (decimal year or 'YYYY-MM-DD'), used to
        label the time axis in calendar time. If None, it is derived from the
        tip dates (`date_lookup`, the tip label, or a 'decimal_year' column in
        `metadata_path`); if no dates are found, the axis shows time before
        the most recent tip instead.
    time_axis_label : str or None
        Label under the time axes. Defaults to 'Year' / 'Date' (calendar
        time) or 'Years before most recent sample'.
    time_axis_fontsize : float or None
        Font size of the time axis tick labels (label is +1). Defaults to
        `legend_fontsize`, or 12.
    shared_time_scale : bool
        If True (default), both trees are drawn on the same time scale (that
        of the deeper tree), aligned at their most recent tip. If False, each
        tree is stretched to the full width independently.
    """
    def _tip_date(label):
        if date_lookup is not None:
            return date_lookup.get(label)
        m = _DATE_SUFFIX_RE.search(label)
        return m.group(1) if m else None

    def _numeric_date(d):
        if d is None:
            return None
        if isinstance(d, (int, float)):
            return float(d)
        return float(datetime.strptime(d, "%Y-%m-%d").toordinal())

    tip_state, tip_color, meta_decimal_year = _read_subtype_metadata(metadata_path, ptxp_col, fim3_col)

    # ---- layout helpers ------------------------------------------------------
    def _root_to_node_dist(node):
        d, n = 0.0, node
        while n.parent_node is not None:
            d += (n.edge_length or 0.0)
            n = n.parent_node
        return d

    def _tip_order(root):
        tips, stack = [], [root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                tips.append(n)
            else:
                stack.extend(reversed(list(n.child_nodes())))
        return tips

    def _ladderize_by_date(node):
        """
        Recursively reorder each node's children by the median tip date
        within each child clade, so tips end up roughly chronologically
        ordered without introducing branch crossings. Returns the list of
        tip dates below `node`.
        """
        if node.is_leaf():
            return [_tip_date(node.taxon.label)]

        child_entries = []
        all_dates = []
        for c in node.child_nodes():
            dates = _ladderize_by_date(c)
            child_entries.append((c, dates))
            all_dates.extend(dates)

        def _median_date(dates):
            valid = sorted(d for d in dates if d is not None)
            return valid[len(valid) // 2] if valid else None

        child_entries.sort(key=lambda ce: (_median_date(ce[1]) is None, _median_date(ce[1])))
        node.set_child_nodes([c for c, _ in child_entries])
        return all_dates

    def _reorder_to_minimise_crossings(node, left_y):
        if node.is_leaf():
            return
        for c in node.child_nodes():
            _reorder_to_minimise_crossings(c, left_y)
        def _med(n):
            ys = [left_y.get(l.taxon.label, 0) for l in n.leaf_iter()]
            return np.median(ys) if ys else 0
        node._child_nodes = sorted(node.child_nodes(), key=_med)

    def _tree_height(tips_in_order):
        return max(_root_to_node_dist(t) for t in tips_in_order) or 1.0

    def _compute_positions(root, tips_in_order, mirror=False, scale_depth=None):
        # x = 1 at the most recent tip; `scale_depth` years span the full
        # width (defaults to this tree's own height, i.e. root at x = 0)
        tip_y = {t: i for i, t in enumerate(tips_in_order)}
        max_depth = _tree_height(tips_in_order)
        scale = scale_depth or max_depth
        pos = {}
        def _rec(n):
            d = 1.0 - (max_depth - _root_to_node_dist(n)) / scale
            x = (1.0 - d) if mirror else d
            if n.is_leaf():
                y = tip_y[n]
            else:
                child_ys = []
                for c in n.child_nodes():
                    _rec(c)
                    child_ys.append(pos[c][1])
                y = (min(child_ys) + max(child_ys)) / 2.0
            pos[n] = (x, y)
        _rec(root)
        return pos, scale

    def _draw_tree(ax, root, pos, node_colors):
        """node_colors: node -> colour of the branch leading to it and of its vertical bar."""
        h_segs, h_cols = [], []
        v_segs, v_cols = [], []
        for node in root.preorder_iter():
            if node.parent_node is not None:
                nx, ny = pos[node]
                px, _ = pos[node.parent_node]
                h_segs.append([(px, ny), (nx, ny)])
                h_cols.append(node_colors.get(node, BLACK))
            if not node.is_leaf():
                cx, _ = pos[node]
                child_ys = [pos[c][1] for c in node.child_nodes()]
                v_segs.append([(cx, min(child_ys)), (cx, max(child_ys))])
                v_cols.append(node_colors.get(node, BLACK))
        ax.add_collection(LineCollection(h_segs, colors=h_cols, linewidths=branch_lw, zorder=2))
        ax.add_collection(LineCollection(v_segs, colors=v_cols, linewidths=branch_lw, zorder=2))

    # ---- load trees and fix tip order ----------------------------------------
    sky_tree = _read_tree(skygrid_tree_path)
    const_tree = _read_tree(constcoal_tree_path)
    sky_root, const_root = sky_tree.seed_node, const_tree.seed_node

    if sort_left_by_date:
        missing = sum(1 for t in sky_root.leaf_iter() if _tip_date(t.taxon.label) is None)
        if missing:
            print(f"Warning: {missing} tip(s) missing a date; ladderized first, in original order.")
        _ladderize_by_date(sky_root)
    sky_tips = _tip_order(sky_root)
    left_y = {t.taxon.label: i for i, t in enumerate(sky_tips)}
    _reorder_to_minimise_crossings(const_root, left_y)
    const_tips = _tip_order(const_root)
    n_tips = len(sky_tips)

    # ---- positions -----------------------------------------------------------
    gap = 0.15
    shared_depth = (max(_tree_height(sky_tips), _tree_height(const_tips))
                    if shared_time_scale else None)
    sky_pos, sky_height = _compute_positions(sky_root, sky_tips, mirror=False,
                                             scale_depth=shared_depth)
    const_pos, const_height = _compute_positions(const_root, const_tips, mirror=True,
                                                 scale_depth=shared_depth)
    sky_pos = {n: (x - 1.0, y) for n, (x, y) in sky_pos.items()}
    const_pos = {n: (x + gap, y) for n, (x, y) in const_pos.items()}

    fig, ax = plt.subplots(figsize=figsize)

    # ---- branch colours by subtype -------------------------------------------
    if color_branches:
        sky_node_colors = _subtype_node_colors(sky_root, tip_state)
        const_node_colors = _subtype_node_colors(const_root, tip_state)
    else:
        sky_node_colors = {}   # _draw_tree falls back to BLACK
        const_node_colors = {}

    # ---- tip / terminal-branch colours by PhyCLIP cluster --------------------
    _unclustered_color = unclustered_color if unclustered_color is not None else GREY
    if cluster_path is not None:
        cdf = pd.read_csv(cluster_path, sep='\t')
        cdf = cdf[[cluster_col, taxa_col]].dropna(subset=[taxa_col])
        taxon_cluster = dict(zip(cdf[taxa_col], cdf[cluster_col]))

        # colour order: first appearance scanning the left tree top-to-bottom
        cluster_order = []
        for t in sky_tips:
            cid = taxon_cluster.get(t.taxon.label)
            if cid is not None and cid not in cluster_order:
                cluster_order.append(cid)
        palette = list(cluster_palette) if cluster_palette is not None else DEFAULT_CLUSTER_PALETTE
        n_needed = len(cluster_order)
        if n_needed <= len(palette):
            # take a contiguous run from the green/blue end of the palette
            picked = palette[-n_needed:]
        else:
            picked = [palette[i % len(palette)] for i in range(n_needed)]
        cluster_color_by_id = dict(zip(cluster_order, picked))

        for tips, node_colors in ((sky_tips, sky_node_colors), (const_tips, const_node_colors)):
            for t in tips:
                cid = taxon_cluster.get(t.taxon.label)
                color = cluster_color_by_id[cid] if cid is not None else _unclustered_color
                tip_color[t.taxon.label] = color
                node_colors[t] = color

        # Fitch-colour the left tree's internal branches by cluster, so
        # mixed-cluster branches stay grey
        if color_branches:
            sky_cluster_state = {
                t.taxon.label: taxon_cluster.get(t.taxon.label, 'unclustered')
                for t in sky_tips
            }
            for n, state in _fitch_coloring(sky_root, sky_cluster_state).items():
                if state is None:
                    sky_node_colors[n] = GREY
                elif state == 'unclustered':
                    sky_node_colors[n] = _unclustered_color
                else:
                    sky_node_colors[n] = cluster_color_by_id[state]

    # ---- tip / terminal-branch colours by sampling date ----------------------
    if color_tips_by_date:
        date_cmap = mcolors.LinearSegmentedColormap.from_list('date_cmap', date_cmap_colors)
        raw_dates = {t.taxon.label: _tip_date(t.taxon.label) for t in sky_tips + const_tips}
        numeric_dates = {label: _numeric_date(d) for label, d in raw_dates.items()}
        valid = [v for v in numeric_dates.values() if v is not None]
        missing = len(numeric_dates) - len(valid)
        if missing:
            print(f"Warning: {missing} tip(s) missing a date for color_tips_by_date; drawn in grey.")
        vmin, vmax = (min(valid), max(valid)) if valid else (0.0, 1.0)
        date_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        dates_are_strings = any(isinstance(d, str) for d in raw_dates.values() if d is not None)

        for label, nd in numeric_dates.items():
            tip_color[label] = date_cmap(date_norm(nd)) if nd is not None else GREY
        for tips, node_colors in ((sky_tips, sky_node_colors), (const_tips, const_node_colors)):
            for t in tips:
                nd = numeric_dates[t.taxon.label]
                if nd is not None:
                    node_colors[t] = date_cmap(date_norm(nd))

    # ---- draw trees, tip dots and connecting lines ---------------------------
    _draw_tree(ax, sky_root, sky_pos, sky_node_colors)
    _draw_tree(ax, const_root, const_pos, const_node_colors)

    # tip dots, offset inward so there's a gap from the branch endpoint
    dot_offset = gap * 0.10
    sky_dot_x, const_dot_x = dot_offset, gap - dot_offset
    for tips, pos, dot_x in ((sky_tips, sky_pos, sky_dot_x), (const_tips, const_pos, const_dot_x)):
        for t in tips:
            ax.scatter(dot_x, pos[t][1], s=tip_size, color=tip_color.get(t.taxon.label, GREY),
                       zorder=5, clip_on=False)

    const_label_y = {t.taxon.label: const_pos[t][1] for t in const_tips}
    conn_segs, conn_colors = [], []
    for t in sky_tips:
        label = t.taxon.label
        if label in const_label_y:
            conn_segs.append([(sky_dot_x, sky_pos[t][1]), (const_dot_x, const_label_y[label])])
            conn_colors.append(tip_color.get(label, GREY))
    ax.add_collection(LineCollection(conn_segs, colors=conn_colors,
                                     linewidths=connection_lw, alpha=connection_alpha,
                                     zorder=3))

    if show_tip_labels:
        for tips, pos, x, ha in ((sky_tips, sky_pos, -0.01, 'right'),
                                 (const_tips, const_pos, gap + 0.01, 'left')):
            for t in tips:
                ax.text(x, pos[t][1], t.taxon.label, ha=ha, va='center',
                        fontsize=tip_fontsize, color=tip_color.get(t.taxon.label, GREY))

    # ---- legend --------------------------------------------------------------
    if metadata_path is not None:
        _legend_fs = legend_fontsize if legend_fontsize is not None else tip_fontsize + 7
        _marker_sz = legend_markersize if legend_markersize is not None else 13
        ax.legend(handles=_subtype_legend_handles(_marker_sz), loc='lower right',
                  bbox_to_anchor=(1.0, 0.0), ncol=1, fontsize=_legend_fs, frameon=False)
    elif cluster_path is not None:
        _legend_fs = legend_fontsize if legend_fontsize is not None else tip_fontsize + 11
        _marker_sz = legend_markersize if legend_markersize is not None else 18
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_color_by_id[c],
                   markersize=_marker_sz, label=f'Cluster {i}')
            for i, c in reversed(list(enumerate(cluster_order)))
        ] + [Line2D([0], [0], marker='o', color='w', markerfacecolor=_unclustered_color,
                    markersize=_marker_sz, label='Unclustered')]
        ax.legend(handles=legend_elements, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=_legend_fs, frameon=False)

    ax.set_xlim(-1.05, gap + 1.05)
    ax.set_ylim(-1, n_tips)
    ax.axis('off')

    # ---- time axes below each tree -------------------------------------------
    # x in [0, 1] within each tree maps linearly to time: root at 0, most
    # recent tip at 1 (mirrored for the right tree)
    if show_time_axis:
        _axis_fs = time_axis_fontsize or legend_fontsize or 12
        mrd = most_recent_date
        if mrd is None:
            tip_years = []
            for t in sky_tips:
                d = _tip_date(t.taxon.label)
                if d is None:
                    d = meta_decimal_year.get(t.taxon.label)
                if d is not None:
                    tip_years.append(_decimal_year(d))
            mrd = max(tip_years) if tip_years else None
        mrd = _decimal_year(mrd) if mrd is not None else None

        axis_y = -1 - 0.01 * n_tips
        tick_len = 0.008 * n_tips
        for height, to_x in (
            (sky_height, lambda frac: -1.0 + frac),
            (const_height, lambda frac: gap + 1.0 - frac),
        ):
            if mrd is not None:
                t0, t1 = mrd - height, mrd
                if height < 3:
                    ticks, labels = _calendar_ticks(t0, t1, maxticks=6)
                else:
                    ticks = MaxNLocator(nbins=5, integer=True).tick_values(t0, t1)
                    labels = [f'{v:.0f}' for v in ticks]
                fracs = [(v - t0) / height for v in ticks]
            else:
                ticks = MaxNLocator(nbins=5).tick_values(0, height)
                labels = [f'{v:g}' for v in ticks]
                fracs = [1.0 - v / height for v in ticks]
            x0, x1 = sorted((to_x(0.0), to_x(1.0)))
            ax.plot([x0, x1], [axis_y, axis_y], color=BLACK, lw=0.8, clip_on=False)
            for frac, label in zip(fracs, labels):
                if -1e-9 <= frac <= 1 + 1e-9:
                    x = to_x(frac)
                    ax.plot([x, x], [axis_y, axis_y - tick_len], color=BLACK,
                            lw=0.8, clip_on=False)
                    ax.text(x, axis_y - 1.5 * tick_len, label, ha='center',
                            va='top', fontsize=_axis_fs, clip_on=False)
        if time_axis_label is None:
            if mrd is None:
                time_axis_label = 'Years before most recent sample'
            else:
                time_axis_label = 'Date' if max(sky_height, const_height) < 3 else 'Year'
        for xc in (-0.5, gap + 0.5):
            ax.text(xc, axis_y - 1.5 * tick_len - 0.035 * n_tips, time_axis_label,
                    ha='center', va='top', fontsize=_axis_fs + 1, clip_on=False)

    # ---- date colorbar -------------------------------------------------------
    if color_tips_by_date:
        sm = plt.cm.ScalarMappable(cmap=date_cmap, norm=date_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=legend_fontsize or 12)
        cbar.ax.tick_params(labelsize=(legend_fontsize or 12) - 1)
        if dates_are_strings:
            ticks = cbar.get_ticks()
            cbar.set_ticklabels([datetime.fromordinal(int(t)).strftime('%Y-%m-%d') for t in ticks])

    if save_path is not None:
        _save_figure(fig, save_path)
    return ax


# =============================================================================
# Topology comparison (Robinson-Foulds distances)
# =============================================================================

def robinson_foulds(tree_path1, tree_path2):
    """
    Compute the Robinson-Foulds distance between two trees (NEXUS or Newick).

    Returns a dict with:
      rf           : unweighted RF distance (number of non-shared splits)
      rf_max       : maximum possible RF distance (2 * (n_taxa - 3))
      rf_normalized: rf / rf_max
    """
    trees = [_read_tree(p) for p in (tree_path1, tree_path2)]
    tns = dendropy.TaxonNamespace()
    for t in trees:
        t.migrate_taxon_namespace(tns)
        t.encode_bipartitions()

    rf = treecompare.symmetric_difference(trees[0], trees[1])
    rf_max = 2 * (len(tns) - 3)
    return {"rf": rf, "rf_max": rf_max, "rf_normalized": rf / rf_max if rf_max > 0 else 0.0}


def load_trees(path, taxon_namespace=None):
    """
    Load a posterior trees file (NEXUS or Newick) and return a list of
    dendropy.Tree objects with encoded bipartitions.

    Pass the same taxon_namespace to multiple calls so that trees from
    different files share one namespace — required for treecompare functions.

    Example
    -------
    tns = dendropy.TaxonNamespace()
    sky_trees   = load_trees(sky_path,   taxon_namespace=tns)
    const_trees = load_trees(const_path, taxon_namespace=tns)
    """
    if taxon_namespace is None:
        taxon_namespace = dendropy.TaxonNamespace()
    trees = dendropy.TreeList.get(
        path=path, schema=_tree_schema(path),
        taxon_namespace=taxon_namespace, preserve_underscores=True,
    )
    for t in trees:
        t.encode_bipartitions()
    return list(trees)


def compute_tree_distances(sky_trees, const_trees, n_between=None):
    """
    Compute Robinson-Foulds (RF) distances between posterior tree samples:
    all pairs within the skygrid trees, all pairs within the constcoal trees,
    and pairs between the two models.

    For large tree sets, use the parallelised scripts/compute_topology_distances.py.

    Parameters
    ----------
    sky_trees, const_trees : list of dendropy.Tree
        Loaded via load_trees() with a shared taxon namespace.
    n_between : int or None
        Subsample this many random cross-model pairs; None = all n_sky × n_const.

    Returns
    -------
    pd.DataFrame with columns: comparison, tree1_idx, tree2_idx, rf
    """
    n_sky, n_const = len(sky_trees), len(const_trees)

    between_pairs = [(i, j) for i in range(n_sky) for j in range(n_const)]
    if n_between is not None and n_between < len(between_pairs):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(between_pairs), size=n_between, replace=False)
        between_pairs = [between_pairs[k] for k in idx]

    comparisons = [
        ('within_skygrid', 'within-skygrid', sky_trees, sky_trees,
         list(combinations(range(n_sky), 2))),
        ('within_constcoal', 'within-constcoal', const_trees, const_trees,
         list(combinations(range(n_const), 2))),
        ('between', 'between-model', sky_trees, const_trees, between_pairs),
    ]

    records = []
    for comparison, description, trees1, trees2, pairs in comparisons:
        print(f"Computing {description} distances ({len(pairs)} pairs) …")
        for k, (i, j) in enumerate(pairs):
            records.append({
                'comparison': comparison,
                'tree1_idx': i, 'tree2_idx': j,
                'rf': treecompare.symmetric_difference(trees1[i], trees2[j]),
            })
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(pairs)}")

    print("Done.")
    return pd.DataFrame(records)


def plot_rf_distributions(
    dist_df,
    normalized=False,
    mode='kde',
    bw_method=None,
    bins=None,
    figsize=(8, 4),
    colors=(SKYGRID_COLOR, CONSTCOAL_COLOR, '#338778'),
    save_path=None,
    fontsize=12,
    tick_fontsize=None,
    legend_fontsize=None,
):
    """
    Plot overlapping density curves (or histograms) of RF distances for the
    three comparison groups produced by compute_tree_distances /
    scripts/compute_topology_distances.py.

    Parameters
    ----------
    normalized : bool
        If True, plot rf_norm instead of rf.
    mode : {'kde', 'hist'}
        'kde' draws smoothed KDE curves (can look artificially bumpy when the
        underlying RF values are sparse/discrete).
        'hist' draws histograms, with bin edges placed at the midpoints
        between consecutive unique values so each discrete RF value gets its
        own bin.
    bw_method : str, scalar, callable, or None
        Bandwidth for mode='kde', passed to scipy's gaussian_kde. None uses
        Scott's rule.
    bins : array-like or None
        Explicit bin edges for mode='hist'. If None, derived from the data.
    """
    if mode not in ('kde', 'hist'):
        raise ValueError("mode must be 'kde' or 'hist'.")

    col = 'rf_norm' if normalized else 'rf'
    xlabel = 'Normalised RF distance' if normalized else 'Robinson-Foulds distance'

    groups = ['within_skygrid', 'within_constcoal', 'between']
    labels = ['Within skygrid', 'Within constcoal', 'Between models']

    fig, ax = plt.subplots(figsize=figsize)

    if mode == 'hist' and bins is None:
        uniq = np.unique(dist_df[col].values)
        if len(uniq) > 1:
            diffs = np.diff(uniq)
            bins = np.concatenate([
                [uniq[0] - diffs[0] / 2],
                uniq[:-1] + diffs / 2,
                [uniq[-1] + diffs[-1] / 2],
            ])
        else:
            bins = np.array([uniq[0] - 0.5, uniq[0] + 0.5])

    for g, label, color in zip(groups, labels, colors):
        vals = dist_df.loc[dist_df['comparison'] == g, col].values
        if len(vals) == 0:
            continue
        if mode == 'kde':
            kde = gaussian_kde(vals, bw_method=bw_method)
            x = np.linspace(vals.min(), vals.max(), 500)
            ax.fill_between(x, kde(x), alpha=0.3, color=color)
            ax.plot(x, kde(x), color=color, label=label, linewidth=1.5)
        else:
            ax.hist(vals, bins=bins, density=True, histtype='bar',
                    alpha=0.3, color=color, edgecolor=color, linewidth=1.2)
            ax.hist(vals, bins=bins, density=True, histtype='step',
                    color=color, linewidth=2.2, label=label)

    _tick_fs = tick_fontsize if tick_fontsize is not None else fontsize - 2
    _legend_fs = legend_fontsize if legend_fontsize is not None else fontsize - 1

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Density', fontsize=fontsize)
    ax.tick_params(labelsize=_tick_fs)
    ax.legend(fontsize=_legend_fs, frameon=False)

    if save_path is not None:
        _save_figure(fig, save_path)
    return ax


# =============================================================================
# Pairwise tip-to-MRCA distances (output of scripts/compute_coalescent_distances.py)
# =============================================================================

def compute_distance_correlation(dist_df, tip='both', method='pearson'):
    """
    Compute the correlation between skygrid and constcoal tip-to-pairwise-MRCA
    median distances.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Output of scripts/compute_coalescent_distances.py.
    tip : str
        'tip1', 'tip2', or 'both' (pools tip1 and tip2 rows).
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    dict with keys: correlation, p_value, n, method
    """
    tips = ('tip1', 'tip2') if tip == 'both' else (tip,)
    df = pd.concat([
        dist_df[[f'skygrid_{t}_to_mrca_median', f'constcoal_{t}_to_mrca_median']].dropna()
        .set_axis(['skygrid', 'constcoal'], axis=1)
        for t in tips
    ], ignore_index=True)

    corr = spearmanr if method == 'spearman' else pearsonr
    r, p = corr(df['skygrid'], df['constcoal'])
    return {'correlation': r, 'p_value': p, 'n': len(df), 'method': method}


def plot_pairwise_distances(
    dist_df,
    tip='both',
    ax=None,
    title=None,
    n_subsample=None,
    point_alpha=0.6,
    errorbar_alpha=0.3,
    markersize=3,
    elinewidth=0.5,
    axis_range=None,
    zoom_range=None,
    n_subsample_zoom=None,
    running_avg=False,
    window_size=500,
    window_overlap=250,
    window_size_zoom=None,
    window_overlap_zoom=None,
    running_avg_style='steps',
    running_avg_color='#c0392b',
    diff_plot=False,
    diff_mode='absolute',
    diff_hpd=True,
    diff_axis_range=None,
    log_scale=False,
    figsize=(10, 7),
    save_path=None,
    fontsize=14,
    tick_fontsize=None,
    label_fontsize=None,
    legend_fontsize=None,
    title_fontsize=None,
    x_label='Skygrid',
    y_label='Const. coal.',
    color_threshold_date=None,
    color_date_cols=('tip1_date', 'tip2_date'),
    color_after=CONSTCOAL_COLOR,
    color_other='#7db87a',
    marker_after='^',
    marker_other='o',
    legend_outside=False,
    legend_markersize=None,
):
    """
    Scatter plot of skygrid (x) vs. constcoal (y) tip-to-pairwise-MRCA distances,
    with 95% HPD errorbars. Optionally adds a running median line, a zoom
    panel and a panel with the constcoal - skygrid difference.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Output of scripts/compute_coalescent_distances.py.
    tip : str
        'tip1', 'tip2', or 'both' (pools tip1 and tip2 rows).
    ax : matplotlib Axes or None
        Only used when neither `zoom_range` nor `diff_plot` is set.
    n_subsample, n_subsample_zoom : int or None
        Number of points drawn in the main / zoom panel (the running median
        and differences are always computed on all points).
    axis_range, zoom_range : tuple or None
        Axis limits of the main panel (default: data range) and of the zoom
        panel (no zoom panel if None).
    running_avg : bool
        If True, add the median of constcoal distances in sliding windows
        (sorted by skygrid distance) of `window_size` points overlapping by
        `window_overlap` points. `window_size_zoom` / `window_overlap_zoom`
        are used for the zoom panel (default: same as main).
    running_avg_style : str
        Kept for backwards compatibility; the running median is always drawn
        as a step function.
    diff_plot : bool
        If True, adds a panel showing the constcoal - skygrid difference per
        point and the median difference per window.
    diff_mode : str
        'absolute' for constcoal - skygrid, 'relative' for (constcoal - skygrid) / skygrid.
    diff_hpd : bool
        If True, draw the per-pair 95% HPD of the difference in the diff panel.
    color_threshold_date : float, str, datetime, or None
        If set, colour points by whether both tips in the pair were sampled
        on/after this date: `color_after` if both tip dates are >= threshold,
        `color_other` otherwise. Requires `dist_df` to have the columns named
        in `color_date_cols` (decimal years). If None, points are grey.
    color_date_cols : tuple of str
        Column names in `dist_df` holding each tip's sampling date.
    color_after, color_other : str
        Hex colours for the two groups when `color_threshold_date` is set.
    legend_outside : bool
        If True, place the legend to the right of the main panel (labels
        wrapped onto more lines to keep it narrow) instead of inside its
        upper-left corner, with larger default font and marker sizes.
    legend_markersize : float or None
        Marker size of the legend entries. Defaults to `markersize` (inside
        legend) or 3 x `markersize`, at least 10 (outside legend).

    Returns
    -------
    ax_main, plus ax_zoom and/or ax_diff if those panels are drawn.
    """
    if color_threshold_date is not None:
        color_threshold_date = _decimal_year(color_threshold_date)
        c1, c2 = color_date_cols
        dist_df = dist_df.copy()
        dist_df['_point_color'] = np.where(
            (dist_df[c1] >= color_threshold_date) & (dist_df[c2] >= color_threshold_date),
            color_after, color_other,
        )

    def _select_and_rename(src, t):
        required = [f'skygrid_{t}_to_mrca_median', f'constcoal_{t}_to_mrca_median']
        optional = [
            f'skygrid_{t}_to_mrca_hpd_lower', f'skygrid_{t}_to_mrca_hpd_upper',
            f'constcoal_{t}_to_mrca_hpd_lower', f'constcoal_{t}_to_mrca_hpd_upper',
            f'abs_diff_{t}_hpd_lower', f'abs_diff_{t}_hpd_upper',
            f'rel_diff_{t}_hpd_lower', f'rel_diff_{t}_hpd_upper',
            '_point_color',
        ]
        cols = required + [c for c in optional if c in src.columns]
        return (src.dropna(subset=required)[cols]
                .rename(columns=lambda c: c.replace(f'_{t}_', '_').replace(f'_{t}', '')))

    # full frame (one row per tip of each pair), subsampled for the scatter only
    tips = ('tip1', 'tip2') if tip == 'both' else (tip,)
    df_full = pd.concat([_select_and_rename(dist_df, t) for t in tips], ignore_index=True)

    _tick_fs = tick_fontsize if tick_fontsize is not None else fontsize - 2
    _label_fs = label_fontsize if label_fontsize is not None else fontsize
    _legend_fs = (legend_fontsize if legend_fontsize is not None
                  else fontsize + 2 if legend_outside else fontsize - 1)
    _legend_ms = (legend_markersize if legend_markersize is not None
                  else max(3 * markersize, 10) if legend_outside else markersize)
    _title_fs = title_fontsize if title_fontsize is not None else fontsize + 1

    if n_subsample is not None and n_subsample < len(df_full):
        df = df_full.sample(n=n_subsample, random_state=0)
    else:
        df = df_full

    if log_scale and zoom_range is not None and zoom_range[0] <= 0:
        raise ValueError(f"zoom_range lower bound must be > 0 for log_scale=True, got {zoom_range[0]}")
    if log_scale and axis_range is not None and axis_range[0] <= 0:
        raise ValueError(f"axis_range lower bound must be > 0 for log_scale=True, got {axis_range[0]}")
    if log_scale:
        df = df[(df['skygrid_to_mrca_median'] > 0) & (df['constcoal_to_mrca_median'] > 0)]

    zoom = zoom_range is not None

    # ---- layout ----
    def _actual_pos(left, bottom, w, h, fw, fh):
        """Figure-fraction position of the square area set_aspect('equal') keeps."""
        side_in = min(w * fw, h * fh)
        aw, ah = side_in / fw, side_in / fh
        return left + (w - aw) / 2, bottom + (h - ah) / 2, aw, ah

    fw, fh = figsize
    fig = plt.figure(figsize=figsize)
    ax_zoom = ax_diff = None
    if zoom:
        # main panel left; zoom panel top right, diff panel below it
        ml, mb, mw, mh = 0.10, 0.10, 0.48, 0.82
        ax_main = fig.add_axes([ml, mb, mw, mh])
        al, ab, aw, ah = _actual_pos(ml, mb, mw, mh, fw, fh)
        gap = 0.085
        right_margin = 0.02
        zoom_left = al + aw + gap
        zoom_w = 1.0 - zoom_left - right_margin
        zoom_h = zoom_w * fw / fh        # square in inches
        zoom_bottom = ab + ah - zoom_h   # top aligned with main panel
        ax_zoom = fig.add_axes([zoom_left, zoom_bottom, zoom_w, zoom_h])
        if diff_plot:
            diff_gap = 0.055
            ax_diff = fig.add_axes([zoom_left, ab, zoom_w, zoom_bottom - ab - diff_gap])
    elif diff_plot:
        # main panel on top, diff panel below
        ml, mw = 0.10, 0.82
        diff_h = 0.12
        mb = 0.10 + diff_h + 0.05
        mh = 0.92 - mb
        ax_main = fig.add_axes([ml, mb, mw, mh])
        al, ab, aw, ah = _actual_pos(ml, mb, mw, mh, fw, fh)
        ax_diff = fig.add_axes([al, 0.10, aw, diff_h], sharex=ax_main)
    elif ax is not None:
        ax_main = ax
        fig = ax_main.figure
    else:
        ax_main = fig.add_axes([0.10, 0.10, 0.82, 0.82])

    def _scatter(ax_, df_):
        x = df_['skygrid_to_mrca_median'].values
        y = df_['constcoal_to_mrca_median'].values

        if 'skygrid_to_mrca_hpd_lower' in df_.columns and 'constcoal_to_mrca_hpd_lower' in df_.columns:
            xlo = df_['skygrid_to_mrca_hpd_lower'].values
            xhi = df_['skygrid_to_mrca_hpd_upper'].values
            ylo = df_['constcoal_to_mrca_hpd_lower'].values
            yhi = df_['constcoal_to_mrca_hpd_upper'].values
            x_segs = np.stack([np.column_stack([xlo, y]), np.column_stack([xhi, y])], axis=1)
            y_segs = np.stack([np.column_stack([x, ylo]), np.column_stack([x, yhi])], axis=1)
            eb_colors = df_['_point_color'].values if '_point_color' in df_.columns else '#aaaaaa'
            for segs in (x_segs, y_segs):
                ax_.add_collection(LineCollection(segs, colors=eb_colors, linewidths=elinewidth,
                                                  alpha=errorbar_alpha, zorder=2, rasterized=True))
        if '_point_color' in df_.columns:
            mask_after = df_['_point_color'].values == color_after
            ax_.scatter(x[mask_after], y[mask_after], s=markersize ** 2, color=color_after,
                        marker=marker_after, alpha=point_alpha, zorder=3, rasterized=True)
            ax_.scatter(x[~mask_after], y[~mask_after], s=markersize ** 2, color=color_other,
                        marker=marker_other, alpha=point_alpha, zorder=3, rasterized=True)
        else:
            ax_.scatter(x, y, s=markersize ** 2, color='#555555',
                        alpha=point_alpha, zorder=3, rasterized=True)

    def _sliding_window_stats(x, y, ws, overlap):
        """Median x, y and difference (with 95% HPD) in windows of `ws` points sorted by x."""
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        step = max(1, ws - overlap)
        rows = []
        for start in range(0, len(x_s) - ws + 1, step):
            xw, yw = x_s[start:start + ws], y_s[start:start + ws]
            d_vals = (yw - xw) / xw if diff_mode == 'relative' else yw - xw
            lo, hi = hpd_interval(d_vals)
            rows.append((np.median(xw), np.median(yw), np.median(d_vals), lo, hi, xw[0], xw[-1]))
        if not rows:
            return tuple(np.array([]) for _ in range(7))
        return tuple(np.array(col) for col in zip(*rows))

    def _step_path(ys, x_lefts, x_rights, x_medians):
        """
        Build (x_path, y_path) for a step function that switches between
        windows halfway between their edges. The first/last segments end
        symmetrically around their window median.
        """
        n = len(ys)
        if n == 0:
            return np.array([]), np.array([])
        splits = (x_rights[:-1] + x_lefts[1:]) / 2
        first_left = (2 * x_medians[0] - splits[0]) if n > 1 else x_lefts[0]
        last_right = (2 * x_medians[-1] - splits[-1]) if n > 1 else x_rights[-1]
        x_path = [first_left]
        y_path = [ys[0]]
        for i in range(n - 1):
            x_path += [splits[i], splits[i]]
            y_path += [ys[i], ys[i + 1]]
        x_path.append(last_right)
        y_path.append(ys[-1])
        return np.array(x_path), np.array(y_path)

    def _df_diff(df_):
        """Per-point x, difference and (optional) difference HPD and colours."""
        x_ = df_['skygrid_to_mrca_median'].values
        y_ = df_['constcoal_to_mrca_median'].values
        if diff_mode == 'relative':
            d_ = np.where(x_ > 0, (y_ - x_) / x_, np.nan)
            lo_col, hi_col = 'rel_diff_hpd_lower', 'rel_diff_hpd_upper'
        else:
            d_ = y_ - x_
            lo_col, hi_col = 'abs_diff_hpd_lower', 'abs_diff_hpd_upper'
        if diff_hpd and lo_col in df_.columns and hi_col in df_.columns:
            d_lo_, d_hi_ = df_[lo_col].values, df_[hi_col].values
        else:
            d_lo_ = d_hi_ = None
        raw_colors_ = df_['_point_color'].values if '_point_color' in df_.columns else None
        return x_, d_, d_lo_, d_hi_, raw_colors_

    def _plot_diff_panel(ax_, win_x, d, raw_x, raw_d, raw_d_lo, raw_d_hi, x_range, raw_colors=None):
        if len(raw_x):
            seg_colors = raw_colors if raw_colors is not None else '#cccccc'
            pt_colors = raw_colors if raw_colors is not None else '#bbbbbb'
            if raw_d_lo is not None and raw_d_hi is not None:
                segs = np.stack([
                    np.column_stack([raw_x, raw_d_lo]),
                    np.column_stack([raw_x, raw_d_hi]),
                ], axis=1)
                ax_.add_collection(LineCollection(segs, colors=seg_colors, linewidths=0.5,
                                                  alpha=0.5, zorder=1, rasterized=True))
            ax_.scatter(raw_x, raw_d, s=1, color=pt_colors, alpha=0.4, zorder=2, rasterized=True)
        ax_.scatter(win_x, d, s=markersize ** 2, color=running_avg_color, zorder=3)
        ax_.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=4)
        if log_scale:
            ax_.set_xscale('log')
        ax_.set_xlim(x_range)
        if diff_axis_range is not None:
            ax_.set_ylim(diff_axis_range)
        ylabel = 'Rel. diff.' if diff_mode == 'relative' else 'Diff. (years)'
        ax_.set_ylabel(ylabel, fontsize=_tick_fs - 1, labelpad=-2)
        ax_.tick_params(labelsize=_tick_fs - 1)

    # ---- window stats on the full data set ----
    x_all = df_full['skygrid_to_mrca_median'].values
    y_all = df_full['constcoal_to_mrca_median'].values

    if running_avg or diff_plot:
        win_x, win_y, win_d, win_dlo, win_dhi, win_xl, win_xr = _sliding_window_stats(
            x_all, y_all, window_size, window_overlap
        )

    # ---- axis limits (computed before drawing so we can filter windows) ----
    if axis_range is not None:
        lims = list(axis_range)
    else:
        val_cols = [c for c in ['skygrid_to_mrca_median', 'constcoal_to_mrca_median',
                                'skygrid_to_mrca_hpd_lower', 'skygrid_to_mrca_hpd_upper',
                                'constcoal_to_mrca_hpd_lower', 'constcoal_to_mrca_hpd_upper']
                    if c in df.columns]
        all_vals = np.concatenate([df[c].values for c in val_cols])
        all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        lims = [all_vals.min(), all_vals.max()]

    # ---- main panel ----
    _scatter(ax_main, df)

    if running_avg or diff_plot:
        # only windows with median within the visible range
        main_sel = np.where((win_x >= lims[0]) & (win_x <= lims[1]))[0]

    if running_avg and len(main_sel):
        xp, yp = _step_path(win_y[main_sel], win_xl[main_sel], win_xr[main_sel], win_x[main_sel])
        ax_main.plot(xp, yp, color=running_avg_color, linewidth=1.5, zorder=5, label='Running median')

    if log_scale:
        ax_main.set_xscale('log')
        ax_main.set_yscale('log')
    ax_main.plot(lims, lims, '--', color='black', linewidth=1, zorder=4)
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    ax_main.set_xlabel(x_label, fontsize=_label_fs)
    ax_main.set_ylabel(y_label, fontsize=_label_fs)
    ax_main.tick_params(labelsize=_tick_fs)
    if title is not None:
        ax_main.set_title(title, fontsize=_title_fs)
    ax_main.set_aspect('equal')

    # ---- legend: errorbar-style marker for the points ----
    _marker_line = Line2D([0], [0], marker='o', color='#555555', markersize=_legend_ms,
                          linestyle='none', alpha=1.0)
    _caplines = (Line2D([0], [0], color='#555555', linewidth=elinewidth),
                 Line2D([0], [0], color='#555555', linewidth=elinewidth))
    _barlines = (LineCollection([[[0, -0.3], [0, 0.3]]], colors=['#555555'], linewidths=[elinewidth]),
                 LineCollection([[[-0.3, 0], [0.3, 0]]], colors=['#555555'], linewidths=[elinewidth]))
    _handle = ErrorbarContainer((_marker_line, _caplines, _barlines), has_xerr=True, has_yerr=True,
                                label='Tip dist. from\npairwise MRCA\n[years]' if legend_outside
                                      else 'Tip dist. from\npairwise MRCA [years]')
    legend_handles = [_handle]
    if color_threshold_date is not None:
        _sep = '\n' if legend_outside else ' '
        legend_handles += [
            Line2D([0], [0], marker=marker_after, color=color_after, markersize=_legend_ms,
                   linestyle='none',
                   label=f"Both tips after{_sep}{_decimal_year_to_datetime(color_threshold_date):%-d. %b. %Y}"),
            Line2D([0], [0], marker=marker_other, color=color_other, markersize=_legend_ms,
                   linestyle='none', label=f'At least one{_sep}before'),
        ]
    if legend_outside:
        ax_main.legend(handles=legend_handles, fontsize=_legend_fs, frameon=False,
                       loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                       handletextpad=0.4, labelspacing=0.9)
    else:
        ax_main.legend(handles=legend_handles, fontsize=_legend_fs, frameon=False,
                       loc='upper left', borderpad=0.5, handletextpad=0.4)

    # ---- diff panel below main (no zoom) ----
    if diff_plot and not zoom:
        bg_x, bg_d, bg_dlo, bg_dhi, bg_colors = _df_diff(df)
        _plot_diff_panel(ax_diff, win_x[main_sel], win_d[main_sel], bg_x, bg_d, bg_dlo, bg_dhi, lims,
                         raw_colors=bg_colors)
        ax_diff.set_xlabel(x_label, fontsize=_label_fs)
        ax_main.set_xlabel('')

    # ---- zoom panel (and diff panel below it) ----
    if zoom:
        in_zoom = ((df_full['skygrid_to_mrca_median'] >= zoom_range[0]) &
                   (df_full['skygrid_to_mrca_median'] <= zoom_range[1]) &
                   (df_full['constcoal_to_mrca_median'] >= zoom_range[0]) &
                   (df_full['constcoal_to_mrca_median'] <= zoom_range[1]))
        df_zoom = df_full[in_zoom]
        if n_subsample_zoom is not None and n_subsample_zoom < len(df_zoom):
            df_zoom = df_zoom.sample(n=n_subsample_zoom, random_state=1)
        _scatter(ax_zoom, df_zoom)

        if running_avg or diff_plot:
            ws_z = window_size_zoom if window_size_zoom is not None else window_size
            ov_z = window_overlap_zoom if window_overlap_zoom is not None else window_overlap
            zwin_x, zwin_y, zwin_d, _, _, zwin_xl, zwin_xr = _sliding_window_stats(
                x_all, y_all, ws_z, ov_z
            )
            # only windows with median within the zoom range
            sel = np.where((zwin_x >= zoom_range[0]) & (zwin_x <= zoom_range[1]))[0]

            if running_avg and len(sel):
                xp, yp = _step_path(zwin_y[sel], zwin_xl[sel], zwin_xr[sel], zwin_x[sel])
                ax_zoom.plot(xp, yp, color=running_avg_color, linewidth=1.5, zorder=5)

            if diff_plot and len(sel):
                bg_x, bg_d, bg_dlo, bg_dhi, bg_colors = _df_diff(df_zoom)
                _plot_diff_panel(ax_diff, zwin_x[sel], zwin_d[sel], bg_x, bg_d, bg_dlo, bg_dhi, zoom_range,
                                 raw_colors=bg_colors)
                ax_diff.set_xlabel(x_label, fontsize=_label_fs)

        if log_scale:
            ax_zoom.set_xscale('log')
            ax_zoom.set_yscale('log')
        ax_zoom.plot(list(zoom_range), list(zoom_range), '--', color='black', linewidth=1, zorder=4)
        ax_zoom.set_xlim(zoom_range)
        ax_zoom.set_ylim(zoom_range)
        ax_zoom.tick_params(labelsize=_tick_fs - 1)
        ax_zoom.set_xlabel('')
        ax_zoom.set_ylabel('')

        # rectangle on main panel marking the zoom region, connected to the zoom panel
        zoom_width = zoom_range[1] - zoom_range[0]
        ax_main.add_patch(Rectangle(
            (zoom_range[0], zoom_range[0]), zoom_width, zoom_width,
            linewidth=1.2, edgecolor=RED, facecolor=mcolors.to_rgba(RED, 0.10),
            linestyle='-', zorder=5,
        ))
        for corner_main, corner_zoom in [
            ((zoom_range[1], zoom_range[1]), (zoom_range[0], zoom_range[1])),
            ((zoom_range[1], zoom_range[0]), (zoom_range[0], zoom_range[0])),
        ]:
            fig.add_artist(ConnectionPatch(
                xyA=corner_main, coordsA=ax_main.transData,
                xyB=corner_zoom, coordsB=ax_zoom.transData,
                color=RED, linewidth=0.8, linestyle='--', zorder=5,
            ))

    if save_path is not None:
        _save_figure(fig, save_path, bbox_inches='tight' if legend_outside else None)

    return tuple(a for a in (ax_main, ax_zoom, ax_diff) if a is not None) if (zoom or diff_plot) else ax_main


# =============================================================================
# Data overview
# =============================================================================

def plot_sampling_neg_cumulative(meta_df, ax=None, title="Negative cumulative distribution of sampling dates"):
    """
    Plot the negative cumulative distribution of sampling dates: for each
    date, how many samples were collected at or after that date.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Must contain a 'decimal_year' column.
    ax : matplotlib Axes or None
    title : str
    """
    dates = meta_df["decimal_year"].sort_values().values
    n = len(dates)
    neg_cum = n - np.arange(1, n + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.step(dates, neg_cum, where="post", color="steelblue", linewidth=1.5)
    ax.set_xlabel("Decimal year")
    ax.set_ylabel("Samples collected at or after date")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax
