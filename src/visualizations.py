"""
helpers.py — Live multi-graph plotting, main-thread only.

No threads, no subprocesses. Call plot_live_value() / plot_live_lidar()
directly in your eval loop — matplotlib stays on the main thread where
TkAgg expects it. plt.pause(0.001) flushes the GUI without blocking.
"""

import signal
import sys
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─── figure state (module-level) ─────────────────────────────────────────────
_fig          = None
_scalar_axes  = {}   # name  -> {"ax", "line", "x", "y"}
_lidar_ax     = None
_lidar_fill   = None
_lidar_line   = None
_lidar_latest = None
_lidar_reg    = False
_max_points   = 400

COLORS = ["#4C8BF5", "#F5A623", "#27AE60"]


# ─── Ctrl+C ───────────────────────────────────────────────────────────────────
def _sigint_handler(sig, frame):
    print("\n⛔  Ctrl+C — closing plots.", flush=True)
    close_live_plots()
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint_handler)


# ─── internal helpers ─────────────────────────────────────────────────────────
def _lidar_to_polar(data):
    n = len(data)
    a = np.linspace(-np.pi, np.pi, n, endpoint=False)
    v = np.asarray(data, dtype=float)
    return np.append(a, a[0]), np.append(v, v[0])


def _rebuild():
    """Tear down and recreate the figure with correct layout."""
    global _fig, _lidar_ax, _lidar_fill, _lidar_line

    if _fig is not None:
        plt.close(_fig)

    n_s = len(_scalar_axes)
    n_l = 1 if _lidar_reg else 0
    n   = max(n_s, 1)

    if n_s == 0 and n_l == 0:
        return

    _fig = plt.figure(figsize=(9, max(3 * n_s, 3)))
    _fig.suptitle("DriveAlign — Live Evaluation", fontsize=11, fontweight="bold")

    if n_l and n_s:
        gs = gridspec.GridSpec(n, 2, figure=_fig,
                               hspace=0.65, wspace=0.45,
                               width_ratios=[1.4, 1])
    elif n_s:
        gs = gridspec.GridSpec(n_s, 1, figure=_fig, hspace=0.65)
    else:
        gs = gridspec.GridSpec(1, 1, figure=_fig)

    # scalar subplots
    for idx, (name, state) in enumerate(_scalar_axes.items()):
        ax = _fig.add_subplot(gs[idx, 0] if (n_l and n_s) else gs[idx])
        (ln,) = ax.plot(state["x"], state["y"],
                        linewidth=1.8, color=COLORS[idx % len(COLORS)])
        ax.set_title(name, fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linestyle="--")
        state["ax"]   = ax
        state["line"] = ln

    # lidar polar subplot
    if _lidar_reg:
        if n_s:
            ax_l = _fig.add_subplot(gs[:n, 1], projection="polar")
        else:
            ax_l = _fig.add_subplot(gs[0], projection="polar")

        ax_l.set_title("Lidar", fontsize=9, fontweight="bold", pad=10)
        ax_l.set_theta_zero_location("N")
        ax_l.set_theta_direction(-1)
        ax_l.set_rlabel_position(135)
        ax_l.tick_params(labelsize=6)
        ax_l.grid(True, alpha=0.2)

        if _lidar_latest is not None:
            a, v = _lidar_to_polar(_lidar_latest)
        else:
            a = np.linspace(-np.pi, np.pi, 2)
            v = np.zeros(2)

        _lidar_ax   = ax_l
        _lidar_fill = ax_l.fill(a, v, alpha=0.25, color="#4C8BF5")[0]
        _lidar_line, = ax_l.plot(a, v, color="#4C8BF5", linewidth=1.5)

    plt.draw()
    plt.pause(0.001)


# ─── public API ───────────────────────────────────────────────────────────────
def plot_live_value(
    value: float,
    graph_name: str,
    step: Optional[int] = None,
    max_points: int     = 400,
) -> None:
    """Record a scalar metric and refresh the plot. Call from main thread."""
    global _fig, _max_points

    _max_points = max_points
    new_graph   = graph_name not in _scalar_axes

    if new_graph:
        _scalar_axes[graph_name] = {"ax": None, "line": None, "x": [], "y": []}

    s = _scalar_axes[graph_name]
    s["x"].append(0 if step is None else int(step))
    s["y"].append(float(np.asarray(value).reshape(-1)[0]))

    if max_points and len(s["x"]) > max_points:
        del s["x"][:-max_points]
        del s["y"][:-max_points]

    if new_graph or _fig is None:
        _rebuild()
        return

    # fast path — just update line data
    if s["line"] is not None:
        s["line"].set_data(s["x"], s["y"])
        s["ax"].relim()
        s["ax"].autoscale_view()

    plt.draw()
    plt.pause(0.001)


def plot_live_lidar(
    lidar_array,
    step: Optional[int] = None,
    max_points: int     = 400,
) -> None:
    """Update the polar lidar subplot. Call from main thread."""
    global _lidar_latest, _lidar_reg, _lidar_fill, _lidar_line, _fig

    data = np.asarray(lidar_array, dtype=float).ravel()
    if len(data) == 0:
        return

    _lidar_latest = data
    was_reg       = _lidar_reg

    if not _lidar_reg:
        _lidar_reg = True

    if not was_reg or _fig is None:
        _rebuild()
        return

    if _lidar_ax is None:
        return

    a, v = _lidar_to_polar(data)

    if _lidar_fill is not None:
        _lidar_fill.remove()

    globals()["_lidar_fill"] = _lidar_ax.fill(a, v, alpha=0.25, color="#4C8BF5")[0]
    _lidar_line.set_data(a, v)
    _lidar_ax.set_rmax(max(float(v.max()), 1.0))

    plt.draw()
    plt.pause(0.001)


def close_live_plots() -> None:
    """Close the figure cleanly."""
    global _fig
    if _fig is not None:
        plt.close(_fig)
        _fig = None