"""Shared plotting style constants for thesis notebooks.

These defaults are anchored to the settings used in:
- notebooks/procedure_policy_bar_grid.ipynb
- notebooks/procedure_policy_bar_grid_best_per_subset.ipynb
"""

# Core font scale (reference style from procedure policy bar notebooks).
FONT_SIZE_SUPTITLE = 12
FONT_SIZE_SUPYLABEL = 11
FONT_SIZE_PANEL_TITLE = 14
FONT_SIZE_XLABEL = 12
FONT_SIZE_YLABEL = 12
FONT_SIZE_XTICKS = 12
FONT_SIZE_YTICKS = 12
FONT_SIZE_LEGEND = 12

# Common y tick spacing for paired-delta bar views.
# Keep interaction-cost deltas coarse and dice deltas finer.
BAR_Y_TICK_STEP_ITERS = 0.1
BAR_Y_TICK_STEP_DICE = 0.02

# Aliases for notebooks that use different naming conventions.
FONT_SIZE_BASE = FONT_SIZE_XLABEL
FONT_SIZE_TICKS = FONT_SIZE_XTICKS
FONT_SIZE_SUBTITLE = FONT_SIZE_PANEL_TITLE
FONT_SIZE_TITLE = FONT_SIZE_SUPTITLE + 4

# Fixed layout controls for policy-position curves (keeps figure geometry stable).
SUBPLOT_LEFT = 0.10
SUBPLOT_RIGHT = 0.985
SUBPLOT_TOP = 0.88
SUBPLOT_BOTTOM = 0.16
SUBPLOT_WSPACE = 0.28
SUBPLOT_HSPACE = 0.30
LEGEND_Y = 0.955

# Metric-specific y-axis tick controls for policy-position curves.
# Use a finer step so final-dice panels always show visible tick labels.
Y_TICK_STEP_DICE = 0.1
Y_TICK_STEP_ITERS = 1.0
Y_TICK_FMT_DICE = "%.2f"
Y_TICK_FMT_ITERS = "%.1f"
