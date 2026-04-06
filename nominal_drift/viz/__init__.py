"""
nominal_drift.viz
==============
Visualisation engine — produces all graphical outputs for both scientific
tracks.  Deliberately separated from the calculation modules so that plots
and animations can be regenerated with different styling without rerunning
expensive simulations.

Planned modules
---------------
profile_plotter   : Static concentration-profile plots (matplotlib PNG/SVG)
animator          : Time-evolving diffusion animations (MP4 / GIF)
heatmap_renderer  : T-t-risk heatmaps
ttt_renderer      : TTT/CCT-style interactive diagrams (Plotly, Phase 2)
"""
