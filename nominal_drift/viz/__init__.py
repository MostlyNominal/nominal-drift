"""
nominal_drift.viz
==============
Visualisation engine — produces all graphical outputs for both scientific
tracks.  Deliberately separated from the calculation modules so that plots
and animations can be regenerated with different styling without rerunning
expensive simulations.

Modules
-------
profile_plotter          : Static concentration-profile plots (matplotlib PNG/SVG)
animator                 : Time-evolving engineering diffusion animations (GIF)
mechanism_animator       : Mechanism-inspired schematic animation (GIF)
microstructure_animator  : Microstructure-inspired particle scene animation (GIF)
species_styles           : Generic element styling (colour, radius, priority)
risk_map                 : TTT/CCT-style sensitization risk heatmaps
"""
