# run_dda_bif

Read a supplied ATL05 file (DDA Bifurcation file created from an ATL03) and generate finetrack heights for top and bottom surfaces of potential melt ponds.

This routine requires an ATL05 generated using test_dda_bif. This routine was created by Jeff Lee and must be compiled within an ASAS environment. test_dda_bif reads an ATL03 file, computes a density value for each photon, computes a signal mask, finds potential bifurcation segments, and creates masks for top and bottom surfaces of bifurcation segments.

This routine reads the ATL05 file, collects photons within bifurcation segments, and computes fine_track segment heights for the top and bottom surfaces of the potential bifurcation segment.

Options for running include: start and end processing time (t0, t1); beam selection; debug plot output; csv summary file output; and source photon selection (all photons in window, or only signal photons).

Path to input ATL05 and ATL03 calibration file are hardcoded within script.

Sample execution:

python run_dda_bif.py -t0 144088547.34 -t1 144088550.0 -b 'gt1l gt1r gt2l gt2r gt3l gt3r' -debug_plots 'y' -source_photons 'signal'
