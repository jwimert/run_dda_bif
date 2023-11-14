##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_dda_bif.py
Jesse Wimert (10/2023)

This routine reads DDA bifurcation ATL05 file and computes
fine_track heights for the top and bottom surfaces of 
bifurcation segments. 

ATL07 is read in order to compare heights


INPUTS:
        -t0 delta time start time to process sea ice segments
        -t1 delta time end time to process sea ice segments
        -b  beam selection, default all [gt1l gt1r gt2l gt2r gt3l gt3r]
        -debug_plots plot individual fine track statistics ('y' or 'n')
        -csv_file provides csv file for each surface ('y' or 'n') 
        -source_photons selects signal photons only ('signal', default), 
         or all photons within window ('all')

OUTPUTS:
        -bif_segment_xxxx.png summary plot of bifurcation surface heights
        -bif_segment_top_xxxx.csv summary of bifurcation top surface heights
        -bif_segment_bot_xxxx.csv summary of bifurcation bottom surface heights
        
        
Steps:

Notes:


sample run:

python run_dda_bif.py -t0 144088547.34 -t1 144088550.0 
  -b 'gt1l gt1r gt2l gt2r gt3l gt3r' -debug_plots 'y' -source_photons 'signal'


"""
#
import os
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fine_trap import fine_track


###
### Function to parse command line arguments
###
 
def parse():
    """ Parse command line input arguments. """

    parser = argparse.ArgumentParser(description='Compute ATL07 sea ice heights')
    parser.add_argument('-t0', action='store', default='0.0', dest='t0',
                        help='start time of processing', type=str)
    parser.add_argument('-t1', action='store', default='0.0', dest='t1',
                        help='plot upper bound', type=str)
    parser.add_argument('-b', action='store', default='gt1r gt1l gt2r gt2l gt3r gt3l',
                        dest='beams', help="List of ATLAS beams [e.g.: \
                        'gt1r gt2r']", type=str)
    parser.add_argument('-debug_plots', action='store', default='n', dest='debug_plots',
                        help="Generate map plot ['y' or 'n', default 'n']", type=str)
    parser.add_argument('-csv_file', action='store', default='n', dest='csv_file',
                        help="Generate csv files ['y' or 'n', default 'n']", type=str)
    parser.add_argument('-no_fpb_corr', action='store', default='n', dest='no_fpb_corr',
                        help="plot heights with no fpb_corr ['y' or 'n', default 'n']", type=str)
    parser.add_argument('-source_photons', action='store', default='signal', 
                        dest='source_photons', help="Use only masked signal photons \
                        or all for finetrack segments ['signal' or 'all', \
                        default 'signal']", type=str)
    inps = parser.parse_args()
    return inps

###
### Print welcome message
###

# os.system('clear')
print(' ')
print('==============================================')
print(' ')
print('run_dda_bif.py')
print(' ')
print(' ')
print('Read DDA bifurcation ATL05 file and computes ')
print('fine_track heights for the top and bottom surfaces of ')
print('bifurcation segments. ')
print(' ')
print(' ')
print('Jesse Wimert')
print('Last update: October 2023')
print(' ')
print('==============================================')
print(' ')
print(' ')


###
### Read command line input parameters
###

inps = parse()
t0 = float(inps.t0)
t1 = float(inps.t1)
beams = (inps.beams.split())
debug_plots = inps.debug_plots
csv_file = inps.csv_file
no_fpb_corr = inps.no_fpb_corr
source_photons = inps.source_photons


###
### Input Files:
###


#
# Set output directory
#

output_directory = '/Users/jwimert/Desktop/summer_arctic_20220726/DDA/finetrack_prep/output_run_dda_bif'
# output_directory = '/Users/jwimert/Desktop/summer_arctic_20220726/DDA/finetrack_prep/output_run_dda_all_photons'

if not (os.path.exists(output_directory)):
  os.makedirs(output_directory)

#
# Set input ATL03 calibrations file
#

ATL03_calibrations = '/Users/jwimert/Desktop/summer_arctic_20220726/DDA/finetrack_prep/atl03_calibrations_and_wf_tables.h5'


#
# Set input ATL05 file
#

ATL05_file = '/Users/jwimert/Desktop/summer_arctic_20220726/DDA/run_atl05/ATL05_20220726163210_05311604_006_02_full.h5'

print('output directory:')
print(output_directory)
print(' ')
print('ATL05_file:')
print(ATL05_file)
print(' ')
print('ATL03 calibrations file:')
print(ATL03_calibrations)
print(' ')
print('=====================================')
print(' ')



###
### Set constants:
###

#
# Set finetrack histogram bins
# (lb/ub_bin : CENTER of lowest/highest bin)
#

bin_size = 0.025
lb_bin = -2.0
ub_bin = 3.5

bins_center = np.arange(lb_bin, ub_bin+bin_size, bin_size)
bins_edges = np.arange(lb_bin - bin_size*0.5, ub_bin + bin_size, bin_size)


#
# Set number of photons to collect for fine track
#

n_collect = 150


#
# Set ordered beams
#

beams_all = "gt1l gt1r gt2l gt2r gt3l gt3r"
beams_all = beams_all.split()


###
### Read CAL19 and expected waveform tables from atl03_calibrations_and_wf_tables.h5
###

##
## Open file
##


hf = h5py.File(ATL03_calibrations, 'r')

#
# Read initial data 
#

atl03_filename = hf.get('atl03_filename')[0]
atl03_filename = atl03_filename.decode()

atl03_orientation = hf.get('orientaion')[0]


#
# Read CAL19 data
#

cal19_dead_time = hf.get('CAL19/dead_time')[:]
cal19_strength = hf.get('CAL19/strength')[:][:]
cal19_width = hf.get('CAL19/width')[:][:]
cal19_fpb_corr = hf.get('CAL19/fpb_corr')[:][:][:]


#
# Read expected waveform tables and data
#

binz = hf.get('exp_wf_table/binz')[:]
mz = hf.get('exp_wf_table/mz')[:]
sdz = hf.get('exp_wf_table/sdz')[:]
wf_table1 = hf.get('exp_wf_table/wf_table1')[:][:][:]
wf_table2 = hf.get('exp_wf_table/wf_table2')[:][:][:]

n_bin = len(binz)
n_mu = len(mz)
n_sig = len(sdz)


#
# Read per beam data
#

dead_time_all = []
strong_weak = []
wf_table_select = []

for beam in beams_all:
    dead_time_all.append(hf.get(beam + '/dead_time')[0])
    strong_weak.append(hf.get(beam + '/strong_weak')[0])
    wf_table_select.append(hf.get(beam + '/wf_table_select')[0])

hf.close()


#
# Print summary of data read in
#

print(' ')
print('ATL03 calibration data summary: ')
print(' ')
print(atl03_filename)
print(' ')
print('Spacecraft orientation:', atl03_orientation)
print(' ')
print('Beam gt, strong/weak flag (1/0), wf_table_select flag, dead_time (ns)')
print(' ')
for ii in np.arange(0,len(beams)):
	print(strong_weak[ii], wf_table_select[ii], dead_time_all[ii])
print(' ')
print('CAL19 arrays:')
print('dead_time',cal19_dead_time.shape)
print('strength', cal19_strength.shape)
print('width',cal19_width.shape)
print('fpb_corr', cal19_fpb_corr.shape)
print(' ')
print(' ')
print('Expected waveform table arrays:')
print('Size of wf_table1')
print(wf_table1.shape)
print('Size of wf_table2')
print(wf_table2.shape)
print(' ')
print('n sigma bins')
print(n_sig)
print(' ')
print('n mu bins')
print(n_mu)
print(' ')
print('n hist bins')
print(n_bin)
print(' ')
print('=====================================')
print(' ')



###
### Start Processing BIF segments
###

print('Start processing BIF segments')


##
## Loop through selected beams
##


for beam in beams:

  print(' ')
  print('=====================================')
  print(' ')
  print(' ')
  print(' PROCESSING BEAM: ')
  print(' ')
  print(beam)
  print(' ')
  print('=====================================')
  print(' ')

  beam_index = beams_all.index(beam)
  
  dead_time = dead_time_all[beam_index]
  if wf_table_select[beam_index] == 1:
    wf_table = wf_table1
  if wf_table_select[beam_index] == 2:
    wf_table = wf_table2
    
    
#
# Create directory for output files
#
 
  beam_directory = output_directory + '/' + str(beam)
  if not (os.path.exists(beam_directory)):
    os.makedirs(beam_directory)

  if (debug_plots == 'y'):
    debug_directory_top = beam_directory + '/debug_top'
    debug_directory_bot = beam_directory + '/debug_bot'
    if not (os.path.exists(debug_directory_top)):
      os.makedirs(debug_directory_top)
    if not (os.path.exists(debug_directory_bot)):
      os.makedirs(debug_directory_bot)
      
#     print('debug directories:')
#     print(debug_directory_top)  
#     print(debug_directory_bot)


##
##
################### Read ATL05 file ###################################
## 
##


#
# Open file
#

  hf = h5py.File(ATL05_file, 'r')

#
# Read data (n_ph)
#

  ATL05_h_ph = hf[beam + '/h_ph'][:]
  ATL05_x_ph = hf[beam + '/x_ph'][:]
  ATL05_t_ph = hf[beam + '/t_ph'][:]
  ATL05_density_ph = hf[beam + '/density_ph'][:]


#
# Read slabs
#
# ATL05_noise_slab = hf[beam + '/noise_slab'][:]
# ATL05_signal_slab = hf[beam + '/signal_slab'][:]

  ATL05_is_multi = hf[beam + '/is_multi'][:]


#
# Read Masks
#
# ATL05_threshold_mask = hf[beam + '/threshold_mask'][:]

  ATL05_signal_mask = hf[beam + '/signal_mask'][:]


#
# Read top/bot surface flags
#

  ATL05_signal_bot = hf[beam + '/signal_bot'][:]
  ATL05_signal_top = hf[beam + '/signal_top'][:]


#
# Read bif segments
#

  ATL05_bif0 = hf[beam + '/bif_segments/bif_beg'][:]
  ATL05_bif1 = hf[beam + '/bif_segments/bif_end'][:]


#
# Read ponds segments
#

  ATL05_ponds_good = hf[beam + '/ponds/good'][:]
  ATL05_ponds_left = hf[beam + '/ponds/left_edge'][:]
  ATL05_ponds_right = hf[beam + '/ponds/right_edge'][:]
  ATL05_ponds_center = hf[beam + '/ponds/center'][:]
  ATL05_ponds_top = hf[beam + '/ponds/top_h'][:]
  ATL05_ponds_max_depth = hf[beam + '/ponds/max_depth'][:]
  ATL05_ponds_mean_depth = hf[beam + '/ponds/mean_depth'][:]

  hf.close()


#
# Save array lengths
#  

  n_photons = len(ATL05_t_ph)
  n_bif = len(ATL05_bif0)
  n_ponds = len(ATL05_ponds_left)
  n_ponds_good = np.count_nonzero(ATL05_ponds_good)


  print(' ')
  print('ATL05 data summary: ')
  print(' ')

  print('delta time:')
  print(ATL05_t_ph[0], ATL05_t_ph[n_photons-1])
  print('dist_x:')
  print(ATL05_x_ph[0], ATL05_x_ph[n_photons-1])
  print(' ')
  
  print('total photons:', n_photons)
  print(' ')

  print('bifurcation segments:', n_bif)
  print(ATL05_bif0[0], '   :   ', ATL05_bif1[n_bif-1])
  print(' ')

  print('number of pond segments, good:')
  print(n_ponds, n_ponds_good)
  print(' ')


##
## Set ATL05 indicies based on t0 and t1
##

  atl05_i0 = 0
  atl05_i1 = 0

  if t0 is None:
    atl05_i0 = 0
    t0 = np.min(ATL05_t_ph)
  else:
    for i in np.arange(0, n_photons - 1):
      if (ATL05_t_ph[i] > t0):
        atl05_i0 = i
        break

  if t0 is None:
    atl05_i1 = n_photons - 1
    t1 = np.max(ATL05_t_ph)
  else:
    for i in np.arange(0, n_photons - 1):
      if (ATL05_t_ph[i] > t1):
        atl05_i1 = i
        break

  atl05_i0 = atl05_i0 - 50
  atl05_i1 = atl05_i1 + 50
  
  if (atl05_i0 < 0):
    atl05_i0 = 0
  if (atl05_i0 > n_photons - 1):
    atl05_i1 = n_photons - 1


#
# Collect BIF segments within timeframe
#

  bif_loc = np.where( (ATL05_bif0[:] > ATL05_x_ph[atl05_i0]) & (ATL05_bif0[:] < ATL05_x_ph[atl05_i1]) )
  bif_loc = np.transpose(bif_loc)
  n_bif_select = (ATL05_bif0[bif_loc].size)


#
# Collect BIF segments within timeframe with pond=good flag
#

  ponds_loc = np.where( (ATL05_ponds_left[:] >= ATL05_x_ph[atl05_i0]) & (ATL05_ponds_left[:] < ATL05_x_ph[atl05_i1]) )
  ponds_good_loc = np.where( (ATL05_ponds_left[:] >= ATL05_x_ph[atl05_i0]) & (ATL05_ponds_left[:] < ATL05_x_ph[atl05_i1]) & (ATL05_ponds_good[:] > 0))


  print(' ')
  print('Subset ATL05 time span')
  print(' ')
  print('delta time:')
  print(ATL05_t_ph[atl05_i0], ATL05_t_ph[atl05_i1])
  print('dist_x:')
  print(ATL05_x_ph[atl05_i0], ATL05_x_ph[atl05_i1])
  print('ATL05 photons spaned:', atl05_i1 - atl05_i0)
  print(' ')
  print('bif segments')
  print(' ')
  print('bifurcation segments within span')
  print(n_bif_select)
  print('number of pond segments, good:')
  print(np.transpose(ponds_loc).size, np.transpose(ponds_good_loc).size)
  print(' ')


###
### Loop through bif segments
###

  for i_bif in bif_loc:

    print(' ')
    print('Processing bif seg:', i_bif, ATL05_bif0[i_bif], ATL05_bif1[i_bif], ATL05_ponds_good[i_bif])
    print(' ')

##
## Collect photons within bif segment
##
    ph_loc_all = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]))
    ph_loc_signal = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) & ATL05_signal_mask[:] > 0)
    ph_loc_signal_ismulti = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) & (ATL05_signal_mask[:] > 0) & (ATL05_is_multi[:] > 0) )

    ph_loc_top = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) & (ATL05_signal_top[:] > 0) & (ATL05_signal_mask[:] > 0))
    ph_loc_bot = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) & (ATL05_signal_bot[:] > 0) & (ATL05_signal_mask[:] > 0))


##
## Compute top and bottom coarse windows:
## Compute coarse mean for top and bottom segment
## Compute mid point between top and bottom coarse segment
## Photon window for top surface : midpoint -> (top_coarse + 3.5)
## Photon window for bottom surface : (bottom_coarse - 2.0) -> (midpoint)
##

    top_seg_coarse = np.mean(ATL05_h_ph[ph_loc_top])
    bot_seg_coarse = np.mean(ATL05_h_ph[ph_loc_bot])

    mid_depth = top_seg_coarse - (top_seg_coarse - bot_seg_coarse)/2.0

    top_window0 = mid_depth
    top_window1 = top_seg_coarse + 3.5

    bot_window0 = bot_seg_coarse - 2.0
    bot_window1 = mid_depth


###
### Collect photons using bif0/bif1, top/bot signal mask, and window created using coarse means
###

##
## Using signal and top/bot mask
##

    if (source_photons == 'signal'):

      photons_finetrack_top = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) \
                                 & (ATL05_signal_top[:] > 0) & (ATL05_h_ph[:] > top_window0) & (ATL05_h_ph[:] < top_window1) )


      photons_finetrack_bot = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) \
                                 & (ATL05_signal_bot[:] > 0) & (ATL05_h_ph[:] > bot_window0) & (ATL05_h_ph[:] < bot_window1) )


##
## Using just coarse window (mimic ASAS)
##

    if (source_photons == 'all'):

      photons_finetrack_top = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) \
                                 & (ATL05_h_ph[:] > top_window0) & (ATL05_h_ph[:] < top_window1) )


      photons_finetrack_bot = np.where( (ATL05_x_ph[:] > ATL05_bif0[i_bif]) & (ATL05_x_ph[:] < ATL05_bif1[i_bif]) \
                                 & (ATL05_h_ph[:] > bot_window0) & (ATL05_h_ph[:] < bot_window1) )


###
### loop through top surface at half steps, call finetrack to calculate segment height
###


    n_shots = 1
    i_step=0
    output_top = {"delta_time":[],
              "dist_x":[],
              "h_surf":[],
              "h_no_fpb_corr":[],
              "w_gauss":[],
              "fpb_corr":[],
              "qual_fit":[]}


    n_photons_top = np.transpose(photons_finetrack_top).size


    print(' ')
    print('process top surface photons: ', 0, ' : ', n_photons_top)
    print(' ')


    for ii in np.arange(0, n_photons_top, n_collect/2):
      i_step = i_step + 1
      i0 = int(ii)
      i1 = int(ii+n_collect-1)
      if (i1 > n_photons_top):
        break
      x_surf_temp = (np.mean(ATL05_x_ph[photons_finetrack_top][i0:i1]))
      t_surf_temp = (np.mean(ATL05_t_ph[photons_finetrack_top][i0:i1]))
      t_span = np.max(ATL05_t_ph[photons_finetrack_top][i0:i1]) - np.min(ATL05_t_ph[photons_finetrack_top][i0:i1])
      n_photons = ATL05_h_ph[photons_finetrack_top][i0:i1].size
      n_shots = int(t_span * 10000.0)


##
## Call finetrack, store output
##

      print(' ')
      print('CALL fine_track, ', i_step)
      print(' ')

      h_surf_temp, w_gauss_temp, fpb_corr, h_fit_qual_flag, error_surface, hist_norm, bins_trim, wf_fit, qtr_h, \
        n_photons_trim, hist_full, hist_trim, wf_table_trim, wf_bins_trim = \
        fine_track(ATL05_h_ph[photons_finetrack_top][i0:i1], top_seg_coarse, n_photons, n_shots, bin_size, lb_bin, ub_bin, wf_table, binz, sdz, mz, cal19_fpb_corr, cal19_width, cal19_strength, cal19_dead_time, dead_time)

      output_top["delta_time"].append(t_surf_temp)
      output_top["dist_x"].append(x_surf_temp)
      output_top["h_surf"].append(h_surf_temp)
      output_top["h_no_fpb_corr"].append(h_surf_temp+fpb_corr)
      output_top["w_gauss"].append(w_gauss_temp)
      output_top["fpb_corr"].append(fpb_corr)
      output_top["qual_fit"].append(h_fit_qual_flag)
       
      if (debug_plots == 'y'):
        
        X_es, Y_es = np.meshgrid(np.arange(sdz.size), np.arange(mz.size))

        es_min0 = np.unravel_index(np.nanargmin(error_surface), error_surface.shape)[0]
        es_min1 = np.unravel_index(np.nanargmin(error_surface), error_surface.shape)[1]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(12, 15), height_ratios=[4, 4, 4, 6])
        fig.suptitle('Fine Track Summary')

        ax0.scatter(ATL05_x_ph[photons_finetrack_top], ATL05_h_ph[photons_finetrack_top], alpha= 0.1, label= 'top surface')
        ax0.scatter(ATL05_x_ph[photons_finetrack_bot], ATL05_h_ph[photons_finetrack_bot], alpha= 0.1, label = 'bot surface')
        ax0.scatter(ATL05_x_ph[photons_finetrack_top][i0:i1], ATL05_h_ph[photons_finetrack_top][i0:i1], alpha= 0.1, label= 'collected photons')
        ax0.scatter(x_surf_temp, h_surf_temp, label= 'fine track surface')
        ax0.hlines(mid_depth, ATL05_bif0[i_bif], ATL05_bif1[i_bif], color='black')
        ax0.title.set_text('ATL03 photon cloud')

        ax1.plot(bins_center[49:109], hist_full[49:109])
        ax1.plot(bins_center[49:109], hist_trim[49:109])
        ax1.title.set_text('Full and trimmed observed waveform')
    
        ax2.plot(bins_trim, hist_norm, label='obs waveform')
        ax2.plot(bins_trim, wf_fit, label='best fit')
        ax2.legend()
        ax2.title.set_text('Observed histogram and best fit expected waveform')

        cont = ax3.contour(X_es, Y_es, error_surface, 75, cmap="OrRd", linestyles="solid")
        ax3.plot([es_min1],[es_min0],np.nanmin(error_surface), markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10, alpha=1.0, label='error_min')
        ax3.title.set_text('Error Surface')
        ax3.contour(X_es,Y_es, error_surface,[qtr_h],linestyles='dashed')
        ax3.text(0.05, 0.05, 'python fit_quality_flag = ' + str(h_fit_qual_flag), transform=ax3.transAxes, size=10, weight='normal', c='black')

        cbar=plt.colorbar(cont)

        output_file = debug_directory_top + '/' + str(i_bif[0]).zfill(4) + '_' + str(i_step).zfill(3) + '.png'
        fig.savefig(output_file, dpi=fig.dpi)
        plt.close('all')
        
#         if (i_step == 7):
#           hf = h5py.File('error_surf_temp.h5', 'w')
#           hf.create_dataset('error_surf', data=error_surface)
#           hf.create_dataset('hist_full',data=hist_full)
#           hf.create_dataset('hist_trim',data=hist_trim)
#           hf.create_dataset('bins_center',data=bins_center)
#           hf.create_dataset('hist_norm',data=hist_norm)
#           hf.create_dataset('wf_fit',data=wf_fit)
#           hf.create_dataset('bins_trim',data=bins_trim)
#           hf.create_dataset('wf_table',data=wf_table)
#           hf.create_dataset('wf_table_trim',data=wf_table_trim)
#           hf.create_dataset('wf_bins_trim',data=wf_bins_trim)
#           hf.close
#           exit()



    
      print(i_step, i0, i1, h_surf_temp, w_gauss_temp, fpb_corr)


###
### loop through bottom surface at half steps, call finetrack to calculate segment height
###

    n_shots = 1
    i_step=0
    output_bot = {"delta_time":[],
              "dist_x":[],
              "h_surf":[],
              "h_no_fpb_corr":[],
              "w_gauss":[],
              "fpb_corr":[],
              "qual_fit":[]}


    n_photons_bot = np.transpose(photons_finetrack_bot).size

    print(' ')
    print('process bot surface photons: ', 0, ' : ', n_photons_bot)
    print(' ')


    for ii in np.arange(0, n_photons_bot, n_collect/2):
      i_step = i_step + 1
      i0 = int(ii)
      i1 = int(ii+n_collect-1)
      if (i1 > n_photons_bot):
        break
      x_surf_temp = (np.mean(ATL05_x_ph[photons_finetrack_bot][i0:i1]))
      t_surf_temp = (np.mean(ATL05_t_ph[photons_finetrack_bot][i0:i1]))
      t_span = np.max(ATL05_t_ph[photons_finetrack_bot][i0:i1]) - np.min(ATL05_t_ph[photons_finetrack_bot][i0:i1])
      n_photons = ATL05_h_ph[photons_finetrack_bot][i0:i1].size
      n_shots = int(t_span * 10000.0)


##
## Call finetrack, store output
##

      h_surf_temp, w_gauss_temp, fpb_corr, h_fit_qual_flag, error_surface, hist_norm, bins_trim, wf_fit, qtr_h, \
        n_photons_trim, hist_full, hist_trim, wf_table_trim, wf_bins_trim = \
        fine_track(ATL05_h_ph[photons_finetrack_bot][i0:i1], bot_seg_coarse, n_photons, n_shots, bin_size, lb_bin, ub_bin, wf_table, binz, sdz, mz, cal19_fpb_corr, cal19_width, cal19_strength, cal19_dead_time, dead_time)

      output_bot["delta_time"].append(t_surf_temp)
      output_bot["dist_x"].append(x_surf_temp)
      output_bot["h_surf"].append(h_surf_temp)
      output_bot["h_no_fpb_corr"].append(h_surf_temp+fpb_corr)
      output_bot["w_gauss"].append(w_gauss_temp)
      output_bot["fpb_corr"].append(fpb_corr)
      output_bot["qual_fit"].append(h_fit_qual_flag)

      if (debug_plots == 'y'):
        
        X_es, Y_es = np.meshgrid(np.arange(sdz.size), np.arange(mz.size))

        es_min0 = np.unravel_index(np.nanargmin(error_surface), error_surface.shape)[0]
        es_min1 = np.unravel_index(np.nanargmin(error_surface), error_surface.shape)[1]

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(12, 15), height_ratios=[4, 4, 4, 6])
        fig.suptitle('Fine Track Summary')

        ax0.scatter(ATL05_x_ph[photons_finetrack_top], ATL05_h_ph[photons_finetrack_top], alpha= 0.1, label= 'top surface')
        ax0.scatter(ATL05_x_ph[photons_finetrack_bot], ATL05_h_ph[photons_finetrack_bot], alpha= 0.1, label = 'bot surface')
        ax0.scatter(ATL05_x_ph[photons_finetrack_bot][i0:i1], ATL05_h_ph[photons_finetrack_bot][i0:i1], alpha= 0.1, label= 'collected photons')
        ax0.scatter(x_surf_temp, h_surf_temp, label= 'fine track surface')
        ax0.hlines(mid_depth, ATL05_bif0[i_bif], ATL05_bif1[i_bif], color='black')
        ax0.title.set_text('ATL03 photon cloud')

        ax1.plot(bins_center[49:109], hist_full[49:109])
        ax1.plot(bins_center[49:109], hist_trim[49:109])
        ax1.title.set_text('Full and trimmed observed waveform')
    
        ax2.plot(bins_trim, hist_norm, label='obs waveform')
        ax2.plot(bins_trim, wf_fit, label='best fit')
        ax2.legend()
        ax2.title.set_text('Observed histogram and best fit expected waveform')

        cont = ax3.contour(X_es, Y_es, error_surface, 75, cmap="OrRd", linestyles="solid")
        ax3.plot([es_min1],[es_min0],np.nanmin(error_surface), markerfacecolor='k', markeredgecolor='k', marker='o', markersize=10, alpha=1.0, label='error_min')
        ax3.title.set_text('Error Surface')
        ax3.contour(X_es,Y_es, error_surface,[qtr_h],linestyles='dashed')
        ax3.text(0.05, 0.05, 'python fit_quality_flag = ' + str(h_fit_qual_flag), transform=ax3.transAxes, size=10, weight='normal', c='black')

        cbar=plt.colorbar(cont)

        output_file = debug_directory_bot + '/' + str(i_bif[0]).zfill(4) + '_' + str(i_step).zfill(3) + '.png'
        fig.savefig(output_file, dpi=fig.dpi)
        plt.close('all')

      print(i_step, i0, i1, h_surf_temp, w_gauss_temp, fpb_corr)


    if (csv_file == 'y'):
      output_df_top = pd.DataFrame(output_top)    
      output_df_bot = pd.DataFrame(output_bot)    
 
      output_file_top = beam_directory + '/bif_segment_top_' + str(i_bif[0]).zfill(4) + '.csv'
      output_file_bot = beam_directory + '/bif_segment_bot_' + str(i_bif[0]).zfill(4) + '.csv'

      output_df_top.to_csv(output_file_top, index=False)
      output_df_bot.to_csv(output_file_bot, index=False)

    

# ###
# ### Plot bifurcate surfaces summary 
# ###
#  
    fig, ax0 = plt.subplots(1, figsize=(7, 5))
    fig.suptitle('bifurcate surfaces')
    ax0.scatter(ATL05_x_ph[photons_finetrack_top], ATL05_h_ph[photons_finetrack_top], alpha= 0.1, label= 'top surface') 
    ax0.scatter(ATL05_x_ph[photons_finetrack_bot], ATL05_h_ph[photons_finetrack_bot], alpha= 0.1, label = 'bot surface')
    ax0.scatter(output_top['dist_x'][:], output_top['h_surf'][:], label= 'fine_track top surface')
    ax0.scatter(output_bot['dist_x'][:], output_bot['h_surf'][:], label = 'fine_track bot surface')
    if (no_fpb_corr == 'y'):
      ax0.scatter(output_top['dist_x'][:], output_top["h_no_fpb_corr"], label= 'fine_track top + fpb_corr')
      ax0.scatter(output_bot['dist_x'][:], output_bot["h_no_fpb_corr"], label = 'fine_track bot + fpb_corr')

    if (ATL05_ponds_good[i_bif] > 0):
      ax0.hlines(ATL05_ponds_top[i_bif], ATL05_ponds_left[i_bif], ATL05_ponds_right[i_bif], linestyle = 'dashed', color='black', label='pond top')
      ax0.hlines(ATL05_ponds_top[i_bif]-ATL05_ponds_max_depth[i_bif], ATL05_ponds_left[i_bif], ATL05_ponds_right[i_bif], linestyle = 'dashed', color='black', label='max_depth')

    ax0.legend(bbox_to_anchor=(0.9, 1.0), loc='upper left')

    output_file = beam_directory + '/bif_segment_' + str(i_bif[0]).zfill(4) + '.png'
    fig.savefig(output_file, dpi=fig.dpi)
    plt.close('all')
