#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess for the example galaxy.

Note that blobby3d.py is required to be in your path -- I append the relative
path using the sys module in this example. Feel free to solve it in your own
way.

The Blobby3D class is very simple (some would say it's a bad use of classes!).
However, it is useful for organisational purposes of the Blobby3D output. In
this script, I created a Blobby3D object and plotted a handful of sample
attributes.

@author: Mathew Varidel

"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..', '..', 'python'))
from blobby3d import Blobby3D
import b3dplot
import b3dcomp
from moments import SpectralModel


b3d = Blobby3D(
        samples_path='posterior_sample.txt',
        data_path='data.txt',
        var_path='var.txt',
        metadata_path='metadata.txt',
        nlines=2)

# choose a sample
sample = 0

# Plot maps for sample
fig, ax = plt.subplots(1, 4)
map_names = ['FLUX0', 'FLUX1', 'V', 'VDISP']
for i in range(4):
    ax[i].set_title(map_names[i])
    ax[i].imshow(
            b3d.maps[sample, i, :, :],
            interpolation='nearest', origin='lower')
fig.tight_layout()

# We can also plot the integrated flux across the wavelength axis for sample
# and compare it to the data
fig, ax = plt.subplots(1, 3)
ax[0].set_title('Preconvolved')
ax[0].imshow(
        b3d.precon_cubes[sample, :, :, :].sum(axis=2),
        interpolation='nearest', origin='lower')


ax[1].set_title('Convolved')
ax[1].imshow(
        b3d.con_cubes[sample, :, :, :].sum(axis=2),
        interpolation='nearest', origin='lower')

ax[2].set_title('Data')
ax[2].imshow(
        b3d.data.sum(axis=2),
        interpolation='nearest', origin='lower')
fig.tight_layout()

# Marginalised samples of flux for each line for sample
b3d.blob_param.loc[sample, ['FLUX0', 'FLUX1']].hist(bins=30)

# Marginalised samples for the mean and standard deviation of the flux for the
# first emission line
b3d.global_param[['FLUX0MU', 'FLUX0SD']].hist()

# Construct a comparison between Blobb3d and a single component fit
sm = SpectralModel(
        lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
        lsf_fwhm=1.59
        )

wave = np.linspace(
        b3d.r_lim[0] + 0.5*b3d.dr,
        b3d.r_lim[1] - 0.5*b3d.dr,
        b3d.naxis[2])

fit_conv, fit_conv_err = sm.fit_cube(
        wave, b3d.con_cubes[sample, :, :, :])

fit_data, fit_data_err = sm.fit_cube(
        wave, b3d.data)

fig, ax = b3d.setup_comparison_maps()
b3d.add_comparison_maps(b3d.maps[sample, :, :], ax, col=0, log_flux=True)
b3d.add_comparison_maps(fit, ax, col=1, log_flux=True)
b3d.add_comparison_maps(fit_data, ax, col=2, log_flux=True)

residuals = fit_conv - fit_data
residuals[0, :, :] = np.log10(fit[0, :, :]) - np.log10(fit_data[0, :, :])
residuals[1, :, :] = np.log10(fit[0, :, :]) - np.log10(fit_data[0, :, :])
b3d.add_comparison_residuals(residuals, ax, col=4)

b3d.update_comparison_clim(ax[0][:3], ax[0][3], pct=100.0, absolute=False)
b3d.update_comparison_clim(ax[1][:3], ax[1][3], pct=100.0, absolute=False)
b3d.update_comparison_clim(ax[2][:3], ax[2][3], pct=100.0, absolute=True)
b3d.update_comparison_clim(ax[3][:3], ax[3][3], pct=100.0, absolute=False)

b3d.update_comparison_clim(ax[0][4], ax[0][5], pct=100.0, absolute=True)
b3d.update_comparison_clim(ax[1][4], ax[1][5], pct=100.0, absolute=True)
b3d.update_comparison_clim(ax[2][4], ax[2][5], pct=100.0, absolute=True)
b3d.update_comparison_clim(ax[3][4], ax[3][5], pct=100.0, absolute=True)
