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
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.join('..', '..', 'python'))
from blobby3d import Blobby3D


def plot_maps(sample, figsize=(9.0, 7.0)):
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    b3d.plot_map(
            ax[0], np.ma.masked_invalid(np.log10(b3d.maps[sample, 0, :, :])),
            title=r'log$_{10}$(H$\alpha$)', colorbar=True, cmap='Greys_r')
    b3d.plot_map(
            ax[1], np.ma.masked_invalid(np.log10(b3d.maps[sample, 1, :, :])),
            title=r'log$_{10}$([NII])', colorbar=True, cmap='Greys_r')
    b3d.plot_map(
            ax[2], b3d.maps[sample, 2, :, :],
            title='$v$', colorbar=True, cmap='RdYlBu_r')
    b3d.plot_map(
            ax[3], b3d.maps[sample, 3, :, :],
            title=r'$\sigma_v$', colorbar=True, cmap='YlOrBr')
    fig.tight_layout()

    return fig, ax


def plot_n2_ha(sample, figsize=(9.0, 7.0),
               min_logflux=-np.inf, only_starforming=True):
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    mask = np.log10(b3d.maps[sample, 0, :, :]) < min_logflux
    mask |= b3d.var.sum(axis=2) <= 0.0

    if only_starforming:
        mask |= b3d.maps[sample, 1, :, :]/b3d.maps[sample, 0, :, :] > 1.0

    mask |= ~np.isfinite(mask)

    b3d.plot_map(
            ax[0], np.ma.masked_where(mask, np.log10(b3d.maps[sample, 0, :, :])),
            title=r'log$_{10}$(H$\alpha$)', colorbar=True, cmap='Greys_r')
    b3d.plot_map(
            ax[1], np.ma.masked_where(mask, np.log10(b3d.maps[sample, 1, :, :])),
            title=r'log$_{10}$([NII])', colorbar=True, cmap='Greys_r')

    n2_ha = b3d.maps[sample, 1, :, :]/b3d.maps[sample, 0, :, :]
    n2_ha = np.ma.masked_where(mask, np.log10(n2_ha))
    b3d.plot_map(
            ax[2], n2_ha,
            title=r'log$_{10}$([NII]/H$\alpha$)',
            colorbar=True, cmap='RdYlBu_r',
            clim=(-1.0, 1.0))
    fig.tight_layout()

    return fig, ax


def plot_cube(sample, path=r'cube.pdf'):
    pdf = mpl.backends.backend_pdf.PdfPages(path)
    fig, ax = plt.subplots()
    for i in list(range(b3d.naxis[0]))[:20]:
        for j in list(range(b3d.naxis[1]))[:20]:
            if b3d.data[i, j, :].sum() > 0.0:
                for l in range(len(ax.lines)):
                    ax.lines.pop()
                ax.plot(b3d.precon_cubes[sample, i, j, :],
                        'k', label='Preconvolved')
                ax.plot(b3d.con_cubes[sample, i, j, :],
                        'r', label='Convolved')
                ax.plot(b3d.data[i, j, :],
                        color='0.5', label='Data')
                fig.legend()
                pdf.savefig(fig)
    plt.close()
    pdf.close()


if __name__ == '__main__':
    b3d = Blobby3D(
            samples_path='sample.txt',
            data_path='data.txt',
            var_path='var.txt',
            metadata_path='metadata.txt',
            nlines=2)

    # choose a sample
    sample = 0

    # plot maps
    plot_maps(sample)
    plot_n2_ha(sample)
#    plot_cube(sample, path=r'/Volumes/ExtHD3/cube.pdf')


    for i in np.random.randint(b3d.nsamples, size=10):
        plot_maps(i)






#    # Plot maps for sample
#    fig, ax = plt.subplots(1, 4)
#    map_names = ['FLUX0', 'FLUX1', 'V', 'VDISP']
#    for i in range(4):
#        ax[i].set_title(map_names[i])
#        ax[i].imshow(
#                b3d.maps[sample, i, :, :],
#                interpolation='nearest', origin='lower')
#    fig.tight_layout()
#
    # We can also plot the integrated flux across the wavelength axis for sample
    # and compare it to the data
#    fig, ax = plt.subplots(1, 3)
#    ax[0].set_title('Preconvolved')
#    ax[0].imshow(
#            b3d.precon_cubes[sample, :, :, :].sum(axis=2),
#            interpolation='nearest', origin='lower')
#
#
#    ax[1].set_title('Convolved')
#    ax[1].imshow(
#            b3d.con_cubes[sample, :, :, :].sum(axis=2),
#            interpolation='nearest', origin='lower')
#
#    ax[2].set_title('Data')
#    ax[2].imshow(
#            b3d.data.sum(axis=2),
#            interpolation='nearest', origin='lower')
#    fig.tight_layout()
#
#    # Marginalised samples of flux for each line for sample
#    b3d.blob_param.loc[sample, ['FLUX0', 'FLUX1']].hist(bins=30)
#
#    # Marginalised samples for the mean and standard deviation of the flux for the
#    # first emission line
#    b3d.global_param[['FLUX0MU', 'FLUX0SD']].hist()
