"""Post analysis for Blobby3D output.

To be used after running Blobby3D to organise the Blobby3D output.


Attributes
----------
Classes: Blobby3D

@author: mathewvaridel

"""

from collections import OrderedDict
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf

import b3dplot
import b3dcomp


class Blobby3D:

    def __init__(
            self, samples_path, data_path, var_path, metadata_path,
            save_maps=True, save_precon=True, save_con=True,
            nlines=1, nsigmad=2):
        self.posterior_path = samples_path
        self.data_path = data_path
        self.var_path = var_path
        self.metadata_path = metadata_path
        self.nlines = nlines
        self.nsigmad = nsigmad

        # import metadata
        metadata = np.loadtxt(metadata_path)
        self.naxis = metadata[:3].astype(int)
        self.sz = self.naxis.prod()
        self.x_lim = metadata[3:5]
        self.y_lim = metadata[5:7]
        self.r_lim = metadata[7:9]
        self.dx = float(np.diff(self.x_lim)/self.naxis[1])
        self.dy = float(np.diff(self.y_lim)/self.naxis[0])
        self.dr = float(np.diff(self.r_lim)/self.naxis[2])

        # import data
        self.data = np.loadtxt(data_path).reshape(self.naxis)
        self.var = np.loadtxt(var_path).reshape(self.naxis)

        # posterior samples
        samples = np.atleast_2d(np.loadtxt(samples_path))
        self.nsamples = samples.shape[0]

        if save_maps:
            self.maps = np.zeros((
                    self.nsamples,
                    self.nlines+2,
                    *self.naxis[:2]))
            map_shp = self.naxis[:2].prod()
            for s in range(self.nsamples):
                # Flux
                for l in range(nlines):
                    self.maps[s, l, :, :] = samples[
                            s, l*map_shp:(1+l)*map_shp].reshape(self.naxis[:2])

                # LoS velocity
                self.maps[s, self.nlines, :, :] = samples[
                        s, self.nlines*map_shp:(1+self.nlines)*map_shp
                        ].reshape(self.naxis[:2])

                # LoS vdisp
                self.maps[s, 1+self.nlines, :, :] = samples[
                        s, (1+self.nlines)*map_shp:(2+self.nlines)*map_shp
                        ].reshape(self.naxis[:2])

        if save_con:
            self.precon_cubes = np.zeros((self.nsamples, *self.naxis))
            st = save_maps*(2+self.nlines)*map_shp
            for s in range(self.nsamples):
                self.precon_cubes[s, :, :, :] = samples[
                        s, st:st+self.sz].reshape(self.naxis)

        if save_precon:
            self.con_cubes = np.zeros((self.nsamples, *self.naxis))
            st = save_maps*(2+self.nlines)*map_shp + save_precon*self.sz
            for s in range(self.nsamples):
                self.con_cubes[s, :, :, :] = samples[
                        s, st:st+self.sz].reshape(self.naxis)

        # blob parameters
        st = save_maps*(2+self.nlines)*map_shp
        st += self.sz*(save_precon + save_precon)
        self.max_blobs = int(samples[0, st+1])

        global_names = ['WD', 'RADMIN', 'RADMAX', 'QMIN']
        for i in range(self.nlines):
            global_names += ['FLUX%iMU' % (i), 'FLUX%iSD' % (i)]
        global_names += [
                'NUMBLOBS',
                'XC', 'YC',
                'DISKFLUX', 'DISKMU',
                'VSYS', 'VMAX', 'VSLOPE', 'VGAMMA', 'VBETA'
                ]
        global_names += ['VDISP%i' % (i) for i in range(self.nsigmad)]
        global_names += ['INC', 'PA', 'SIGMA0', 'SIGMA1']
        global_param = np.concatenate((
                samples[:, st+2:st+7+2*self.nlines],
                samples[:, -13-self.nsigmad:]), axis=1)
        self.global_param = pd.DataFrame(global_param, columns=global_names)

        if self.max_blobs > 0:
            blob_names = ['RC', 'THETAC', 'W', 'Q', 'PHI']
            blob_names += ['FLUX%i' % (i) for i in range(self.nlines)]
            n_bparam = len(blob_names)
            self.blob_param = np.zeros(
                    (self.nsamples*self.max_blobs, n_bparam))
            st_bparam = st+7+2*self.nlines
            end_bparam = -13-self.nsigmad
            for s in range(self.nsamples):
                row_st = self.max_blobs*s
                row_end = self.max_blobs*(s + 1)
                sblob_param = samples[s, st_bparam:end_bparam]
                sblob_param = sblob_param.reshape(
                        (self.max_blobs, n_bparam), order='F')
                self.blob_param[row_st:row_end, :] = sblob_param

            self.blob_param = pd.DataFrame(self.blob_param, columns=blob_names)

            self.blob_param['SAMPLE'] = np.repeat(
                    np.arange(self.nsamples), self.max_blobs)
            self.blob_param['BLOB'] = np.tile(
                    np.arange(self.max_blobs), self.nsamples)
            self.blob_param.set_index(['SAMPLE', 'BLOB'], inplace=True)
            self.blob_param = self.blob_param[self.blob_param['RC'] > 0.0]

    def plot_global_marginalised(self, save_file=None):
        if isinstance(save_file, str):
            pdf_file = mpl.backends.backend_pdf.PdfPages(save_file)

        for key in self.global_param.keys():
            fig = plt.figure()
            ax = plt.gca()
            ax.hist(self.global_param[key])
            ax.set_title(r'%s' % (str(key)))
            if save_file:
                pdf_file.savefig(fig)
                plt.close()

        if isinstance(save_file, str):
            pdf_file.close()

    def plot_map(self, ax, map_2d, **kwargs):
        """Plot individual map to a given axes object."""
        b3dplot.plot_map(ax, map_2d, **kwargs)

    def setup_comparison_maps(
            self, figsize=(10.0, 10.0), log_flux=True, **kwargs):
        """Setup comparsion maps for a given sample."""
        fig, ax = b3dcomp.setup_comparison_maps(
                comp_shape=(2 + self.nlines, 3),
                map_shape=self.naxis[:2],
                figsize=figsize,
                **kwargs)

        return fig, ax

    def add_comparison_maps(self, maps, ax, col, log_flux=False):
        """Add a column worth of maps to the comparison maps."""
        for line in range(self.nlines):
            if log_flux:
                flux_map = np.log10(maps[line, :, :])
            else:
                flux_map = maps[line, :, :]
            self.plot_map(ax[line][col], flux_map, cmap=b3dplot.cmap.flux)
        self.plot_map(
                ax[self.nlines][col], maps[self.nlines, :, :],
                cmap=b3dplot.cmap.v)
        self.plot_map(
                ax[self.nlines+1][col], maps[self.nlines+1, :, :],
                cmap=b3dplot.cmap.vdisp)

    def add_comparison_residuals(self, maps, ax, col, log_flux=False):
        """Add a column worth of maps to the comparison maps."""
        for line in range(self.nlines):
            if log_flux:
                flux_map = np.log10(maps[line, :, :])
            else:
                flux_map = maps[line, :, :]
            self.plot_map(ax[line][col], flux_map, cmap=b3dplot.cmap.residuals)
        self.plot_map(
                ax[self.nlines][col], maps[self.nlines, :, :],
                cmap=b3dplot.cmap.residuals)
        self.plot_map(
                ax[self.nlines+1][col], maps[self.nlines+1, :, :],
                cmap=b3dplot.cmap.residuals)

    def update_comparison_clim(
            self, ax, cax,
            pct=100.0, absolute=False, **cb_kwargs):
        """Update each row colour scale for comparison plots."""
        if isinstance(ax, (list, tuple)):
            data = np.array([a.images[0]._A.data for a in ax])
            clim = b3dcomp.map_limits(data, pct, absolute)
            for a in ax:
                a.images[0].set_clim(clim)
            b3dcomp.colorbar(ax[-1].images[0], cax, clim, **cb_kwargs)
        else:
            data = ax.images[0]._A.data
            clim = b3dcomp.map_limits(data, pct, absolute)
            ax.images[0].set_clim(clim)
            b3dcomp.colorbar(ax.images[0], cax, clim, **cb_kwargs)

    def update_comparison_colorbar(self, ax, **cb_kwargs):
        """
        Update colorbars in accordance with corresponding maps.

        ax : list of matplotlib.axes
            The right-most axis is the colorbar. The preceding maps are the
            maps with the corresponding clim.
        """
        clims = [a.images[0].get_clim() for a in ax[:-1]]
        for clim in clims:
            assert clim == clims[0]

        b3dcomp.colorbar(ax[-1], clims[0], **cb_kwargs)
