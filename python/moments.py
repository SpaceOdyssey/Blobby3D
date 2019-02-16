#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fitting kinematic moments.

@author: mathewvaridel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

from const import PhysicalConstants

# constants
C = PhysicalConstants.C


class SpectralModel:

    def __init__(self, lines, lsf_fwhm, baseline_order=None):
        self.lines = lines
        self.nlines = 1 + len(lines)//2

        self.lsf_fwhm = lsf_fwhm
        self.lsf_sigma = lsf_fwhm/np.sqrt(8.0*np.log(2.0))

        self.baseline_order = baseline_order

        self.nparam = self.nlines + 2
        if self.baseline_order is not None:
            self.nparam += 1 + self.baseline_order

    def calculate(self, wavelength, *param):
        if self.baseline_order is None:
            model = self._gas_model(wavelength, param)
        else:
            model = self._gas_model(
                    wavelength,
                    param[:-self.baseline_order])
            model += self._baseline_model(
                    wavelength,
                    param[self.baseline_order:])

        return model

    def fit_spaxel(self, wavelength, data, var=None, bounds=None):
        if bounds is None:
            # line flux bounds
            bounds = [
                    [1e-9]*self.nlines,
                    [np.inf]*self.nlines
                    ]

            # v, vdisp bounds
            bounds[0] += [-np.inf, 1e-9]
            bounds[1] += [np.inf, np.inf]

            if self.baseline_order is not None:
                bounds[0] += [-np.inf]*(self.baseline_order + 1)
                bounds[1] += [np.inf]*(self.baseline_order + 1)

        if (var is None) & np.any(data != 0.0):
            data_tmp = data[:]
            w_tmp = wavelength[:]
            sigma_tmp = None
        elif (var is None) & np.all(data == 0.0):
            popt = np.zeros(self.nparam)*np.nan
            pcov = np.zeros(self.nparam)*np.nan
            return popt, pcov
        elif np.any(var > 0.0):
            data_valid = var > 0.0
            data_tmp = data[data_valid]
            w_tmp = wavelength[data_valid]
            sigma_tmp = np.sqrt(var[data_valid])
        else:
            popt = np.zeros(self.nparam)*np.nan
            pcov = np.zeros(self.nparam)*np.nan
            return popt, pcov

        guess = self._guess(wavelength, data)

        try:
            popt, pcov = curve_fit(
                    self.calculate,
                    w_tmp,
                    data_tmp,
                    sigma=sigma_tmp,
                    bounds=bounds,
                    p0=guess,
                    )

            pcov = pcov.diagonal()
        except RuntimeError:
            # Occurs when curve_fit fails to converge
            popt = np.zeros(guess.size)*np.nan
            pcov = np.zeros(guess.size)*np.nan

        return popt, pcov

    def fit_cube(self, wavelength, data, var=None, wave_axis=2, **fit_kwargs):
        if wave_axis == 2:
            fit = np.zeros((self.nparam, *data.shape[:2]))*np.nan
            fit_err = np.zeros((self.nparam, *data.shape[:2]))*np.nan
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    fit[:, i, j], fit_err[:, i, j] = self.fit_spaxel(
                            wavelength, data[i, j, :], var,
                            **fit_kwargs)
        elif wave_axis == 0:
            fit = np.zeros((self.nparam, *data.shape[1:]))*np.nan
            fit_err = np.zeros((self.nparam, *data.shape[1:]))*np.nan
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    fit[:, i, j], fit_err[:, i, j] = self.fit_spaxel(
                            wavelength, data[:, i, j], var,
                            **fit_kwargs)
        else:
            raise ValueError('Wave axis needs to be 0 or 2.')

        return fit, fit_err

    def _guess(self, wavelength, data, lambda_win=10.0):
        """Guess parameters using method of moments.

        data : array-like
        lambda_win : floatp
            Window to estimate parameters
        """
        dwave = wavelength[1] - wavelength[0]
        wave_left = wavelength - 0.5*dwave
        wave_right = wavelength + 0.5*dwave

        guess = np.zeros(self.nlines + 2)

        tmp_v = np.zeros(self.nlines)
        tmp_vdisp = np.zeros(self.nlines)
        for i, line in enumerate(self.lines):
            win = (
                (wave_right >= line[0] - lambda_win)
                & (wave_left <= line[0] + lambda_win)
                )
            win_data = data[win]
            win_wave = wavelength[win]

            guess[i] = max(1e-9, win_data.sum())

            weights = win_data - np.min(win_data)
            mean_wave = np.average(win_wave, weights=weights)
            tmp_v[i] = (mean_wave/line[0] - 1.0)*C

            # TODO : Think below is wrong + it's a biased estimator
            tmp_vdisp[i] = np.sqrt(np.average(
                    (win_wave - mean_wave)**2, weights=weights))
            tmp_vdisp[i] = max(
                    1e-9,
                    np.sqrt(tmp_vdisp[i]**2 - self.lsf_sigma**2)
                    )
            tmp_vdisp[i] *= C/line[0]

        guess[-2] = np.average(tmp_v, weights=guess[:-2])
        guess[-1] = np.average(tmp_vdisp, weights=guess[:-2])

        return guess

    def _gas_model(self, wave, gas_param):
        model = np.zeros(len(wave))

        rel_lambda = 1.0 + gas_param[-2]/C
        rel_lambda_sigma = gas_param[-1]/C

        for i, line in enumerate(self.lines):
            # model first line
            line_wave = line[0]
            line_flux = gas_param[i]

            lam = rel_lambda*line_wave
            lam_sigma = rel_lambda_sigma*line_wave

            model += self._gas_line_model(
                    wave, line_flux, lam, lam_sigma)

            nclines = len(line)//2
            for i in range(nclines):
                line_wave = line[1+2*i]
                factor = line[2+2*i]
                lam = rel_lambda*line_wave
                lam_sigma = rel_lambda_sigma*line_wave
                model += self._gas_line_model(
                        wave, factor*line_flux, lam, lam_sigma)

        return model

    def _gas_line_model(self, wave, flux, lam, lam_sigma):
        dwave = wave[1] - wave[0]
        wave_left = wave - 0.5*dwave
        wave_right = wave + 0.5*dwave

        var = lam_sigma**2 + self.lsf_sigma**2

        cdf_left = 0.5*erf((wave_left - lam)/np.sqrt(2.0*var))
        cdf_right = 0.5*erf((wave_right - lam)/np.sqrt(2.0*var))
        return flux*(cdf_right - cdf_left)  # /dx

    def _baseline_model(self, wave, baseline_param):
        pass


if __name__ == '__main__':
    import os

    from blobby3d import Blobby3D

    root = r'/Users/mathewvaridel/Google Drive/Code/Uni/Blobby3D/Examples/106717'
    b3d = Blobby3D(
        samples_path=os.path.join(root, 'posterior_sample.txt'),
        data_path=os.path.join(root, 'data.txt'),
        var_path=os.path.join(root, 'var.txt'),
        metadata_path=os.path.join(root, 'metadata.txt'),
        nlines=2)

    sm = SpectralModel(
            lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
            lsf_fwhm=1.61
            )

    wave = np.linspace(b3d.r_lim[0] + 0.5*b3d.dr,
                       b3d.r_lim[1] - 0.5*b3d.dr,
                       b3d.naxis[2])

    i, j = 15, 15

    guess = sm._guess(wave, b3d.data[i, j, :])
    guess_model = sm._gas_model(wave, guess)

    fit, fit_err = sm.fit_spaxel(wave, b3d.data[i, j, :])
    fit_model = sm.calculate(wave, *fit)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(wave, guess_model)
    plt.plot(wave, fit_model)
    plt.plot(wave, b3d.data[i, j, :])

    fit_cube, fit_cube_err = sm.fit_cube(wave, b3d.data)








#class Moments:
#
#    def __init__(self, lsf_fwhm):
#        """
#        Class to calculate first 3 moments of each spaxel within cube.
#
#        Attributes
#        ----------
#        lsf_fwhm : float
#            FWHM of LSF in wavelength units.
#        lsf_sigma : float
#            1 sigma of LSF in wavelength units assuming Gaussian conversion
#            from FWHM to 1 sigma.
#        """
#        self.lsf_fwhm = lsf_fwhm
#        self.lsf_sigma = lsf_fwhm/np.sqrt(8.0*np.log(2.0))
#
#    def calc_moments(
#            self, cube, w,
#            baseline_order=None, var_cube=None,
#            wave_axis=0, debug=False
#            ):
#        """
#        Calculate 2D maps for the first 3 moments assuming Gaussian emission
#        line function.
#
#        cube : 3D numpy.array
#            3D cube.
#        w : array-like
#            Representing array wavelengths.
#        baseline_order : int, default None
#            Order to represent baseline. Default None assumes the data is
#            baseline subtracted (ie. baseline is constant at y=0.0).
#        var_cube : 3D numpy.array
#            Variance cube.
#        wave_axis : int, default 0
#            Spectral axis.
#        debug : bool, default False
#            Plot 2D maps for testing purposes.
#
#        Returns
#        -------
#        flux : 2D numpy.ma.core.MaskedArray
#        vel : 2D numpy.ma.core.MaskedArray
#        vdisp : 2D numpy.ma.core.MaskedArray
#        baseline : 3D numpy.ma.core.MaskedArray
#        """
#        bounds = [
#                [1e-9, -np.inf, 1e-9],
#                [np.inf, np.inf, np.inf]
#                ]
#        if baseline_order is not None:
#            bounds[0] += [-np.inf]*(baseline_order + 1)
#            bounds[1] += [np.inf]*(baseline_order + 1)
#
#        cube = swap_axes(cube, wave_axis)
#        if var_cube is not None:
#            var_cube = swap_axes(var_cube, wave_axis)
#
#        flux = np.zeros(cube.shape[1:])*np.nan
#        vel = np.zeros(cube.shape[1:])*np.nan
#        vdisp = np.zeros(cube.shape[1:])*np.nan
#        flux_err = np.zeros(cube.shape[1:])*np.nan
#        if baseline_order is not None:
#            baseline = np.zeros(cube.shape)
#
#        x0_guess = HA
#        sigma_guess = 25.0*HA/C
#        baseline_guess = 0.0
#        for i in range(cube.shape[1]):
#            for j in range(cube.shape[2]):
#
#                if var_cube is None:
#                    cube_tmp = cube[:, i, j]
#                    w_tmp = w
#                    sigma_tmp = None
#                elif np.any(var_cube[:, i, j] > 0.0):
#                    sigma_valid = var_cube[:, i, j] > 0.0
#                    cube_tmp = cube[sigma_valid, i, j]
#                    w_tmp = w[sigma_valid]
#                    sigma_tmp = np.sqrt(var_cube[sigma_valid, i, j])
#                else:
#                    continue
#
#                # Initial position
#                amp_guess = max(np.sum(cube_tmp), 1e-9)
#                p0 = [amp_guess, x0_guess, sigma_guess]
#                if baseline_order is not None:
#                    p0 += [baseline_guess]*(baseline_order + 1)
#
#                try:
#                    popt, pcov = curve_fit(
#                            self.spectral_model,
#                            w_tmp,
#                            cube_tmp,
#                            p0=p0,
#                            bounds=bounds,
#                            sigma=sigma_tmp
#                            )
#                    flux[i, j] = popt[0]
#                    vel[i, j] = C*(popt[1]/HA - 1.0)
#                    vdisp[i, j] = popt[2]*C/HA
#                    flux_err[i, j] = np.sqrt(pcov[0, 0])
#
#                    # baseline
#                    if baseline_order is not None:
#                        baseline[:, i, j] = self._baseline(w, popt[3:])
#
#                    if debug:
#                        print('POPT', popt)
#                        plt.figure()
#                        plt.plot(w_tmp, cube_tmp)
#                        plt.plot(w_tmp, self.spectral_model(w_tmp, *popt))
#                        plt.show()
#
#                except RuntimeError:
#                    # RuntimeError occurs on optimisation failure.
#                    pass
#
#        # mask
#        flux = np.ma.masked_where(~np.isfinite(flux), flux)
#        vel = np.ma.masked_where(~np.isfinite(vel), vel)
#        vdisp = np.ma.masked_where(~np.isfinite(vdisp), vdisp)
#
#        if debug:
#            self.plot_moments(flux, vel, vdisp)
#
#        if baseline_order is None:
#            return flux, vel, vdisp, flux_err
#        else:
#            return flux, vel, vdisp, baseline, flux_err
#
#    def plot_moments(
#            self, flux, vel, vdisp,
#            show=True, save_path=None
#            ):
#        """
#        Plot 2D maps for moments.
#
#        Parameters
#        ----------
#        flux : numpy.array
#            2D map for flux.
#        vel : numpy.array
#            2D map for vel.
#        vdisp : numpy.array
#            2D map for vdisp.
#        show : bool, default True
#            Show figure.
#        save_path : str, default None
#            Save figure to provided path. Default None does not save figure.
#        """
#        plt.subplots(1, 3)
#
#        plt.subplot(1, 3, 1)
#        ax = plt.gca()
#        self._plot_moment(ax, flux)
#
#        plt.subplot(1, 3, 2)
#        ax = plt.gca()
#        self._plot_moment(ax, vel)
#
#        plt.subplot(1, 3, 3)
#        ax = plt.gca()
#        self._plot_moment(ax, vdisp)
#
#        plt.tight_layout()
#
#        if save_path is not None:
#            plt.savefig(save_path)
#
#        if show:
#            plt.show()
#        else:
#            plt.close()
#
#    def spectral_model(self, x, *param):
#        """
#        Construct Gaussian function.
#
#        Parameters
#        ----------
#        x : array-like
#            1D array.
#        *param : array-like
#            Parameters passed to represent Gaussian function.
#            param[0] : float
#                Integrated flux across x.
#            param[1] : float
#                Position of the peak.
#            param[2] : float
#                1 sigma.
#            param[3]-param[N] : float
#                Polynomial baseline parameter values. In ascending order.
#
#        Returns
#        -------
#        y : numpy.array
#            Gaussian function + Baseline in 1D.
#        """
#        flux = param[0]
#        x0 = param[1]
#        sigma = param[2]
#
#        dx = x[1] - x[0]
#        var = sigma**2 + self.lsf_sigma**2
#
#        # I don't understand the LZIFU /dx term?
#        cdf_left = 0.5*erf((x - 0.5*dx - x0)/np.sqrt(2.0*var))
#        cdf_right = 0.5*erf((x + 0.5*dx - x0)/np.sqrt(2.0*var))
#        gauss = flux*(cdf_right - cdf_left)  # /dx
#
#        # Baseline contribution
#        if len(param) == 3:
#            return gauss
#        else:
#            base_param = param[3:]
#            baseline = self._baseline(x, base_param)
#            return gauss + baseline
#
#    def _gas_model(self, lines, line_param):
#        """
#        Fit gas emission line(s) model.
#        """
#        pass
#
#    def _baseline(self, x, base_param):
#        """
#        Construct baseline.
#
#        Parameters
#        ----------
#        x : array-like
#        base_param : array-like
#            Polynomial baseline parameter values. In ascending order.
#
#        Returns
#        -------
#        baseline : numpy.array
#            Polynomial representing baseline.
#        """
#        baseline = np.ones(len(x))*base_param[0]
#
#        # Higher order polynomial
#        if len(base_param) > 1:
#            for i, param in enumerate(base_param[1:]):
#                baseline += param*x**(i + 1)
#
#        return baseline
#
#    def _plot_moment(self, ax, moment):
#        """
#        Plot 2D map for a particular moment.
#
#        Parameters
#        ----------
#        ax : matplotlib.axes object
#            Axes to plot moment.
#        moment : 2D numpy.array
#            2D map for a particular moment.
#        """
#        im = ax.imshow(moment, origin='lower')
#        divider = make_axes_locatable(ax)
#        cax = divider.append_axes('right', size='10%', pad=0.03)
#        plt.colorbar(im, cax=cax)
