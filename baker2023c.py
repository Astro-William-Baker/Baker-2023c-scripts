#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract quantities for Roberto and run analysis.

Created on Tue May  4 13:45:44 2021
Author: Will Baker
"""

import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import ascii
from scipy.stats import binned_statistic

# local modules
from contours import cont
from rf_parts import basic_RF, boots_RFR_updated
from stat_util import outlier_cut
from pcc_arrow import arrow_error_new


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def set_plot_style():
    """Apply consistent matplotlib style settings."""
    rcParams.update({
        "lines.linewidth": 2,
        "lines.markersize": 3,
        "errorbar.capsize": 3,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 7,
        "xtick.minor.size": 4,
        "xtick.direction": "in",
        "ytick.major.size": 7,
        "ytick.minor.size": 4,
        "ytick.direction": "in",
        "mathtext.rm": "serif",
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "axes.ymargin": 0.5,
    })


def percentiles(y, level=16):
    """Return percentile of array y at given level."""
    return np.percentile(y, level)


# ----------------------------------------------------------------------
# Main analysis class
# ----------------------------------------------------------------------

class PlotsNew:
    def __init__(self, filepath="/Users/will/Downloads/dyn.dat"):
        self.data = ascii.read(filepath)

        Z = np.array(self.data["met_re"], float)
        Mstar = np.array(self.data["mstar_sum_re"], float)
        sfr = np.array(self.data["sfr_sum_re"], float)
        sfr_cat = np.array(self.data["sfr_cat"], float)
        Mdyn = np.array(self.data["mdyn"], float)
        r_e = np.array(self.data["col5"], float)
        sigma_e = np.array(self.data["sigma_e"], float)
        actual_mstar = np.array(self.data["mstar_cat"], float)

        idx = np.where(
            (Mstar > 1) & (actual_mstar > 1) &
            (Mdyn > 1) & (Mdyn < 100) &
            (Z > 0.1) & (sfr > -12) & (sfr < 10)
        )[0]

        self.mstar_sum, self.mstar_cat, self.Mdyn, self.r_e, self.Z, \
        self.sfr, self.sfr_cat, self.sigma_e = outlier_cut(
            [Mstar[idx], actual_mstar[idx], Mdyn[idx], r_e[idx],
             Z[idx], sfr[idx], sfr_cat[idx], sigma_e[idx]],
            dev=5
        )

    # ------------------------------------------------------------------
    def hex(self):
        """Hexbin plots of stellar mass vs. dynamical mass and grav parameter."""
        set_plot_style()

        mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat = (
            self.mstar_sum, self.Mdyn, self.Z,
            self.r_e, self.sfr, self.sfr_cat
        )
        idx = np.where((mstar_sum > 8.5) & (Z > 8.4))[0]
        mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat = (
            mstar_sum[idx], Mdyn[idx], Z[idx],
            r_e[idx], sfr[idx], sfr_cat[idx]
        )

        print(f"Sample size: {len(mstar_sum)}")
        Nsize = 25

        # --- Mass–Dyn plot
        fig, ax = plt.subplots(figsize=(9, 7))
        pl1 = ax.hexbin(mstar_sum, Mdyn, C=Z, cmap="Spectral_r",
                        gridsize=15, mincnt=5, reduce_C_function=np.median)
        cbar = fig.colorbar(pl1)
        cbar.set_label(r"12+log(O/H)", fontsize=Nsize)
        cbar.ax.tick_params(labelsize=16)

        f, xx, yy = cont(mstar_sum, Mdyn)
        ax.contour(xx, yy, f, levels=5, colors="grey")

        ax.set_xlabel(r"$\log(M_{*}/M_\odot)$", fontsize=Nsize)
        ax.set_ylabel(r"$\log(M_{dyn}/M_\odot)$", fontsize=Nsize)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid()

        angles = arrow_error_new(pl1.get_offsets()[:, 0],
                                 pl1.get_offsets()[:, 1],
                                 pl1.get_array())
        ax.text(0.2, 0.9,
                rf"Arrow Angle $\theta={90-angles.angle:.1f}\pm{angles.angle_error:.1f}^\circ$",
                fontsize=20, ha="center", va="center", transform=ax.transAxes)
        dx, dy, x1, y1 = angles.dim()
        ax.quiver(x1, y1, dx, dy, scale=5)
        fig.savefig("Plots/Dyn/mass_dyn_z.pdf")

        # --- Mass–Grav plot
        grav = Mdyn - r_e
        fig, ax = plt.subplots(figsize=(9, 7))
        pl2 = ax.hexbin(mstar_sum, grav, C=Z, cmap="Spectral_r",
                        gridsize=15, mincnt=5, reduce_C_function=np.median)
        cbar = fig.colorbar(pl2)
        cbar.set_label(r"12+log(O/H)", fontsize=Nsize)
        cbar.ax.tick_params(labelsize=16)

        f, xx, yy = cont(mstar_sum, grav)
        ax.contour(xx, yy, f, levels=5, colors="grey")

        ax.set_xlabel(r"$\log(M_{*}/M_\odot)$", fontsize=Nsize)
        ax.set_ylabel(r"$\log(\phi [M_\odot/kpc])$", fontsize=Nsize)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid()

        angles = arrow_error_new(pl2.get_offsets()[:, 0],
                                 pl2.get_offsets()[:, 1],
                                 pl2.get_array())
        ax.text(0.2, 0.9,
                rf"Arrow Angle $\theta={90-angles.angle:.1f}\pm{angles.angle_error:.1f}^\circ$",
                fontsize=20, ha="center", va="center", transform=ax.transAxes)
        dx, dy, x1, y1 = angles.dim()
        ax.quiver(x1, y1, dx, dy, scale=5)
        fig.savefig("Plots/Dyn/mass_grav_z.pdf")

    # ------------------------------------------------------------------
    def RF_all(self):
        """Random forest regression with all parameters."""
        set_plot_style()

        mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat, sigma_e, mstar_cat = (
            self.mstar_sum, self.Mdyn, self.Z,
            self.r_e, self.sfr, self.sfr_cat,
            self.sigma_e, self.mstar_cat
        )
        idx = np.where((self.mstar_sum > 8.5) & (self.Z > 8.4))[0]
        mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat, sigma_e, mstar_cat = (
            mstar_sum[idx], Mdyn[idx], Z[idx],
            r_e[idx], sfr[idx], sfr_cat[idx], sigma_e[idx], mstar_cat[idx]
        )

        grav = Mdyn - r_e
        grav_mass = mstar_cat - r_e
        print(f"Sample size: {len(mstar_sum)}")

        ran = np.random.uniform(0, 1., len(Z))
        data = np.array([Z, mstar_sum, sfr, Mdyn, grav,
                         ran, sigma_e, r_e, grav_mass])
        performance, mse_tr, mse_va, length = basic_RF(
            data, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False
        )
        performance, errors = boots_RFR_updated(
            data, n_est=200, min_samp_leaf=15, max_dep=70, n_times=100, wide=False
        )

        varnames = np.array([
            r"$M_*$", "SFR", r"$M_{dyn}$", r"$\phi$",
            "random", r"$\sigma_e$", r"$r_e$", r"$\Phi_{*}$"
        ])

        idx = np.argsort(performance)[::-1]
        fig, ax = plt.subplots(figsize=(5, 5))
        loc = np.arange(length.shape[1])

        ax.bar(loc, performance[idx], yerr=errors[:, idx], width=0.6,
               edgecolor="#0F2080", facecolor="w", hatch="\\\\",
               lw=2, label=f"MSE train,test = [{mse_tr:.3f}, {mse_va:.3f}]")
        ax.set_xticks(loc)
        ax.set_xticklabels(varnames[idx], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Relative importance")
        ax.text(0.6, 0.7, "Parameter importance \nin determining \n12+log(O/H)",
                fontsize=16, ha="center", va="center", transform=ax.transAxes)
        ax.legend(fontsize=11)
        fig.savefig("Plots/Dyn/Z_total_performance_all.pdf", dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    def scat_no_color(self):
        """Scatter plots of Z relations without color coding."""
        set_plot_style()
        Nsize = 20

        idx = np.where((self.mstar_sum > 8.5) & (self.Z > 8.4))[0]
        mstar, Mdyn, Z, sfr_cat, sfr = (
            self.mstar_sum[idx], self.Mdyn[idx], self.Z[idx],
            self.sfr_cat[idx], self.sfr[idx]
        )
        r_e = self.r_e[idx]
        grav = Mdyn - r_e
        print(f"Sample size: {len(mstar)}")

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5.2))
        color, alph = "violet", 0.4

        # --- Stellar mass vs Z
        ax[0].scatter(mstar, Z, color=color, alpha=alph, s=20)
        ax[0].set_xlabel(r"$\log(M_{*}/M_\odot)$", fontsize=Nsize)
        ax[0].set_ylabel(r"12+log(O/H)", fontsize=Nsize)

        # binned statistics
        bin_means, _, _ = binned_statistic(mstar, [mstar, Z], statistic="median", bins=7)
        bin_low, _, _ = binned_statistic(mstar, [mstar, Z],
                                         statistic=lambda Z: np.percentile(Z, 16), bins=7)
        bin_high, _, _ = binned_statistic(mstar, [mstar, Z],
                                          statistic=lambda Z: np.percentile(Z, 84), bins=7)
        bin_std, _, _ = binned_statistic(mstar, [mstar, Z], statistic="std", bins=7)
        number_per_bin, _, _ = binned_statistic(mstar, [mstar, Z], statistic="count", bins=7)

        av_std = np.mean(bin_std[1])
        av_std_weighted = np.average(bin_std[1], weights=number_per_bin[1])
        errs = np.array([bin_means[1] - bin_low[1], bin_high[1] - bin_means[1]])

        ax[0].errorbar(bin_means[0], bin_means[1], yerr=errs,
                       color="black", marker="s", ms=7)
        ax[0].set_title("MZR", fontsize=20)
        ax[0].text(0.75, 0.25, f"<σ> = {av_std:.3f}", fontsize=17,
                   ha="center", va="center", transform=ax[0].transAxes)
        ax[0].text(0.75, 0.15, f"<σW> = {av_std_weighted:.3f}", fontsize=16,
                   ha="center", va="center", transform=ax[0].transAxes)
        sns.kdeplot(x=mstar, y=Z, ax=ax[0], thresh=.05, color="grey", levels=5)

        # (similar cleanup for ax[1] and ax[2] ...)

        fig.savefig("Plots/Dyn/combined.pdf", dpi=300, bbox_inches="tight")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    print("Started successfully")

    p = PlotsNew()
    p.hex()
    p.RF_all()
    p.scat_no_color()

    print(f"This took {time.time()-start_time:.2f} seconds!")

