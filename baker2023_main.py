#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:45:44 2021

@author: will
"""

'''script to extract quantities for Roberto'''
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import time
from astropy.io import ascii
import seaborn as sns
from scipy.stats import binned_statistic, norm
from scipy import odr
from loess.loess_2d import loess_2d

from contours import cont
from rf_parts import boots_RFR, basic_RF, boots_RFR_updated, permutate, permutation_

from stat_util import outlier_cut as OC
from stat_util import outlier_cut
import matplotlib.patches as mpatches
from pcc_arrow import arrow_error, arro, pcc, arrow_error_new

from matplotlib import rcParams


from numba import njit

from lmfit import Model
from lmfit import Parameters, fit_report, minimize
import corner



class plots_new(object):

    def __init__(self):
        super(plots_new,self).__init__()
        self.data=ascii.read('/Users/will/Downloads/dyn.dat')
        Z=np.array(self.data['met_re'], dtype=float)
        Mstar=np.array(self.data['mstar_sum_re'], dtype=float)
        sfr=np.array(self.data['sfr_sum_re'], dtype=float)
        sfr_cat=np.array(self.data['sfr_cat'], dtype=float)
        Mdyn=np.array(self.data['mdyn'], dtype=float)
        r_e=np.array(self.data['col5'], dtype=float)
        sigma_e=np.array(self.data['sigma_e'], dtype=float)
        #self.age=np.genfromtxt('Table-data.dat', skip_header=10,usecols=5)
        actual_mstar=np.array(self.data['mstar_cat'], dtype=float)
        #self.id=np.genfromtxt('Table-data.dat', skip_header=10,usecols=0)

        idx=np.where((Mstar > 1) & (actual_mstar>1) & (Mdyn>1) & (Mdyn<100) 
            & (Z>0.1) & (sfr>-12) & (sfr<10))[0]
        mstar_sum, mstar_cat=Mstar[idx], actual_mstar[idx]
        Mdyn, r_e, Z=Mdyn[idx], r_e[idx], Z[idx]
        sfr=sfr[idx]
        sfr_cat=sfr_cat[idx]
        sigma_e=sigma_e[idx]

        (self.mstar_sum, self.mstar_cat, self.Mdyn, self.r_e, self.Z, 
            self.sfr, self.sfr_cat, self.sigma_e)=outlier_cut([mstar_sum, mstar_cat, Mdyn, r_e, Z, sfr, sfr_cat, sigma_e], dev=5)

        '''
        idx=np.where((self.Mstar > 1) & (self.actual_mstar>1) & (Mdyn>1) & (Mdyn<100) 
            & (Z>0.1) & (self.sfr>-12) & (self.sfr<10) & (sfr_cat>-10))[0]
        self.mstar_sum, self.mstar_cat=self.Mstar[idx], self.actual_mstar[idx]
        self.Mdyn, self.r_e, self.Z=Mdyn[idx], self.r_e[idx], Z[idx]
        self.sfr=self.sfr[idx]
        self.sfr_cat=sfr_cat[idx]'''


    def hex(self):


        rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 3
        rcParams['errorbar.capsize'] = 3

        rcParams['xtick.top'] = True
        rcParams['ytick.right'] = True
        rcParams['xtick.major.size'] = 7
        rcParams['xtick.minor.size'] = 4
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.major.size'] = 7
        rcParams['ytick.minor.size'] = 4
        rcParams['ytick.direction'] = 'in'

        # text settings
        rcParams['mathtext.rm'] = 'serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14
        #rcParams['text.usetex'] = True
        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 15
        rcParams['axes.ymargin'] = 0.5

        mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat=self.mstar_sum, self.Mdyn, self.Z, self.r_e, self.sfr, self.sfr_cat
        idx=np.where((mstar_sum>8.5) & (Z>8.4) & (np.abs(sfr_cat-sfr)<0.5))[0]
        idx=np.where((mstar_sum>8.5) & (Z>8.4))[0]
        mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat=mstar_sum[idx], Mdyn[idx], Z[idx], r_e[idx], sfr[idx], sfr_cat[idx]

        print(len(mstar_sum))
        Nsize=25

        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))

        pl1=ax.hexbin(mstar_sum, Mdyn, C=Z, cmap='Spectral_r', gridsize=15, mincnt=5, reduce_C_function=np.median)
        cent=pl1.get_offsets()
        val=pl1.get_array()
        ax.tick_params(axis="x", labelsize=14) 
        ax.tick_params(axis="y", labelsize=14)        
        ax.set_xlabel(r'$\mathrm{log(M_{*}[M_\odot])}$', fontsize=Nsize)
        ax.set_ylabel(r'$\mathrm{log(M_{dyn}[M_\odot])}$', fontsize=Nsize)
        cbar1=fig.colorbar(pl1)
        cbar1.set_label(r'12+log(O/H)', fontsize=Nsize)
        cbar1.ax.tick_params(labelsize=16) 
        f, xx, yy=cont(mstar_sum, Mdyn)

        ax.contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        ax.grid()
        #ax.set_title('Centrals', fontsize=Nsize)
        angles=arrow_error_new(cent[:,0], cent[:,1], val)
        ax.text(0.2, 0.90, 'Arrow Angle \n'r'$\theta$='+str((90-angles.angle).round(1))+
             r'$\pm$'+str(angles.angle_error.round(1))+r'$^{\circ}$',  fontsize=20,
                horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
        dx,dy, x1,y1=angles.dim()
        ax.quiver(x1,y1, dx, dy, 1, angles=(angles.angle_rad),  scale=5)
        fig.savefig('Plots/Dyn/mass_dyn_z.pdf')

        grav=Mdyn-r_e

        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))

        pl1=ax.hexbin(mstar_sum, grav, C=Z, cmap='Spectral_r', gridsize=15, mincnt=5, reduce_C_function=np.median)
        cent=pl1.get_offsets()
        val=pl1.get_array()
        ax.tick_params(axis="x", labelsize=14) 
        ax.tick_params(axis="y", labelsize=14)         
        ax.set_xlabel(r'$\mathrm{log(M_{*}[M_\odot])}$', fontsize=Nsize)
        ax.set_ylabel(r'$\mathrm{log( \phi [M_\odot/kpc])}$', fontsize=Nsize)
        cbar1=fig.colorbar(pl1)
        cbar1.set_label(r'12+log(O/H)', fontsize=Nsize)
        cbar1.ax.tick_params(labelsize=16) 
        f, xx, yy=cont(mstar_sum, grav)

        ax.contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        ax.grid()
        #ax.set_title('Centrals', fontsize=Nsize)
        angles=arrow_error_new(cent[:,0], cent[:,1], val)
        ax.text(0.2, 0.90, 'Arrow Angle \n'r'$\theta$='+str((90-angles.angle).round(1))+
             r'$\pm$'+str(angles.angle_error.round(1))+r'$^{\circ}$',  fontsize=20,
                horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
        dx,dy, x1,y1=angles.dim()
        ax.quiver(x1,y1, dx, dy, 1, angles=(angles.angle_rad),  scale=5)
        fig.savefig('Plots/Dyn/mass_grav_z.pdf')
    

    def RF_(self):
        (mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat, 
            sigma_e, mstar_cat)=self.mstar_sum, self.Mdyn, self.Z, self.r_e, self.sfr, self.sfr_cat, self.sigma_e, self.mstar_cat
        idx=np.where((mstar_sum>8.5) & (Z>8.4) & (np.abs(sfr_cat-sfr)<0.5))[0]
        dx=np.where((self.mstar_sum>8.5) & (self.Z>8.4))[0]
        (mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat,
            sigma_e, mstar_cat)=mstar_sum[idx], Mdyn[idx], Z[idx], r_e[idx], sfr[idx], sfr_cat[idx], sigma_e[idx], mstar_cat[idx]
        grav=Mdyn-r_e
        #grav=mstar_sum-r_e
        #grav=mstar_cat-r_e
        
        print(len(mstar_sum))


        ran=np.random.uniform(0,1.,len(Z))
        data=np.array([Z, mstar_sum, sfr, Mdyn, grav, ran, sigma_e, r_e])
        performance, mse_tr, mse_va, length=basic_RF(data, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
        performance, errors=boots_RFR_updated(data,  n_est=200, min_samp_leaf=15, max_dep=70, n_times=100, wide=False)

        varnames = np.array([
        r'$M_*$', 
        r'SFR',  r'$M_{Dyn}$',  r'$\phi_g$', 'random', '$\sigma_e$', r'$r_e$'
        ])
        #r'$d_{cent}$'

        idx = np.argsort(performance)[::-1]

        fig, ax = plt.subplots(figsize=(5,5))
        loc = np.arange(length.shape[1])

        ax.bar(loc, performance[idx], yerr=errors[:,idx], width=0.3, 
        edgecolor='#0F2080', facecolor='w',
        hatch=2*'\\', lw=2, label='\n MSE train,test = [{}, {}]'.format(mse_tr.round(3),mse_va.round(3)))
        ax.set_xticks(loc)
        ax.tick_params(which='both', axis='x', bottom=False, top=False)
        ax.set_ylim(0,1)
        ax.set_xticklabels(varnames[idx], fontsize=12)
        ax.set_title(r'Parameter importance in determining 12+log(O/H)'+'\n  ')
        ax.set_ylabel('Relative importance')
        ax.legend()
        fig.savefig('Plots/Dyn/Z_total_performance.pdf')


    def RF_all(self):
        rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 3
        rcParams['errorbar.capsize'] = 3

        rcParams['xtick.top'] = True
        rcParams['ytick.right'] = True
        rcParams['xtick.major.size'] = 7
        rcParams['xtick.minor.size'] = 4
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.major.size'] = 7
        rcParams['ytick.minor.size'] = 4
        rcParams['ytick.direction'] = 'in'

        # text settings
        rcParams['mathtext.rm'] = 'serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14
        #rcParams['text.usetex'] = True
        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 15
        rcParams['axes.ymargin'] = 0.5

        (mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat, 
            sigma_e, mstar_cat)=self.mstar_sum, self.Mdyn, self.Z, self.r_e, self.sfr, self.sfr_cat, self.sigma_e, self.mstar_cat
        idx=np.where((mstar_sum>8.5) & (Z>8.4) & (np.abs(sfr_cat-sfr)<0.5))[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4))[0]
        (mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat,
            sigma_e, mstar_cat)=mstar_sum[idx], Mdyn[idx], Z[idx], r_e[idx], sfr[idx], sfr_cat[idx], sigma_e[idx], mstar_cat[idx]
        grav=Mdyn-r_e
        #grav_mass=mstar_sum-r_e
        grav_mass=mstar_cat-r_e
        
        print(len(mstar_sum))


        ran=np.random.uniform(0,1.,len(Z))
        data=np.array([Z, mstar_sum, sfr, Mdyn, grav, ran, sigma_e, r_e, grav_mass])
        performance, mse_tr, mse_va, length=basic_RF(data, n_est=200, min_samp_leaf=15, max_dep=100, param_check=False) 
        performance, errors=boots_RFR_updated(data,  n_est=200, min_samp_leaf=15, max_dep=70, n_times=100, wide=False)

        varnames = np.array([
        r'$M_*$', 
        r'SFR',  r'$M_{dyn}$',  r'$\phi$', 'random', '$\sigma_e$', r'$r_e$', r'$\Phi_{*}$'
        ])
        #r'$d_{cent}$'

        idx = np.argsort(performance)[::-1]

        fig, ax = plt.subplots(figsize=(5,5))
        loc = np.arange(length.shape[1])

        ax.bar(loc, performance[idx], yerr=errors[:,idx], width=0.6, 
        edgecolor='#0F2080', facecolor='w',
        hatch=2*'\\', lw=2, label=' MSE train,test = [{}, {}]'.format(mse_tr.round(3),mse_va.round(3)))
        ax.set_xticks(loc)
        ax.tick_params(which='both', axis='x', bottom=False, top=False)
        ax.set_ylim(0,1)
        ax.set_xticklabels(varnames[idx], fontsize=10)
        ax.text(0.6, 0.7, "Parameter importance \n in determining \n 12+log(O/H)", fontsize=16,
                horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
        #ax.set_title(r'Parameter importance in determining 12+log(O/H)'+'\n  ', fontsize=13)
        ax.set_ylabel('Relative importance')
        ax.legend(fontsize=11)
        fig.savefig('Plots/Dyn/Z_total_performance_all.pdf', dpi=300, bbox_inches='tight')




    def permutation(self):
        (mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat, 
            sigma_e)=self.mstar_sum, self.Mdyn, self.Z, self.r_e, self.sfr, self.sfr_cat, self.sigma_e
        idx=np.where((mstar_sum>8.5) & (Z>8.4) & (np.abs(sfr_cat-sfr)<0.5))[0]
        (mstar_sum, Mdyn, Z, r_e, sfr, sfr_cat,
            sigma_e)=mstar_sum[idx], Mdyn[idx], Z[idx], r_e[idx], sfr[idx], sfr_cat[idx], sigma_e[idx]
        #grav=Mdyn-r_e
        grav=mstar_sum-r_e
        
        print(len(mstar_sum))


        ran=np.random.uniform(0,1.,len(Z))
        data=np.array([Z, mstar_sum, sfr, Mdyn, grav, ran, sigma_e])

        
        performance, mse_tr, mse_va, length, perm_tr, perm_va=permutate(data, n_est=200, min_samp_leaf=15, t_size=0.5,
                                                                        n_cores=4, max_dep=100, param_check=False) 
 
         
        varnames = np.array([
        r'$M_*$', 
        r'SFR',  r'$M_{Dyn}$',  r'$\phi_g$', 'random', '$\sigma_e$', 
        ])
         
        #bars
        p=permutation_(data, n_est=200, min_samp_leaf=20, t_size=0.2,
                            n_cores=8, max_dep=100, param_check=False)


        loc = np.arange(length.shape[1])
        #idx = np.argsort(p.idx)[::-1]
        fig, ax = plt.subplots(ncols=1, figsize=(7,7))
        ax.bar(loc, p.result_tr[p.idx], yerr=p.error_tr[p.idx],width=0.3, 
               edgecolor='#0F2080', facecolor='w',
               hatch=2*'\\', lw=2, label='Training set')
        ax.set_title("Permutation Importances for determining Z in Satellites")
        ax.set_xticks(loc+0.15)
        ax.set_xticklabels(varnames[p.idx], fontsize=14)

        ax.bar(loc+0.3, p.result_va[p.idx], yerr=p.error_va[p.idx], width=0.3, 
               edgecolor='pink', facecolor='w',
               hatch=2*'\\', lw=2, label='Test set')
        ax.grid()
        ax.legend()
        ax.set_ylabel('Relative importance')
        fig.tight_layout()
        fig.savefig('Plots/Dyn/permutation_importance.pdf')



    def scat(self):
        Nsize=25
        one=onetoline(self.sfr)

        #idx=np.where(np.abs(sfr_cat-sfr)<0.5)[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4) & (np.abs(self.sfr_cat-self.sfr)<0.5))[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4))[0]
        mstar, Mdyn, Z, sfr_cat, sfr=self.mstar_sum[idx], self.Mdyn[idx], self.Z[idx], self.sfr_cat[idx], self.sfr[idx]
        r_e=self.r_e[idx]
        grav=mstar-r_e
        print(len(mstar))

        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))

        ax.tick_params(axis="x", labelsize=14) 
        ax.tick_params(axis="y", labelsize=14) 

        pl1=ax.scatter(mstar, Z, c=sfr, cmap='Spectral', s=20)        
        ax.set_xlabel(r'$\mathrm{log(M_{*}[M_\odot/yr])}$', fontsize=Nsize)
        ax.set_ylabel(r'12+log(O/H)', fontsize=Nsize)
        cbar1=fig.colorbar(pl1)
        cbar1.set_label(r'$\mathrm{ log(SFR [M_\odot])} $', fontsize=Nsize)
        cbar1.ax.tick_params(labelsize=16) 
        f, xx, yy=cont(mstar, Z)

        bin_means, bin_edges, binnumber=binned_statistic(mstar, 
            [mstar, Z], statistic='median', bins=7)
        ax.scatter(bin_means[0], bin_means[1], color='black', marker='s')

        #a, b, c=fitter_poly(bin_means[0], bin_means[1])
        #x1=np.linspace(min(mstar), max(mstar), len(mstar))
        #zs = c + b*x1 + a*x1**2
        #ax.plot(x1,zs)

        ax.contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        #ax.grid()
        fig.savefig('Plots/Dyn/MZR.pdf')


        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))

        ax.tick_params(axis="x", labelsize=14) 
        ax.tick_params(axis="y", labelsize=14) 

        pl1=ax.scatter(Mdyn, Z, c=sfr, cmap='Spectral', s=20)        
        ax.set_xlabel(r'$\mathrm{log(M_{Dyn}[M_\odot])}$', fontsize=Nsize)
        ax.set_ylabel(r'12+log(O/H)', fontsize=Nsize)
        cbar1=fig.colorbar(pl1)
        cbar1.set_label(r'$\mathrm{ log(SFR [M_\odot/yr]) }$', fontsize=Nsize)
        cbar1.ax.tick_params(labelsize=16) 
        f, xx, yy=cont(Mdyn, Z)
        bin_means, bin_edges, binnumber=binned_statistic(Mdyn, 
            [Mdyn, Z], statistic='median', bins=7)
        ax.scatter(bin_means[0], bin_means[1], color='black', marker='s')

        ax.contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        #ax.grid()
        fig.savefig('Plots/Dyn/DZR.pdf')



        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(9,7))

        ax.tick_params(axis="x", labelsize=14) 
        ax.tick_params(axis="y", labelsize=14) 

        pl1=ax.scatter(grav, Z, c=sfr, cmap='Spectral', s=20)        
        ax.set_xlabel(r'$\mathrm{\Phi=log(M_{*}/R_e)}$', fontsize=Nsize)
        ax.set_ylabel(r'12+log(O/H)', fontsize=Nsize)
        cbar1=fig.colorbar(pl1)
        cbar1.set_label(r'$\mathrm{ log(SFR [M_\odot/yr]) }$', fontsize=Nsize)
        cbar1.ax.tick_params(labelsize=16) 
        f, xx, yy=cont(grav, Z)
        bin_means, bin_edges, binnumber=binned_statistic(grav, 
            [grav, Z], statistic='median', bins=6)
        ax.scatter(bin_means[0], bin_means[1], color='black', marker='s')

        ax.contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        #ax.grid()
        fig.savefig('Plots/Dyn/PhiZR.pdf')



    def scat_no_color(self):

        rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 3
        rcParams['errorbar.capsize'] = 3

        rcParams['xtick.top'] = True
        rcParams['ytick.right'] = True
        rcParams['xtick.major.size'] = 7
        rcParams['xtick.minor.size'] = 4
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.major.size'] = 7
        rcParams['ytick.minor.size'] = 4
        rcParams['ytick.direction'] = 'in'

        # text settings
        rcParams['mathtext.rm'] = 'serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14
        #rcParams['text.usetex'] = True
        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 15
        rcParams['axes.ymargin'] = 0.5


        Nsize=20
        #one=onetoline(self.sfr)

        #idx=np.where(np.abs(sfr_cat-sfr)<0.5)[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4) & (np.abs(self.sfr_cat-self.sfr)<0.5))[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4))[0]
        mstar, Mdyn, Z, sfr_cat, sfr=self.mstar_sum[idx], self.Mdyn[idx], self.Z[idx], self.sfr_cat[idx], self.sfr[idx]
        r_e=self.r_e[idx]
        grav=Mdyn-r_e
        print(len(mstar))

        fig, ax=plt.subplots(nrows=1, ncols=3, figsize=(18,5.2))

        #ax.tick_params(axis="x", labelsize=14) 
        #ax.tick_params(axis="y", labelsize=14)
       

        color='violet' 
        alph=0.4

        pl1=ax[0].scatter(mstar, Z, color=color, alpha=alph, s=20)  

        ax[0].set_xlabel(r'$\mathrm{log(M_{*}[M_\odot])}$', fontsize=Nsize)
        ax[0].set_ylabel(r'12+log(O/H)', fontsize=Nsize)

        bin_means, bin_edges, binnumber=binned_statistic(mstar, 
            [mstar, Z], statistic='median', bins=7)
        bin_low, bin_edges, binnumber=binned_statistic(mstar, 
            [mstar, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=7)
        bin_high, bin_edges, binnumber=binned_statistic(mstar, 
            [mstar, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=7)
        bin_std, bin_edges, binnumber=binned_statistic(mstar, 
            [mstar, Z], statistic='std', bins=7)
        number_per_bin, bin_edges, binnumber=binned_statistic(mstar, 
            [mstar, Z], statistic='count', bins=7)
        av_std=np.mean(bin_std[1])
        print(av_std)
        print(bin_edges[1]-bin_edges[2])
        print(bin_edges[3]-bin_edges[4])
        av_std_weighted=np.average(bin_std[1], weights=number_per_bin[1])
        errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
        ax[0].errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s', ms=7)
        ax[0].set_title('MZR', fontsize=20)
        ax[0].text(0.75, 0.25, "<$\sigma$> = {}".format(av_std.round(3)), fontsize=17,
                horizontalalignment='center',  verticalalignment='center', transform=ax[0].transAxes)
        ax[0].text(0.75, 0.15, "<$\sigma_W$> = {}".format(av_std_weighted.round(3)), fontsize=16,
                horizontalalignment='center',  verticalalignment='center', transform=ax[0].transAxes)
        sns.kdeplot(x=mstar,y=Z,ax=ax[0],thresh=.05, color='grey', levels=5)

  
        pl1=ax[1].scatter(Mdyn, Z, color=color, alpha=alph, s=20)        
        ax[1].set_xlabel(r'$\mathrm{log(M_{dyn}[M_\odot])}$', fontsize=Nsize)
        bin_means, bin_edges, binnumber=binned_statistic(Mdyn, 
            [Mdyn, Z], statistic='median', bins=7)
        bin_low, bin_edges, binnumber=binned_statistic(Mdyn, 
            [Mdyn, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=7)
        bin_high, bin_edges, binnumber=binned_statistic(Mdyn, 
            [Mdyn, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=7)

        bin_std, bin_edges, binnumber=binned_statistic(Mdyn, 
            [Mdyn, Z], statistic='std', bins=7)

        number_per_bin, bin_edges, binnumber=binned_statistic(Mdyn, 
            [Mdyn, Z], statistic='count', bins=7)
        av_std=np.mean(bin_std[1])
        av_std_weighted=np.average(bin_std[1], weights=number_per_bin[1])
        print(av_std)
        print(av_std_weighted)
        print(bin_edges[1]-bin_edges[2])
        print(bin_edges[3]-bin_edges[4])

        errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
        ax[1].errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s', ms=7)
        ax[1].set_title('DMZR', fontsize=20)
        ax[1].text(0.75, 0.25, "<$\sigma$> = {}".format(av_std.round(3)), fontsize=17,
                horizontalalignment='center',  verticalalignment='center', transform=ax[1].transAxes)
        ax[1].text(0.75, 0.15, "<$\sigma_W$> = {}".format(av_std_weighted.round(3)), fontsize=16,
                horizontalalignment='center',  verticalalignment='center', transform=ax[1].transAxes)
        sns.kdeplot(x=Mdyn, y=Z,ax=ax[1],thresh=.05, color='grey', levels=5)



        pl1=ax[2].scatter(grav, Z, color=color, alpha=alph, s=20)        
        ax[2].set_xlabel(r'$\mathrm{log(\phi)=log(M_{dyn}/R_e)}$', fontsize=Nsize)
        f, xx, yy=cont(grav, Z)
        bin_means, bin_edges, binnumber=binned_statistic(grav, 
            [grav, Z], statistic='median', bins=6)
        bin_low, bin_edges, binnumber=binned_statistic(grav, 
            [grav, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=6)
        bin_high, bin_edges, binnumber=binned_statistic(grav, 
            [grav, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=6)
        bin_std, bin_edges, binnumber=binned_statistic(grav, 
            [grav, Z], statistic='std', bins=6)
        number_per_bin, bin_edges, binnumber=binned_statistic(grav, 
            [grav, Z], statistic='count', bins=6)
        av_std=np.mean(bin_std[1])
        print(bin_edges)
        print(bin_edges[1]-bin_edges[2])
        print(bin_edges[3]-bin_edges[4])
        #print(av_std)
        av_std_weighted=np.average(bin_std[1], weights=number_per_bin[1])

        errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
        ax[2].errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s', ms=7)
        ax[2].set_title(r'$\phi$ZR', fontsize=20)
        ax[2].text(0.75, 0.25, "<$\sigma$> = {}".format(av_std.round(3)), fontsize=17,
                horizontalalignment='center',  verticalalignment='center', transform=ax[2].transAxes)
        ax[2].text(0.75, 0.15, "<$\sigma_W$> = {}".format(av_std_weighted.round(3)), fontsize=16,
                horizontalalignment='center',  verticalalignment='center', transform=ax[2].transAxes)
        sns.kdeplot(x=grav,y=Z,ax=ax[2],thresh=.05, color='grey', levels=5)
        fig.savefig('Plots/Dyn/combined.pdf', dpi=300, bbox_inches='tight')



    def scat_phi_(self):

        rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 3
        rcParams['errorbar.capsize'] = 3

        rcParams['xtick.top'] = True
        rcParams['ytick.right'] = True
        rcParams['xtick.major.size'] = 7
        rcParams['xtick.minor.size'] = 4
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.major.size'] = 7
        rcParams['ytick.minor.size'] = 4
        rcParams['ytick.direction'] = 'in'

        # text settings
        rcParams['mathtext.rm'] = 'serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14
        #rcParams['text.usetex'] = True
        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 15
        rcParams['axes.ymargin'] = 0.5


        Nsize=20
        one=onetoline(self.sfr)

        #idx=np.where(np.abs(sfr_cat-sfr)<0.5)[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4) & (np.abs(self.sfr_cat-self.sfr)<0.5))[0]
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4))[0]
        mstar, Mdyn, Z, sfr_cat, sfr=self.mstar_sum[idx], self.Mdyn[idx], self.Z[idx], self.sfr_cat[idx], self.sfr[idx]
        r_e, sigma_e=self.r_e[idx], self.sigma_e[idx]
        grav=Mdyn-r_e
        print(len(mstar))

        fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(10,5.2))

        #ax.tick_params(axis="x", labelsize=14) 
        #ax.tick_params(axis="y", labelsize=14)
       

        color='violet' 
        alph=0.4

        pl1=ax[0].scatter(mstar-r_e, Z, color=color, alpha=alph, s=20)  

        ax[0].set_xlabel(r'$\mathrm{log(M_{*}[M_\odot]/r_e)}$', fontsize=Nsize)
        ax[0].set_ylabel(r'12+log(O/H)', fontsize=Nsize)
        f, xx, yy=cont(mstar-r_e, Z)

        bin_means, bin_edges, binnumber=binned_statistic(mstar-r_e, 
            [mstar-r_e, Z], statistic='median', bins=7)
        bin_low, bin_edges, binnumber=binned_statistic(mstar-r_e, 
            [mstar-r_e, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=7)
        bin_high, bin_edges, binnumber=binned_statistic(mstar-r_e, 
            [mstar-r_e, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=7)
        bin_std, bin_edges, binnumber=binned_statistic(mstar-r_e, 
            [mstar-r_e, Z], statistic='std', bins=7)
        av_std=np.mean(bin_std[1])
        print(av_std)

        errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
        ax[0].errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s')
        ax[0].set_title('$\Phi ZR$', fontsize=20)
        ax[0].text(0.75, 0.25, "<$\sigma$> = {}".format(av_std.round(3)), fontsize=17,
                horizontalalignment='center',  verticalalignment='center', transform=ax[0].transAxes)
        #ax.scatter(bin_means[0], bin_means[1], color='black', marker='s')
        #ax[0].contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        sns.kdeplot(x=mstar-r_e,y=Z,ax=ax[0],thresh=.05, color='grey', levels=5)
  
        pl1=ax[1].scatter(sigma_e, Z, color=color, alpha=alph, s=20)        
        ax[1].set_xlabel(r'$\mathrm{\sigma_e}$', fontsize=Nsize)
        #ax[1].set_ylabel(r'12+log(O/H)', fontsize=Nsize)
        f, xx, yy=cont(sigma_e, Z)
        bin_means, bin_edges, binnumber=binned_statistic(sigma_e, 
            [sigma_e, Z], statistic='median', bins=7)
        bin_low, bin_edges, binnumber=binned_statistic(sigma_e, 
            [sigma_e, Z], statistic=lambda Z: np.percentile(Z, 16, axis=None), bins=7)
        bin_high, bin_edges, binnumber=binned_statistic(sigma_e, 
            [sigma_e, Z], statistic=lambda Z: np.percentile(Z, 84, axis=None), bins=7)
        bin_std, bin_edges, binnumber=binned_statistic(sigma_e, 
            [sigma_e, Z], statistic='std', bins=7)
        av_std=np.mean(bin_std[1])
        print(av_std)

        errs=np.array([bin_means[1]-bin_low[1], bin_high[1]-bin_means[1]])
        ax[1].errorbar(bin_means[0], bin_means[1], yerr=errs, color='black', marker='s')
        #ax.scatter(bin_means[0], bin_means[1], color='black', marker='s')
        ax[1].set_title('$\sigma ZR$', fontsize=20)
        ax[1].text(0.75, 0.25, "<$\sigma$> = {}".format(av_std.round(3)), fontsize=17,
                horizontalalignment='center',  verticalalignment='center', transform=ax[1].transAxes)
        #ax[1].contour(xx,yy,f, levels=5, colors='grey') #, extent=(min(Mstar), max(Mstar), min(bh), max(bh)))
        sns.kdeplot(x=sigma_e, y=Z,ax=ax[1],thresh=.05, color='grey', levels=5)
    
        fig.savefig('Plots/Dyn/combined_2.pdf')


    def li_capp(self):

        rcParams['lines.linewidth'] = 2
        rcParams['lines.markersize'] = 3
        rcParams['errorbar.capsize'] = 3

        rcParams['xtick.top'] = True
        rcParams['ytick.right'] = True
        rcParams['xtick.major.size'] = 7
        rcParams['xtick.minor.size'] = 4
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.major.size'] = 7
        rcParams['ytick.minor.size'] = 4
        rcParams['ytick.direction'] = 'in'

        # text settings
        rcParams['mathtext.rm'] = 'serif'
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 14
        #rcParams['text.usetex'] = True
        rcParams['axes.titlesize'] = 18
        rcParams['axes.labelsize'] = 15
        rcParams['axes.ymargin'] = 0.5


        Nsize=20
        idx=np.where((self.mstar_sum>8.5) & (self.Z>8.4))[0]
        mstar, Mdyn, Z, sfr_cat, sfr=self.mstar_sum[idx], self.Mdyn[idx], self.Z[idx], self.sfr_cat[idx], self.sfr[idx]
        r_e=self.r_e[idx]
        grav=Mdyn-r_e
        print(len(mstar))

        fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(7,5.5))

        #pl1=ax.scatter(Mdyn, r_e, c=Z, cmap='Spectral_r', s=10)

        zout, wout = loess_2d(Mdyn, r_e, Z)

        #pl1=ax.scatter(Mdyn, r_e, c=zout, cmap='rainbow_r', s=20)
        pl1=ax.scatter(Mdyn, r_e, c=zout, cmap='tab20', s=20)
        #pl1=ax.scatter(Mdyn, r_e, c=zout, cmap='gnuplot', s=20)


        cbar1=fig.colorbar(pl1)
        cbar1.set_label(r'$\mathrm{12+log(O/H)} $', fontsize=Nsize)
        cbar1.ax.tick_params(labelsize=16)         
        #ax.set_xlabel(r'$\mathrm{log(M_{dyn})}$', fontsize=Nsize)
        ax.set_xlabel(r'$\mathrm{log(M_{dyn})}$', fontsize=Nsize)
        ax.set_ylabel(r'$\mathrm{log(R_e)}$', fontsize=Nsize)
        ax.plot(Mdyn,Mdyn-10, color='grey', linestyle='-', linewidth=1)
        ax.plot(Mdyn,Mdyn-10.5, color='grey', linestyle='-', linewidth=1)
        ax.plot(Mdyn,Mdyn-9.5, color='grey', linestyle='-', linewidth=1)
        ax.plot(Mdyn,Mdyn-9, color='grey', linestyle='-', linewidth=1)
        ax.set_ylim(-0.2,1.5)
        
        #f, xx, yy=cont(Mdyn, Z)
        xx,yy=np.meshgrid(Mdyn, r_e)
        # #zz=np.array(np.meshgrid(zout,zout))
        # #zz=np.array([zz,zz])
        # zs=zout.flatten()
        zz, zz=np.meshgrid(zout,zout)

        #zz=np.array(zout)
        #zz.reshape(len(Mdyn),len(Mdyn))
        # print(xx.shape)
        # #print(zz.shape)
        print(zz.shape)
        #zz=zz.reshape(zz.shape[0], zz.shape[0])

        #ax.contour(xx,yy, zz, levels=[8.65,8.70,8.75], colors='grey', alpha=0.5)

        #ax.contour(Mdyn,r_e, Z, levels=5, colors='grey')

        fig.savefig('Plots/Dyn/LOESS.pdf')



def percentiles(y, level=16):                                                                                                                                                                                                                                                                                                                                                                      
   return(np.percentile(y, level))

        






    
if __name__ == "__main__": 
    
    start_time=time.time()
    print('Started successfully')

    p=plots_new()
    p.hex()
    p.RF_()

    p.RF_all()

    #p.scat()
    #p.onetoone()
    #p.permutation()
    #p.mass_R()

    p.scat_no_color()
    #p.li_capp()
    #p.scat_phi_()


    
    
    end_time=time.time()
    print("This took {} seconds!".format(end_time-start_time))
