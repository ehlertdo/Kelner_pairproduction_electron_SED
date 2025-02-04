import numpy as np
from numba import njit
import sys
import astropy.units as u
import astropy.constants as const
# from astropy.table import QTable
# import matplotlib.pyplot as plt
# plt.rcParams.update({'text.usetex': True, 'axes.formatter.use_mathtext': True, "font.size": 14, "lines.linewidth": 1})

MEC2 = (const.m_e * const.c**2).to(u.eV)


@njit
def BH_differential_crosssection(Z, cosT_, k, E_, E2, p_, p2):
    r0 = const.a0 * const.alpha**2  # classical electron radius

    D_ = E_ - p_ * cosT_
    T = np.sqrt(k**2 + p_**2 - 2 * k * p_ * cosT_)
    Y = (2 / p_**2) * np.log((E2 * E_ + p2 * p_ + 1) / k)
    y2 = 1 / p2 * np.log((E2 + p2) / (E2 - p2))
    d2T = np.log((T + p2) / (T - p2))

    prefactor = const.alpha * Z**2 * r0**2 * p_ * p2 / (2 * k**3) / MEC2

    T1 = -4 * (1 - cosT_**2) * (2 * E_**2 + 1) / (p_**2 * D_**4)
    T2 = (5 * E_**2 - 2 * E2 * E_ + 3) / (p_**2 * D_**2)
    T3 = (p_**2 - k**2) / (T**2 * D_**2)
    T4 = 2 * E2 / (p_**2 * D_)
    T5_1 = 2 * E_ * (1 - cosT_**2) * (3 * k + p_**2 * E2) / D_**4
    T5_2 = (2 * E_**2 * (E_**2 + E2**2) - 7 * E_**2 - 3 * E2 * E_ - E2**2 + 1) / D_**2
    T5_3 = k * (E_**2 - E_ * E2 - 1) / D_
    T5 = Y / (p_ * p2) * (T5_1 + T5_2 + T5_3)
    T6_1 = 2 / D_**2
    T6_2 = 3 * k / D_
    T6_3 = k * (p_**2 - k**2) / (T**2 * D_)
    T6 = - d2T / (p2 * T) * (T6_1 - T6_2 - T6_3)
    T7 = - 2 * y2 / D_

    return prefactor * (T1 + T2 + T3 + T4 + T5 + T6 + T7)


@njit
def outer_integral(Z, lf_nuc, Ee, eps_arr, eps_max, deps, f_pho):
    # ==================== OUTER INTEGRAL ====================
    eps_min = (lf_nuc + Ee)**2 / (4 * lf_nuc**2 * Ee)
    eps_mask = (eps_arr >= eps_min) * (eps_arr <= eps_max)

    outer = 0  # [1 / eV / s]
    # INTEGRATION LOWER LIMITS -----
    # middle
    k_min = (lf_nuc + Ee)**2 / (2 * lf_nuc * Ee)
    # inner
    Emin_ = (lf_nuc**2 + Ee**2) / (2 * lf_nuc * Ee)

    for n, eps in enumerate(eps_arr[eps_mask]):
        # sys.stdout.write('\r\touter: %.2f | %i' % (n / len(eps_arr[eps_mask]), n))
        # sys.stdout.flush()
        # integral 1: over all eps (LAB frame photon energies)

        middle = middle_integral(Z, lf_nuc, Ee, eps, k_min, Emin_)
        # units: middle [m^3 * eV / s] * (deps / eps^2 / MEC2) [1 / eV] * f_pho [1 / m^3 / eV] -> [1 / eV / s]
        outer += middle * deps[eps_mask][n] / eps**2 / MEC2 * f_pho[eps_mask][n]

    return outer


@njit
def middle_integral(Z, lf_nuc, Ee, eps, k_min, Emin_):
    # ==================== MIDDLE INTEGRAL ====================
    # integral 2: over all scattering angles (nucleus - photon); (-> all k)

    k_max   = 2 * lf_nuc * eps
    k_arr   = np.logspace(np.log10(k_min), np.log10(k_max), 20 + 1)
    dk      = np.ediff1d(k_arr)
    k_arr   = (k_arr[1:] + k_arr[:-1]) / 2
    middle  = 0  # [m^3 * eV / s]

    for i, k in enumerate(k_arr):
        inner = inner_integral(Z, lf_nuc, Ee, k, Emin_)  # [m^3 / s / eV]
        # units: inner [m^3 / s / eV] * (dk * k * MEC2^2) [eV^2] -> [m^3 * eV / s]
        middle += inner * (dk[i] * MEC2) * (k * MEC2)

    return middle


@njit
def inner_integral(Z, lf_nuc, Ee, k, Emin_):
    # ==================== INNER INTEGRAL ====================
    # integral 3: over all electron energies (in nucleus rest frame - NRF)

    # electron
    Emax_   = k - 1
    E_      = np.logspace(np.log10(Emin_), np.log10(Emax_), 20)# * MEC2
    dE_     = np.ediff1d(E_)
    E_      = (E_[1:] + E_[:-1]) / 2
    p_      = np.sqrt(E_**2 - 1)  # momentum in NRF

    cosT_ = (E_ - Ee / lf_nuc) / p_  # angle between momentum of incident photon and produced electron in NRF
    
    # positron
    E2 = k - E_# - (2 * const.m_e * const.c**2)
    p2 = np.sqrt(E2**2 - 1)  # momentum in NRF

    W = BH_differential_crosssection(Z, cosT_, k, E_, E2, p_, p2)  # [m^2 / eV]

    # units: W [m^2 / eV] / (1 / c) [m / s] -> inner [m^3 / s / eV]
    inner = np.nan_to_num((dE_ / (p_ / const.c) * W)[(E_ >= Emin_) * (E_ <= Emax_)], nan=0).sum()
    # print('inner', inner)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.plot(E_, W.to(u.barn / u.eV))

    # ax.axvline(Emin_, color='grey', linestyle='dashed')
    # ax.axvline(Emax_, color='grey', linestyle='dotted')
    # ax.set_xlim(Emin_ / 2, Emax_ * 2)
    # ax.set_xlabel(r'$\displaystyle E / m_e\,c^2$')
    # ax.set_ylabel(r'$\displaystyle\frac{d^2\,\sigma}{d\,E\_\,d(\cos\theta)}~\mathrm{[barn]}$')
    # # ax.loglog()
    # ax.set_xscale('log')
    # plt.show()
    # # plt.savefig('BH_diff_cross.pdf', bbox_inches='tight')

    return inner


def main():
    # PHOTON FIELD ===========================================
    inpath = 'fields/dnde_CMB.csv'
    data_pho = np.loadtxt(inpath, skiprows=3, delimiter=',').T
    # --------------------------------------------------------
    field = inpath.split('_')[-1].split('.')[0]
    f_pho = (data_pho[1][1:] + data_pho[1][:-1]) / 2 / u.m**3 / u.eV
    eps_arr = (data_pho[0] * u.Hz * const.h).to(u.eV) / MEC2
    deps = np.ediff1d(eps_arr)
    eps_arr = (eps_arr[1:] + eps_arr[:-1]) / 2
    eps_max = eps_arr[f_pho > 1e-300 / u.m**3 / u.eV].max()
    # ========================================================

    # NUCLEUS ----------------------------------
    A, Z = 1, 1
    N_Ep = 70
    E_nuc_arr = np.logspace(15, 22, N_Ep) * u.eV
    # ------------------------------------------

    N_Ee = 170
    Ee_arr = np.logspace(7.0028719, 23.9028719, N_Ee) * u.eV / MEC2

    # array for SED output (same format as required for CRPropa3-Data)
    data_write = np.zeros((3, N_Ep * N_Ee))
    data_write[0] = np.repeat(E_nuc_arr, repeats=N_Ee)
    data_write[1] = np.tile(Ee_arr * MEC2, reps=N_Ep)

    for ni, E_nuc in enumerate(E_nuc_arr):
        print('\n Ep: %.2f | %.1E eV' % ((ni + 1) / N_Ep, E_nuc.value))
        # ------------------
        lf_nuc = E_nuc  / (const.m_p * const.c**2 * A).to(u.eV) - 1
        beta_nuc = np.sqrt(1 - 1 / lf_nuc**2)
        # ------------------

        dNdEe = np.zeros(len(Ee_arr)) / u.eV / u.s

        for m, Ee in enumerate(Ee_arr):
            sys.stdout.write('\r\tEe: %.2f | %.1E eV' % ((m + 1) / len(Ee_arr), Ee * MEC2.value))
            sys.stdout.flush()

            outer = outer_integral(Z, lf_nuc, Ee, eps_arr, eps_max, deps, f_pho)  # [1 / eV / s]
            dNdEe[m] = (1 / (2 * lf_nuc**3) * outer) / u.eV / u.s  # [1 / eV / s]

        data_write[2][ni * N_Ee:(ni + 1) * N_Ee] = dNdEe

        # dNdEe_qt = QTable([Ee_arr * MEC2, dNdEe], names=['Ee', 'dNdEe'])
        # dNdEe_qt.write('electron_SED_%s.fits' % field, format='fits', overwrite=True)

    np.savetxt('electron_SED_%s.txt' % field, data_write.T)


if __name__ == '__main__':
    main()
