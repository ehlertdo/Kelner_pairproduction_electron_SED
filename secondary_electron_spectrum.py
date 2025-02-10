import numpy as np
from numba import njit

# =============================
# = CONSTANTS (from PDG 2024) =
# -----------------------------
# speed of light [m / s]
c = 299792458
# electron mass [eV]
MEC2 = 510998.95
# proton mass [eV]
MPC2 = 938272088.16
# Planck constant [eV / Hz]
h = 4.135667696923859e-15
# fine-structure constant
alpha = 7.2973525693e-3
# classical electron radius [m]
re = 2.8179403262e-15
# =============================


@njit
def BH_differential_crosssection(Z, cosT_, k, E_, E2, p_, p2):
    D_ = E_ - p_ * cosT_
    T = np.sqrt(k**2 + p_**2 - 2 * k * p_ * cosT_)
    Y = (2 / p_**2) * np.log((E2 * E_ + p2 * p_ + 1) / k)
    y2 = 1 / p2 * np.log((E2 + p2) / (E2 - p2))
    d2T = np.log((T + p2) / (T - p2))

    prefactor = alpha * Z**2 * re**2 * p_ * p2 / (2 * k**3) / MEC2

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

    return np.nan_to_num(prefactor * (T1 + T2 + T3 + T4 + T5 + T6 + T7), nan=0.)


@njit
def inner_integrand(E_, Ee, lf_nuc, k, Z):
    p_      = np.sqrt(E_**2 - 1)  # momentum in NRF

    cosT_ = (E_ - Ee / lf_nuc) / p_  # angle between momentum of incident photon and produced electron in NRF
    
    # positron
    E2 = k - E_
    p2 = np.sqrt(E2**2 - 1)  # momentum in NRF

    W = BH_differential_crosssection(Z, cosT_, k, E_, E2, p_, p2)  # [m^2 / eV]

    return W / (p_ / c)


@njit
def inner_midpoint(E_, dE_, Ee, lf_nuc, k, Z):
    return (dE_ * inner_integrand((E_[1:] + E_[:-1]) / 2, Ee, lf_nuc, k, Z)).sum()


@njit
def inner_simpson(E_, dE_, Ee, lf_nuc, k, Z):
    # left
    left = inner_integrand(E_[:-1], Ee, lf_nuc, k, Z)
    # middle
    middle = inner_integrand((E_[1:] + E_[:-1]) / 2, Ee, lf_nuc, k, Z)
    # right
    right = inner_integrand(E_[1:], Ee, lf_nuc, k, Z)

    return (dE_ / 6 * (left + 4 * middle + right)).sum()


@njit
def inner_integral(Z, lf_nuc, Ee, k, Emin_):
    # ==================== INNER INTEGRAL ====================
    # integral 3: over all electron energies (in nucleus rest frame - NRF)
    Emax_   = k - 1
    E_      = np.logspace(np.log10(Emin_), np.log10(Emax_), 20)
    dE_     = np.ediff1d(E_)

    inner = inner_midpoint(E_, dE_, Ee, lf_nuc, k, Z)

    # units: W [m^2 / eV] / (1 / c) [m / s] -> inner [m^3 / s / eV]
    return inner


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
def outer_integral(Z, lf_nuc, Ee, eps_max, eps_arr, deps, f_pho):
    # INTEGRATION LOWER LIMITS -----
    # middle
    k_min = (lf_nuc + Ee)**2 / (2 * lf_nuc * Ee)
    # inner
    Emin_ = (lf_nuc**2 + Ee**2) / (2 * lf_nuc * Ee)

    # ==================== OUTER INTEGRAL ====================
    # integral 1: over all eps (LAB frame photon energies)
    eps_min = (lf_nuc + Ee)**2 / (4 * lf_nuc**2 * Ee)
    eps_mask = (eps_arr >= eps_min) * (eps_arr <= eps_max)
    outer = 0  # [1 / eV / s]

    for n, eps in enumerate(eps_arr[eps_mask]):
        middle = middle_integral(Z, lf_nuc, Ee, eps, k_min, Emin_)
        # units: middle [m^3 * eV / s] * (deps / eps^2 / MEC2) [1 / eV] * f_pho [1 / m^3 / eV] -> [1 / eV / s]
        outer += middle * deps[eps_mask][n] / eps**2 / MEC2 * f_pho[eps_mask][n]

    return outer


def loop_electron_energies(A, Z, Ee_arr, lf_nuc, eps_max, eps_arr, deps, f_pho):
    dNdEe = np.zeros(len(Ee_arr))  # [1 / eV / s]

    for m, Ee in enumerate(Ee_arr):
        outer = outer_integral(Z, lf_nuc, Ee, eps_max, eps_arr, deps, f_pho)  # [1 / eV / s]
        dNdEe[m] = (1 / (2 * lf_nuc**3) * outer)

    return dNdEe  # [1 / eV / s]


def main():
    # PHOTON FIELD ===========================================
    # (should be in two-column format)
    # (col 1: nu [Hz], col 2: dN/dV/dE [1 / m^3 / eV])
    inpath = 'fields/'
    field = 'dnde_CMB.csv'
    data_pho = np.loadtxt(inpath + field, skiprows=3, delimiter=',').T
    # --------------------------------------------------------
    field = field.split('_')[-1].split('.')[0]
    f_pho = (data_pho[1][1:] + data_pho[1][:-1]) / 2  # [1 / m^3 / eV]
    eps_arr = (data_pho[0] * h) / MEC2
    deps = np.ediff1d(eps_arr)
    eps_arr = (eps_arr[1:] + eps_arr[:-1]) / 2
    eps_max = eps_arr[f_pho > 1e-300].max()
    # ========================================================

    # NUCLEUS ----------------------------------
    A, Z = 1, 1
    N_Ep = 70
    E_nuc_arr = np.logspace(15, 22, N_Ep)
    # ------------------------------------------

    N_Ee = 170
    Ee_arr = np.logspace(7.0028719, 23.9028719, N_Ee) / MEC2

    # array for SED output (same format as required for CRPropa3-data)
    data_write = np.zeros((3, N_Ep * N_Ee))
    data_write[0] = np.repeat(E_nuc_arr, repeats=N_Ee)
    data_write[1] = np.tile(Ee_arr * MEC2, reps=N_Ep)

    for ni, E_nuc in enumerate(E_nuc_arr):
        print('Ep: %.2f | %.1E eV' % ((ni + 1) / N_Ep, E_nuc))
        # nucleon Lorentz factor
        lf_nuc = E_nuc  / (MPC2 * A) - 1

        dNdEe = loop_electron_energies(A, Z, Ee_arr, lf_nuc, eps_max, eps_arr, deps, f_pho)

        data_write[2][ni * N_Ee:(ni + 1) * N_Ee] = dNdEe

        # from astropy.table import QTable
        # import astropy.units as u
        # dNdEe_qt = QTable([Ee_arr * MEC2 * u.eV, dNdEe / u.eV / u.s], names=['Ee', 'dNdEe'])
        # dNdEe_qt.write('electron_SED_%s_%.1EeV.fits' % (field, E_nuc), format='fits', overwrite=True)

    np.savetxt(inpath + 'pair_spectrum_%s.table' % field[0:3], data_write.T)


if __name__ == '__main__':
    main()
