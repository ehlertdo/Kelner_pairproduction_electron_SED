import numpy as np
import ctypes
so_file = './kelner.so'
kelner = ctypes.CDLL(so_file)
c_double_p = ctypes.POINTER(ctypes.c_double)
kelner.inner_integral.restype = ctypes.c_double
kelner.middle_integral.restype = ctypes.c_double
kelner.outer_integral.restype = ctypes.c_double

# =============================
# = CONSTANTS (from PDG 2024) =
# -----------------------------
# electron mass [eV]
MEC2 = 510998.95
# proton mass [eV]
MPC2 = 938272088.16
# Planck constant [eV / Hz]
h = 4.135667696923859e-15
# =============================


def loop_electron_energies(A, Z, Ee_arr, lf_nuc, eps_max, eps_arr, deps, f_pho):
    dNdEe = np.zeros(len(Ee_arr))  # [1 / eV / s]

    for m, Ee in enumerate(Ee_arr):
        dNdEe[m] = kelner.outer_integral(ctypes.c_int(Z), ctypes.c_double(lf_nuc), ctypes.c_double(Ee), ctypes.c_double(eps_max), eps_arr.astype(np.float64).ctypes.data_as(c_double_p), deps.ctypes.data_as(c_double_p), f_pho.ctypes.data_as(c_double_p), ctypes.c_int(len(eps_arr)))

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
        # dNdEe_qt.write('spectra/electron_SED_%s_%.1EeV.fits' % (field, E_nuc), format='fits', overwrite=True)

    np.savetxt('spectra/pair_spectrum_%s.table' % field[0:3], data_write.T)


if __name__ == '__main__':
    main()
