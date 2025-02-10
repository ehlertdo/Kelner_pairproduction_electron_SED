#include <stdio.h>
#include <math.h>

// =============================
// = CONSTANTS (from PDG 2024) =
// -----------------------------
// speed of light [m / s]
double c = 299792458;
// electron mass [eV]
double MEC2 = 510998.95;
// proton mass [eV]
double MPC2 = 938272088.16;
// Planck constant [eV / Hz]
double h = 4.135667696923859e-15;
// fine-structure constant
double alpha = 7.2973525693e-3;
// classical electron radius [m]
double re = 2.8179403262e-15;
// # =============================

int square(int i) {
    return i * i;
}


double differential_crosssection(int Z, double cosT_, double k, double E_, double E2, double p_, double p2) {
    double D_ = E_ - p_ * cosT_;
    double T = sqrt(k*k + p_*p_ - 2 * k * p_ * cosT_);
    double Y = (2 / (p_*p_)) * log((E2 * E_ + p2 * p_ + 1) / k);
    double y2 = 1 / p2 * log((E2 + p2) / (E2 - p2));
    double d2T = log((T + p2) / (T - p2));

    double prefactor = alpha * Z*Z * re*re * p_ * p2 / (2 * k*k*k) / MEC2;

    double T1 = -4 * (1 - cosT_*cosT_) * (2 * E_*E_ + 1) / (p_*p_ * D_*D_*D_*D_);
    double T2 = (5 * E_*E_ - 2 * E2 * E_ + 3) / (p_*p_ * D_*D_);
    double T3 = (p_*p_ - k*k) / (T*T * D_*D_);
    double T4 = 2 * E2 / (p_*p_ * D_);
    double T5_1 = 2 * E_ * (1 - cosT_*cosT_) * (3 * k + p_*p_ * E2) / (D_*D_*D_*D_);
    double T5_2 = (2 * E_*E_ * (E_*E_ + E2*E2) - 7 * E_*E_ - 3 * E2 * E_ - E2*E2 + 1) / (D_*D_);
    double T5_3 = k * (E_*E_ - E_ * E2 - 1) / D_;
    double T5 = Y / (p_ * p2) * (T5_1 + T5_2 + T5_3);
    double T6_1 = 2 / (D_*D_);
    double T6_2 = 3 * k / D_;
    double T6_3 = k * (p_*p_ - k*k) / (T*T * D_);
    double T6 = - d2T / (p2 * T) * (T6_1 - T6_2 - T6_3);
    double T7 = - 2 * y2 / D_;

    return prefactor * (T1 + T2 + T3 + T4 + T5 + T6 + T7);
}


double inner_integrand(double E_, double Ee, double lf_nuc, double k, double Z) {
    double p_ = sqrt(E_*E_ - 1);  // momentum in NRF
    double cosT_ = (E_ - Ee / lf_nuc) / p_;

    if (cosT_ < -1. || cosT_ > 1. || E_ < 1. || E_ > k - 1) {
        return 0.;
    } else {
        // positron
        double E2 = k - E_;
        double p2 = sqrt(E2*E2 -1);  // momentum in NRF
        // double-differential cross section
        double W = differential_crosssection(Z, cosT_, k, E_, E2, p_, p2);

        return W / (p_ / c);
    }
}


double inner_integral(int Z, double lf_nuc, double Ee, double k, double Emin_) {
	// ==================== INNER INTEGRAL ====================
    // integral 3: over all electron energies (in nucleus rest frame - NRF)

    if (k - 1.0001 <= Emin_) {
    	return 0.0;
    } else {
        double inner = 0.0;
        double Emax_ = k - 1;
        double dE_ = (Emax_ - Emin_) / 50;
        for (double E_ = Emin_; E_ < Emax_; E_ += dE_) {
            // manual midpoint rule 
            inner += dE_ * inner_integrand(E_ + dE_ / 2, Ee, lf_nuc, k, Z);

            // // manual Simpson's rule
            // double left = inner_integrand(E_, Ee, lf_nuc, k, Z);
            // double middle = inner_integrand(E_ + dE_ / 2, Ee, lf_nuc, k, Z);
            // double right = inner_integrand(E_ + dE_, Ee, lf_nuc, k, Z);
            // inner += dE_ / 6 * (left + 4 * middle + right);  
        }
        return inner;
    }
}


double middle_integral(int Z, double lf_nuc, double Ee, double eps, double k_min, double Emin_) {
    // ==================== MIDDLE INTEGRAL ====================
    // integral 2: over all scattering angles (nucleus - photon); (-> all k)
    double middle = 0.0;

    double k_max   = 2 * lf_nuc * eps;
    double dk = (k_max - k_min) / 20;
    for (double k = k_min; k < k_max; k += dk) {
        // midpoint rule
        middle += inner_integral(Z, lf_nuc, Ee, k + dk / 2, Emin_) * dk * MEC2 * k * MEC2;

    }
    return middle;
}


double outer_integral(int Z, double lf_nuc, double Ee, double *eps_arr, double eps_max, double *deps, double *f_pho, int N_eps) {
    // INTEGRATION LOWER LIMITS -----
    // middle
    double k_min = pow((lf_nuc + Ee), 2) / (2 * lf_nuc * Ee);
    // inner
    double Emin_ = (lf_nuc*lf_nuc + Ee*Ee) / (2 * lf_nuc * Ee);

    // ==================== OUTER INTEGRAL ====================
    // integral 1: over all eps (LAB frame photon energies)
    double eps_min = k_min / (2 * lf_nuc);

    double outer = 0.0;

    for (int i = 0; i < N_eps ; i++) {
        if (eps_arr[i] >= eps_min && eps_arr[i] <= eps_max) {
            double middle = middle_integral(Z, lf_nuc, Ee, eps_arr[i], k_min, Emin_);
            outer += middle * deps[i] / (eps_arr[i]*eps_arr[i]) / MEC2 * f_pho[i];
        }
    }
    return 1 / (2 * lf_nuc*lf_nuc*lf_nuc) * outer;
}


// double *loop_electron_energies(int A, int Z, double lf_nuc, double eps_max, double *eps_arr, double *deps, double *f_pho, double N_eps, double *dNdEe) {
//     double Ee_min = pow(10, 7.0028719) /  MEC2;
//     double Ee_max = pow(10, 23.9028719) / MEC2;
//     double dEe = (Ee_max - Ee_min) / 170;

//     // for (double Ee = Ee_min; Ee < Ee_max; Ee += dEe) {
//     for (int i = 0; i < 170; i++) {
//         printf("%i", i);
//         dNdEe[i] = outer_integral(Z, lf_nuc, Ee_min + i * dEe, eps_arr, eps_max, deps, f_pho, N_eps);
//     }
//     return dNdEe;

// }
