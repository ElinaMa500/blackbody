import os
import numpy as np
import matplotlib.pyplot as plt

# constants (lab-script values)
h, c, kB = 6.626e-34, 3.00e8, 1.381e-23
SIG_LAM = 0.02                      # wavelength uncertainty (um)

def planck(lam_um, T):
    """B_λ with λ converted to meter."""
    l = lam_um * 1e-6
    return 2 * h * c**2 / l**5 / (np.exp(h * c / (l * kB * T)) - 1)

# ---------------------------------------------------------------- data
# wavelength (μm) and prism dispersion dλ/dn (Table 1 of lab script)
lam = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
                2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5])
dldn = np.array([0.581, 0.853, 1.131, 1.378, 1.570, 1.697, 1.764, 1.782,
                 1.765, 1.725, 1.670, 1.607, 1.541, 1.474, 1.408, 1.343,
                 1.281, 1.222, 1.166, 1.113, 1.063, 1.005, 0.970, 0.927,
                 0.886, 0.848, 0.811, 0.777, 0.744, 0.712])

# thermopile readings (V): background, ±bg, full signal, ±full
# np.nan = missed point (5.00 V at 1.1 um)
# use claude to extract the data out

runs = [
  dict(V=7.00, TR=2088, col="C3", mk="o", rank=1,
   bg=[0.003, 0.010, 0.007, 0.008, 0.008, 0.008, 0.011, 0.010, 0.011, 0.011,
       0.011, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012,
       0.012, 0.012, 0.012, 0.012, 0.013, 0.013, 0.013, 0.013, 0.014, 0.014],
   sbg=[0.003, 0.005, 0.001, 0.002, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002,
        0.002, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.002, 0.002, 0.002, 0.002, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002],
   full=[0.010, 0.020, 0.035, 0.057, 0.096, 0.137, 0.168, 0.190, 0.197, 0.195,
         0.189, 0.177, 0.161, 0.141, 0.127, 0.114, 0.100, 0.089, 0.078, 0.069,
         0.057, 0.046, 0.036, 0.031, 0.026, 0.026, 0.025, 0.024, 0.022, 0.022],
   sfull=[0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001,
          0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.002,
          0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.002, 0.001]),
  dict(V=5.00, TR=1806, col="C2", mk="o", rank=1,
   bg=[0.015, 0.015, 0.015, 0.016, 0.015, 0.015, 0.016, 0.016, 0.017, 0.017,
       0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017,
       0.017, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.017, 0.017, 0.017],
   sbg=[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.001, 0.001,
        0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001],
   full=[0.016, 0.018, 0.020, 0.023, 0.030, np.nan, 0.048, 0.056, 0.060, 0.063,
         0.064, 0.064, 0.059, 0.056, 0.052, 0.048, 0.045, 0.041, 0.037, 0.035,
         0.032, 0.027, 0.025, 0.023, 0.021, 0.021, 0.020, 0.020, 0.020, 0.019],
   sfull=[0.002, 0.002, 0.002, 0.002, 0.001, np.nan, 0.002, 0.002, 0.002, 0.002,
          0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
          0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002]),
  # second-highest point (1.8 um) used for 4.00 V peak,
  dict(V=4.00, TR=1633, col="C0", mk="o", rank=2,
   bg=[0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017,
       0.017, 0.018, 0.018, 0.018, 0.018, 0.018, 0.017, 0.017, 0.017, 0.017,
       0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017],
   sbg=[0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.002, 0.001, 0.002, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002],
   full=[0.017, 0.017, 0.017, 0.019, 0.022, 0.026, 0.030, 0.033, 0.037, 0.038,
         0.041, 0.040, 0.040, 0.038, 0.037, 0.035, 0.032, 0.032, 0.030, 0.027,
         0.025, 0.023, 0.022, 0.020, 0.020, 0.020, 0.019, 0.019, 0.018, 0.018],
   sfull=[0.002, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002,
          0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002,
          0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),
  dict(V=3.99, TR=1631, col="C9", mk="s", rank=1,          # reverse scan
   bg=[0.016, 0.016, 0.016, 0.016, 0.016, 0.017, 0.017, 0.017, 0.017, 0.017,
       0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017,
       0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 0.017],
   sbg=[0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
   full=[0.016, 0.016, 0.018, 0.019, 0.022, 0.026, 0.032, 0.034, 0.038, 0.039,
         0.039, 0.040, 0.040, 0.038, 0.036, 0.033, 0.032, 0.030, 0.029, 0.027,
         0.025, 0.022, 0.021, 0.020, 0.019, 0.019, 0.019, 0.019, 0.019, 0.019],
   sfull=[0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001,
          0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
          0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002]),
]

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(12.5, 6))
leg_h, leg_l = [], []
ll = np.linspace(0.55, 3.55, 3001)                    # fine grid for theory

for r in runs:
    bg, sbg = np.array(r["bg"]), np.array(r["sbg"])
    full, sfull = np.array(r["full"]), np.array(r["sfull"])
    TR, col, mk = r["TR"], r["col"], r["mk"]

    # pure filament signal, corrected for prism dispersion, in mV
    y  = (full - bg) / dldn * 1e3
    sy = np.sqrt(sbg**2 + sfull**2) / dldn * 1e3
    ok = ~np.isnan(y)                                 # skip missed point
    l_, y_, sy_ = lam[ok], y[ok], sy[ok]

    # measured lambda_max = highest (or 2nd-highest) data point
    lmax = l_[np.argsort(y_)[-r["rank"]]]
    r.update(l_=l_, y_=y_, sy_=sy_, lmax=lmax)        # keep for 2nd graph

    # Planck at T_R, scaled so theory peak = measured peak (no fit)
    B = planck(ll, TR)
    B_scaled = B / B.max() * y_.max()
    th_pk = ll[np.argmax(B)]

    label = f"supply {r['V']:.2f}V" + (" reverse scan" if mk == "s" else "")
    eb = ax.errorbar(l_, y_, yerr=sy_, xerr=SIG_LAM, fmt=mk, ms=4,
                     capsize=2, color=col,
                     mfc=col if mk == "o" else "none")
    vl = ax.axvline(lmax, color=col, ls="--", lw=1.3)         # measured peak
    pl, = ax.plot(ll, B_scaled, color=col, lw=1.4, alpha=.85,
                  ls="-" if mk == "o" else "-.")
    tl = ax.axvline(th_pk, color=col, ls=":", lw=1.6, alpha=.9)  # theory peak

    leg_h += [eb, vl, pl, tl]
    leg_l += [label,
              f"  measured λ$_{{max}}$ = {lmax:.2f} ± {SIG_LAM:.2f} µm",
              f"  Planck at T$_R$ = {TR:.0f} K",
              f"  theoretical peak λ = {th_pk:.2f} µm"]

ax.axvspan(2.9, 3.1, color="grey", alpha=.15)
ax.text(3.0, 65, "quartz\nabsorption", ha="center", fontsize=9,
        color="dimgrey")
ax.set_xlabel("Wavelength λ (µm)")
ax.set_ylabel("Corrected signal  (mV) / (dλ/dn)")
ax.set_title("Exp 2: corrected spectra with Planck curves at T$_R$ "
             "(scaled to measured peak)")
ax.grid(alpha=.3)
ax.legend(leg_h, leg_l, fontsize=9, loc="upper left",
          bbox_to_anchor=(1.01, 1), borderaxespad=0)
fig.tight_layout()
os.makedirs("graphs", exist_ok=True)       # make the folder if it isn't there
path_spectra = "graphs/exp2_spectra.png"   # path to save the spectra graph
fig.savefig(path_spectra, dpi=110)

# ---------------------------------------------------------------- graph 2
# Same data, Planck curve at T_Wien = b / lambda_max instead of T_R
WIEN_B = 2897.8                     # Wien's constant (um K)
fig2, ax2 = plt.subplots(figsize=(12.5, 6))
leg_h, leg_l = [], []

for r in runs:
    l_, y_, sy_, lmax = r["l_"], r["y_"], r["sy_"], r["lmax"]
    col, mk = r["col"], r["mk"]

    TW = WIEN_B / lmax                      # Wien temperature
    sTW = TW * SIG_LAM / lmax               # its uncertainty

    # Planck at T_Wien, scaled so theory peak = measured peak (no fit)
    B = planck(ll, TW)
    B_scaled = B / B.max() * y_.max()

    label = f"supply {r['V']:.2f}V" + (" reverse scan" if mk == "s" else "")
    eb = ax2.errorbar(l_, y_, yerr=sy_, xerr=SIG_LAM, fmt=mk, ms=4,
                      capsize=2, color=col,
                      mfc=col if mk == "o" else "none")
    vl = ax2.axvline(lmax, color=col, ls="--", lw=1.3)        # measured peak
    pl, = ax2.plot(ll, B_scaled, color=col, lw=1.4, alpha=.85,
                   ls="-" if mk == "o" else "-.")

    leg_h += [eb, vl, pl]
    leg_l += [label,
              f"  measured λ$_{{max}}$ = {lmax:.2f} ± {SIG_LAM:.2f} µm",
              f"  Planck at T$_{{Wien}}$ = {TW:.0f} ± {sTW:.0f} K "
              f"(T$_R$ = {r['TR']} K)"]

ax2.axvspan(2.9, 3.1, color="grey", alpha=.15)
ax2.text(3.0, 65, "quartz\nabsorption", ha="center", fontsize=9,
         color="dimgrey")
ax2.set_xlabel("Wavelength λ (µm)")
ax2.set_ylabel("Corrected signal  (mV) / (dλ/dn)")
ax2.set_title("Exp 2: corrected spectra with Planck curves at T$_{Wien}$ "
              "(scaled to measured peak)")
ax2.grid(alpha=.3)
ax2.legend(leg_h, leg_l, fontsize=9, loc="upper left",
           bbox_to_anchor=(1.01, 1), borderaxespad=0)
fig2.tight_layout()
path_wien = "graphs/exp2_spectra_TWien.png"  # path to save the T_Wien graph
fig2.savefig(path_wien, dpi=110)
plt.show()