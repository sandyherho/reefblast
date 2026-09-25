# Supplementary Materials "Spall Failure of Coral Skeleton beneath Gas-Laden Canopies:An Idealized Blast-Fishing Model"

[![DOI](https://zenodo.org/badge/1380313579.svg)](https://doi.org/10.5281/zenodo.22884080)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.24-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.10-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![Lint](https://img.shields.io/badge/pycodestyle%20%7C%20pydocstyle-clean-1E7B7B?style=flat-square)](https://peps.python.org/pep-0008/)
[![License: MIT](https://img.shields.io/badge/License-MIT-A31F34?style=flat-square)](LICENSE)

Supplementary code for a reduced-order model of how free gas in a coral
canopy changes the loading that an improvised fishing charge delivers to
reef skeleton.

**Authors:** Sandy H. S. Herho, Agus W. Jatmiko, Rizki D. Permana, Iwan P. Anwar,
Alfita P. Handayani, Faruq Khadami, Karina A. Sujatmiko, Sri Y. Cahyarini, and Dasapta E. Irawan

<p align="center">
  <img src="outputs/animations/anim01_diptych.gif" width="100%" alt="one charge over a gas-poor and a gas-rich canopy"><br>
  <sub>One charge, two canopies: void fraction 10<sup>-5</sup> (left) and 10<sup>-2</sup> (right). Amber compression, blue tension.</sub>
</p>

<table>
  <tr>
    <td align="center"><img src="outputs/animations/anim02_diel.gif" height="240" alt="diel reach"></td>
    <td align="center"><img src="outputs/animations/anim03_bubbles.gif" height="240" alt="bubble phase portraits"></td>
    <td align="center"><img src="outputs/animations/anim05_thickness.gif" height="240" alt="critical thickness"></td>
  </tr>
  <tr>
    <td align="center"><sub>Reach of one charge over a day</sub></td>
    <td align="center"><sub>Forty canopy bubbles under one pulse</sub></td>
    <td align="center"><sub>Critical plate thickness as range closes</sub></td>
  </tr>
</table>

<p align="center">
  <img src="outputs/animations/anim04_reverberation.gif" width="80%" alt="reverberation in the reef column"><br>
  <sub>Reverberation in water, canopy, plate, canopy; the bar converges to the impulse invariant.</sub>
</p>

## Key results

Gas and liquid compliances of the canopy are equal at

```math
p^* = \frac{\alpha K_l}{1-\alpha},
```

about 2.3 MPa at $`\alpha = 10^{-3}`$. Blast-strength shocks above $`p^*`$
see a nearly transparent canopy.

For any lossless stack between water and a half-space of impedance
$`Z_b`$, the transmitted impulse is exactly $`2Z_b/(Z_b+Z_w)`$ times the
incident impulse. Gas redistributes the pulse in time but cannot change
its impulse.

A plate of thickness $`d`$ carries first-reflection tension if and only if

```math
\frac{Z_c}{Z_s} < \tanh\delta, \qquad \delta = \frac{d}{c_s\theta}, \qquad d_c = c_s\theta\,\mathrm{artanh}\frac{Z_c}{Z_s}.
```

| 1 kg TNT equivalent | value |
| :-- | :-- |
| critical thickness, water-backed, R = 2 to 10 m | 10.7 to 15.3 cm |
| critical thickness, void fraction 3e-2 | 3.5 to 5.3 cm |
| spall radius, 12 cm plate, void fraction 1e-5 to 2.7e-2 | 1.75 to 5.70 m |
| crush radius, same plate | 3.37 to 2.51 m |
| noon spall reach, peak void fraction 1e-2 / 3e-2 (night 0.9 m) | 4.1 / 5.6 m |
| 2D median tension-to-compression ratio, 1e-5 to 1e-2 | 0.14 to 0.19 |

Every canopy effect requires void fractions of order 1e-3 or more.

## Verification

| check | result |
| :-- | :-- |
| Goupillaud vs ray sum vs transfer matrix, max abs | 2.2e-16, 4.4e-16 |
| impulse invariant, three solvers, relative | 2.2e-16 |
| plate first reflection vs closed form | 2.2e-16 |
| Hugoniot to Wood limit, slope | 0.9998 |
| Keller-Miksis nonlinear frequency shift, slope (expected 2) | 1.978 |
| DOP853 vs Radau | 5e-13 |
| 2D self-convergence orders | 2.59, 0.81 |

Full residuals are in `outputs/reports/verification.txt`.

## Run

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

About fifteen minutes on one core. The 2D runs and the convergence study
cache to `outputs/cache/` and are skipped on reruns.

## Layout

```
reefblast/   core, canopy, stack, damage, bubble, fdtd, diel, scenario,
             plotting, anim, io_utils
scripts/     fig00-fig08, anim01-anim05, make_reports, run_all
outputs/     figures (PDF, 600 dpi PNG), animations (GIF),
             data (CSV per panel), reports (plain text)
```

## Limitations

The canopy void fraction is a free parameter; no measurement in a coral
canopy is known to the authors. Normal incidence, a frozen secant canopy
impedance, and a shear-free skeleton are assumed, and skeletal constants
are illustrative, so every threshold scales with them. The 2D line source
is exact only in the far field and is used for amplitude-free ratios,
never for thresholds. Details are in `outputs/reports/open_items.txt`.