# Supplementary Materials: **Photosynthetic gas in coral canopies and the reach of blast-fishing shocks**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.24-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.10-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%E2%89%A53.7-11557C?style=flat-square)](https://matplotlib.org)
[![Pillow](https://img.shields.io/badge/Pillow-%E2%89%A510.0-4B6C8C?style=flat-square)](https://python-pillow.org)
[![pycodestyle](https://img.shields.io/badge/pycodestyle-clean-1E7B7B?style=flat-square)](https://peps.python.org/pep-0008/)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-clean-1E7B7B?style=flat-square)](https://peps.python.org/pep-0257/)
[![License: MIT](https://img.shields.io/badge/License-MIT-A31F34?style=flat-square)](LICENSE)
[![No external data](https://img.shields.io/badge/data-none%20required-7D8CA3?style=flat-square)](#)

A reduced-order model of how free gas in a coral canopy changes the loading
that an improvised fishing charge delivers to reef skeleton. Analysis only:
no field data, no calibration.

<div align="center">
  <img src="outputs/animations/anim01_diptych.gif" alt="one charge, two canopies" width="760" />
</div>

## Model

A canopy of seawater carrying void fraction $\alpha$ is compressed by an
overpressure $\Delta p$. With liquid bulk modulus $K_l$ and polytropic gas,

$$v_0-v_1=\frac{v_{l0}\,\Delta p}{K_l}+v_{g0}\left[1-\left(\frac{p_0}{p_0+\Delta p}\right)^{1/\kappa}\right],\qquad U^2=\frac{v_0^2\,\Delta p}{v_0-v_1},\qquad Z_c=\frac{U}{v_0}.$$

$U$ reduces to the Wood speed as $\Delta p\to0$. The incident pulse is the
similitude form $P_m e^{-t/\theta}$ with $P_m$ and $\theta$ set by charge mass
$W$ and slant range $R$. The reef is a normal-incidence column
water | canopy | skeletal plate | canopy, solved exactly in equal-travel-time
cells. Keller-Miksis dynamics test whether canopy bubbles can follow the
shock, and a two-dimensional linear solver shows the same contrast over a
branching thicket and a tabular plate.

## Results

**Blast-strength shocks see through the canopy.** Gas and liquid compliance
are equal at

$$p^*=\frac{\alpha K_l}{1-\alpha},$$

2.3 MPa at $\alpha=10^{-3}$. Above $p^*$ the canopy is nearly transparent,
so its acoustic (Wood) speed greatly overstates its effect on a blast. To
leading order in $p_0/p^*$ the shock speed at $p^*$ is
$c_l/\sqrt2(1-\alpha)$.

**The canopy cannot change the impulse.** For any lossless stack between
water and a half-space of impedance $Z_b$, the transmitted impulse is
exactly $2Z_b/(Z_b+Z_w)$ times the incident impulse. The canopy only
redistributes it in time; the peak just after the $N$-th reverberation is
$T_1T_2P\,(r^{N+1}-q^{N+1})/(r-q)$ with $r=e^{-2\tau/\theta}$ and
$q=R_{cs}R_{cw}$.

**What gas does change is the back face.** A plate of thickness $d$ carries
first-reflection tension if and only if

$$\frac{Z_c}{Z_s}<\tanh\delta,\qquad \delta=\frac{d}{c_s\theta},\qquad d_c=c_s\theta\,\operatorname{artanh}\frac{Z_c}{Z_s}.$$

| quantity, 1 kg TNT equivalent | value |
| --- | --- |
| $d_c$ with water behind the plate, $R=2\ldots10$ m | 10.7 to 15.3 cm |
| $d_c$ at $\alpha=3\times10^{-2}$ | 3.5 to 5.3 cm |
| spall radius, 12 cm plate, $\alpha=10^{-5}\to2.7\times10^{-2}$ | 1.75 to 5.70 m |
| crush radius, same plate | 3.37 to 2.51 m |
| noon horizontal spall reach, charge 1.5 m above plates, $\alpha_{\max}=10^{-2}$, $3\times10^{-2}$ | 4.1 m, 5.6 m (0.9 m at night) |
| 2D median tension-to-compression ratio in skeleton, $\alpha=10^{-5}\to10^{-2}$ | 0.14 to 0.19 |

The effect needs $\alpha\gtrsim10^{-3}$. At $\alpha_{\max}=10^{-3}$ the diel
change in reach is 0.94 to 1.29 m. A step front drives canopy bubbles past
equilibrium even when $\theta f_M\gg1$ (minimum volume 5 to 32 % of the
static value), so the equilibrium Hugoniot does not bound instantaneous
compaction.

<table>
<tr>
<td><img src="outputs/animations/anim02_diel.gif" width="270" /></td>
<td><img src="outputs/animations/anim03_bubbles.gif" width="270" /></td>
<td><img src="outputs/animations/anim05_thickness.gif" width="280" /></td>
</tr>
<tr>
<td align="center">reach of one charge through a day</td>
<td align="center">forty canopy bubbles under one pulse</td>
<td align="center">critical thickness as range closes</td>
</tr>
</table>

<div align="center">
  <img src="outputs/animations/anim04_reverberation.gif" alt="reverberation" width="620" />
</div>

## Numerics

Three layered solvers with no shared code path: Goupillaud scattering in
equal-travel-time cells, the closed-form ray sum, and propagator matrices
inverted by FFT. Keller-Miksis is integrated with DOP853 and checked
against Radau. The 2D solver is a staggered-grid scheme with Cerjan
sponges and a line source whose volume rate is the half-integral of the
target pulse.

| check | result |
| --- | --- |
| Goupillaud vs ray sum, max abs | $2.2\times10^{-16}$ |
| Goupillaud vs transfer matrix, max abs | $4.4\times10^{-16}$ |
| impulse invariant, all three solvers, relative | $2.2\times10^{-16}$ |
| plate first-reflection profile vs closed form | $2.2\times10^{-16}$ |
| Hugoniot to Wood, slope in $\Delta p$ | 0.9998 |
| Keller-Miksis nonlinear frequency shift, slope (expected 2) | 1.978 |
| DOP853 vs Radau, period and minimum radius | $5\times10^{-13}$, $1.4\times10^{-12}$ |
| 2D self-convergence, observed orders | 2.59, 0.81 |
| 2D impulse invariant, relative | $7\times10^{-6}$, window-limited |

The 2D order approaches one because buoyancy is averaged arithmetically
across discontinuous interfaces.

## Units

SI throughout. Figures 2 and 8(a) use $P$ and $\theta$ as units; Figure 4
and animation 3 use $R_0$, $p_0$ and $R_0\sqrt{\rho/p_0}$. Figure 7 and
animation 1 have an arbitrary source amplitude and report only ratios.
Material constants are round illustrative values; see
`outputs/reports/parameters.txt`.

## Run

```bash
pip install -r requirements.txt   # or: pip install -e .
python scripts/run_all.py
```

About fifteen minutes on one core, most of it in the 2D runs and the
convergence study, both of which cache to `outputs/cache/` and are skipped
on a rerun. Scripts run standalone once their cache exists. Lint with
`pycodestyle reefblast/ scripts/` and `pydocstyle reefblast/ scripts/`.

## Layout

```
reefblast/
  core.py        parameters, similitude source, crossover p* and R*
  canopy.py      Wood speed, bubbly Hugoniot, secant impedance
  stack.py       Goupillaud, ray sum, transfer matrix, closed forms
  damage.py      batched plate stresses over sweeps, damage radii
  bubble.py      Keller-Miksis, Minnaert frequency, static radius
  fdtd.py        2D acoustics, reef scene, half-integral line source
  diel.py        prescribed diel void fraction
  scenario.py    reference configuration shared by every script
  plotting.py    figure style, palette, PDF and 600 dpi PNG export
  anim.py        dark style, signed colour map, GIF export
  io_utils.py    CSV, report and cache writers
scripts/
  fig00 .. fig08        one script per figure
  anim01 .. anim05      one script per animation
  make_reports.py       plain-text reports
  run_all.py            regenerate everything
outputs/
  figures/       vector PDF and 600 dpi PNG
  animations/    looping GIFs
  data/          every panel as CSV
  reports/       closed forms, verification, parameters,
                 figure notes, open items
```

Figures carry no titles and no in-panel annotation; values are in
`outputs/reports/figure_notes.txt`. The animations carry labels and a
clock because they are meant to be read without a caption.

## Limitations

The canopy void fraction is a free parameter. Diel oxygen bubbles are
documented acoustically in seagrass meadows, but no measurement in a coral
canopy is known to the authors, and every canopy effect reported here
requires $\alpha\gtrsim10^{-3}$.

Normal incidence is assumed at every range, the canopy impedance is frozen
at its secant value for the incident peak, the skeleton has no shear
rigidity, and damage is judged by a single peak stress. The
surface-reflected rarefaction is omitted because bulk cavitation caps it
near $-p_0$, well below the assumed tensile strength.

Skeleton density, sound speed and strengths are illustrative, and every
threshold scales with them. The onset criterion and the impulse invariant
do not.

The 2D runs use a line source that reproduces the pulse only in the far
field; its negative near field is visible above the reef in Figure 7 and
animation 1. They are used for patterns and amplitude-free ratios, never
for thresholds.

The model began from the expectation that the water-canopy reflection
changes sign at a critical void fraction. It does not, since any free gas
lowers the canopy impedance below water. The surviving mechanism is at the
back face of the skeleton.
