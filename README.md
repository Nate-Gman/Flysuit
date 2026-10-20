# FLYSUIT — Mjalnor'MV1.17

A complete, interactive **3D model + physics simulator** of the
Mjalnor'MV1.17 hybrid exoskeleton — one suit that operates as a **combat flight
suit**, a **space suit**, and a **dive suit**. Every dimension is real (mm / SI)
and every subsystem is driven by math and physics, not scripted constants.

> One-size-fits-all telescoping frame (fits a 3′3″–7′3″, 65–420 lb pilot; the
> suit mesh itself conforms to the pilot — no tailoring) · **100% scale:**
> the model's crown-to-ground span equals the pilot height exactly at any
> size, and the assembled suit adds only ~24 mm of helmet-dome + boot-sole
> clearance — verified by self-test · 4-layer sealed armor
> (7.5 mm total) in **accumulative stacking** — each layer sits exactly its
> thickness above the body; outer shell stands ~7 mm proud, verified by
> self-test · conformal-wrapped chest/back/ab plates on a smooth elliptical
> shell, flush panel lines, contour detail · 72 flush RMF plasma-clutch pods (25 mm dia coil discs — 30
> backpack + 12 forearm + 12 calf + 12 thigh + 6 collar ring — bayonet-swap
> cartridges, iris-sealed in vacuum/underwater, flush-mounted within 1-inch
> profile) whose ~9.4× lobes merge into one ~1 m coupling region · 6 steering
> discs (2 palm gimbaled + 2 heel fixed + 2 backpack vectored) · D-He3 fusion
> reactor 55 kW (~85 yr pellet endurance) drives every regime: air MHD
> ion-drag, Titan/Mars gas (no O₂ needed), underwater ducted waterjets,
> vacuum water plasma jet (Isp ~2550 s), solar-wind magnetic sail · Archangel
> glide membrane — a compact gliding suit that is PART OF THE SUIT (3.5 m
> deployed span, arm-to-body + leg-to-leg membranes, cape-fold stow; clutch
> pulses to climb, the membrane glides) · vacuum-sealed helmet with wraparound
> visor + fusion-powered closed-loop life support (Sabatier + electrolysis,
> indefinite endurance; 24 h backup O₂/LiOH) · neural BCI · dual-redundant
> real-time OS · buoyancy control + decompression computer · radiation
> dosimetry + RMF magnetic shielding · three-tier meteoroid defense (plasma
> sheath ≤300 J, energy-shield sentry ≤50 kJ, self-healing Whipple bumper) ·
> directed-energy defense: 92%-reflective ceramic face + ablation vs lasers,
> RMF shunt ≤120 kJ + capacitor vs plasma bolts, 700 kPa blast attenuation —
> and Argent-drain: absorbed energy recharges the capacitor and battery ·
> cold-gas RCS with water-steam fallback · ISA atmosphere · **field-serviceable
> architecture**: per-pod health degrades gracefully (−1.4% thrust per lost
> pod, never a drive failure), 8-min bayonet LRU swaps, 8-unit maintenance
> prognostics with next-service forecast, and signed A/B firmware updates on
> the dual-redundant OS for decades of upgrades.

The simulator is a complete **single-file build** in [flysuit.py](flysuit.py)
(~17,400 lines) — the 16 component showcases are integrated directly into it.
[showcase.py](showcase.py) remains only as a compatibility re-export shim for
older imports. Design intent is in [Projectgoal.md](Projectgoal.md) and
[officialgoal.md](officialgoal.md). For the architecture and the engineering
model, see [overview.md](overview.md).

---

## Install & run

```sh
python3 -m pip install pygame numpy
python flysuit.py                 # open the interactive 3D viewer
```

`numpy` is required; `pygame` is only needed for the interactive viewer — all
the headless report/test modes below run without it.

### Command-line modes

| Command | What it does |
|---|---|
| `python flysuit.py` | Interactive viewer (MODEL / FLIGHT / LAYERS) |
| `python flysuit.py --selftest` | Headless build + render + full physics/system self-test |
| `python flysuit.py --impact-test` | Ballistic impact absorption table (9 mm → artillery) |
| `python flysuit.py --stress-test` | Steps the suit through all 11 environments |
| `python flysuit.py --os-test` | SuitRTOS deep diagnostic (boot, tasks, crypto, BCI) |
| `python flysuit.py --feasibility` | Real-world build feasibility report |
| `python flysuit.py --export-obj` | Write `flysuit.obj` + `flysuit.mtl` and exit |
| `python flysuit.py --layers` | Start in the layer-by-layer exploded view |

Extra flags: `--detail <mult>` (mesh density), `--pilot-height <m>`,
`--pilot-weight <kg>` — auto-sizes the telescoping frame; inputs are clamped to
the rated fit envelope (0.99–2.21 m, 29.5–190.5 kg).

---

## Controls (interactive viewer)

**Switch views:** `TAB` cycles MODEL ↔ FLIGHT ↔ LAYERS. `H` help · `F1`
instructions · `I` info · `P` pause · `O` export OBJ · `ESC` quit.

**MODEL** — mouse orbit · wheel zoom · right-drag pan · `1`/`2`/`3` full /
exploded / assembly · `L` labels · `E` explode · `X` section cut · `,` `.`
isolate parts · `U` part browser · `B` blueprint · `0` suit overview · `S`
showcase (16 detailed component views with math proof) · `[` `]` cycle showcase.

**FLIGHT** — `UP`/`DOWN` throttle · `SPACE` max + afterburner · `C` cut ·
`W/S` pitch · `A/D` roll · `←/→` yaw · `I/J/K/L` thrust vector · `Z` altitude-hold · `V` hover · `B` auto-hover ·
`N` auto-level · `Y` **pulse-glide autopilot** (clutch pulse climb → membrane
glide cycle) · `6` deploy/stow glide membrane · `[` `]` change environment · `R`
respawn. Gamepad supported.

**LAYERS** — `1`–`4` toggle the four armor layers · `E` explode · `[` `]` focus.

**Combat / test** — `F` or click fire · `T` spawn target · `G` defense AI ·
`J` 200 ft jump · `5` EMP strike · `4` armor-damage test · `K` heat overlay.

**Upgrades** — `F2` stealth · `F3` night vision · `F4` thermal/FLIR · `F5`
thermal mode · `F6` DEA muscle · `F7` STF test · `F8` crypto · `F9` flares ·
`F10` energy shield · `F11` tactical shield · `F12` stun · `Scroll Lock` beacon
· `Pause` maglev climb.

**Window** — resizable; all panels, HUD bars, and text scale dynamically with
window size. Scroll the mouse wheel over info/showcase/help/instructions panels
to scroll their content.

---

## Verifying a change

The self-tests assert real behavior, so run them after any edit:

```sh
python flysuit.py --selftest      # must end with "ALL TESTS PASSED"
python flysuit.py --impact-test   # absorption must VARY by threat (ballistic limit)
python flysuit.py --stress-test   # must be stable across all 11 environments
```

**Design rule:** every part operates mechanically from math — nothing is faked.
Material dimensions and ratings are constants; *outcomes* (impact absorption,
buoyancy, radiation dose, punch force, hit resolution, fuel burn, induced drag,
g-load, rotational dynamics) are always computed from physics equations. If
you add a system, trace every number it reports back to a formula and add a
self-test assertion for it.

### Physics models (mechanically accurate)

| Domain | Equation / Model |
|---|---|
| Plasma clutch (air) | Merged ~1 m coupling region; F = (2ρAP²)^⅓·coupling·k_seed actuator-disk + real plasma diagnostics (ω_p, v_A, λ_D, Rm, β); works on any gas — no O₂ needed (Mars/Titan OK) |
| Plasma clutch (water) | Pods sealed; 6 steering discs run ducted waterjets on ambient water — power-limited momentum flux + cavitation derate |
| Plasma clutch (space) | Vacuum: water plasma jet (F = 2ηP/v_exh, Isp ~2550 s, 4.9 km/s Δv on 22 kg H₂O); ambient plasma: frozen-in magnetic sail; solar wind boost + ISM braking legs |
| Fusion power | D-He3 aneutronic, 55 kW continuous, sponge-lattice core, ~85 yr pellet endurance (primary + backup) |
| Water reaction mass | Drawn only by the vacuum jet (2 kg suit tank → 20 L reserve bladder); total mass decreases as it depletes |
| Wing lift | L = ½ρv²S·CL(α), linear to stall, flat-plate post-stall |
| Induced drag | CDi = CL²/(π·AR·e), Oswald e=0.85, aspect ratio from span²/area |
| Parasitic drag | D = ½ρv²·Cd·A, environment-specific coefficients |
| Impact pressure | PSI = (F/A)/6895 (Pa→PSI), F = m·v/t_stop |
| G-load | \|a_total\| / g (full 3-D acceleration vector, not just vertical) |
| Rotational dynamics | α = τ/I (torque = thrust × moment arm, per-axis inertia tensor) |
| Atmosphere | ISA troposphere + stratosphere lapse rate; Mars/Titan/ocean/space models |

---

## Repo layout

| Path | Purpose |
|---|---|
| `flysuit.py` | The entire simulator (~17,400 lines: specs, systems, physics, renderer, tests, showcases) |
| `showcase.py` | Compatibility shim — re-exports the showcase builders integrated in `flysuit.py` |
| `overview.md` | Architecture + engineering model deep-dive |
| `Projectgoal.md` / `officialgoal.md` | Original design brief and goals |
| `flysuit.obj` / `flysuit.mtl` | Exported 3D model (regenerate with `--export-obj`) |
| `ReferenceCode/` | Prior software-renderer projects used as the engine reference |
