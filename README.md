# FLYSUIT — Mjalnor'MV1.17

A complete, single-file, interactive **3D model + physics simulator** of the
Mjalnor'MV1.17 hybrid exoskeleton — one suit that operates as a **combat flight
suit**, a **space suit**, and a **dive suit**. Every dimension is real (mm / SI)
and every subsystem is driven by math and physics, not scripted constants.

> One-size-fits-all telescoping frame (fits a 3′3″–7′3″, 65–420 lb pilot) ·
> 4-layer armor · 48 micro-turbofan swarm turbines (Jet-A1 fuel, SFC-modeled) ·
> deployable gliding wings (induced drag + Oswald efficiency) · vacuum-sealed
> helmet · neural BCI · dual-redundant real-time OS · buoyancy control +
> decompression computer · radiation dosimetry + cold-gas RCS · ISA atmosphere.

Everything lives in [flysuit.py](flysuit.py) (~9,700 lines). Design intent is in
[Projectgoal.md](Projectgoal.md) and [officialgoal.md](officialgoal.md). For the
architecture and the engineering model, see [overview.md](overview.md).

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
`--pilot-weight <kg>` (auto-sizes the telescoping frame).

---

## Controls (interactive viewer)

**Switch views:** `TAB` cycles MODEL ↔ FLIGHT ↔ LAYERS. `H` help · `I` info ·
`M` math · `P` pause · `O` export OBJ · `ESC` quit.

**MODEL** — mouse orbit · wheel zoom · right-drag pan · `1`/`2`/`3` full /
exploded / assembly · `L` labels · `,` `.` isolate parts.

**FLIGHT** — `UP`/`DOWN` throttle · `SPACE` max + afterburner · `C` cut ·
`W/S/A/D/Q/E` vector thrust · `Z` altitude-hold · `V` hover · `B` auto-hover ·
`N` auto-level · `6` deploy/stow wings · `[` `]` change environment · `R`
respawn. Gamepad supported.

**LAYERS** — `1`–`4` toggle the four armor layers · `E` explode · `[` `]` focus.

**Combat / test** — `F` or click fire · `T` spawn target · `G` defense AI ·
`J` 200 ft jump · `5` EMP strike · `4` armor-damage test · `K` heat overlay.

**Upgrades** — `F2` stealth · `F3` vision modes · `F4`/`F5` grapple · `F6`
parachute · `F7`/`F8` drone swarm · `F9` flares · `F10` energy shield · `F11`
tactical shield · `F12` stun · `Scroll Lock` beacon · `Pause` maglev climb.

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
| Turbine thrust | Scales with air density ratio (ρ/ρ₀); zero in vacuum → RCS cold-gas |
| Fuel consumption | SFC × thrust → kg/s burn rate; total mass decreases as fuel depletes |
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
| `flysuit.py` | The entire simulator (specs, systems, physics, renderer, tests) |
| `overview.md` | Architecture + engineering model deep-dive |
| `Projectgoal.md` / `officialgoal.md` | Original design brief and goals |
| `flysuit.obj` / `flysuit.mtl` | Exported 3D model (regenerate with `--export-obj`) |
| `ReferenceCode/` | Prior software-renderer projects used as the engine reference |
