# FLYSUIT — Mjalnor'MV1.17

A complete, interactive **3D model + physics simulator** of the
Mjalnor'MV1.17 hybrid exoskeleton — one suit that operates as a **combat flight
suit**, a **space suit**, and a **dive suit**. Every dimension is real (mm / SI)
and every subsystem is driven by math and physics, not scripted constants.

> One-size-fits-all telescoping frame (optimized for a 6′2″–6′10″ pilot,
> subfits down to 5′0″; 65–420 lb; the
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
> cold-gas RCS with water-steam fallback · ISA atmosphere · full tactical
> suite (stealth camo, 5-mode vision, grapple, drones, countermeasures,
> tactical + energy shields, stun, beacon, maglev, emergency parachute) ·
> 33-phrase voice command dispatch · **field-serviceable
> architecture**: per-pod health degrades gracefully (−1.4% thrust per lost
> pod, never a drive failure), 8-min bayonet LRU swaps, 8-unit maintenance
> prognostics with next-service forecast, and signed A/B firmware updates on
> the dual-redundant OS for decades of upgrades.

The simulator is a complete **single-file build** in [flysuit.py](flysuit.py)
(~18,400 lines) — the 16 component showcases are integrated directly into it.
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
| `python flysuit.py --selftest` | Headless build + render + full physics/system self-test (must end `ALL TESTS PASSED`) |
| `python flysuit.py --impact-test` | Ballistic impact absorption table (9 mm → artillery) |
| `python flysuit.py --stress-test` | Steps the suit through all 11 environments |
| `python flysuit.py --os-test` | SuitRTOS deep diagnostic (boot, tasks, crypto, BCI) |
| `python flysuit.py --energy-test` | Directed-energy weapon defense test (laser/plasma/blast) |
| `python flysuit.py --extreme-test` | Extreme-envelope fuzz test (degenerate inputs, corrupted state, all environments — must end `ZERO FAILURES`) |
| `python flysuit.py --feasibility` | Real-world build feasibility report |
| `python flysuit.py --export-obj` | Write `flysuit.obj` + `flysuit.mtl` and exit |
| `python flysuit.py --layers` | Start in the layer-by-layer exploded view |

Extra flags: `--detail <mult>` (mesh density), `--pilot-height <m>`,
`--pilot-weight <kg>` — auto-sizes the telescoping frame; inputs are clamped to
the rated fit envelope (1.52–2.09 m, 29.5–190.5 kg; optimized band
1.88–2.08 m).

### Fit envelope

| Band | Height | Weight | Frame state |
|---|---|---|---|
| Rated subfit floor | 1.52 m (5′0″) | 29.5–190.5 kg | torso −184 mm / limb −138 mm retraction |
| Optimized band | 1.88–2.08 m (6′2″–6′10″) | 29.5–190.5 kg | actuators 0.36–0.65 stroke (max rigidity margin) |
| Design center | 1.98 m (6′6″) | 79.4 kg reference | actuators mid-stroke |
| Rated ceiling | 2.09 m (6′10″) | 29.5–190.5 kg | torso +44 mm / limb +33 mm extension |

Travel budget: torso ±355.6 mm, limbs ±254.0 mm telescoping; 64 Ti nodes,
20 pivots, 14 linear actuators, 24 auto-tensioning straps.

---

## Controls (interactive viewer)

**Switch views:** `TAB` cycles MODEL ↔ FLIGHT ↔ LAYERS. `H` help · `F1`
instructions · `I` info · `P` pause · `O` export OBJ · `ESC`/`Q` quit.

**MODEL** — mouse orbit · wheel zoom · right-drag pan · `1`/`2`/`3` full /
exploded / assembly · `L` labels · `K` auto-rotate · `E` explode · `X` section
cut · `,` `.` isolate parts · `U` part browser · `B` blueprint · `0` suit
overview · `S` showcase (16 detailed component views with math proof) · `[` `]`
cycle showcase.

**FLIGHT** — `UP`/`DOWN` throttle · `SPACE` max + afterburner · `C` cut ·
`V` cruise 30% · `W/S` pitch · `A/D` roll · `←/→` yaw · `I/J/K/L` thrust
vector · `F6` pitch invert · `Z` altitude-hold · `B` auto-hover ·
`N` auto-level · `Y` **pulse-glide autopilot** (clutch pulse climb → membrane
glide cycle) · `F1`–`F5` autopilot HOVER/LEVEL/LAND/ORBIT/NAVIGATE ·
`6` deploy/stow glide membrane · `[` `]` change environment · `8` chase cam ·
`9` FPS overlay · `H` heat overlay · `R` respawn. Gamepad supported.

**LAYERS** — `1`–`4` toggle the four armor layers · `E` explode · `[` `]` focus.

**Combat / test** — `F` or click fire · `T` spawn target · `U` auto-targets ·
`M` combat scenario (waves) · `G` defense AI · `J` 200 ft jump (charge, then
execute) · `5` EMP strike · `4` armor-damage test · `7` add squad member.

**Subsystems (model-mode F-keys)** — `F2` stealth · `F3` vision mode cycle ·
`F4` grapple fire/release · `F5` grapple winch · `F6` parachute · `F7`/`F8`
drones launch/recall · `F9` countermeasures · `F10` energy shield · `F11`
tactical shield · `F12` stun · `Scroll Lock` beacon · `Pause` maglev climb.

**Gamepad** — left stick pitch+roll · right stick yaw+thrust vector · RT/LT
throttle · `A` afterburner · `B` cut · `X` fire · `Y` wings · `LB`/`RB` cycle
environment · `Back` auto-hover · `Start` auto-level · `D-pad up` jump.

**Window** — resizable; all panels, HUD bars, and text scale dynamically with
window size. Scroll the mouse wheel over info/showcase/help/instructions panels
to scroll their content.

---

## Voice commands

`VoiceCommandSystem` is a full **subvocal command pipeline** — 33 phrases,
80 ms latency spec, 97 % accuracy model, longest-match-first phrase resolution
(so *"execute jump"* is never shadowed by *"jump"*). `voice.command(text,
state)` runs recognize → dispatch against the live suit — every phrase performs
the same action as its keybind. Unrecognized phrases dispatch nothing.

| Spoken phrase | Action |
|---|---|
| "stealth on" / "stealth off" | toggle adaptive camouflage |
| "night vision" / "thermal" / "sonar" / "x-ray" / "normal vision" | set helmet vision mode |
| "deploy shield" / "stow shield" | tactical shield deploy/stow |
| "energy shield" | activate eshield barrier |
| "fire grapple" / "release grapple" / "winch" | grapple fire / release / winch toggle |
| "launch drones" / "launch scout" / "recall drones" | full swarm / single scout / recall |
| "deploy parachute" | emergency canopy |
| "countermeasures" | deploy flare |
| "stun" | fire contact stun |
| "beacon on" | 406 MHz emergency beacon |
| "maglev on" / "maglev off" | maglev climb toggle |
| "afterburner" | AB toggle |
| "deploy wings" / "stow wings" | glide membrane deploy/stow |
| "pulse glide" | pulse-climb-glide autopilot |
| "auto hover" / "auto level" | hover / attitude-level autopilot |
| "defense mode" | arm/disarm martial-arts defense AI |
| "fire" | fire at locked target |
| "jump" / "execute jump" | charge fibers / execute 200 ft jump |
| "reset" | reset position + attitude |

---

## Subsystem status readouts

Every one of the **31 subsystems** exposes a live status string
(`status`/`status_str`/`readiness`) — self-test asserts none can ship without
one. Examples at boot:

```
atmosphere      EARTH  P=101.3 kPa  wind=0.0 m/s  turb=0.00  solar=0 W/m2
auto_aim        SCANNING  targets=0 tracked=0 hits=0/0 (0%)
beacon          STANDBY
countermeasures remaining=12 deployed=0 active=0 defeated=0 auto-defeat=on
defense         IDLE stance=neutral punches=0 blocks=0 neutralized=0 force=300 lbs
dive            SURFACE
drones          0/4 active 4 stowed
edefense        drained=0.0 kJ ablated=0.0 mm shunted=0 burnthroughs=0
eshield         READY 100%
frame           T=0mm S=0MPa L=0/20 OK
grapple         STOWED
helmet          Z1x normal sig=-80dB
jump            READY jumps=0 max=200 ft
life_support    STANDBY (atmosphere)
maglev          OFF
maintenance     ready=100% worst=clutch_pods 100% LRUs=8
muscle          V=0.0kV F=0N STF=normal H=100%
neural          idle(0%) Q=100% loss=0.0%
parachute       PACKED
plasma          idle F=0.00N water=100% pellets=61yr
power           SOC=100% V=48.0V P=0W H=0W
regen           0.0Wh recovered
rtos            up=0s tasks=20 checks=0 viruses=0 failovers=0 restores=0
space           ATMOSPHERE
stealth         OFF 0% panels=48 draw=0W RCS=1.000m2
stun            READY 50/50
thermal         AUTO skin=37.0C PCM=100% P=0W
tshield         STOWED
vision          NRM objects=0 AR=on
voice           LISTENING 33/50 words
ai_copilot      AI:standby | rec:hold
```

The HUD additionally drives **27 status lights** every tick (power, thermal,
seal, turbines, wings, iris seal, plasma, defense, aim-lock, OS failover,
stealth, vision, grapple, parachute, drones, countermeasures, eshield,
tshield, stun, beacon, maglev, muscle, neural, life support, helmet seal,
frame stress, power shed) — self-test asserts every declared light is
assigned inside `SuitState.update`, so an indicator can never silently go dead.

---

## Environments

11 presets selected with `[` `]`; every field (density, gravity, O₂ fraction,
temperature, wind, weather, planet, radiation, plasma density/speed,
surface density) feeds the models:

| Environment | ρ (kg/m³) | g | O₂ frac | T (K) | Wind | Weather | Notes |
|---|---|---|---|---|---|---|---|
| Earth Sea Level | 1.225 | 9.81 | 1.00 | 288.15 | 3 | clear | baseline |
| Earth 10k ft | 0.905 | 9.78 | 0.74 | 268.15 | 8 | clear | |
| Earth 28k ft | 0.418 | 9.72 | 0.34 | 228.15 | 15 | clear | |
| Earth Stratosphere | 0.089 | 9.71 | 0.21 | 216.65 | 30 | clear | 1e3 iono attenuation |
| Mars Surface | 0.020 | 3.71 | 0.016 | 210.65 | 12 | dust | clutch flies on CO₂ |
| Mars Orbit | 0 | 3.71 | 0.016 | 2.7 | 0 | vacuum | plasma 6/cm³ @450 km/s |
| Titan Surface | 5.270 | 1.35 | 4.3 | 93.65 | 0.5 | methane | T/W ~2.4 — best flying in the set |
| Titan Lakes | 450.0 | 1.35 | 4.3 | 93.65 | 0.3 | methane | liquid CH₄ swim |
| Underwater 100m | 997.0 | 9.81 | 0 | 283.15 | 0.3 | current | seawater |
| Deep Ocean 4km | 1029.0 | 9.81 | 0 | 275.15 | 0.1 | current | seawater |
| Vacuum/Space | 0 | 0 | 0 | 2.7 | 0 | vacuum | plasma sail regime |

`WEATHER_TYPES` (8) validates every weather tag; unknown tags degrade to
`clear` instead of silently skipping weather physics.

---

## Verifying a change

The self-tests assert real behavior, so run them after any edit:

```sh
python flysuit.py --selftest      # must end with "ALL TESTS PASSED"
python flysuit.py --impact-test   # absorption must VARY by threat (ballistic limit)
python flysuit.py --stress-test   # must be stable across all 11 environments
python flysuit.py --extreme-test  # must end with "ZERO FAILURES"
```

**Design rule:** every part operates mechanically from math — nothing is faked.
Material dimensions and ratings are constants; *outcomes* (impact absorption,
buoyancy, radiation dose, punch force, hit resolution, fuel burn, induced drag,
g-load, rotational dynamics) are always computed from physics equations. If
you add a system, trace every number it reports back to a formula and add a
self-test assertion for it.

### What `--selftest` asserts

Build sweep at 3 pilot corners · 16-showcase integration · one-size
conformance + optimized-band telescoping margins · 100% scale crown-to-ground ·
T/W, impact absorption (600 k PSI → 158 PSI felt), jump energy, battery
runtime · plasma thrust in air/vacuum/underwater + Mars/Titan · voyage
endurance + interplanetary destination table · iris-seal regime gating ·
PART_DB structure + counts vs DIMS + mesh bijection · RTOS tasks, CRC
failover, virus injection/recovery · auto-aim lock + firing solution ·
defense-AI cycle · jump assist · 6-DOF integration · dive NDL/deco/ppO₂/
narcosis/auto-trim · radiation dose + MMOD + RCS depletion · eshield
activation/cooldown/sentry · stealth cycle · grapple hold + 500 lb
line-break physics · parachute mass-scaled descent · thermal soak · EMP
shielding · self-heal vascular repair · muscle-force jump integration ·
pulse-glide autopilot cycle · component weight budget (15 items = 23.0 kg
exactly) · anatomy-basis measurements · performance-spec audit vs live model
· **DIMS coverage audit** — all 303 spec keys referenced in code ·
armor-layer stack sum · air-MHD + vacuum-jet energy/momentum conservation ·
9 material-property plausibility checks · 10 failure-mode cascades ·
**voice coverage** — all 33 phrases recognize + dispatch + negative reject ·
environment table integrity · subsystem status coverage (31) · HUD light
coverage (27 driven).

### What `--extreme-test` asserts

| Check | Content |
|---|---|
| 1 | 160-build fit-envelope sweep (in/around/beyond rated sizes, NaN/Inf heights + weights) — no exceptions, all clamped, all geometry finite |
| 2 | 20 telescoping `set_pilot` calls — travel limits never exceeded |
| 3 | 264 poisoned-state `update` + 6-DOF `step` calls (NaN/Inf injection × 11 envs × 3 pilots × degenerate dt incl. 0/neg/inf/NaN) + recovery tick — state finite throughout |
| 3b | 24 targeted corruption probes (bad env_idx types, poisoned RTOS clock, poisoned scrubber, GUARD-state degenerate dt, 0-g/NaN/huge jumps, pathological sail/brake args, plasma at degenerate ρ/v, corrupted dive tissue/depth) |
| 4 | 22 env×pilot physics sweeps — every query finite |
| 5 | Ballistic-limit sanity — absorption varies with threat; NaN impact input sanitized |
| 6 | OBJ+MTL export round-trip at both envelope corners + headless render at corners across all explode levels |

### Physics models (mechanically accurate)

| Domain | Equation / Model |
|---|---|
| Plasma clutch (air) | Merged ~1 m coupling region; F = (2ρAP²)^⅓·coupling·k_seed actuator-disk + real plasma diagnostics (ω_p, v_A, λ_D, Rm, β); works on any gas — no O₂ needed (Mars/Titan OK) |
| Plasma clutch (water) | Pods sealed; 6 steering discs run ducted waterjets on ambient water — power-limited momentum flux + cavitation derate |
| Plasma clutch (space) | Vacuum: water plasma jet (F = 2ηP/v_exh, Isp ~2550 s, 4.9 km/s Δv on 22 kg H₂O); ambient plasma: frozen-in magnetic sail; solar wind boost + ISM braking legs |
| Fusion power | D-He3 aneutronic, 55 kW continuous, sponge-lattice core, ~85 yr pellet endurance (primary + backup) |
| Water reaction mass | Drawn only by the vacuum jet (2 kg suit tank → 20 L reserve bladder); total mass decreases as it depletes |
| Wing lift | L = ½ρv²S·CL(α), linear to stall, flat-plate post-stall; membrane area scales with stature² |
| Induced drag | CDi = CL²/(π·AR·e), Oswald e=0.85, aspect ratio from span²/area |
| Parasitic drag | D = ½ρv²·Cd·A, environment-specific coefficients; frontal area = ¼ Mosteller BSA of the live pilot |
| Impact pressure | PSI = (F/A)/6895 (Pa→PSI), F = m·v/t_stop; frame telescoping extends t_stop 20% |
| G-load | \|a_total\| / g (full 3-D acceleration vector, not just vertical) |
| Rotational dynamics | α = τ/I; inertia tensor + moment arm scale with live pilot mass·stature² |
| Atmosphere | ISA troposphere + stratosphere lapse rate; Mars/Titan/ocean/space models |
| Parachute | Terminal velocity √(2W/ρC_dA) from live suspended mass — 4.8 m/s at min pilot, 9.7 m/s at max (honest heavy-pilot limit) |
| Grapple | Line tension = hang + centripetal + hoist terms; enforced 500 lb tensile limit — line breaks, refuses re-fire |
| Dive trim | Pilot-scaled hull displacement + SAC rate; BCD solver incl. ballast; heavy pilot exceeds trim capacity honestly |
| Thermal | Pilot-scaled metabolism, body thermal capacitance, Mosteller surface area |
| Metabolism/CO₂ | Scaled by live pilot mass ratio |
| EMP / energy defense | Faraday attenuation, ceramic reflectivity + ablation, RMF shunt, Argent-drain recharge |

---

## Repo layout

| Path | Purpose |
|---|---|
| `flysuit.py` | The entire simulator (~18,400 lines: specs, systems, physics, renderer, tests, showcases, audits) |
| `showcase.py` | Compatibility shim — re-exports the showcase builders integrated in `flysuit.py` |
| `overview.md` | Architecture + engineering model deep-dive |
| `Projectgoal.md` / `officialgoal.md` | Original design brief and goals |
| `flysuit.obj` / `flysuit.mtl` | Exported 3D model (regenerate with `--export-obj`) |
| `_render_check.py` / `_parts_bbox.py` | Headless render + bounding-box utility scripts |
| `ReferenceCode/` | Prior software-renderer projects used as the engine reference |

---

## Complete specification registry

Every one of the **303 `DIMS` keys** is listed below, grouped by subsystem.
These are the design parameters consumed by the live model — the self-test's
coverage audit asserts each is referenced in code, so nothing here is cosmetic.

### Pilot, anatomy & fit

| Key | Value |
|---|---|
| `pilot_min_height_m` | 1.52 (rated subfit floor, 5′0″) |
| `pilot_max_height_m` | 2.09 (rated ceiling, 6′10″) |
| `pilot_opt_min_m` | 1.88 (optimized band start, 6′2″) |
| `pilot_opt_max_m` | 2.08 (optimized band end, ~6′10″) |
| `pilot_min_weight_kg` | 29.5 (65 lb) |
| `pilot_max_weight_kg` | 190.5 (420 lb) |
| `ref_height_m` | 1.98 (design-center pilot, 6′6″) |
| `ref_weight_kg` | 79.4 |
| `ref_shoulder_m` | 0.46 |
| `ref_chest_m` | 0.30 |
| `ref_hip_m` | 0.38 |
| `ref_arm_len_m` | 0.63 |
| `ref_leg_len_m` | 0.85 |
| `ref_torso_len_m` | 0.52 |
| `anatomy_ref_height_m` | 1.73 (basis height for BODY_PROPORTIONS) |

`BODY_PROPORTIONS` (fractions of height at the 1.73 m basis): head 0.125 · neck
0.045 · torso 0.285 · upper arm 0.195 · forearm 0.155 · hand 0.11 · hip 0.048 ·
thigh 0.265 · shin 0.26 · foot 0.155 · shoulder_w 0.335 · hip_w 0.205 ·
chest_d 0.215 · sole 0.045.

### Armor stack & materials

| Key | Value |
|---|---|
| `inner_thick_mm` | 0.8 (spandex-nylon sensor suit) |
| `middle_thick_mm` | 2.4 (DEA+STF muscle) |
| `middle_stop_psi` | 80,000 base |
| `middle_leak_frac` | 0.06 |
| `intermediate_thick_mm` | 1.2 (auxetic) |
| `outer_thick_mm` | 2.5 (graphene-UHMWPE) |
| `outer_max_psi` | 600,000 |
| `outer_nij_level` | IV |
| `outer_heat_tol_c` | 2,000 |
| `outer_laser_tol_kw` | 10 |
| `outer_stop_psi` | 520,000 |
| `outer_leak_frac` | 0.015 |
| `faraday_thick_mm` | 0.05 (embedded EMP mesh) |
| `total_thick_mm` | 7.5 |
| `dea_sublayers` | 3 |
| `dea_contraction_ms` | 20 |
| `dea_strain_pct` | 50 |
| `stf_max_psi` | 150,000 |
| `stf_stiffen_mult` | 1.9 |
| `muscle_boost_x` | 15 |
| `fiber_density_jump_x` | 5 |
| `fiber_density_lift_x` | 2 |
| `punch_kinetic_chain_x` | 3.0 |
| `auxetic_poisson` | −0.75 |
| `auxetic_cell_type` | REC-Star |
| `auxetic_rel_density` | 0.20 |
| `auxetic_energy_abs_pct` | 65 |
| `coating_type` | NiP + graphene NP |
| `ceramic_type` | alumina-aerogel hybrid |
| `inter_stop_psi` | 150,000 |
| `inter_leak_frac` | 0.03 |
| `body_impact_area_m2` | 1.5 |

### Frame & telescoping

| Key | Value |
|---|---|
| `frame_tube_od_mm` | 22.0 |
| `frame_tube_id_mm` | 18.0 |
| `frame_node_count` | 64 |
| `frame_pivot_count` | 20 |
| `frame_actuator_count` | 14 |
| `frame_strap_count` | 24 |
| `frame_material` | Toray T1100G carbon fiber |
| `frame_node_material` | Ti-6Al-4V (Markforged Metal X) |
| `torso_telescope_mm` | 355.6 (±14″) |
| `limb_telescope_mm` | 254.0 (±10″) |

### Helmet, visor & life support

| Key | Value |
|---|---|
| `helmet_od_mm` | 225.0 (scales with head) |
| `helmet_wall_mm` | 5.0 |
| `helmet_weight_kg` | 1.0 |
| `visor_thickness_mm` | 12.0 |
| `visor_max_force_lbs` | 15,000 |
| `visor_zoom_optical` | 10× |
| `visor_zoom_digital` | 100× |
| `visor_fov_deg` | 180 |
| `seal_depth_m` | 1,000 |
| `seal_vacuum` | True |
| `co2_scrub_hours` | 24 |
| `o2_capacity_hours` | 24 |
| `life_recycler_power_w` | 250 |
| `life_sabatier_temp_c` | 300 |
| `life_electrolysis_o2_lpm` | 3.5 |
| `life_water_recycle_pct` | 98 |
| `life_contaminant_filter` | catalytic_oxidizer |
| `life_backup_o2_kg` | 0.5 |
| `life_backup_lioh_kg` | 0.3 |

### Propulsion — RMF plasma clutch

| Key | Value |
|---|---|
| `turbine_count` | 72 |
| `turbine_d_mm` | 25.0 |
| `turbine_len_mm` | 90.0 |
| `turbine_rpm_max` | 45,000 (field-ripple equivalent) |
| `turbine_thrust_dry_lbf` | 14 (legacy jet-frame rating) |
| `turbine_thrust_ab_lbf` | 23 |
| `turbine_bypass_ratio` | 6 |
| `turbine_bearing` | active magnetic (RMF disc) |
| `turbine_sfc_dry_lb_lbh` | 0.0 (no fuel in air) |
| `turbine_sfc_ab_lb_lbh` | 0.0 |
| `turbine_backpack` | 30 |
| `turbine_forearm` | 12 |
| `turbine_calf` | 12 |
| `turbine_thigh` | 12 |
| `turbine_helmet` | 6 |
| `turbine_vector_deg` | 120 |
| `turbine_seal_s` | 2.0 (iris) |
| `turbine_cartridge_s` | 30.0 (bayonet swap) |
| `pod_design_life_h` | 5,000 |
| `pod_lru_swap_min` | 8.0 |
| `fuel_type` | H₂O reaction mass |
| `fuel_density_kg_l` | 1.0 |
| `fuel_capacity_l` | 20.0 |

### Propulsion — plasma discs, fusion & space drive

| Key | Value |
|---|---|
| `plasma_disc_d_mm` | 25.0 |
| `plasma_disc_count` | 6 |
| `plasma_palm_count` | 2 (gimbaled) |
| `plasma_heel_count` | 2 (fixed) |
| `plasma_backpack_count` | 2 (vectored) |
| `plasma_gimbal_deg` | 30 |
| `plasma_backpack_vector_deg` | 45 |
| `plasma_rmf_b0_t` | 0.8 |
| `plasma_ion_seed` | 2.8e-06 (2.8 ppm) |
| `plasma_jet_v_ms` | 25,000 (Isp ≈ 2,550 s) |
| `plasma_jet_eff` | 0.6 |
| `plasma_propellant_kg` | 2.0 (suit tank) |
| `plasma_backup_fuel_kg` | 0.2 |
| `plasma_clutch_r_m` | 1.0 (merged coupling region) |
| `plasma_lobe_ratio` | 9.4 |
| `plasma_clutch_eff` | 0.95 |
| `plasma_ab_boost` | 1.3 |
| `plasma_shunt_max_kj` | 120 (plasma-bolt defense) |
| `fusion_type` | D-He3 aneutronic compact reactor (sponge-lattice) |
| `fusion_power_kw` | 55.0 continuous |
| `fusion_pellet_g` | 500.0 |

### Wings (Archangel glide membrane)

| Key | Value |
|---|---|
| `wing_span_m` | 3.5 |
| `wing_area_sqft` | 22 (design pilot; scales with stature²) |
| `wing_ld_ratio` | 12 |
| `wing_deploy_s` | 0.5 |
| `wing_min_sink_ms` | 3.0 |
| `wing_membrane` | aerogel/graphene weave |
| `wing_q_max_pa` | 25,000 |
| `wing_rib_material` | nitinol memory alloy |

### Power & thermal

| Key | Value |
|---|---|
| `battery_type` | solid-state Li-S |
| `battery_wh` | 1,200 |
| `battery_wh_kg` | 550 |
| `battery_life_hours` | 24 |
| `piezo_harvest_pct` | 40 |
| `solar_nanofiber` | True |
| `thermal_inner_temp_c` | 37.0 |
| `thermal_range_lo_c` | −100.0 |
| `thermal_range_hi_c` | +100.0 |
| `thermal_heating_kw` | 12 |
| `thermal_cooling_kw` | 3 |

### BCI, OS & neural

| Key | Value |
|---|---|
| `bci_latency_ms` | 17 |
| `bci_type` | EEG/EMG hybrid |
| `bci_crypto` | lattice-based post-quantum |
| `bci_air_gapped` | True |
| `os_type` | SuitRTOS dual-redundant |
| `os_check_ms` | 10 |
| `os_failover_ms` | 5 |
| `os_uptime_pct` | 99.999 |
| `os_ai_model` | Vera 3.0 (Llama-3.1-70B 4-bit) |

### Performance envelope

| Key | Value |
|---|---|
| `perf_jump_vertical_ft` | 200 |
| `perf_punch_lbs` | 10,000 |
| `perf_lift_overhead_lbs` | 3,000 |
| `perf_deadlift_lbs` | 6,000 |
| `perf_push_pull_lbs` | 10,000 |
| `perf_run_mph` | 60 |
| `perf_safe_fall_ft` | 1,000 |
| `perf_aim_range_miles` | 4 |
| `perf_aim_accuracy_pct` | 98 |
| `perf_max_speed_mph` | 420 |
| `perf_climb_ft_min` | 18,000 |
| `perf_ceiling_ft` | 28,000 |
| `perf_hover_hours` | 0.35 (fusion-fed; battery-only ≈ 2 min) |

### Stealth, vision & tactical

| Key | Value |
|---|---|
| `stealth_electrochromic_panels` | 48 |
| `stealth_ir_reduction_pct` | 85 |
| `stealth_radar_reduction_db` | 15 |
| `stealth_visual_match_pct` | 92 |
| `stealth_power_draw_w` | 120 |
| `stealth_activate_s` | 0.5 |
| `stealth_cooldown_s` | 1.0 |
| `vision_modes` | normal / night / thermal / sonar / xray |
| `vision_switch_ms` | 50 |
| `vision_night_range_m` | 300 |
| `vision_thermal_range_m` | 500 |
| `vision_thermal_sensitivity` | 0.05 K |
| `vision_sonar_range_m` | 200 |
| `vision_xray_penetration_mm` | 50 |
| `vision_ar_overlay` | True |
| `grapple_range_m` | 50 |
| `grapple_line_diameter_mm` | 3 |
| `grapple_tensile_lbs` | 500 (enforced — line breaks) |
| `grapple_winch_speed_mps` | 8 |
| `grapple_anchor_force_n` | 5,000 |
| `grapple_deploy_s` | 0.3 |
| `grapple_weight_kg` | 0.5 |
| `parachute_canopy_sqft` | 300 |
| `parachute_deploy_s` | 1.5 |
| `parachute_descent_mps` | 7 (nominal; live = √(2W/ρC_dA)) |
| `parachute_min_alt_m` | 15 |
| `parachute_auto_deploy` | True |
| `parachute_weight_kg` | 0.8 |
| `drone_count` | 4 |
| `drone_weight_g` | 50 each |
| `drone_flight_time_min` | 15 |
| `drone_range_km` | 2 (link-loss auto-return) |
| `drone_speed_mph` | 35 |
| `drone_camera` | 4K30 |
| `drone_deploy_s` | 2.0 |
| `drone_weight_kg` | 0.2 total |
| `cm_count` | 12 |
| `cm_types` | flare / chaff / decoy |
| `cm_deploy_ms` | 50 |
| `cm_effectiveness_pct` | 85 |
| `cm_burn_time_s` | 4.5 |
| `cm_decoy_radius_m` | 30 |
| `cm_weight_kg` | 0.3 |
| `eshield_max_charge_kj` | 50 |
| `eshield_recharge_kj_s` | 5 |
| `eshield_activate_ms` | 20 |
| `eshield_duration_s` | 3.0 |
| `eshield_cooldown_s` | 8.0 |
| `eshield_effectiveness_pct` | 95 |
| `eshield_radius_m` | 0.8 |
| `eshield_weight_kg` | 0.4 |
| `regen_impact_pct` | 15 |
| `regen_landing_wh` | 2.0 |
| `regen_combat_wh` | 0.5 |
| `regen_brake_pct` | 8 |
| `tshield_height_m` | 0.9 |
| `tshield_width_m` | 0.6 |
| `tshield_weight_kg` | 2.0 |
| `tshield_nij_level` | IV |
| `tshield_max_psi` | 600,000 |
| `tshield_deploy_s` | 0.4 |
| `tshield_folds_flat` | True |
| `stun_voltage_v` | 50,000 |
| `stun_current_ma` | 1.3 |
| `stun_pulse_s` | 5.0 |
| `stun_charge_s` | 1.5 |
| `stun_range_m` | 0.3 |
| `stun_shots` | 50 |
| `stun_weight_kg` | 0.1 |
| `heal_vascular_channels` | 200 (8-region concurrency) |
| `heal_repair_pct_s` | 25 |
| `heal_structural` | True |
| `heal_polymer_ml` | 50 |
| `heal_tiers` | 3 |
| `maglev_adhesion_n` | 1,200 per pad (2,400 N hold) |
| `maglev_power_w` | 80 |
| `maglev_max_speed_mps` | 3.0 |
| `maglev_activate_s` | 0.5 |
| `maglev_weight_kg` | 0.3 |
| `voice_vocab_size` | 50 |
| `voice_latency_ms` | 80 |
| `voice_accuracy_pct` | 97 |
| `voice_subvocal` | True |
| `beacon_freq_mhz` | 406.0 (COSPAS-SARSAT) |
| `beacon_homing_mhz` | 121.5 |
| `beacon_battery_h` | 72 |
| `beacon_range_km` | 20,000 |
| `beacon_gps_accuracy_m` | 5 |
| `beacon_weight_kg` | 0.1 |

### Underwater jet & dive

| Key | Value |
|---|---|
| `uw_jet_thrust_lbf` | 35 per disc |
| `uw_jet_max_speed_mph` | 40 |
| `uw_jet_intake_mm` | 25 |
| `uw_jet_activate_s` | 0.8 (spool) |
| `uw_jet_cavitation_pct` | 15 derate |
| `dive_bcd_volume_l` | 30.0 |
| `dive_ballast_kg` | 6.0 |
| `dive_max_depth_m` | 1,000 |
| `dive_gas_liters` | 2,400 |
| `dive_sac_lpm` | 18 (× mass ratio) |
| `dive_fo2` | 0.21 |
| `dive_ppo2_max_ata` | 1.6 |
| `dive_ascent_rate_max_m_min` | 9.0 |
| `dive_narcosis_depth_m` | 30.0 (EAD) |
| `dive_tissue_compartments` | 6 |

### Space systems

| Key | Value |
|---|---|
| `rad_shield_g_cm2` | 12.0 areal |
| `rad_gcr_msv_day` | 0.66 |
| `rad_leo_msv_day` | 0.3 |
| `rad_career_limit_msv` | 1,000 |
| `rad_spe_warn_msv_h` | 5.0 |
| `rad_field_deflect_pct` | 45 |
| `rcs_propellant_kg` | 2.0 GN₂ |
| `rcs_isp_s` | 65 |
| `rcs_steam_isp_s` | 45 (water fallback) |
| `rcs_thrust_lbf` | 2.0 |
| `rcs_thruster_count` | 12 |
| `mmod_shield_layers` | 6 (Whipple) |
| `mmod_flux_m2_s` | 3e-06 |
| `mmod_exposed_area_m2` | 2.0 |
| `mmod_sheath_cutoff_j` | 300 |
| `mmod_rated_j` | 5,000 |
| `mmod_heal_pct_h` | 0.1 |
| `laser_reflect_pct` | 92 |
| `laser_ablate_j_cm2` | 450 |
| `energy_drain_eff` | 0.35 (Argent-drain coupling) |
| `blast_att_kpa` | 700 |

### Weight budget (sums to `weight_total_kg` = 23.0 exactly — audited)

| Item | kg | Item | kg |
|---|---|---|---|
| `weight_frame_kg` | 1.8 | `weight_wings_kg` | 1.0 |
| `weight_inner_kg` | 0.3 | `weight_fuel_system_kg` | 1.0 |
| `weight_faraday_kg` | 0.2 | `weight_thermal_kg` | 0.5 |
| `weight_fibers_kg` | 1.8 | `weight_power_kg` | 2.2 |
| `weight_auxetic_kg` | 0.6 | `weight_plasma_kg` | 5.6 |
| `weight_armor_kg` | 1.5 | `weight_electronics_kg` | 0.4 |
| `weight_helmet_kg` | 1.0 | `weight_enhancements_kg` | 0.8 |
| `weight_turbines_kg` | 4.3 | **Total** | **23.0** |

### LRU design lives (`lru_design_life_h` — 8 line-replaceable units)

| LRU | Design life |
|---|---|
| `clutch_pods` | 5,000 h |
| `water_pumps` | 8,000 h |
| `scrubber_bed` | 500 h |
| `electrolyzer` | 4,000 h |
| `battery_pack` | 2,000 h |
| `fusion_cell` | 175,000 h (~20 yr) |
| `suit_shell` | 50,000 h |
| `drone_airframe` | 500 h |

### Interstellar destination table (`MISSION_DESTS`)

| Destination | Distance |
|---|---|
| Alpha Centauri | 4.367 ly |
| Barnard's Star | 5.96 ly |
| Wolf 359 | 7.86 ly |
| Sirius | 8.6 ly |
| TRAPPIST-1 | 40.7 ly |
| Galactic Centre | 26,670 ly |
| Across the Milky Way | 100,000 ly |

(Computed mission legs — solar-wind boost to ~30 km/s cruise, ISM magsail
braking — are printed by the plasma showcase and cross-checked in self-test.)

### Showcase inventory (16 integrated views)

| # | Showcase | Scale |
|---|---|---|
| 1 | INNER SUIT | fiber/molecular |
| 2 | DEA-STF MUSCLE FIBER | micro |
| 3 | OUTER ARMOR | molecular/layer |
| 4 | RMF PLASMA-CLUTCH POD | component |
| 5 | HELMET | component |
| 6 | DEA POLYMER CHAIN | atomic (1e7×) |
| 7 | STF NANOPARTICLES | nano (1e6×) |
| 8 | AUXETIC UNIT CELL | structural (200×) |
| 9 | GRAPHENE LATTICE | atomic (1e8×) |
| 10 | UHMWPE MOLECULAR CHAIN | atomic (1e7×) |
| 11 | CERAMIC GRAIN STRUCTURE | nano (5000×) |
| 12 | CFRP FRAME | component |
| 13 | ARCHANGEL GLIDE MEMBRANE | compact glide suit |
| 14 | POWER SYSTEM | component |
| 15 | NEURAL BCI + AI | component |
| 16 | PLASMA THRUSTERS + FUSION CELL | full detail |
