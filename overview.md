# Mjalnor'MV1.17 — Engineering & Architecture Overview

This document explains *how the suit works* in the simulator: the three operating
modes, the layered architecture, every subsystem, and the physics that ties them
together. For install/run/controls see [README.md](README.md).

The model is a **single-file build**: [flysuit.py](flysuit.py) (~18,400 lines)
contains the simulator core (specs, systems, physics, renderer, tests) *and*
the 16 detailed component showcases with 3D meshes, math proof, and true-scale
specifications (integrated from the former showcase.py, which remains only as
a compatibility re-export shim). The code is organized top-to-bottom as:
**`DIMS`** (the SI/mm specification — the single source of truth, 303 keys) →
**`PART_DB`** (rich per-part metadata: blueprint, build details, connectors,
materials, performance, sub-components) → **`ENVIRONMENTS`/`WEATHER_TYPES`**
→ **system classes** → **`SuitState`** (owns and steps every subsystem each
tick) → **`SuitPhysics`** (6-DOF integrator) → **renderers / HUD** → **`App`**.

The suit's visual design is **smooth and aerodynamic** with a sealed, continuous
armor skin. The torso and limbs use smooth elliptical cross-section primitives
(`_smooth_torso`, `_smooth_limb`) with high segment counts (24 torso, 20 limbs)
for a form-fitting, contoured surface. The four armor layers use **accumulative
stacking** — every layer shares one base profile and sits exactly its
cumulative thickness above the pilot body (0.8 mm sensor skin → +2.4 mm DEA-STF
muscle → +0.05 mm Faraday mesh → +1.2 mm auxetic → +2.5 mm armor = ~7.5 mm
total standoff, verified by self-test), so the suit is ultra-slim with no dead
air between layers. Major armor panels (chest, back, abdomen) are
**_conformal_plate** grids wrapped onto the shell surface rather than flat
slabs. Recessed panel-line grooves, subtle color variation between sections,
shoulder pauldrons, bicep/tricep contour panels, thigh/shin detail plates,
smooth joint-cover spheres at shoulders and hips, and an aerodynamic helmet
(225 mm OD nominal — it telescopes with the pilot's head) with wraparound
visor and cheek guards complete the look.

**The model is a 100%-to-scale digital prototype.** `body_segments()` normalizes
the standing chain (head→neck→torso→hip→thigh→shin→sole) so the body's
crown-to-ground span equals `pilot_height` *exactly* at any size in the
1.52–2.09 m fit envelope — verified by self-test. The assembled suit adds only
documented clearances (~24 mm: helmet dome over the crown + armored boot sole).
The gauntlets cover wrist-to-fingertip (`seg["hand"]`), the inner layer covers
hands and feet, and `build_suit` clamps pilot dimensions to the rated envelope
(1.52–2.09 m, 29.5–190.5 kg) so out-of-range input cannot produce an
out-of-spec suit.
The Archangel glide membrane is a compact **gliding suit — part of the suit
itself, not literal wings**: two arm-to-body membranes and one leg-to-leg
membrane unfold from the armor seams (cape-like) and tension along nitinol
battens; stowed, everything folds flat against the back armor (<25 mm).

---

## The three suit modes

The suit is a single machine that adapts to its environment. Mode is determined
each frame from throttle, wings, and ambient density (`SuitState.flight_mode`).

### 1. Combat flight suit
VTOL and cruise flight on the **RMF plasma-clutch drive** (Main.py
Gman's-117 physics): 72 flush coil pods (25 mm dia discs on the armor
surface, within the 1-inch / 25.4 mm profile — 30 backpack + 12 forearm +
12 calf + 12 thigh + 6 collar ring) project overlapping lobes that
merge into a single ~1 m effective coupling region. The clutch ionises a
~2.8 ppm seed of the ambient air and collisionally drags the full air column:
F = (2ρAP²)^⅓·coupling·k_seed (actuator disk), where the coupling comes from
real plasma diagnostics (electron/ion plasma frequency, Alfvén speed, Debye
length, Spitzer conductivity, magnetic Reynolds number, plasma β). The drive
needs **no oxygen** — it ionises whatever gas is present (CO₂ on Mars, N₂/CH₄
on Titan), so the same pods fly on other planets. Thrust scales with the
cube-root of density×power², so thin air is honestly marginal (T/W 0.70 at
stratospheric density). The Archangel
glide membrane — a compact gliding suit that is part of the suit (3.5 m
deployed span, 22 ft², 12:1 glide; arm-to-body + leg-to-leg membranes with
wrist-to-tip sleeves) — provides efficient range extension with proper induced
drag (CDi = CL²/(π·AR·e), Oswald e = 0.85); the deployed membrane area scales
with pilot stature² so lift, sink rate, and solar harvest all follow the pilot.
The jetpack flight profile is a
**pulse-climb-glide cycle** (autopilot key `Y`): the clutch pulses at reactor
overdrive, with a ballistic cutoff so the suit arrives at the top of a
200 m altitude band at ~zero climb rate), the field cuts, the membrane deploys
and glides at best L/D back down to the band floor — tracked by clutch
energy expenditure (kJ per pulse). Wing angle-of-attack is measured against the flight
path (a vertical climb is post-stall) and lift acts perpendicular to the
airflow, so climbs and glides trim like real aerodynamics. A full combat
suite includes Kalman-tracked auto-aim with ballistic firing
solutions, a martial-arts defense AI, energy and tactical shields, a recon drone
swarm, countermeasures, a non-lethal stun, and a grappling winch. Muscle fibers
and the frame distribute g-loads so the pilot survives high-g maneuvers.

### 2. Space suit
In vacuum the pod **iris covers seal** (no leaks) and propulsion shifts to the
fusion-powered **water plasma jet** (F = 2ηP/v_exh, Isp ~2550 s): a 2 kg suit
tank feeds first, then the 20 L conformal water-reserve bladder — ~4.9 km/s
delta-v for injection and maneuvers — plus a **cold-gas RCS**
with a real propellant budget (2 kg GN2); when GN2 is spent the jets fall back
to flash-steamed reserve water at lower Isp, so attitude authority lasts as
long as the water reserve. In ambient plasma (ionosphere, solar wind) the same clutch field
runs as a **magnetic sail** (frozen-in flow, Rm ≫ 1), which the mission model
integrates as a continuous-thrust boost from Earth's orbit to the 90 AU
termination shock (~20 yr, ~30 km/s cruise) and an ISM magsail braking leg —
the suit can ride the solar wind to another planet. A 55 kW **aneutronic D-He3
fusion cell** provides continuous power
(sponge-lattice core, direct energy conversion), feeding the clutch coils and
charging the battery. Pellet burn gives ~85 yr total endurance (primary +
backup) — effectively endless for any single mission, but honestly finite:
when both stores run out the cell goes dark, the clutch falls back to
battery-only thrust at a 10 C discharge limit, and a fully dead suit cannot
fly. A **radiation dosimeter** integrates galactic-cosmic-ray and
low-Earth-orbit dose with stochastic solar-particle events, attenuated by the
suit's areal shielding, and tracks cumulative mission dose vs. career limit. Life
support seals, scrubs CO₂, generates O₂, and holds cabin pressure. The
closed-loop **chemical recycling system** (fusion-powered, 250 W) provides
indefinite life support: a 4-bed molecular sieve captures CO₂, a Sabatier
reactor converts it to water (CO₂ + 4H₂ → CH₄ + 2H₂O), water electrolysis
produces breathable O₂ (2H₂O → 2H₂ + O₂), and a 98%-efficient water recycler
reclaims condensate and sweat. Backup compressed O₂ (24 h) and LiOH canister
(24 h) cover emergencies when the fusion cell is offline. Micrometeoroids are
handled by a **three-tier active defense**: the energized RMF clutch field
maintains a plasma sheath that vaporizes sub-mm grains (≲300 J), the energy
shield's sentry mode discharges its capacitor to intercept mm-class grains up
to ~45 kJ (≈4 mm at 30 km/s), and a 6-layer **Whipple bumper** with
self-sealing, fusion-powered regeneration absorbs what survives — penetrations
punch a real seal leak into the life-support loop. The same energized field
doubles as a **magnetic radiation shield**, deflecting ~45% of the
charged-particle flux on top of the areal mass shielding.

### 3. Dive suit
A variable-volume **buoyancy control device (BCD)** auto-trims to neutral at any
depth, and a **decompression computer** models inert-gas loading across six
Bühlmann-style tissue compartments to produce the no-decompression limit, deco-
stop ceiling, ascent-rate warnings, oxygen-toxicity (ppO₂ / CNS clock), nitrogen
narcosis (equivalent narcotic depth, `NARC!` flag past the 30 m rating), and
depth-scaled gas consumption. Hull displacement scales with pilot mass
(~20 L shell + pilot/1400), SAC rate scales by mass ratio, and a 6 kg ballast
trim weight is included in both the auto-trim solver and net buoyancy — so a
max-weight pilot pegs the 30 L BCD at 100 % and still nets ~−275 N, an honest
capacity limit rather than a cosmetic one. The
sealed helmet and hull are rated to 1,000 m — and the rating is *enforced*:
hydrostatic pressure is computed per ocean world (Earth seawater, Titan
methane), the margin zone to 2× depth accrues compression damage over
minutes, and beyond 2× the hull implodes catastrophically. Breaches vent
honestly too: a ruptured seal relaxes cabin pressure toward ambient at a rate
set by hole size — a full breach empties the ~20 L loop in seconds while
makeup gas can fight only a pinhole, and hypoxia is judged by O₂ partial
pressure, not just mix fraction.

---

## Layered armor architecture

Four concentric layers, **7.5 mm total** (6.9 mm materials + 0.6 mm gaps), each
a real material with real properties. A 0.05 mm Faraday mesh is embedded within
the middle layer. All layers use smooth elliptical cross-sections with high
segment counts for a sealed, form-fitting armor skin. All back-mounted components
(battery, fusion cell, water tank, AI processor, fuel pumps, fuel lines) are
conformal flush-mounted panels within the 1-inch (25.4 mm) profile. The outer
shell features flush detail panels (bicep, tricep, forearm, thigh, shin), pronounced shoulder
pauldrons, smooth joint covers at shoulders and hips, and recessed panel-line
grooves between armor sections with subtle color variation.

| Layer | Thickness | Material | Function |
|---|---|---|---|
| Inner | 0.8 mm | spandex-nylon sensor suit | EEG/EMG sensing, comfort |
| Middle | 2.4 mm | tripled DEA + STF muscle fibers | strength (15×), impact stiffening |
| Intermediate | 1.2 mm | foam-filled auxetic metamaterial | energy densification |
| Outer | 2.5 mm | graphene-UHMWPE panels (NIJ IV+) | ballistic + thermal + laser |

**Impact model (real ballistic limit).** A peak contact pressure propagates
through the stack layer-by-layer: each layer defeats pressure up to a *stopping
capacity* derived from its material (plastic work / densification), and any excess
penetrates while a small back-face-coupling fraction of the stopped pressure still
transmits. The three layers sum to ~750 k PSI, so the suit **fully arrests up to a
.50 BMG (~600 k PSI)** and is progressively **overmatched by 20 mm+ autocannon**.
STF shear-stickening and armor integrity feed back into the layer capacities, so
absorption genuinely varies with threat, damage, and state — see `--impact-test`.

---

## Subsystem inventory

All of these are classes in `flysuit.py`, instantiated and stepped by `SuitState`.
Every subsystem exposes a live `status`/`status_str`/`readiness` string —
self-test asserts all 31 do, so a subsystem can never be invisible to the pilot.
27 HUD status lights are assigned every tick and audited the same way.

**Core suit**
- `FrameSystem` — CFRP telescoping exoskeleton, 64 Ti nodes, 14 Maxon actuators, 20 pivots, 24 straps, 64 load cells; magneto-rheological dampers lock joints above 4 g; `set_pilot` auto-sizes within ±355.6 mm torso / ±254.0 mm limb travel and reports actuator-stroke fraction.
- `MuscleFiberSystem` — dielectric-elastomer actuators (voltage → Maxwell-stress force) with shear-thickening-fluid impact stiffening; drives strength, jump, and punch. Full anatomical fiber mesh: pecs, abs, obliques, lats, erector spinae, trapezius, deltoids (3 heads), biceps, triceps, forearm flexors/extensors, quads (4 heads), hamstrings, glutes, calves, tibialis, soleus.
- `ThermalManagementSystem` — glycol loop + phase-change material + resistive backup; holds skin temp across ±100 °C. Metabolic heat, body thermal capacitance, and wetted area all scale from the live pilot's mass and Mosteller BSA.
- `NeuralInterfaceSystem` — EEG/EMG BCI (64ch EEG + 16ch EMG), band-power intent decoding, post-quantum crypto handshake, <17 ms latency.
- `PowerManagementSystem` — solid-state Li-S battery (1,200 Wh @550 Wh/kg), piezo/solar harvesting (solar tracks deployed membrane area), voltage sag, load shedding (critical systems preserved), D-He3 fusion generation (55 kW continuous).
- `PlasmaThrusterSystem` — the sole flight drive (Main.py plasma-clutch model): 72 flush RMF pod lobes merge into a ~1 m coupling region; 6 steering discs (2 palm gimbaled + 2 heel fixed + 2 backpack vectored). Regimes: air MHD ion-drag on any gas, underwater ducted waterjets, vacuum water plasma jet (Isp ~2550 s, ~4.9 km/s Δv), ambient-plasma magnetic sail; solar-wind boost + ISM braking mission profiles; 2 kg water tank + 20 L reserve, ~85 yr pellet endurance. Per-pod health degrades gracefully (a failed pod costs ~1.4% thrust, never the drive); pods are 8-min bayonet LRU swaps; MMOD spall can arc a coil.
- `LifeSupportSystem` — fusion-powered closed-loop chemical recycling (Sabatier + electrolysis + water recycler), backup O₂/LiOH, seal pressure, humidity, CO₂ scrubber cycles; auto-activates in vacuum/underwater; `SEAL-OK`/`SEAL-LOW` readout.
- `HelmetSystem` — vacuum/underwater seal, zoom visor (10× optical / 100× digital), comms signal, night/thermal vision flags, recording.
- `FuelSystem` — 20 L conformal Kevlar **water reserve** bladder feeding the vacuum plasma jet (suit tank first), dual-redundant electric pumps, self-sealing coating, AN-8 lines; reserve mass depletes in flight and reduces total mass.
- `SuitRTOS` — dual-redundant real-time OS: 20 scheduled tasks (17 critical), CRC integrity voting, failover, virus detection + ROM restore + purge, post-quantum crypto (Kyber-1024 handshake), and signed A/B firmware updates staged on the shadow bank with rollback.
- `MaintenanceSystem` — LRU prognostics on 8 line-replaceable units: per-unit operating hours vs design life coupled to live damage (pod wear, battery cycles, armor integrity); readiness score, next-service forecast, field swaps.

**Environment (mode) systems**
- `AtmosphericModel` — ISA Earth, Mars, Titan, ocean, and space models; wind/turbulence/gusts, solar irradiance, radio attenuation, hydrostatic pressure; weather tags validated against `WEATHER_TYPES` (unknown → `clear`).
- `DiveSystem` — BCD buoyancy auto-trim + Bühlmann decompression computer + ppO₂/CNS clock + narcosis EAD + pilot-scaled hull displacement, SAC rate, and 6 kg ballast.
- `SpaceSystem` — radiation dosimetry + magnetic field shielding, cold-gas RCS with steam fallback + Tsiolkovsky Δv, three-tier MMOD defense (plasma sheath → energy-shield sentry → Whipple bumper) with breach→leak wiring.
- `EnergyShield` — capacitor-discharge reactive shield: held barrier (activate, 95 % stoppable-energy cap, 7 s cooldown), sentry directed-discharge (hypervelocity grains), recharge/cooldown timers stepped every tick.
- `DirectedEnergyDefense` — laser fluence defense (92 % reflective ceramic + ablation), plasma-bolt shunt through RMF field + capacitor (≤120 kJ), 700 kPa blast attenuation, and Argent-drain: absorbed directed energy recharges the eshield capacitor then battery (~35 % coupling).

**Combat / "fight mode"**
- `AutoAimSystem` + `KalmanTracker` + `Target3D` — multi-target tracking with IFF transponder query on every new lock, threat-priority scoring, inbound-missile dodge vectors (time-to-closest-approach, not heading cones), ballistic firing solutions with time-of-flight lead, hit/miss bookkeeping.
- `DefenseAI` — martial-arts state machine (IDLE→ALERT→GUARD→COUNTER→NEUTRALIZED); punch force driven live by the DEA arm fibers toward the 10,000 lbf rating.
- `JumpAssist` — charged 200 ft vertical jump; charge accumulates on command, muscle leg force and local gravity fold into the trajectory, non-finite charge fizzles instead of wedging.
- `StealthMode` — electrochromic adaptive camouflage (48 panels): activation ramp, IR signature + visual match percentage, radar cross-section reduction, re-trigger cooldown.
- `TacticalShield` — deployable NIJ-rated forearm plate sized to the pilot (h×w), over-pressure handling halves protection honestly, folds flat.
- `StunSystem` — contact-range stun: voltage/current/pulse-duration specs, finite charge pool with recharge timer, range-gated firing.
- `CountermeasureSystem` — 12-cartridge dispenser (flares/chaff/holographic decoys): per-type deployment timing, decoy radius, active-ordinance tracking, auto-defeat engages only threats outside the deploy window.
- `DroneSwarm` — 4 micro-drones (50 g, 4K): launch-all or single-scout launch, link-loss auto-return at rated range/speed, deployment ramp, recall, per-drone camera/state tracking.
- `GrappleSystem` — 500 lb line, 50 m range, 8 m/s winch: anchor grip = min(tensile, anchor force); live tension model (hang + centripetal + hoist terms vs pilot+suit+fuel mass and local gravity) snaps the line past rating and refuses re-fire.

**Utility / survival**
- `VisionModeSystem` — 5 helmet modes (normal/night/thermal/sonar/x-ray) with per-mode HUD icons, switch-time ramp, thermal sensitivity and x-ray penetration specs, AR overlay; cycle or direct-select via voice.
- `EmergencyParachute` — deploy/animate/open/jettison; descent rate is computed terminal velocity √(2W/ρC_dA) of the live suspended mass vs the 7 m/s nominal spec.
- `RegenSystem` — regenerative capture: combat energy, braking deceleration, and landing-impact recovery back into the battery SOC.
- `VoiceCommandSystem` — subvocal command pipeline: 33-phrase vocabulary (50-word spec), 80 ms latency, 97 % accuracy model, longest-match-first recognition, `execute()` dispatches every phrase to the same action as its keybind, `command()` runs recognize→dispatch end-to-end; unrecognized phrases dispatch nothing.
- `EmergencyBeacon` — 406 MHz COSPAS-SARSAT + 121.5 MHz homer: GPS-accuracy fix, rated range, battery budget, activate/deactivate.
- `MaglevMode` — magnetic climb: 2×1,200 N adhesion hold (≈244 kg limit — overload detaches honestly), ferromagnetic surface detection.
- `AICopilotSystem` ("Vera 3.0") — combat assessment, recommended action, spider-sense proximity alarm, system-alert severity tracking, auto-assist flags.
- `AtmosphericModel` — also provides solar-irradiance and radio-attenuation feeds to power/comms.

**Simulation core**
- `SuitState` — the live state; `update(dt)` steps every subsystem in order, drives all 27 HUD lights, and hardens every external input (non-finite dt, corrupted dims, out-of-range env).
- `SuitPhysics` — 6-DOF translational + rotational integrator: thrust, drag, lift, gravity, buoyancy, wind, muscle assist, g-limiting; environment-aware (air / water / vacuum); every pilot-dependent property (frontal area, inertia, moment arm, impact area, suit volume, wing area) is a live function of pilot height/mass.
- `Mesh` / `Part` / `SuitRenderer` / `FlightRenderer` / `App` — geometry, HUD, and the interactive shell. The full suit model is 1,031 meshes / 28,934 vertices across 14 parts. The renderer uses a y-up coordinate system with smooth elliptical mesh primitives (`_smooth_torso`, `_smooth_limb`, `_helmet_shell`, `_pauldron`, `_sphere`), flush armor plates (`_armor_plate`), anatomical muscle fiber strands (`_fiber_strand`), and recessed panel lines (`_panel_line`). The UI is fully responsive: all panels scale with window size via `pygame.VIDEORESIZE`, and help/instructions/showcase/info panels support mouse-wheel scrolling. Each of the 14 parts includes inline `MATH PROOF` sections in its spec list, documenting weight derivation, compactness (≤25.4 mm), and physics validation.
- **Showcase system** (integrated in `flysuit.py`) — 16 detailed component views: inner suit, middle layer, outer armor, clutch pod (bayonet cartridge + iris seal), helmet, DEA/STF/auxetic/graphene/UHMWPE/ceramic atomic-scale, CFRP frame, Archangel glide membrane, plasma thrusters + fusion cell, power system, and neural BCI + AI co-pilot. Each includes 3D meshes, sub-components, math proof, and true-scale reference dots.

---

## Pilot-scaled physics (live, not constants)

Everything that depends on pilot size reads the live pilot — a mid-flight
`set_pilot` can never leave stale physics, and corrupted dimensions fall back
to the reference pilot:

- **Frontal area** — ¼ Mosteller body-surface estimate from live height+mass
  (0.28 → 0.83 m² across the envelope) feeding drag and wind-gust force.
- **Suit displacement / hull volume** — mass-proportional (0.05 → 0.20 m³)
  for buoyancy, plus the dive system's pilot-scaled hull.
- **Inertia tensor & moment arm** — I ∝ m·L², arm = 0.28 × stature; a 190 kg
  pilot carries ~2.3× the rotational inertia of a 30 kg one.
- **Impact / silhouette area** — h×√w (0.80 → 2.81 m²) for landing and
  water-entry pressure.
- **Wing area & solar harvest** — membrane area scales with stature²
  (1.20 → 2.28 m²); lift, induced drag, glide ratio, and 200 W-class
  photovoltaic harvest all track it.
- **Metabolism, thermal capacitance, breathing rate** — mass-ratio scaled
  (clamped), so O₂ draw, CO₂ production, and heat soak all follow the pilot.
- **Parachute descent** — terminal velocity of the actual suspended mass.
- **Grapple tension** — live mass+gravity; a 470 lb pilot snap-loads a 500 lb
  line under hoist, a light pilot winches fine.
- **Punch force** — DEA arm-fiber output + kinetic-chain multiplier, not a
  typed 10,000 lbf.

---

## Specification-integrity audits

The `--selftest` run ends with a battery of audits proving declared
specification data is actually consumed — a spec nobody reads fails the build:

- **DIMS coverage** — a static scan asserts every one of the 303 `DIMS` keys
  is referenced in program code (nested `lru_design_life_h` keys excepted —
  they are consumed via the dict itself).
- **Component weight budget** — the 15 `weight_*_kg` line items must sum
  exactly to `weight_total_kg` (23.0 kg).
- **PART_DB integrity** — all 14 parts carry the full 12-field record
  (blueprint, build_details, category, connectors, description, dimensions,
  icon, layer, materials, performance, sub_components, weight_kg);
  sub-component counts reconcile with DIMS (72 clutch pods, 6 plasma discs,
  frame nodes/pivots/actuators/straps); every DB key maps to a built mesh and
  vice versa.
- **Anatomy basis** — `body_segments(1.73 m)` reproduces the reference
  measurements within tolerance.
- **Performance audit** — hover draw/endurance, AB climb margin, push/pull
  rating cross-checked against the live thrust model (the 0.35 h hover spec is
  reported honestly as fusion-feed-dependent: battery-only hover ≈ 2 min).
- **Environment table** — all 11 tuples finite, weather tags ∈ WEATHER_TYPES.
- **Voice coverage** — all 33 phrases recognize + dispatch to real actions;
  an unknown phrase must dispatch nothing.
- **Status coverage** — all 31 subsystems expose a readout; all 27 HUD lights
  are assigned inside `SuitState.update`.
- **Conservation** — air-MHD and vacuum-jet energy/momentum audited; armor
  layer sum = 7.5 mm; clutch T/W in both regimes.
- **Materials** — 9 material properties checked against published bounds.
- **Failure modes** — 10 documented cascades (trigger/effect/mitigation/
  recovery).

The `--extreme-test` harness then proves none of it breaks: 160 fit-envelope
builds, 264 poisoned updates + 6-DOF steps across all 11 environments and
three pilot extremes with degenerate `dt`, 24 targeted corruption probes,
22 physics sweeps, ballistic-limit sanity, and OBJ export + headless render
at both corners — `ZERO FAILURES` is the pass criterion.

---

## Physics that couples everything

`SuitPhysics.step()` is where the systems become one machine — nothing is
cosmetic:

- **Propulsion is environment-aware.** One plasma-clutch drive, four regimes:
  in atmosphere it runs MHD ion-drag on the merged ~1 m coupling region
  (F = (2ρAP²)^⅓·coupling·k_seed — thrust falls with density cube-root, so
  stratosphere is honestly marginal; needs ambient gas, not O₂). Underwater the
  pod covers seal and the six steering discs run ducted waterjets on ambient
  water. In vacuum the covers seal and the fusion-powered water plasma jet
  draws the 2 kg tank then the 20 L reserve (~4.9 km/s Δv), with cold-gas RCS
  trim. In ambient plasma the clutch runs as a magnetic sail (frozen-in flow).
  Battery voltage sag and load shedding further reduce thrust via a (V/V_nom)²
  multiplier.
- **Water reaction mass is Tsiolkovsky-driven.** The vacuum jet consumes stored
  water at mdot = F/v_exh; `total_mass` is a live property that includes
  remaining water — so the suit gets lighter as the tank depletes, affecting
  T/W, drag, and inertia. Air and underwater flight consume no stored mass.
- **Wing aerodynamics are proper.** Lift follows L = ½ρv²S·CL(α) with linear CL
  up to stall angle then flat-plate post-stall. Induced drag is computed from
  CDi = CL²/(π·AR·e) with Oswald efficiency e = 0.85 and aspect ratio from
  span²/area — the fundamental drag penalty for generating lift. Both parasitic
  and induced drag are summed in the drag vector.
- **Impact pressure is correctly converted.** PSI = (F/A)/6895 where F = m·v/t_stop
  and A is the live pilot-scaled presented area. Frame telescoping extends
  t_stop by 20%, and STF stiffening + multi-layer absorption reduce felt
  pressure.
- **G-load uses full 3-D acceleration.** g = |a_total| / g, not just the vertical
  component — so lateral maneuvering and drag deceleration both count.
- **Rotational dynamics use torque/inertia.** Angular acceleration α = τ/I where
  τ = thrust × moment arm and I is a pilot-scaled per-axis inertia tensor.
  Neural interface signal quality modulates response speed. Aerodynamic
  damping scales with speed × density.
- **Buoyancy** underwater uses the dive system's *variable* BCD displacement, so
  the suit can hold neutral instead of simply sinking.
- **G-loads** are limited by the frame (locked joints raise the safe ceiling) to
  protect the pilot; muscle leg force adds to jumps and high-g events.
- **Pilot state feeds back:** poor neural signal quality adds control noise; high
  CO₂ / low O₂ or extreme skin temperature degrade control authority.
- **Corrupted state cannot crash it:** NaN/Inf positions, velocities, dt,
  pilot dims, and env indices are sanitized at the boundary every tick; the
  same fuzz runs in `--extreme-test`.

---

## Key specifications

| Domain | Spec |
|---|---|
| Pilot range | 1.52–2.09 m (5′0″–6′10″), optimized band 1.88–2.08 m (6′2″–6′10″) centered on a 1.98 m design pilot; 29.5–190.5 kg (65–420 lb), telescoping; crown-to-ground = pilot height exactly (100% scale, self-test verified) |
| Propulsion | 72 flush RMF plasma-clutch pods (25 mm dia, bayonet-swap, iris-sealed; 30 backpack + 12 forearm + 12 calf + 12 thigh + 6 collar ring, ~1 m merged coupling region) + 6 steering discs (2 palm gimbaled + 2 heel fixed + 2 backpack vectored) |
| Drive regimes | Air MHD ion-drag ~2.0 kN sea-level (Mars ~0.5 kN, Titan ~2.8 kN); waterjet ~2.6 kN; vacuum water jet (Isp ~2550 s, ~4.9 km/s Δv); solar-wind sail + ISM braking |
| Water | 2 kg suit tank + 20 L reserve bladder — vacuum-jet reaction mass only (air/water flight use ambient medium) |
| Wings | Glide membrane — part of the suit (not literal wings): 3.5 m deployed span, 22 ft² (2.0 m²) at design pilot scaling with stature², 12:1 glide, arm-to-body + leg-to-leg + wrist-to-tip sleeves, stows flat (cape fold) |
| Power | 1,200 Wh solid-state Li-S (550 Wh/kg) + piezo/solar harvesting + 55 kW D-He3 fusion (continuous, primary) |
| Thermal | ±100 °C, 12 kW heat / 3 kW cool, skin held ~37 °C |
| Flight envelope | 420 mph, 28,000 ft ceiling (marginal in thin air), reactor-powered hover, pulse-glide autopilot (key Y), 200 ft jump |
| Armor | 7.5 mm total stack, outer 600 k PSI (NIJ IV+), stack ballistic limit ~750 k PSI |
| Helmet | 225 mm OD nominal (telescopes with pilot head), graphene-UHMWPE shell, wraparound visor (10×/100× zoom), life support |
| Tactical | Stealth 48-panel camo; 5 vision modes; eshield 95% cap/45 kJ sentry; NIJ tac shield (pilot-sized); 4-drone swarm (range-limited link); 12-CM dispenser; contact stun; grapple 500 lb × 50 m × 8 m/s winch; beacon 406/121.5 MHz; maglev 2,400 N hold; 33-phrase voice dispatch |
| Dive | 30 L BCD, 6 kg ballast, 2,400 L gas, 1,000 m rating, 6-compartment deco model, ppO₂ ≤ 1.6 ata, 30 m narcosis warning |
| Space | 12 g/cm² shielding + RMF magnetic deflection (~45% charged flux), GCR ~0.66 mSv/day, 12 RCS jets (2 kg GN2, steam fallback on water reserve), MMOD: sheath ≤300 J + sentry ≤50 kJ + 6-layer Whipple, 1,000 mSv career limit |
| OS | SuitRTOS: 20 tasks (17 critical), CRC voting, failover, virus recovery, Kyber-1024, A/B firmware; 8-LRU maintenance prognostics |
| Environments | 11 presets: Earth SL → stratosphere, Mars, Titan, ocean 4 km, vacuum; 8 weather types |
| Showcases | 16 detailed component views (integrated in flysuit.py) with 3D meshes, math proof, true-scale refs |

---

## The all-mechanical rule

The governing rule for this project: **every part must operate mechanically from
math and reality — nothing faked.** Design *parameters* (dimensions, material
strengths, capacities) are legitimate constants. Reported *outcomes* — impact
absorption, buoyancy, radiation dose, punch force, hit resolution — must always be
computed from those parameters and the current state. The `--selftest`,
`--impact-test`, `--stress-test`, and `--extreme-test` harnesses exist to keep
the model honest — and the specification-integrity audits exist to keep the
*declaration* honest: a spec that no code consumes, a command that dispatches
nothing, a status light that never updates, or a part no mesh implements all
fail the build. Add an assertion whenever you add a system.

---

## Appendix A — Part registry (`PART_DB`, 14 parts)

Every built part carries a 12-field record — description, category, layer,
materials, weight, blueprint, icon, sub_components, dimensions, connectors,
performance, build_details — asserted complete by self-test. The 14 keys are
bijective with the built mesh parts; sub-component counts reconcile with DIMS
hardware counts (72 clutch pods, 6 plasma discs, 64 nodes / 20 pivots /
14 actuators / 24 straps).

| Part | Category | Weight | Sub-components | Build |
|---|---|---|---|---|
| `inner_suit` | Base Layer | 0.3 kg | 5 groups / 68 units (64-ch EMG grid, PCM capsules, moisture network, antimicrobial, harness) | 5 min · Easy — hand-donned |
| `faraday` | EM Shielding + Power Bus | 0.2 kg | 5 groups / 7 units (Cu-graphene mesh, 48V 200A spine bus, shoulder/hip branches) | 10 min · Medium — 4 Nm spine clamp |
| `middle_layer` | Artificial Muscle + Impact Absorption | 1.8 kg | 6 groups / 183 units (48 DEA bundles, 24 STF bladders, 8 jump clusters, CNT electrodes) | 25 min · Hard — STF injection + 200 V HV test |
| `intermediate` | Impact Distribution + Insulation | 0.6 kg | 5 groups / 39 units (REC-Star lattice, R-12 foam fill, 12 Ti junction nodes, stitching) | 15 min · Medium — pressurized foam fill |
| `outer_armor` | Ballistic Armor | 1.5 kg | 8 groups / 36 units (12 modular graphene-UHMWPE panels, ceramic strike faces, coatings) | 20 min · Medium — panel torque sequence |
| `frame` | Structural Frame | 1.8 kg | 6 groups / 202 units (64 Ti nodes, 20 pivots, 14 actuators, 24 straps, 64 load cells, MR dampers) | 45 min · Expert — node torque + actuator cal |
| `helmet` | Helmet + Life Support | 1.0 kg | 13 groups / 18 units (shell, 12 mm visor, neck ring, HUD, cheek guards, comms, seals) | 15 min · Medium — seal test + HUD align |
| `turbines` | Propulsion (clutch pods) | 4.3 kg | 11 groups / 304 units (72 pods in 5 groups, iris seals, phase steerers, CAN buses, cartridge mounts) | 60 min · Expert — 0.1° phase alignment |
| `wings` | Glide Membrane (part of suit) | 1.0 kg | 8 groups / 26 units (8 nitinol battens, 3 membrane panels, seam locks, heaters, actuators) | 30 min · Hard — tension gauge + lock test |
| `plasma` | Plasma Propulsion + Fusion | 5.6 kg | 9 groups / 17 units (6 discs + gimbal/vector mounts, fusion cell, 2 kg water tank, pellet store) | 60 min · Expert — vacuum chamber test |
| `power` | Power System | 2.2 kg | 7 groups / 16 units (2×600 Wh hot-swap packs, fusion feed, piezo harvesters, solar film, BMS) | 10 min · Easy — thumb-screw bay |
| `neural` | Neural Interface + AI | 0.3 kg | 6 groups / 6 units (EEG cap, DSP+NPU, Vera 3.0, Kyber-1024 crypto, data diode) | 20 min · Expert — key gen + calibration |
| `thermal` | Thermal Regulation | 0.5 kg | 6 groups / 12 units (Dragon Skin PCM, 6 Peltier junctions, glycol loop, radiator fins, solar lens) | 15 min · Medium — 2 bar loop test |
| `fuel_system` | Propellant Reserve (water) | 1.0 kg | 7 groups / 12 units (20 L Kevlar bladder, dual 0.5 LPM pumps, AN-8 lines, level sensors, self-seal) | 20 min · Hard — 25 Nm AN-8 + pressure test |

Suit total (structure + systems): **23.0 kg** — the 15-item `weight_*_kg`
budget is asserted to sum exactly to `weight_total_kg`. Field-serviceability:
clutch pods 8-min bayonet swaps, batteries tool-less hot-swap, all other parts
≤ 60 min with documented tools/torques.

## Appendix B — Failure-mode cascades (10 documented)

The self-test prints the full failure-mode analysis — every cascade specifies a
trigger threshold, the physical effect, the in-suit mitigation, and the
recovery path:

| # | Mode | Trigger | Effect | Mitigation | Recovery |
|---|---|---|---|---|---|
| F1 | Outer armor penetration | Ballistic impact > 520 k PSI | Ceramic face shatters, UHMWPE delaminates | Energy transfers to auxetic (150 k PSI) | Self-heal < 25 mm in 2 s; panel swap if larger |
| F2 | Auxetic densification | Residual > 150 k PSI | Lattice cells crush to solid, foam ruptures | Energy transfers to DEA-STF (80 k/152 k stiffened) | Lattice permanently deformed; post-mission replacement |
| F3 | DEA fiber rupture | Residual > 80 k base / 152 k stiffened | Elastomer tears, STF bladder bursts, CNT electrodes sever | Load redistributes to adjacent panels | Panel swap + STF refill; 24-panel redundancy |
| F4 | Frame structural failure | > 50 G axial or > 500 N actuator overload | CFRP delamination, Ti node fracture, actuator seizure | 24 Spectra straps redistribute; telescoping locks | Frame replacement; mission abort + extraction |
| F5 | Clutch pod failure | Coil short, driver fault, debris strike | Zero field contribution locally | Remaining 71 pods compensate; phase redistribute (~−1.4 % thrust) | Bayonet cartridge swap; flyable at 71/72 |
| F6 | Power failure | Thermal runaway, BMS fault, bus short | 48 V bus lost to actuators/turbines/neural/thermal | Hot-swap backup (5 s); piezo emergency power | Battery swap; degraded mode 40 % thrust, no AB |
| F7 | Neural interface loss | EEG degradation, NPU fault, crypto failure | Thought-to-action control lost | AI copilot holds hover; manual fallback | Recalibrate; safe-mode manual only |
| F8 | Thermal runaway | Skin > 42 °C or < 30 °C despite regulation | Peltier failure, loop blockage, aerogel degradation | Radiator fins deploy; PCM absorbs 12 kJ/kg | Thermal service; abort if uncontrolled |
| F9 | Helmet seal breach | Visor crack > 5 mm, neck ring/O-ring failure | Pressure lost; O₂ out, CO₂/water in | Self-sealing polymer + 5 min emergency O₂ | Immediate egress; helmet replacement |
| F10 | Water reserve breach | Penetration > 12 mm, line/pump failure | Reserve water loss → vacuum Δv reduced (air flight unaffected) | Self-seal < 12 mm; dual pumps/lines; suit tank drawn first | Bladder repair; curtailed vacuum ops if > 50 % lost |

## Appendix C — Model physics constants

Module-level constants used by the plasma, orbital, and mission math — all SI:

| Constant | Value | Used by |
|---|---|---|
| `_ELEM_E` | 1.602e-19 C | plasma frequency, Debye length |
| `_EPS0` | 8.854e-12 F/m | plasma diagnostics |
| `_MU0` | 1.257e-6 H/m | Alfvén speed, Rm, magnetic sail |
| `_M_E` / `_M_PROTON` / `_M_AIR` | 9.109e-31 / 1.673e-27 / 4.81e-26 kg | plasma β, ion drag |
| `_E_ION_AIR` | 2.403e-18 J (15 eV) | air ionization budget |
| `_SEED_HALF` | 4e-7 | ion-seed half-life model |
| `_DHE3_M_KG` / `_DHE3_E_J` | 8.353e-27 kg / 2.932e-12 J | pellet burn rate → ~85 yr endurance |
| `_AU_M` / `_LY_M` / `_YEAR_S` | 1.496e11 m / 9.461e15 m / 31,557,600 s | mission legs |
| `_GM_SUN` | 1.327e20 m³/s² | orbital injection/escape math |
| `_V_EARTH_ORB` / `_V_ARRIVE` | 29.78 / 1.0 km/s | boost & arrival budgets |
| `_SW_V` / `_SW_N_1AU` | 450 km/s / 6 cm⁻³ | solar-wind sail force |
| `_ISM_N` | 0.1 cm⁻³ | ISM magsail braking leg |
| `_TS_AU` / `_VOYAGER1_V` | 90 AU / 17 km/s | termination-shock benchmark |
| `MM` / `UM` / `NM` / `SC_*` | unit + showcase scale factors | mm-native geometry, atomic-scale showcases |

The destination table (`MISSION_DESTS`: Alpha Cen 4.367 ly, Barnard's 5.96 ly,
Wolf 359 7.86 ly, Sirius 8.6 ly, TRAPPIST-1 40.7 ly, Galactic Centre 26.67 kly,
across-the-Milky-Way 100 kly) is exercised by the plasma showcase's voyage
integration — solar-wind boost from Earth's orbit to the 90 AU termination
shock (~30 km/s cruise), then ISM braking — and cross-checked in self-test.

## Appendix D — Self-test section index

`--selftest` executes, in order: build sweep (3 pilot corners) → part/scale
checks → armor stack & ballistic math → DEA/STF/muscle → frame telescoping &
optimized-band margins → plasma clutch all-regime thrust + voyage endurance +
iris gating → fusion/battery power + weight budget → thermal soak → life
support & recycling → helmet/seal → dive (NDL/deco/ppO₂/CNS/narcosis/trim)
→ space (dosimetry/MMOD tiers/RCS/steam fallback/mission legs) → wings &
pulse-glide → jump/defense/auto-aim/IFF → stealth/vision/eshield/tshield/
stun/CM/drones/grapple/parachute/beacon/maglev/voice/regen/heal/maintenance
→ RTOS (tasks/CRC/failover/virus/A-B firmware) → conservation & material
audits → failure-mode table → DIMS coverage → PART_DB integrity → environment
integrity → voice coverage → status-light coverage → export/render checks.
Any failed assertion prints `FAIL:` and ends the run non-zero; a clean run
ends `ALL TESTS PASSED`.
