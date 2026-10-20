# Mjalnor'MV1.17 — Engineering & Architecture Overview

This document explains *how the suit works* in the simulator: the three operating
modes, the layered architecture, every subsystem, and the physics that ties them
together. For install/run/controls see [README.md](README.md).

The model is a **single-file build**: [flysuit.py](flysuit.py) (~17,400 lines)
contains the simulator core (specs, systems, physics, renderer, tests) *and*
the 16 detailed component showcases with 3D meshes, math proof, and true-scale
specifications (integrated from the former showcase.py, which remains only as
a compatibility re-export shim). The code is organized top-to-bottom as:
**`DIMS`** (the SI/mm specification — the single source of truth) → **system
classes** → **`SuitState`** (owns and steps every subsystem each tick) →
**`SuitPhysics`** (6-DOF integrator) → **renderers / HUD** → **`App`**.

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
0.99–2.21 m fit envelope — verified by self-test. The assembled suit adds only
documented clearances (~24 mm: helmet dome over the crown + armored boot sole).
The gauntlets cover wrist-to-fingertip (`seg["hand"]`), the inner layer covers
hands and feet, and `build_suit` clamps pilot dimensions to the rated envelope
(0.99–2.21 m, 29.5–190.5 kg) so out-of-range input cannot produce an
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
drag (CDi = CL²/(π·AR·e), Oswald e = 0.85). The jetpack flight profile is a
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
narcosis (equivalent narcotic depth), and depth-scaled gas consumption. The
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

**Core suit**
- `FrameSystem` — CFRP telescoping exoskeleton, 64 Ti nodes, 14 Maxon actuators, 20 joints; magneto-rheological dampers lock joints above 4 g; auto-sizes to pilot.
- `MuscleFiberSystem` — dielectric-elastomer actuators (voltage → Maxwell-stress force) with shear-thickening-fluid impact stiffening; drives strength, jump, and punch. Full anatomical fiber mesh: pecs, abs, obliques, lats, erector spinae, trapezius, deltoids (3 heads), biceps, triceps, forearm flexors/extensors, quads (4 heads), hamstrings, glutes, calves, tibialis, soleus.
- `ThermalManagementSystem` — glycol loop + phase-change material + resistive backup; holds skin temp across ±100 °C.
- `NeuralInterfaceSystem` — EEG/EMG BCI, intent decoding, post-quantum crypto, <17 ms latency.
- `PowerManagementSystem` — solid-state Li-S battery, piezo/solar harvesting, voltage sag, load shedding (critical systems preserved), D-He3 fusion generation (55 kW continuous).
- `PlasmaThrusterSystem` — the sole flight drive (Main.py plasma-clutch model): 72 flush RMF pod lobes merge into a ~1 m coupling region; 6 steering discs (2 palm gimbaled + 2 heel fixed + 2 backpack vectored). Regimes: air MHD ion-drag on any gas, underwater ducted waterjets, vacuum water plasma jet (Isp ~2550 s, ~4.9 km/s Δv), ambient-plasma magnetic sail; solar-wind boost + ISM braking mission profiles; 2 kg water tank + 20 L reserve, ~85 yr pellet endurance. Per-pod health degrades gracefully (a failed pod costs ~1.4% thrust, never the drive); pods are 8-min bayonet LRU swaps; MMOD spall can arc a coil.
- `LifeSupportSystem` — fusion-powered closed-loop chemical recycling (Sabatier + electrolysis + water recycler), backup O₂/LiOH, seal pressure, humidity; auto-activates in vacuum/underwater.
- `HelmetSystem` — vacuum/underwater seal, 20-mile zoom visor, comms, night/thermal vision.
- `FuelSystem` — 20 L conformal Kevlar **water reserve** bladder feeding the vacuum plasma jet (suit tank first), dual-redundant electric pumps, self-sealing coating, AN-8 lines; reserve mass depletes in flight and reduces total mass.
- `SuitRTOS` — dual-redundant real-time OS: task scheduling, CRC integrity voting, failover, virus detection, post-quantum crypto (Kyber-1024 handshake), and signed A/B firmware updates staged on the shadow bank with rollback.
- `MaintenanceSystem` — LRU prognostics: per-unit operating hours vs design life coupled to live damage (pod wear, battery cycles, armor integrity); readiness score, next-service forecast, field swaps.

**Environment (mode) systems**
- `AtmosphericModel` — ISA Earth, Mars, Titan, ocean, and space; wind/turbulence, solar, radio attenuation.
- `DiveSystem` — BCD buoyancy auto-trim + Bühlmann decompression computer + ppO₂/narcosis/gas.
- `SpaceSystem` — radiation dosimetry + magnetic field shielding, cold-gas RCS with steam fallback + Tsiolkovsky Δv, three-tier MMOD defense (plasma sheath → energy-shield sentry → Whipple bumper) with breach→leak wiring.
- `EnergyShield` — capacitor-discharge reactive shield: held barrier (activate), sentry directed-discharge (hypervelocity grains), recharge/cooldown timers stepped every tick.
- `DirectedEnergyDefense` — laser fluence defense (92% reflective ceramic + ablation), plasma-bolt shunt through RMF field + capacitor (≤120 kJ), 700 kPa blast attenuation, and Argent-drain: absorbed directed energy recharges the eshield capacitor then battery (~35% coupling).

**Combat / "fight mode"**
- `AutoAimSystem` + `KalmanTracker` + `Target3D` — multi-target tracking, IFF, lead/ballistic firing solutions, missile-dodge.
- `DefenseAI` — martial-arts state machine; punch force driven live by the DEA arm fibers.
- `JumpAssist` — charged 200 ft vertical jump with muscle assist.
- `EnergyShield`, `TacticalShield`, `StunSystem`, `CountermeasureSystem`, `DroneSwarm`, `GrappleSystem`, `StealthMode`.

**Utility / survival**
- `VisionModeSystem`, `EmergencyParachute`, `RegenSystem`, `VoiceCommandSystem`, `EmergencyBeacon`, `MaglevMode`.

**Simulation core**
- `SuitState` — the live state; `update(dt)` steps every subsystem in order.
- `SuitPhysics` — 6-DOF translational + rotational integrator: thrust, drag, lift, gravity, buoyancy, wind, muscle assist, g-limiting; environment-aware (air / water / vacuum).
- `Mesh` / `Part` / `SuitRenderer` / `FlightRenderer` / `App` — geometry, HUD, and the interactive shell. The full suit model is 1,031 meshes / 28,934 vertices across 14 parts. The renderer uses a y-up coordinate system with smooth elliptical mesh primitives (`_smooth_torso`, `_smooth_limb`, `_helmet_shell`, `_pauldron`, `_sphere`), flush armor plates (`_armor_plate`), anatomical muscle fiber strands (`_fiber_strand`), and recessed panel lines (`_panel_line`). The UI is fully responsive: all panels scale with window size via `pygame.VIDEORESIZE`, and help/instructions/showcase/info panels support mouse-wheel scrolling. Each of the 14 parts includes inline `MATH PROOF` sections in its spec list, documenting weight derivation, compactness (≤25.4 mm), and physics validation.
- **Showcase system** (integrated in `flysuit.py`) — 16 detailed component views: inner suit, middle layer, outer armor, clutch pod (bayonet cartridge + iris seal), helmet, DEA/STF/auxetic/graphene/UHMWPE/ceramic atomic-scale, CFRP frame, Archangel glide membrane, plasma thrusters + fusion cell, power system, and neural BCI + AI co-pilot. Each includes 3D meshes, sub-components, math proof, and true-scale reference dots.

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
  and A is the body presented area (1.5 m²). Frame telescoping extends t_stop by
  20%, and STF stiffening + multi-layer absorption reduce felt pressure.
- **G-load uses full 3-D acceleration.** g = |a_total| / g, not just the vertical
  component — so lateral maneuvering and drag deceleration both count.
- **Rotational dynamics use torque/inertia.** Angular acceleration α = τ/I where
  τ = thrust × moment arm (0.5 m to pod clusters) and I is a per-axis inertia
  tensor [0.8, 1.2, 0.6] kg·m². Neural interface signal quality modulates
  response speed. Aerodynamic damping scales with speed × density.
- **Buoyancy** underwater uses the dive system's *variable* BCD displacement, so
  the suit can hold neutral instead of simply sinking.
- **G-loads** are limited by the frame (locked joints raise the safe ceiling) to
  protect the pilot; muscle leg force adds to jumps and high-g events.
- **Pilot state feeds back:** poor neural signal quality adds control noise; high
  CO₂ / low O₂ or extreme skin temperature degrade control authority.

---

## Key specifications

| Domain | Spec |
|---|---|
| Pilot range | 0.99–2.21 m (3′3″–7′3″), 29.5–190.5 kg (65–420 lb), telescoping; crown-to-ground = pilot height exactly (100% scale, self-test verified) |
| Propulsion | 72 flush RMF plasma-clutch pods (25 mm dia, bayonet-swap, iris-sealed; 30 backpack + 12 forearm + 12 calf + 12 thigh + 6 collar ring, ~1 m merged coupling region) + 6 steering discs (2 palm gimbaled + 2 heel fixed + 2 backpack vectored) |
| Drive regimes | Air MHD ion-drag ~2.0 kN sea-level (Mars ~0.5 kN, Titan ~2.8 kN); waterjet ~2.6 kN; vacuum water jet (Isp ~2550 s, ~4.9 km/s Δv); solar-wind sail + ISM braking |
| Water | 2 kg suit tank + 20 L reserve bladder — vacuum-jet reaction mass only (air/water flight use ambient medium) |
| Wings | Glide membrane — part of the suit (not literal wings): 3.5 m deployed span, 22 ft² (2.0 m²), 12:1 glide, arm-to-body + leg-to-leg + wrist-to-tip sleeves, stows flat (cape fold) |
| Power | 1,200 Wh solid-state Li-S (550 Wh/kg) + piezo/solar harvesting + 55 kW D-He3 fusion (continuous, primary) |
| Thermal | ±100 °C, 12 kW heat / 3 kW cool, skin held ~37 °C |
| Flight envelope | 420 mph, 28,000 ft ceiling (marginal in thin air), reactor-powered hover, pulse-glide autopilot (key Y), 200 ft jump |
| Armor | 7.5 mm total stack, outer 600 k PSI (NIJ IV+), stack ballistic limit ~750 k PSI |
| Helmet | 225 mm OD nominal (telescopes with pilot head), graphene-UHMWPE shell, wraparound visor (20-mile zoom), life support |
| Dive | 30 L BCD, 1,000 m rating, 6-compartment deco model, ppO₂ ≤ 1.6 ata |
| Space | 12 g/cm² shielding + RMF magnetic deflection (~45% charged flux), GCR ~0.66 mSv/day, 12 RCS jets (2 kg GN2, steam fallback on water reserve), MMOD: sheath ≤300 J + sentry ≤50 kJ + 6-layer Whipple, 1,000 mSv career limit |
| Environments | 11 presets: Earth SL → stratosphere, Mars, Titan, ocean 4 km, vacuum |
| Showcases | 16 detailed component views (integrated in flysuit.py) with 3D meshes, math proof, true-scale refs |

---

## The all-mechanical rule

The governing rule for this project: **every part must operate mechanically from
math and reality — nothing faked.** Design *parameters* (dimensions, material
strengths, capacities) are legitimate constants. Reported *outcomes* — impact
absorption, buoyancy, radiation dose, punch force, hit resolution — must always be
computed from those parameters and the current state. The `--selftest`,
`--impact-test`, and `--stress-test` harnesses exist to keep the model honest:
add an assertion whenever you add a system.
