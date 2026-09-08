# Mjalnor'MV1.17 — Engineering & Architecture Overview

This document explains *how the suit works* in the simulator: the three operating
modes, the layered architecture, every subsystem, and the physics that ties them
together. For install/run/controls see [README.md](README.md).

The model is a **single-file build**: [flysuit.py](flysuit.py) (~15,300 lines)
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
for a form-fitting, contoured surface. Outer armor plates are flush chamfered
panels with recessed panel-line grooves and subtle color variation between
sections. Pronounced shoulder pauldrons, bicep/tricep contour panels, thigh/shin
detail plates, smooth joint-cover spheres at shoulders and hips, and an
aerodynamic helmet (225 mm OD) with wraparound visor and cheek guards complete the look.
The Archangel glide membrane is a compact **gliding suit — part of the suit
itself, not literal wings**: two arm-to-body membranes and one leg-to-leg
membrane unfold from the armor seams (cape-like) and tension along nitinol
battens; stowed, everything folds flat against the back armor (<25 mm).

---

## The three suit modes

The suit is a single machine that adapts to its environment. Mode is determined
each frame from throttle, wings, and ambient density (`SuitState.flight_mode`).

### 1. Combat flight suit
VTOL and cruise flight on 72 thrust-vectoring micro-turbofans (25 mm dia, 14 lbf
dry / 23 lbf AB each, 300 k rpm, 393 m/s tip speed) burning Jet-A1
fuel (SFC-modeled: 1.0 lb/(lbf·h) dry, 1.8 AB), with a 72 L conformal
self-sealing bladder (~58 kg fuel). Turbines are flush-mounted 4 mm discs on the
armor surface, staying within the 1-inch (25.4 mm) suit profile. Thrust scales
with air density (mass flow rate ρ/ρ₀), so at 28 k ft thrust drops to ~34% of
sea-level. The Archangel
glide membrane — a compact gliding suit that is part of the suit (3.5 m
deployed span, 22 ft², 12:1 glide; arm-to-body + leg-to-leg membranes with
wrist-to-tip sleeves) — provides efficient range extension with proper induced
drag (CDi = CL²/(π·AR·e), Oswald e = 0.85). The jetpack flight profile is a
**pulse-climb-glide cycle** (autopilot key `Y`): the turbine swarm pulses
(afterburner, with a ballistic cutoff so the suit arrives at the top of a
200 m altitude band at ~zero climb rate), engines cut, the membrane deploys
and glides at best L/D back down to the band floor — repeating until the 2 kg
landing fuel reserve. Wing angle-of-attack is measured against the flight
path (a vertical climb is post-stall) and lift acts perpendicular to the
airflow, so climbs and glides trim like real aerodynamics. A full combat
suite includes Kalman-tracked auto-aim with ballistic firing
solutions, a martial-arts defense AI, energy and tactical shields, a recon drone
swarm, countermeasures, a non-lethal stun, and a grappling winch. Muscle fibers
and the frame distribute g-loads so the pilot survives high-g maneuvers. Hover
endurance is ~21 min at hover thrust; the pulse-glide cycle roughly doubles
endurance while covering real range (~2.4 km per cycle).

### 2. Space suit
In vacuum the air-breathing turbines **iris-seal shut** (no fuel burn, no leaks) —
maneuvering falls to **6 plasma thruster discs** (25 mm dia, 2 palm gimbaled + 2 heel fixed +
2 backpack vectored, D-He3 fusion-water jet, Isp ~2550 s) and a **cold-gas RCS**
with a real propellant budget (empty tank ⇒ no attitude
authority). The 2 backpack discs are the primary interplanetary voyage drive.
A 12 kW **aneutronic D-He3 fusion cell** provides continuous power
(sponge-lattice core, direct energy conversion), feeding the plasma discs and
charging the battery. Pellet burn rate is ~1.8 g/year, giving ~56 yr on the
100 g primary store + ~28 yr on the 50g backup reserve (~84 yr total) —
effectively endless for any single mission. A **radiation dosimeter** integrates galactic-cosmic-ray and
low-Earth-orbit dose with stochastic solar-particle events, attenuated by the
suit's areal shielding, and tracks cumulative mission dose vs. career limit. Life
support seals, scrubs CO₂, generates O₂, and holds cabin pressure. The
closed-loop **chemical recycling system** (fusion-powered, 250 W) provides
indefinite life support: a 4-bed molecular sieve captures CO₂, a Sabatier
reactor converts it to water (CO₂ + 4H₂ → CH₄ + 2H₂O), water electrolysis
produces breathable O₂ (2H₂O → 2H₂ + O₂), and a 98%-efficient water recycler
reclaims condensate and sweat. Backup compressed O₂ (24 h) and LiOH canister
(24 h) cover emergencies when the fusion cell is offline. A
Whipple/MMOD shield erodes under micrometeoroid flux.

### 3. Dive suit
A variable-volume **buoyancy control device (BCD)** auto-trims to neutral at any
depth, and a **decompression computer** models inert-gas loading across six
Bühlmann-style tissue compartments to produce the no-decompression limit, deco-
stop ceiling, ascent-rate warnings, oxygen-toxicity (ppO₂ / CNS clock), nitrogen
narcosis (equivalent narcotic depth), and depth-scaled gas consumption. The
sealed helmet and hull are rated to 1,000 m.

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
- `PowerManagementSystem` — solid-state Li-S battery, piezo/solar harvesting, voltage sag, load shedding (critical systems preserved), D-He3 fusion generation (12 kW continuous).
- `PlasmaThrusterSystem` — 6 plasma discs (25 mm dia, 2 palm gimbaled + 2 heel fixed + 2 backpack vectored, D-He3 fusion-powered) powered by the fusion cell: air MHD ion-drag, space magnetic-sail, vacuum fusion-water jet (Isp ~2550 s); 2 kg water propellant, ~84 yr pellet endurance (primary + backup), per-group thrust tracking.
- `LifeSupportSystem` — fusion-powered closed-loop chemical recycling (Sabatier + electrolysis + water recycler), backup O₂/LiOH, seal pressure, humidity; auto-activates in vacuum/underwater.
- `HelmetSystem` — vacuum/underwater seal, 20-mile zoom visor, comms, night/thermal vision.
- `FuelSystem` — 72 L conformal Kevlar bladder (Jet-A1, ~58 kg), dual-redundant electric pumps, self-sealing coating, AN-8 lines; fuel mass depletes in flight and reduces total mass.
- `SuitRTOS` — dual-redundant real-time OS: task scheduling, CRC integrity voting, failover, virus detection.

**Environment (mode) systems**
- `AtmosphericModel` — ISA Earth, Mars, Titan, ocean, and space; wind/turbulence, solar, radio attenuation.
- `DiveSystem` — BCD buoyancy auto-trim + Bühlmann decompression computer + ppO₂/narcosis/gas.
- `SpaceSystem` — radiation dosimetry + shielding, cold-gas RCS propellant + Tsiolkovsky Δv, MMOD flux/shield.

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
- `Mesh` / `Part` / `SuitRenderer` / `FlightRenderer` / `App` — geometry, HUD, and the interactive shell. The full suit model is 904 meshes / 27,628 vertices across 14 parts. The renderer uses a y-up coordinate system with smooth elliptical mesh primitives (`_smooth_torso`, `_smooth_limb`, `_helmet_shell`, `_pauldron`, `_sphere`), flush armor plates (`_armor_plate`), anatomical muscle fiber strands (`_fiber_strand`), and recessed panel lines (`_panel_line`). The UI is fully responsive: all panels scale with window size via `pygame.VIDEORESIZE`, and help/instructions/showcase/info panels support mouse-wheel scrolling. Each of the 14 parts includes inline `MATH PROOF` sections in its spec list, documenting weight derivation, compactness (≤25.4 mm), and physics validation.
- **Showcase system** (integrated in `flysuit.py`) — 16 detailed component views: inner suit, middle layer, outer armor, turbine (bayonet cartridge + iris seal), helmet, DEA/STF/auxetic/graphene/UHMWPE/ceramic atomic-scale, CFRP frame, Archangel glide membrane, plasma thrusters + fusion cell, power system, and neural BCI + AI co-pilot. Each includes 3D meshes, sub-components, math proof, and true-scale reference dots.

---

## Physics that couples everything

`SuitPhysics.step()` is where the systems become one machine — nothing is
cosmetic:

- **Propulsion is environment-aware.** Turbine thrust scales with air density
  ratio (ρ/ρ₀) — the correct mass-flow-rate physics for air-breathing engines.
  At 28 k ft (~34% sea-level density), thrust drops to ~34%. In vacuum or
  underwater, turbine iris covers **seal shut** (no fuel burn, no leaks) and
  propulsion falls to **plasma thrusters** (6 discs: palm gimbaled for attitude,
  heel fixed for translation, backpack vectored for voyage thrust; fusion-water
  jet in vacuum, sealed underwater) plus RCS cold-gas. Plasma discs add MHD ion-drag thrust in air
  (actuator-disk on the air column, no water consumed). Battery voltage sag and
  load shedding further reduce thrust via a (V/V_nom)² multiplier.
- **Fuel consumption is SFC-driven.** Specific fuel consumption (1.02 lb/(lbf·h)
  dry, 1.85 AB) × total thrust × throttle → kg/s burn rate. Fuel mass depletes
  over time, and `total_mass` is a live property that includes remaining fuel —
  so the suit gets lighter as it flies, affecting T/W, drag, and inertia.
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
  τ = thrust × moment arm (0.5 m to turbine clusters) and I is a per-axis inertia
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
| Pilot range | 0.99–2.21 m (3′3″–7′3″), 29.5–190.5 kg (65–420 lb), telescoping |
| Propulsion | 72 micro-turbofans (25 mm dia, 14 lbf dry / 23 lbf AB, 300 k rpm, 393 m/s tip speed), SFC 1.0/1.8 lb/(lbf·h), bayonet-swap cartridges, iris-sealed, flush-mounted |
| Plasma | 6 discs (25 mm dia, 2 palm + 2 heel + 2 backpack, D-He3 fusion: air MHD, space sail, vacuum jet (Isp ~2550 s), 2 kg water, ~84 yr pellets (primary + backup) |
| Fuel | 72 L Jet-A1 (~58 kg), conformal self-sealing bladder, ~21 min hover; pulse-glide ~doubles endurance (~100 km) |
| Wings | Glide membrane — part of the suit (not literal wings): 3.5 m deployed span, 22 ft² (2.0 m²), 12:1 glide, arm-to-body + leg-to-leg + wrist-to-tip sleeves, stows flat (cape fold) |
| Power | 1,200 Wh solid-state Li-S (550 Wh/kg) + piezo/solar harvesting + 12 kW D-He3 fusion (continuous, primary) |
| Thermal | ±100 °C, 12 kW heat / 3 kW cool, skin held ~37 °C |
| Flight envelope | 420 mph, 28,000 ft ceiling, 21.3 min hover, pulse-glide autopilot (key Y), 200 ft jump |
| Armor | 7.5 mm total stack, outer 600 k PSI (NIJ IV+), stack ballistic limit ~750 k PSI |
| Helmet | 225 mm OD, graphene-UHMWPE shell, wraparound visor (20-mile zoom), flush turbines + life support |
| Dive | 15 L BCD, 1,000 m rating, 6-compartment deco model, ppO₂ ≤ 1.6 ata |
| Space | 12 g/cm² shielding, GCR ~0.66 mSv/day, 12 cold-gas RCS jets (2 kg GN2, ~12 m/s Δv), 1,000 mSv career limit |
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
