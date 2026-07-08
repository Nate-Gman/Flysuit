# Mjalnor'MV1.17 — Engineering & Architecture Overview

This document explains *how the suit works* in the simulator: the three operating
modes, the layered architecture, every subsystem, and the physics that ties them
together. For install/run/controls see [README.md](README.md).

The model spans two files: [flysuit.py](flysuit.py) (~10,800 lines) contains the
simulator core (specs, systems, physics, renderer, tests), and
[showcase.py](showcase.py) (~1,500 lines) contains 15 detailed component
showcases with 3D meshes, math proof, and true-scale specifications. The code is
organized top-to-bottom as: **`DIMS`** (the SI/mm specification — the single
source of truth) → **system classes** → **`SuitState`** (owns and steps every
subsystem each tick) → **`SuitPhysics`** (6-DOF integrator) → **renderers / HUD**
→ **`App`**.

The suit's visual design is **angular and faceted** (MJOLNIR/Iron Man style),
not round or blobby. Inner and middle layers use hexagonal cross-section torso
and limb primitives (`_angular_torso`, `_angular_limb`), and outer armor plates
are chamfered with visible gaps at joints for a sleek, forged appearance.

---

## The three suit modes

The suit is a single machine that adapts to its environment. Mode is determined
each frame from throttle, wings, and ambient density (`SuitState.flight_mode`).

### 1. Combat flight suit
VTOL and cruise flight on 48 thrust-vectoring micro-turbofans burning Jet-A1
fuel (SFC-modeled: 1.02 lb/(lbf·h) dry, 1.85 AB), with a 72 L conformal
self-sealing bladder (~58 kg fuel). Thrust scales with air density (mass flow
rate ρ/ρ₀), so at 28 k ft thrust drops to ~34% of sea-level. Compact deployable
gliding wings (3.5 m span, 22 ft², 12:1 glide) provide efficient range extension
with proper induced drag (CDi = CL²/(π·AR·e), Oswald e = 0.85) — turbines
maintain airspeed, wings provide lift. Wings stow flat against back armor. A
full combat suite includes Kalman-tracked auto-aim with ballistic firing
solutions, a martial-arts defense AI, energy and tactical shields, a recon drone
swarm, countermeasures, a non-lethal stun, and a grappling winch. Muscle fibers
and the frame distribute g-loads so the pilot survives high-g maneuvers. Hover
endurance is ~26 min at full dry thrust; climb-glide cycling extends this to
4–7 hours.

### 2. Space suit
In vacuum the air-breathing turbines produce **zero** thrust — maneuvering falls
to a **cold-gas RCS** with a real propellant budget (empty tank ⇒ no attitude
authority). A **radiation dosimeter** integrates galactic-cosmic-ray and low-
Earth-orbit dose with stochastic solar-particle events, attenuated by the suit's
areal shielding, and tracks cumulative mission dose vs. career limit. Life support
seals, scrubs CO₂, generates O₂, and holds cabin pressure. A Whipple/MMOD shield
erodes under micrometeoroid flux.

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
the middle layer. All layers use angular hexagonal cross-sections for a
MJOLNIR/Iron Man aesthetic.

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
- `MuscleFiberSystem` — dielectric-elastomer actuators (voltage → Maxwell-stress force) with shear-thickening-fluid impact stiffening; drives strength, jump, and punch.
- `ThermalManagementSystem` — glycol loop + phase-change material + resistive backup; holds skin temp across ±100 °C.
- `NeuralInterfaceSystem` — EEG/EMG BCI, intent decoding, post-quantum crypto, <17 ms latency.
- `PowerManagementSystem` — solid-state Li-S battery, piezo/solar harvesting, voltage sag, load shedding (critical systems preserved).
- `LifeSupportSystem` — CO₂ scrubber, O₂ generation, seal pressure, humidity; auto-activates in vacuum/underwater.
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
- `Mesh` / `Part` / `SuitRenderer` / `FlightRenderer` / `App` — geometry, HUD, and the interactive shell. The UI is fully responsive: all panels scale with window size via `pygame.VIDEORESIZE`, and help/instructions/showcase/info panels support mouse-wheel scrolling.
- **Showcase system** (`showcase.py`) — 15 detailed component views: inner suit, middle layer, outer armor, turbine, helmet, DEA/STF/auxetic/graphene/UHMWPE/ceramic atomic-scale, CFRP frame, Archangel wings, power system, and neural BCI + AI co-pilot. Each includes 3D meshes, sub-components, math proof, and true-scale reference dots.

---

## Physics that couples everything

`SuitPhysics.step()` is where the systems become one machine — nothing is
cosmetic:

- **Propulsion is environment-aware.** Turbine thrust scales with air density
  ratio (ρ/ρ₀) — the correct mass-flow-rate physics for air-breathing engines.
  At 28 k ft (~34% sea-level density), thrust drops to ~34%. In vacuum, thrust
  is zero and only the RCS (gated on remaining cold-gas propellant) provides
  attitude/translation. Battery voltage sag and load shedding further reduce
  thrust via a (V/V_nom)² multiplier.
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
| Propulsion | 48 micro-turbofans, 68 lbf dry / 112 lbf AB, 180 k rpm, SFC 1.02/1.85 lb/(lbf·h) |
| Fuel | 72 L Jet-A1 (~58 kg), conformal self-sealing bladder, ~26 min hover / 4–7 hr climb-glide |
| Wings | 3.5 m span, 22 ft² (2.0 m²), 12:1 glide, compact turbine-assisted, stows flat |
| Power | 1,200 Wh solid-state Li-S (550 Wh/kg) + piezo/solar harvesting |
| Thermal | ±100 °C, 12 kW heat / 3 kW cool, skin held ~37 °C |
| Flight envelope | 420 mph, 28,000 ft ceiling, 5.8 h hover, 200 ft jump |
| Armor | 7.5 mm total stack, outer 600 k PSI (NIJ IV+), stack ballistic limit ~750 k PSI |
| Dive | 15 L BCD, 1,000 m rating, 6-compartment deco model, ppO₂ ≤ 1.6 ata |
| Space | 12 g/cm² shielding, GCR ~0.66 mSv/day, 2 kg cold-gas RCS, 1,000 mSv career limit |
| Environments | 11 presets: Earth SL → stratosphere, Mars, Titan, ocean 4 km, vacuum |
| Showcases | 15 detailed component views with 3D meshes, math proof, and true-scale refs |

---

## The all-mechanical rule

The governing rule for this project: **every part must operate mechanically from
math and reality — nothing faked.** Design *parameters* (dimensions, material
strengths, capacities) are legitimate constants. Reported *outcomes* — impact
absorption, buoyancy, radiation dose, punch force, hit resolution — must always be
computed from those parameters and the current state. The `--selftest`,
`--impact-test`, and `--stress-test` harnesses exist to keep the model honest:
add an assertion whenever you add a system.
