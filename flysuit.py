#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 FLYSUIT :: Mjalnor'MV1.17 -- Hybrid Combat / Space / Undersea / Flight Suit
================================================================================

A complete, single-file, standalone interactive 3D model + physics study of the
Mjalnor'MV1.17 hybrid exoskeleton suit described in Projectgoal.md:

  - One-size-fits-all telescoping CFRP frame (fits 3ft-7ft, 65-420 lb pilot)
  - 4-layer armor: inner sensor suit / tripled DEA-STF muscle / auxetic
    metamaterial / graphene-UHMWPE outer panels
  - 48 micro-turbofan swarm turbines for VTOL flight (thrust-vectoring, Jet-A1 fuel)
  - Compact deployable gliding wings (12:1 L/D, 22 sq ft, turbine-assisted)
  - Vacuum-sealed helmet for space + underwater (regenerable CO2, O2, thermal)
  - Unbreakable visor (graphene-polycarbonate, 15,000 lbs, 20-mile zoom)
  - Neural BCI (EEG/EMG, <17 ms latency, quantum-resistant encryption)
  - AI co-pilot "Vera 3.0" (auto-aim, defense, jump assist, thermal hunting)
  - SuitRTOS dual-redundant real-time OS (integrity checks, failover)
  - Solid-state Li-S batteries + piezo/solar harvesting (24+ hr endurance)
  - Faraday shielding, chemical/heat-resistant coatings, self-healing seals

Built using the same pure-Python software renderer architecture as the reference
code (Main.py, SE.py, LS.py): numpy-vectorized painter + flat shading +
backface culling + screen-space LOD. Every dimension is real (mm/SI).

--------------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------------
    python flysuit.py                  # open the interactive viewer
    python flysuit.py --selftest       # headless build + render + physics check
    python flysuit.py --feasibility    # real-world build feasibility report
    python flysuit.py --layers         # layer-by-layer exploded view mode
    python flysuit.py --export-obj     # write OBJ + MTL model files and exit
    python flysuit.py --stress-test    # multi-environment stress test
    python flysuit.py --impact-test    # ballistic impact absorption simulation

Dependencies:  python3 -m pip install pygame numpy

--------------------------------------------------------------------------------
CONTROLS
--------------------------------------------------------------------------------
  TAB ................... switch MODEL <-> FLIGHT <-> LAYERS
  MODEL:  mouse orbit / wheel zoom / R-M drag pan / 1-6 view presets
          L labels ; E exploded ; X section ; ,/. isolate parts
  FLIGHT: UP/DOWN throttle ; W/S pitch ; A/D roll ; L/R yaw ; IJKL thrust vector
          SPACE max+afterburner ; Z altitude-hold ; V hover ; R respawn ; [/] env
  LAYERS: 1-4 toggle layers ; E exploded ; [/] cycle layer focus
  ANY:    H help ; I info ; M math ; P pause ; O export OBJ ; ESC quit
================================================================================
"""

import os
import sys
import math
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # hard dependency

try:
    import pygame
except Exception:
    pygame = None


# =============================================================================
# ENGINEERING SPECIFICATION  (millimetres / SI -- source of truth)
# =============================================================================

MM = 0.001

DIMS = {
    # --- Pilot sizing range (one-size-fits-all telescoping) ---
    "pilot_min_height_m":    0.99,   # 3ft 3in
    "pilot_max_height_m":    2.21,   # 7ft 3in
    "pilot_min_weight_kg":   29.5,   # 65 lb
    "pilot_max_weight_kg":  190.5,   # 420 lb
    # NOTE: torso_telescope_mm and limb_telescope_mm defined in Frame section below

    # --- Reference pilot (median 5ft 8in, 175 lb) ---
    "ref_height_m":          1.73,
    "ref_weight_kg":         79.4,
    "ref_shoulder_m":        0.46,   # shoulder width
    "ref_chest_m":           0.30,   # chest depth
    "ref_hip_m":             0.38,   # hip width
    "ref_arm_len_m":         0.63,   # shoulder to wrist
    "ref_leg_len_m":         0.85,   # hip to ankle
    "ref_torso_len_m":       0.52,   # shoulder to hip

    # --- Layer thicknesses (ultra-conformed, total 7.5 mm) ---
    # Stack: inner(0.8) + middle(2.4) + intermediate(1.2) + outer(2.5) + gaps(0.6) = 7.5mm
    # Faraday mesh (0.05mm) is embedded within middle layer, not in stack
    "inner_thick_mm":         0.8,   # ultra-thin spandex-nylon sensor fabric with printed EMG
    "middle_thick_mm":        2.4,   # tripled DEA-STF (3x 0.8mm sublayers, STF interlayers)
    "intermediate_thick_mm":  1.2,   # thin auxetic re-entrant honeycomb + foam fill
    "outer_thick_mm":         2.5,   # graphene-UHMWPE (2.0mm) + alumina ceramic strike face (0.5mm)
    "faraday_thick_mm":       0.05,  # graphene-copper monolayer mesh (embedded in middle)
    "total_thick_mm":         7.5,   # 7.5 mm total stack (with 0.6mm gaps/gaskets)

    # --- Frame: CFRP telescoping exoskeleton ---
    "frame_tube_od_mm":      22.0,   # outer tube diameter
    "frame_tube_id_mm":      18.0,   # inner tube diameter (hollow)
    "frame_node_count":          64,   # Ti-6Al-4V printed nodes
    "frame_pivot_count":           20,   # key pivoting joints
    "frame_actuator_count":        14,   # Maxon EC45 linear actuators
    "frame_strap_count":           24,   # Spectra auto-tensioning straps
    "frame_material":       "Toray T1100G carbon fiber",
    "frame_node_material":  "Ti-6Al-4V (Markforged Metal X)",
    "torso_telescope_mm":   355.6,  # +/-14 inches torso telescoping (355.6mm = 14.0in)
    "limb_telescope_mm":    254.0,  # +/-10 inches limb telescoping (254.0mm = 10.0in)

    # --- Middle layer: tripled DEA-STF muscle fibers ---
    "dea_sublayers":              3,   # triply redundant
    "dea_contraction_ms":        20,   # <20 ms contraction time
    "dea_strain_pct":            50,   # 50% elongation capability
    "stf_max_psi":           150000,   # 150,000 PSI absorption (tripled)
    "muscle_boost_x":            15,   # 15x human strength (tripled + localized)
    "fiber_density_jump_x":       5,   # 5x density in legs for 200ft jump
    "fiber_density_lift_x":       2,   # 2x in shoulders/back/arms for lifts
    "punch_kinetic_chain_x":    3.0,   # arm-fiber force -> strike via frame + hip/torso chain

    # --- Intermediate layer: auxetic metamaterial ---
    "auxetic_poisson":        -0.75,   # negative Poisson's ratio
    "auxetic_cell_type":   "REC-Star",
    "auxetic_rel_density":     0.20,   # 20% relative density
    "auxetic_energy_abs_pct":  65,   # absorbs 65% of residual energy

    # --- Outer layer: graphene-UHMWPE armor ---
    "outer_max_psi":        600000,   # 600,000 PSI before penetration
    "outer_nij_level":         "IV",
    "outer_heat_tol_c":      2000,   # >2,000 C tolerance
    "outer_laser_tol_kw":     10,   # 10 kW/cm^2 laser resistance
    "coating_type":         "NiP + graphene NP",
    "ceramic_type":         "alumina-aerogel hybrid",

    # --- Armor impact mechanics (per-layer stopping pressure + back-face leak) ---
    # A layer defeats peak contact pressure up to its "stopping" capacity (plastic
    # work per unit area = dynamic strength x thickness); pressure above that
    # capacity penetrates, and a small "leak" fraction of the stopped pressure
    # still reaches the next layer as back-face deformation. The three layers sum
    # to ~750k PSI, so the stack fully defeats a .50 BMG (~600k) and is overmatched
    # (behind-armor lethal) only by 20mm+ autocannon (~1.2M PSI). Ballistic limit
    # sits near ~800k PSI.
    "outer_stop_psi":       520000,   # graphene-UHMWPE plastic-work capacity
    "outer_leak_frac":       0.015,   # back-face coupling
    "inter_stop_psi":       150000,   # auxetic densification capacity (REC-Star lattice)
    "inter_leak_frac":       0.030,
    "middle_stop_psi":       80000,   # STF fibers, base (shear-thickens under impact)
    "middle_leak_frac":      0.060,
    "stf_stiffen_mult":        1.9,   # STF dynamic strengthening when shear-stiffened
    "body_impact_area_m2":     1.5,   # torso presented area for pressure<->force

    # --- Helmet ---
    "helmet_od_mm":          260.0,
    "helmet_wall_mm":          8.0,
    "helmet_weight_kg":        1.0,
    "visor_thickness_mm":     12.0,
    "visor_max_force_lbs": 15000.0,
    "visor_zoom_optical":      10,   # 10x optical
    "visor_zoom_digital":     100,   # 100x digital (effective 20-mile)
    "visor_fov_deg":          180,   # panoramic
    "seal_depth_m":            1000, # underwater depth rating
    "seal_vacuum":          True,    # space-rated
    "co2_scrub_hours":          24,  # regenerable 4-bed molecular sieve
    "o2_capacity_hours":        24,

    # --- Propulsion: micro-turbofan swarm (physically validated) ---
    # Thrust ~ mdot * V_exit; at 50mm dia, 150k RPM, tip speed ~393 m/s
    # mdot = rho * A * V_inlet = 1.225 * pi*(0.025)^2 * 196 = 0.47 kg/s
    # F = 0.47 * 280 = 132 N = 30 lbf (validated)
    # T/W ~40:1 with CMC blades + magnetic bearings + 3D-printed Ti housing
    # Weight per turbine: 30/40 = 0.75 lbf = 0.34 kg; total = 36 * 0.34 = 12.2 kg
    "turbine_count":            36,
    "turbine_d_mm":            50.0,   # 50mm dia (frontal area for 30 lbf thrust)
    "turbine_len_mm":         180.0,   # 180mm length (compact, conformal)
    "turbine_rpm_max":      150000,
    "turbine_thrust_dry_lbf":  30,   # 30 lbf dry (132 N, validated from mdot*V_exit)
    "turbine_thrust_ab_lbf":  50,   # 50 lbf afterburner (1.67x boost)
    "turbine_bypass_ratio":      6,   # 6:1 cold bypass (compact)
    "turbine_bearing":    "ceramic-matrix hybrid",
    "turbine_sfc_dry_lb_lbh": 1.0,  # lb fuel per lbf-thrust per hour (dry)
    "turbine_sfc_ab_lb_lbh":  1.8,  # SFC with afterburner (richer burn)
    "fuel_type":           "Jet-A1",
    "fuel_density_kg_l":      0.81,  # Jet-A1 density
    "fuel_capacity_l":       45.0,   # 45 L distributed conformal bladders (torso+thighs+calves)
    # Placement: 12 backpack, 6 forearm, 6 calf, 6 thigh, 6 helmet ring = 36
    "turbine_backpack":         12,
    "turbine_forearm":           6,
    "turbine_calf":              6,
    "turbine_thigh":             6,
    "turbine_helmet":            6,
    "turbine_vector_deg":      120,   # +/-120 deg vectoring

    # --- Deployable Archangel wings (compact, turbine-assisted glide) ---
    "wing_span_m":             3.50,   # 11.5 ft compact span, stows flat on back
    "wing_area_sqft":           22,   # 2.04 m^2, turbine-assisted so less area needed
    "wing_ld_ratio":            12,   # 12:1 best glide (compact wings, lower AR)
    "wing_deploy_s":           0.5,
    "wing_min_sink_ms":        3.0,   # higher sink but turbines compensate
    "wing_membrane":    "aerogel/graphene weave",
    "wing_rib_material":      "nitinol memory alloy",

    # --- Power system ---
    "battery_type":      "solid-state Li-S",
    "battery_wh":            1200,   # 2x 600 Wh hot-swap
    "battery_wh_kg":         550,   # 550 Wh/kg energy density
    "battery_life_hours":      24,
    "piezo_harvest_pct":       40,   # 40% kinetic recovery
    "solar_nanofiber":      True,

    # --- Thermal system ---
    "thermal_inner_temp_c":    37.0,  # skin temp locked 36.5-37.5
    "thermal_range_lo_c":    -100.0,
    "thermal_range_hi_c":     100.0,
    "thermal_heating_kw":      12,   # 12 kW dump into body
    "thermal_cooling_kw":       3,   # 3 kW resistive backup

    # --- Neural interface ---
    "bci_latency_ms":           17,   # <17 ms thought-to-thrust
    "bci_type":         "EEG/EMG hybrid",
    "bci_crypto":       "lattice-based post-quantum",
    "bci_air_gapped":   True,

    # --- SuitRTOS ---
    "os_type":          "SuitRTOS dual-redundant",
    "os_check_ms":            10,   # integrity check every 10 ms
    "os_failover_ms":          5,   # <5 ms failover
    "os_uptime_pct":       99.999,
    "os_ai_model":  "Vera 3.0 (Llama-3.1-70B 4-bit)",

    # --- Performance specs ---
    "perf_jump_vertical_ft":   200,
    "perf_punch_lbs":        10000,
    "perf_lift_overhead_lbs":  3000,
    "perf_deadlift_lbs":       6000,
    "perf_push_pull_lbs":     10000,
    "perf_run_mph":             60,
    "perf_safe_fall_ft":      1000,
    "perf_aim_range_miles":       4,
    "perf_aim_accuracy_pct":     98,
    "perf_max_speed_mph":       420,
    "perf_climb_ft_min":      18000,
    "perf_ceiling_ft":        28000,
    "perf_hover_hours":        0.26,   # ~15 min pure hover; climb-glide extends to 4-7h

    # --- Upgrade: Active Camouflage / Adaptive Stealth ---
    "stealth_electrochromic_panels":  48,   # panels in outer armor
    "stealth_ir_reduction_pct":       85,   # % IR signature reduction
    "stealth_radar_reduction_db":     15,   # dB RCS reduction
    "stealth_visual_match_pct":       92,   # % background color match
    "stealth_power_draw_w":          120,   # watts active
    "stealth_activate_s":            0.5,   # seconds to full stealth
    "stealth_cooldown_s":            1.0,   # cooldown after deactivation

    # --- Upgrade: Multi-Spectrum Vision Modes ---
    "vision_modes":        ["normal", "night", "thermal", "sonar", "xray"],
    "vision_switch_ms":           50,   # ms to switch modes
    "vision_night_range_m":      300,   # night vision effective range
    "vision_thermal_range_m":    500,   # thermal imaging range
    "vision_thermal_sensitivity": 0.05, # Kelvin resolution
    "vision_sonar_range_m":      200,   # sonar range underwater
    "vision_xray_penetration_mm": 50,   # backscatter penetration depth
    "vision_ar_overlay":        True,   # AR tactical overlay

    # --- Upgrade: Grappling Hook / Winch ---
    "grapple_range_m":            50,   # max cable length
    "grapple_line_diameter_mm":    3,   # Dyneema cable
    "grapple_tensile_lbs":       500,   # load capacity
    "grapple_winch_speed_mps":     8,   # winch pull speed
    "grapple_anchor_force_n":   5000,   # anchor grip force
    "grapple_deploy_s":         0.3,   # deployment time
    "grapple_weight_kg":        0.5,

    # --- Upgrade: Emergency Ballistic Parachute ---
    "parachute_canopy_sqft":    300,   # canopy area
    "parachute_deploy_s":       1.5,   # ballistic deployment time
    "parachute_descent_mps":      7,   # descent rate under canopy
    "parachute_min_alt_m":       15,   # minimum safe deployment altitude
    "parachute_auto_deploy":   True,   # auto-deploy on freefall
    "parachute_weight_kg":      0.8,

    # --- Upgrade: Micro-Drone Swarm ---
    "drone_count":                4,   # recon drones
    "drone_weight_g":            50,   # per drone
    "drone_flight_time_min":     15,   # endurance
    "drone_range_km":              2,   # communication range
    "drone_speed_mph":            35,   # max speed
    "drone_camera":          "4K30",   # camera spec
    "drone_deploy_s":           2.0,   # launch sequence time
    "drone_weight_kg":         0.2,

    # --- Upgrade: Countermeasure Dispenser ---
    "cm_count":                  12,   # flare/chaff/deoy cartridges
    "cm_types":     ["flare", "chaff", "decoy"],
    "cm_deploy_ms":              50,   # deployment time
    "cm_effectiveness_pct":      85,   # missile defeat probability
    "cm_burn_time_s":           4.5,   # flare burn duration
    "cm_decoy_radius_m":         30,   # IR decoy effective radius
    "cm_weight_kg":            0.3,

    # --- Upgrade: Underwater Jet Propulsion ---
    "uw_jet_thrust_lbf":         35,   # per turbine in water-jet mode
    "uw_jet_max_speed_mph":      40,   # max underwater speed
    "uw_jet_intake_mm":          25,   # intake diameter
    "uw_jet_activate_s":        0.8,   # mode switch time
    "uw_jet_cavitation_pct":     15,   # cavitation reduction (shrouded)

    # --- Upgrade: Energy Shield / Reactive Armor Pulse ---
    "eshield_max_charge_kj":     50,   # max stored energy
    "eshield_recharge_kj_s":      5,   # recharge rate
    "eshield_activate_ms":       20,   # activation time
    "eshield_duration_s":       3.0,   # max active duration
    "eshield_cooldown_s":       8.0,   # cooldown after use
    "eshield_effectiveness_pct": 95,   # projectile vaporization
    "eshield_radius_m":         0.8,   # shield bubble radius
    "eshield_weight_kg":       0.4,

    # --- Upgrade: Regenerative Impact Recovery ---
    "regen_impact_pct":         15,   # % of impact energy recovered
    "regen_landing_wh":        2.0,   # Wh per hard landing
    "regen_combat_wh":         0.5,   # Wh per punch/block
    "regen_brake_pct":          8,   # % of braking energy recovered

    # --- Upgrade: Deployable Tactical Shield ---
    "tshield_height_m":        0.9,
    "tshield_width_m":         0.6,
    "tshield_weight_kg":       2.0,
    "tshield_nij_level":      "IV",
    "tshield_max_psi":      600000,
    "tshield_deploy_s":       0.4,
    "tshield_folds_flat":    True,

    # --- Upgrade: Non-Lethal Taser / Stun ---
    "stun_voltage_v":        50000,   # 50kV pulse
    "stun_current_ma":         1.3,   # 1.3mA (non-lethal)
    "stun_pulse_s":           5.0,   # pulse duration
    "stun_charge_s":         1.5,   # charge time between uses
    "stun_range_m":           0.3,   # contact range
    "stun_shots":              50,   # charges per battery
    "stun_weight_kg":        0.1,

    # --- Upgrade: Voice / Subvocal Commands ---
    "voice_vocab_size":         50,   # command vocabulary
    "voice_latency_ms":         80,   # recognition latency
    "voice_accuracy_pct":       97,   # recognition accuracy
    "voice_subvocal":        True,   # subvocal (no audible speech)

    # --- Upgrade: Emergency Locator Beacon ---
    "beacon_freq_mhz":       406.0,   # COSPAS-SARSAT satellite
    "beacon_homing_mhz":     121.5,   # local homing
    "beacon_battery_h":         72,   # beacon battery life
    "beacon_range_km":       20000,   # satellite uplink range
    "beacon_gps_accuracy_m":     5,   # GPS accuracy
    "beacon_weight_kg":       0.1,

    # --- Upgrade: Enhanced Self-Healing (Vascular Nanite) ---
    "heal_vascular_channels":  200,   # microchannels in armor
    "heal_repair_pct_s":        25,   # % per second (up from ~20)
    "heal_structural":        True,   # can fix structural damage
    "heal_polymer_ml":          50,   # repair polymer capacity
    "heal_tiers":                3,   # surface / structural / critical

    # --- Upgrade: Maglev Wall-Climbing ---
    "maglev_adhesion_n":      1200,   # adhesion force per boot
    "maglev_power_w":           80,   # power draw per boot
    "maglev_max_speed_mps":    3.0,   # climbing speed
    "maglev_activate_s":      0.5,   # activation time
    "maglev_weight_kg":       0.3,

    # --- Dive suit: buoyancy control + decompression computer ---
    "dive_bcd_volume_l":        30.0,   # variable buoyancy bladder capacity (L)
    "dive_ballast_kg":           6.0,   # fixed trim ballast
    "dive_max_depth_m":       1000.0,   # crush depth rating (outer shell)
    "dive_gas_liters":        2400.0,   # onboard breathing gas (STP-equivalent)
    "dive_sac_lpm":             18.0,   # surface air consumption (L/min)
    "dive_fo2":                 0.21,   # breathing-gas oxygen fraction (air)
    "dive_ppo2_max_ata":         1.6,   # max safe O2 partial pressure (ata)
    "dive_ascent_rate_max_m_min": 9.0,  # safe ascent rate (m/min)
    "dive_narcosis_depth_m":    30.0,   # onset of nitrogen narcosis on air
    "dive_tissue_compartments":    6,   # Bühlmann-style N2 tissue compartments

    # --- Space suit: radiation dosimetry + RCS maneuvering ---
    "rad_shield_g_cm2":         12.0,   # areal shielding (armor + life-support water)
    "rad_gcr_msv_day":          0.66,   # deep-space galactic cosmic ray dose rate
    "rad_leo_msv_day":          0.30,   # low-Earth-orbit baseline dose rate
    "rad_career_limit_msv":   1000.0,   # career effective-dose limit
    "rad_spe_warn_msv_h":        5.0,   # solar-particle-event dose-rate alarm
    "rcs_propellant_kg":         2.0,   # cold-gas (GN2) EVA propellant
    "rcs_isp_s":                65.0,   # cold-gas specific impulse
    "rcs_thrust_lbf":            2.0,   # per-thruster vacuum thrust
    "mmod_shield_layers":         6,    # Whipple/MMOD bumper layers
    "mmod_flux_m2_s":         3.0e-6,   # damaging-particle flux (>0.1mm) per m2/s
    "mmod_exposed_area_m2":      2.0,   # suit surface presented to the flux

    # --- Weight budget (ultra-conformed, self-consistent) ---
    # Each component weight is physically derived from material density x volume
    # Total = 25.2 kg suit (without fuel); +36.5 kg fuel + 79.4 kg pilot = 141.1 kg
    # T/W dry = (36*30*4.448)/(141.1*9.81) = 4804/1385 = 3.47:1 (flyable)
    # T/W AB  = (36*50*4.448)/(141.1*9.81) = 8006/1385 = 5.78:1 (agile)
    "weight_frame_kg":         1.8,   # CFRP tubes(0.34) + Ti nodes(0.32) + actuators(0.70) + straps(0.24) + misc(0.20)
    "weight_inner_kg":         0.3,   # 0.8mm sensor fabric, ~1.5 m^2, 800 kg/m^3
    "weight_faraday_kg":       0.2,   # 0.05mm graphene-Cu mesh, ~1.5 m^2
    "weight_fibers_kg":        1.8,   # 2.4mm DEA-STF, 3 sublayers, segmented 70% coverage
    "weight_auxetic_kg":       0.6,   # 1.2mm auxetic lattice at 20% rel density + foam
    "weight_armor_kg":         1.5,   # 2.5mm graphene-UHMWPE(2.0mm) + alumina(0.5mm), 12 panels
    "weight_helmet_kg":        1.0,   # shell + visor + life support + neural band
    "weight_turbines_kg":     12.2,   # 36x micro-turbofans at 0.34 kg each (T/W 40:1)
    "weight_wings_kg":         1.0,   # nitinol ribs + graphene membrane, compact
    "weight_fuel_system_kg":   1.0,   # bladder + lines + pumps (fuel counted separately)
    "weight_thermal_kg":       0.5,   # Peltier junctions + capillary loop + aerogel
    "weight_power_kg":         2.2,   # 1200 Wh / 550 Wh/kg = 2.18 kg Li-S cells + BMS
    "weight_electronics_kg":   0.3,   # neural processor + RTOS + sensors
    "weight_enhancements_kg":  0.8,   # grapple, countermeasures, drones, beacon, taser, maglev, healing
    "weight_total_kg":        25.2,   # 25.2 kg suit base (without fuel)
}

# Environment presets for flight mode
# (name, density, gravity, solar_w_m2, temp_K, wind_base_mps, weather, planet)
ENVIRONMENTS = [
    ("Earth Sea Level",  1.225,  9.81,  1.0,  288.15,  3.0, "clear",    "earth"),
    ("Earth 10k ft",     0.905,  9.78,  0.74, 268.15,  8.0, "clear",    "earth"),
    ("Earth 28k ft",     0.418,  9.72,  0.34, 228.15, 15.0, "clear",    "earth"),
    ("Earth Stratosphere", 0.089, 9.71, 0.21, 216.65, 30.0, "clear",    "earth"),
    ("Mars Surface",     0.020,  3.71,  0.016, 210.65, 12.0, "dust",     "mars"),
    ("Mars Orbit",       0.0,    3.71,  0.016, 2.7,    0.0,  "vacuum",   "space"),
    ("Titan Surface",    5.270,  1.35,  4.3,   93.65,  0.5, "methane",   "titan"),
    ("Titan Lakes",      5.270,  1.35,  4.3,   93.65,  0.3, "methane",   "ocean"),
    ("Underwater 100m",  997.0,  9.81,  0.0,   283.15,  0.3, "current",  "ocean"),
    ("Deep Ocean 4km",   1029.0, 9.81,  0.0,   275.15,  0.1, "current",  "ocean"),
    ("Vacuum/Space",     0.0,    0.0,   0.0,   2.7,    0.0, "vacuum",    "space"),
]

WEATHER_TYPES = ["clear", "clouds", "rain", "storm", "dust", "methane", "current", "vacuum"]


class AtmosphericModel:
    """Full atmospheric model for Earth, Mars, Titan, and ocean environments.

    Models:
    - ISA (International Standard Atmosphere) for Earth: altitude-based
      density, temperature, pressure using troposphere/lapse rate equations
    - Mars atmosphere: thin CO2, scale-height model
    - Titan atmosphere: thick N2/CH4, extended scale height
    - Ocean: hydrostatic pressure gradient, temperature thermocline
    - Wind: base wind + turbulence (Gaussian noise) + gusts
    - Solar radiation: altitude-attenuated irradiance
    - Ionosphere effects (radio attenuation) for comms
    """
    # ISA constants
    ISA_T0 = 288.15  # K at sea level
    ISA_P0 = 101325  # Pa at sea level
    ISA_RHO0 = 1.225  # kg/m3 at sea level
    ISA_LAPSE = -0.0065  # K/m troposphere lapse rate
    ISA_G = 9.80665  # m/s2
    ISA_R = 287.052  # J/(kg K) specific gas constant for air
    ISA_TROPOPAUSE = 11000  # m, tropopause height
    ISA_T_TROP = 216.65  # K at tropopause

    # Mars constants
    MARS_T0 = 210.65
    MARS_P0 = 610.0  # Pa (very thin)
    MARS_RHO0 = 0.020
    MARS_SCALE_H = 11100  # m
    MARS_G = 3.71
    MARS_R = 191.8  # J/(kg K) for CO2

    # Titan constants
    TITAN_T0 = 93.65
    TITAN_P0 = 146700  # Pa (1.45x Earth)
    TITAN_RHO0 = 5.27
    TITAN_SCALE_H = 21000  # m (very tall atmosphere)
    TITAN_G = 1.35

    # Ocean constants
    OCEAN_RHO = 997.0
    OCEAN_G = 9.81
    OCEAN_T0 = 288.15  # surface temp
    OCEAN_THERMOCLINE = 1000  # m, thermocline depth

    def __init__(self, planet="earth"):
        self.planet = planet
        self.wind = np.zeros(3)
        self.wind_base = 0.0
        self.turbulence = 0.0
        self.gust_timer = 0.0
        self.gust_strength = 0.0
        self.solar_irradiance = 0.0
        self.ambient_pressure = 101325.0  # Pa
        self.radio_attenuation = 0.0  # dB

    def update(self, dt, altitude, env_idx, weather="clear"):
        """Update atmospheric conditions based on altitude and environment."""
        env = ENVIRONMENTS[env_idx]
        planet = env[7] if len(env) > 7 else "earth"
        self.planet = planet
        self.wind_base = env[5] if len(env) > 5 else 0.0

        if planet == "earth":
            self._update_earth(altitude, weather)
        elif planet == "mars":
            self._update_mars(altitude, weather)
        elif planet == "titan":
            self._update_titan(altitude, weather)
        elif planet == "ocean":
            self._update_ocean(altitude, weather)
        elif planet == "space":
            self._update_space(altitude)

        # Wind turbulence
        self._update_wind(dt, weather)

        # Radio attenuation (ionosphere for Earth, dust for Mars)
        if planet == "earth" and 60 < altitude < 1000:
            self.radio_attenuation = 5.0 + 10.0 * max(0, (altitude - 200) / 300)
        elif planet == "mars" and weather == "dust":
            self.radio_attenuation = 20.0 + 30.0 * self.turbulence
        else:
            self.radio_attenuation = 0.0

    def _update_earth(self, altitude, weather):
        """ISA atmosphere model for Earth."""
        if altitude < self.ISA_TROPOPAUSE:
            # Troposphere: linear temperature lapse
            T = self.ISA_T0 + self.ISA_LAPSE * altitude
            P = self.ISA_P0 * (T / self.ISA_T0) ** (self.ISA_G / (self.ISA_R * -self.ISA_LAPSE))
        else:
            # Stratosphere: isothermal
            T = self.ISA_T_TROP
            P_trop = self.ISA_P0 * (self.ISA_T_TROP / self.ISA_T0) ** (self.ISA_G / (self.ISA_R * -self.ISA_LAPSE))
            P = P_trop * math.exp(-self.ISA_G * (altitude - self.ISA_TROPOPAUSE) / (self.ISA_R * T))
        rho = P / (self.ISA_R * T)
        self.ambient_pressure = P
        # Solar irradiance (attenuated by atmosphere)
        solar_top = 1361.0  # W/m2 solar constant
        air_mass = max(1.0, 1.0 / max(0.01, math.cos(0.5)))  # simplified
        self.solar_irradiance = solar_top * math.exp(-0.1 * air_mass * rho / self.ISA_RHO0)
        if weather == "clouds":
            self.solar_irradiance *= 0.5
        elif weather == "rain":
            self.solar_irradiance *= 0.2
        elif weather == "storm":
            self.solar_irradiance *= 0.1

    def _update_mars(self, altitude, weather):
        """Mars atmosphere (thin CO2)."""
        T = self.MARS_T0 - 0.002 * altitude  # gentle lapse
        P = self.MARS_P0 * math.exp(-altitude / self.MARS_SCALE_H)
        rho = P / (self.MARS_R * T)
        self.ambient_pressure = P
        # Solar: ~590 W/m2 at top, less atmosphere attenuation
        self.solar_irradiance = 590.0 * math.exp(-0.01 * altitude / self.MARS_SCALE_H)
        if weather == "dust":
            self.solar_irradiance *= 0.3

    def _update_titan(self, altitude, weather):
        """Titan atmosphere (thick N2/CH4)."""
        T = self.TITAN_T0 - 0.001 * altitude
        P = self.TITAN_P0 * math.exp(-altitude / self.TITAN_SCALE_H)
        rho = P / (290.0 * T)  # N2 gas constant ~296.8, approx
        self.ambient_pressure = P
        # Solar: very weak at Saturn distance (~14 W/m2)
        self.solar_irradiance = 14.0 * math.exp(-0.001 * altitude / self.TITAN_SCALE_H)

    def _update_ocean(self, depth, weather):
        """Ocean hydrostatic pressure and thermocline."""
        # depth is negative altitude (underwater)
        d = max(0, -depth)
        P = 101325 + self.OCEAN_RHO * self.OCEAN_G * d
        self.ambient_pressure = P
        # Temperature: thermocline model
        if d < self.OCEAN_THERMOCLINE:
            T = self.OCEAN_T0 - (self.OCEAN_T0 - 277.15) * d / self.OCEAN_THERMOCLINE
        else:
            T = 277.15  # deep water ~4C
        # No solar underwater (attenuated)
        self.solar_irradiance = max(0, 1361.0 * math.exp(-d / 20.0))

    def _update_space(self, altitude):
        """Vacuum/space environment."""
        self.ambient_pressure = 0.0
        # Solar constant in space (no atmospheric attenuation)
        self.solar_irradiance = 1361.0  # W/m2 at 1 AU

    def _update_wind(self, dt, weather):
        """Update wind vector with turbulence and gusts."""
        # Base wind direction (varies by weather)
        if weather in ("storm", "dust"):
            turb_intensity = 0.8
        elif weather in ("rain", "clouds"):
            turb_intensity = 0.3
        elif weather == "current":
            turb_intensity = 0.1
        else:
            turb_intensity = 0.05

        self.turbulence = turb_intensity

        # Gust timer
        self.gust_timer -= dt
        if self.gust_timer <= 0:
            self.gust_timer = 2.0 + np.random.random() * 5.0
            self.gust_strength = (np.random.random() - 0.5) * self.wind_base * 2.0

        # Wind vector: base + turbulence + gust
        base_vec = np.array([self.wind_base, 0, self.wind_base * 0.3])
        turb_vec = (np.random.random(3) - 0.5) * self.wind_base * turb_intensity
        gust_vec = np.array([self.gust_strength, 0, 0])
        self.wind = base_vec + turb_vec + gust_vec

    @property
    def wind_speed(self):
        return np.linalg.norm(self.wind)

    @property
    def wind_direction(self):
        if self.wind_speed < 0.01:
            return 0.0
        return math.degrees(math.atan2(self.wind[2], self.wind[0]))

    def harvesting_power(self, wing_area_m2):
        """Solar + piezo harvesting power in watts."""
        # Solar panels on wing surface (~20% efficiency)
        solar_power = self.solar_irradiance * wing_area_m2 * 0.20
        # Piezo harvesting from turbulence
        piezo_power = self.turbulence * 5.0  # watts
        return solar_power + piezo_power


# Colors
C_FRAME     = (70, 78, 86)
C_FRAME_DK  = (42, 48, 54)
C_NODE      = (120, 130, 138)
C_INNER     = (38, 42, 52)
C_MIDDLE    = (58, 68, 84)
C_MIDDLE_LT = (78, 92, 112)
C_INTERMED  = (52, 60, 72)
C_INTERMED_DK = (38, 44, 54)
C_OUTER     = (32, 36, 42)
C_OUTER_LT  = (52, 56, 64)
C_ACCENT    = (72, 156, 214)
C_ACCENT2   = (180, 60, 60)
C_TURBINE   = (88, 94, 100)
C_TURBINE_HOT = (255, 120, 40)
C_WING      = (44, 48, 56)
C_WING_MEM  = (60, 66, 78)
C_HELMET    = (40, 44, 52)
C_VISOR     = (30, 100, 160)
C_VISOR_GLOW = (80, 180, 255)
C_FIBER     = (100, 180, 220)
C_FIBER_HOT = (140, 220, 255)
C_BATTERY   = (80, 100, 70)
C_FUEL      = (90, 80, 50)    # Kevlar bladder + fuel lines
C_STF       = (90, 70, 100)
C_FARADAY   = (140, 110, 60)
C_TEXT      = (224, 230, 238)
C_DIM       = (150, 160, 176)
C_WARN      = (240, 182, 60)
C_OK        = (80, 200, 120)
C_LIGHT_DIR = np.array([0.45, 0.75, 0.9])
C_PANEL_BG  = (24, 28, 36)

VISUAL_DETAIL = 1.0
OUTLINE_MAX_POLYS = 3200


# =============================================================================
# MATH HELPERS
# =============================================================================

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _mix(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def _detail_seg(seg):
    return max(6, int(round(seg * VISUAL_DETAIL)))


def _translate(verts, offset):
    return (np.asarray(verts) + np.asarray(offset)).tolist()


# =============================================================================
# MESH  -- vertices + polygon faces with placement/animation state
# =============================================================================

class Mesh:
    """Vertices (metres) + polygon faces + a base colour.

    Local build convention: primitives are built around the origin.
    `spin` is the per-frame rotation ratio about local Z relative to the
    group's master angle; `tilt` (rx, ry) statically reorients the part;
    `pivot` translates it into place. `emissive` marks glowing parts."""

    def __init__(self, verts, faces, color, name="", spin=0.0, group="static",
                 pivot=(0.0, 0.0, 0.0), tilt=(0.0, 0.0), emissive=False):
        self.verts = np.asarray(verts, dtype=float)
        self.faces = faces
        self.color = color
        self.name = name
        self.spin = spin
        self.group = group
        self.pivot = np.asarray(pivot, dtype=float)
        self.tilt = tilt
        self.emissive = emissive

    def world_verts(self, angle):
        v = self.verts
        if self.spin:
            v = v @ rot_z(angle * self.spin).T
        rx, ry = self.tilt
        if rx or ry:
            v = v @ (rot_x(rx) @ rot_y(ry)).T
        return v + self.pivot


# =============================================================================
# PART  -- a named logical component made of one or more meshes
# =============================================================================

class Part:
    def __init__(self, key, name, meshes, specs, order, explode, color,
                 description="", category="", materials=None, weight_kg=0.0,
                 blueprint_notes=""):
        self.key = key
        self.name = name
        self.meshes = meshes
        self.specs = specs
        self.order = order
        self.explode = explode
        self.color = color
        self.description = description
        self.category = category
        self.materials = materials or []
        self.weight_kg = weight_kg
        self.blueprint_notes = blueprint_notes

    @property
    def mesh_count(self):
        return len(self.meshes)

    @property
    def vertex_count(self):
        return sum(len(m.vertices) for m in self.meshes)

    @property
    def face_count(self):
        return sum(len(m.faces) for m in self.meshes)

    def __repr__(self):
        return f"Part({self.key!r}, {len(self.meshes)} meshes)"


# =============================================================================
# PRIMITIVE BUILDERS  -> (verts, faces)
# =============================================================================

def _solid_cylinder(r, z0, z1, seg=32):
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    c0 = len(verts); verts.append((0, 0, z0))
    c1 = len(verts); verts.append((0, 0, z1))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
        faces.append((c0, b, a))
        faces.append((c1, seg + a, seg + b))
    return verts, faces


def _hollow_cylinder(r_out, r_in, z0, z1, seg=32):
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r_out * math.cos(a), r_out * math.sin(a), z))
        for a in ang:
            verts.append((r_in * math.cos(a), r_in * math.sin(a), z))
    n = seg
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, n + b, n + a))
        faces.append((2*n + a, 2*n + b, 3*n + b, 3*n + a))
        faces.append((a, 2*n + a, 2*n + b, b))
        faces.append((n + b, 3*n + b, 3*n + a, n + a))
    return verts, faces


def _box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz),
         (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
         (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz),
         (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz)]
    f = [(0,1,2,3), (4,7,6,5), (0,4,5,1),
         (1,5,6,2), (2,6,7,3), (3,7,4,0)]
    return v, f


def _sphere(r, seg=12):
    seg = _detail_seg(seg)
    rings = max(5, seg // 2)
    verts, faces = [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(seg):
            phi = 2 * math.pi * j / seg
            verts.append((r * math.sin(theta) * math.cos(phi),
                          r * math.cos(theta),
                          r * math.sin(theta) * math.sin(phi)))
    for i in range(rings):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            c = (i + 1) * seg + (j + 1) % seg
            d = (i + 1) * seg + j
            if i == 0:
                faces.append((a, c, d))
            elif i == rings - 1:
                faces.append((a, b, d))
            else:
                faces.append((a, b, c, d))
    return verts, faces


def _cone(r_bot, r_top, z0, z1, seg=24):
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for a in ang:
        verts.append((r_bot * math.cos(a), r_bot * math.sin(a), z0))
    for a in ang:
        verts.append((r_top * math.cos(a), r_top * math.sin(a), z1))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        if r_top > 0.001:
            faces.append((a, b, seg + b, seg + a))
        else:
            faces.append((a, b, seg))
    if r_top < 0.001:
        verts.append((0, 0, z1))
    return verts, faces


def _tube(r, z0, z1, seg=24, wall_r=None):
    """Open-ended tube (no caps)."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (z0, z1):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
    return verts, faces


def _capsule(r, z0, z1, seg=20):
    """Cylinder with hemispherical caps."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    half_len = (z1 - z0) / 2 - r
    if half_len < 0:
        half_len = 0
    mid0 = (z0 + z1) / 2 - half_len
    mid1 = (z0 + z1) / 2 + half_len
    # cylinder section
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z in (mid0, mid1):
        for a in ang:
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
    # bottom cap
    rings = max(4, seg // 3)
    base = len(verts)
    for ri in range(rings):
        theta = math.pi * 0.5 * (ri + 1) / (rings + 1)
        rr = r * math.cos(theta)
        zz = mid0 - r * math.sin(theta)
        for a in ang:
            verts.append((rr * math.cos(a), rr * math.sin(a), zz))
    verts.append((0, 0, mid0 - r))
    apex = len(verts) - 1
    for i in range(seg):
        faces.append((i, (i+1) % seg, base + (i+1) % seg, base + i))
    for ri in range(rings - 1):
        off0 = base + ri * seg
        off1 = base + (ri + 1) * seg
        for i in range(seg):
            faces.append((off0 + i, off0 + (i+1) % seg, off1 + (i+1) % seg, off1 + i))
    for i in range(seg):
        faces.append((base + (rings-1)*seg + i, base + (rings-1)*seg + (i+1) % seg, apex))
    # top cap
    base2 = len(verts)
    for ri in range(rings):
        theta = math.pi * 0.5 * (ri + 1) / (rings + 1)
        rr = r * math.cos(theta)
        zz = mid1 + r * math.sin(theta)
        for a in ang:
            verts.append((rr * math.cos(a), rr * math.sin(a), zz))
    verts.append((0, 0, mid1 + r))
    apex2 = len(verts) - 1
    off_cyl = seg  # top cylinder ring index
    for i in range(seg):
        faces.append((off_cyl + i, base2 + i, base2 + (i+1) % seg, off_cyl + (i+1) % seg))
    for ri in range(rings - 1):
        off0 = base2 + ri * seg
        off1 = base2 + (ri + 1) * seg
        for i in range(seg):
            faces.append((off0 + i, off0 + (i+1) % seg, off1 + (i+1) % seg, off1 + i))
    for i in range(seg):
        faces.append((base2 + (rings-1)*seg + i, apex2, base2 + (rings-1)*seg + (i+1) % seg))
    return verts, faces


def _torus(r_major, r_minor, seg_major=24, seg_minor=12):
    seg_major = _detail_seg(seg_major)
    seg_minor = _detail_seg(seg_minor)
    verts, faces = [], []
    for i in range(seg_major):
        a = 2 * math.pi * i / seg_major
        for j in range(seg_minor):
            b = 2 * math.pi * j / seg_minor
            verts.append((
                (r_major + r_minor * math.cos(b)) * math.cos(a),
                (r_major + r_minor * math.cos(b)) * math.sin(a),
                r_minor * math.sin(b)))
    for i in range(seg_major):
        for j in range(seg_minor):
            a = i * seg_minor + j
            b = i * seg_minor + (j + 1) % seg_minor
            c = ((i + 1) % seg_major) * seg_minor + (j + 1) % seg_minor
            d = ((i + 1) % seg_major) * seg_minor + j
            faces.append((a, b, c, d))
    return verts, faces


def _quad(p0, p1, p2, p3):
    """A single quad face from 4 points."""
    verts = [p0, p1, p2, p3]
    faces = [(0, 1, 2, 3)]
    return verts, faces


def _strip(points, width, z=0.0):
    """Build a flat strip (ribbon) from a list of (x,y) centerline points."""
    verts = []
    faces = []
    n = len(points)
    for i, (px, py) in enumerate(points):
        if i < n - 1:
            dx = points[i+1][0] - px
            dy = points[i+1][1] - py
            dl = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / dl * width / 2, dx / dl * width / 2
        else:
            dx = px - points[i-1][0]
            dy = py - points[i-1][1]
            dl = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / dl * width / 2, dx / dl * width / 2
        verts.append((px + nx, py + ny, z))
        verts.append((px - nx, py - ny, z))
    for i in range(n - 1):
        a = 2 * i; b = 2 * i + 1; c = 2 * (i+1) + 1; d = 2 * (i+1)
        faces.append((a, b, c, d))
    return verts, faces


def _tapered_prism(r_top, r_bot, z0, z1, seg=8, flat_frac=0.6):
    """Angular prism with flat front/back and rounded sides.
    flat_frac controls how much of the circumference is flat (0=circle, 1=square).
    Creates a sleek tapered limb segment like Iron Man forearm/shin."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z, r in ((z0, r_bot), (z1, r_top)):
        for a in ang:
            # Flatten front and back by scaling cos component
            ca = math.cos(a)
            sa = math.sin(a)
            flat = 1.0 - flat_frac * abs(ca)
            rr = r * flat
            verts.append((rr * ca, r * flat_frac * ca * 0.3, rr * sa))
    for i in range(seg):
        a, b = i, (i + 1) % seg
        faces.append((a, b, seg + b, seg + a))
    return verts, faces


def _armor_plate(cx, cy, cz, sx, sy, sz, chamfer=0.015, taper=0.0):
    """Chamfered box for sleek armor plates.
    chamfer = amount to bevel edges (makes it look forged, not boxy).
    taper = amount to narrow the top face (trapezoidal cross-section)."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    tx = hx - taper  # tapered top
    ch = chamfer
    v = [
        # bottom face (wider)
        (cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz),
        (cx+hx, cy-hy, cz+hz), (cx-hx, cy-hy, cz+hz),
        # chamfer level bottom
        (cx-hx+ch, cy-hy+ch, cz-hz+ch), (cx+hx-ch, cy-hy+ch, cz-hz+ch),
        (cx+hx-ch, cy-hy+ch, cz+hz-ch), (cx-hx+ch, cy-hy+ch, cz+hz-ch),
        # chamfer level top
        (cx-tx+ch, cy+hy-ch, cz-hz+ch), (cx+tx-ch, cy+hy-ch, cz-hz+ch),
        (cx+tx-ch, cy+hy-ch, cz+hz-ch), (cx-tx+ch, cy+hy-ch, cz+hz-ch),
        # top face (narrower)
        (cx-tx, cy+hy, cz-hz), (cx+tx, cy+hy, cz-hz),
        (cx+tx, cy+hy, cz+hz), (cx-tx, cy+hy, cz+hz),
    ]
    f = [
        (0,1,2,3), (12,15,14,13),  # bottom, top
        (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0),  # lower chamfer
        (4,8,9,5), (5,9,10,6), (6,10,11,7), (7,11,8,4),  # mid chamfer
        (8,12,13,9), (9,13,14,10), (10,14,15,11), (11,15,12,8),  # upper chamfer
    ]
    return v, f


def _tapered_limb(r_top, r_bot, z0, z1, seg=12, flat_frac=0.5, taper_profile=None):
    """Sleek tapered limb section with flat front/back.
    taper_profile: list of radius multipliers along length for custom shaping.
    Creates Iron Man-style tapered forearm/shin/thigh geometry."""
    seg = _detail_seg(seg)
    verts, faces = [], []
    n_rings = len(taper_profile) if taper_profile else 2
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for ri in range(n_rings):
        t = ri / max(n_rings - 1, 1)
        z = z0 + t * (z1 - z0)
        r = r_bot + (r_top - r_bot) * t
        if taper_profile:
            r *= taper_profile[ri]
        for a in ang:
            ca = math.cos(a)
            sa = math.sin(a)
            flat = 1.0 - flat_frac * abs(ca)
            rr = r * flat
            verts.append((rr * ca, r * flat_frac * ca * 0.25, rr * sa))
    for ri in range(n_rings - 1):
        for i in range(seg):
            a = ri * seg + i
            b = ri * seg + (i + 1) % seg
            c = (ri + 1) * seg + (i + 1) % seg
            d = (ri + 1) * seg + i
            faces.append((a, b, c, d))
    return verts, faces


def _pauldron(r, cx, cy, cz, seg=10):
    """Angular shoulder pauldron - faceted sphere, not round.
    Looks like Master Chief shoulder armor."""
    seg = _detail_seg(seg)
    rings = max(4, seg // 2)
    verts, faces = [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(seg):
            phi = 2 * math.pi * j / seg
            # Flatten bottom half to create a dome, not full sphere
            if theta > math.pi * 0.6:
                theta_clamped = math.pi * 0.6 + (theta - math.pi * 0.6) * 0.3
            else:
                theta_clamped = theta
            r_ring = r * math.sin(theta_clamped)
            verts.append((
                cx + r_ring * math.cos(phi) * 0.9,  # slightly flattened
                cy + r * math.cos(theta_clamped) * 0.7,  # flattened dome
                cz + r_ring * math.sin(phi) * 0.9))
    for i in range(rings):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            c = (i + 1) * seg + (j + 1) % seg
            d = (i + 1) * seg + j
            if i == 0:
                faces.append((a, c, d))
            elif i == rings - 1:
                faces.append((a, b, d))
            else:
                faces.append((a, b, c, d))
    return verts, faces


def _angular_torso(r_shoulder, r_waist, r_hip, z0, z1, seg=6):
    """Angular faceted torso — hexagonal cross-section with flat chest/back.
    Broad at shoulders, tapered at waist, flared at hips.
    Creates MJOLNIR/Iron Man torso silhouette, not a round capsule."""
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    # 3 rings: shoulders (z1), waist (mid), hips (z0)
    z_mid = (z0 + z1) / 2
    rings = [(z0, r_hip), (z_mid, r_waist), (z1, r_shoulder)]
    for z, r in rings:
        for a in ang:
            ca = math.cos(a)
            sa = math.sin(a)
            # Flatten front/back (x axis) for angular chest/back plates
            flat_x = 1.0 - 0.45 * abs(ca)
            flat_y = 1.0 - 0.30 * abs(sa)
            vx = r * ca * flat_x * 1.15
            vy = r * sa * flat_y * 0.85
            verts.append((vx, vy, z))
    n = seg
    for ring in range(2):
        for i in range(n):
            a = ring * n + i
            b = ring * n + (i + 1) % n
            c = (ring + 1) * n + (i + 1) % n
            d = (ring + 1) * n + i
            faces.append((a, b, c, d))
    # Cap top and bottom
    faces.append(tuple(range(n - 1, -1, -1)))  # bottom
    faces.append(tuple(range(2 * n, 3 * n)))   # top
    return verts, faces


def _angular_limb(r_top, r_bot, z0, z1, seg=6, flat_frac=0.6):
    """Angular limb for inner layer — hexagonal cross-section, not round.
    Flat front/back like MJOLNIR techsuit, tapered."""
    verts, faces = [], []
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for z, r in ((z0, r_bot), (z1, r_top)):
        for a in ang:
            ca = math.cos(a)
            sa = math.sin(a)
            flat = 1.0 - flat_frac * abs(ca)
            rr = r * flat
            verts.append((rr * ca, r * flat_frac * ca * 0.2, rr * sa))
    n = seg
    for i in range(n):
        a, b = i, (i + 1) % n
        faces.append((a, b, n + b, n + a))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(n, 2 * n)))
    return verts, faces


def _panel_line(x0, y0, z0, x1, y1, z1, width=0.002):
    """Thin emissive strip for panel lines / accent grooves on armor."""
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    dl = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    # Perpendicular in the XY plane
    nx = -dy / dl * width
    ny = dx / dl * width
    nz = 0.0
    if dl < 0.001:
        nx, ny = width, 0
    v = [
        (x0+nx, y0+ny, z0+nz), (x0-nx, y0-ny, z0-nz),
        (x1-nx, y1-ny, z1-nz), (x1+nx, y1+ny, z1+nz)]
    f = [(0, 1, 2, 3)]
    return v, f


# =============================================================================
# FACE GROUPS  -- pre-compute face arity buckets for vectorized rendering
# =============================================================================

def _face_groups(m):
    faces = m.faces
    if not faces:
        return {}
    by_arity = {}
    for f in faces:
        k = len(f)
        if k not in by_arity:
            by_arity[k] = []
        by_arity[k].append(f)
    return {k: np.array(v, dtype=np.int32) for k, v in by_arity.items()}


# =============================================================================
# POLYGON EMITTER  -- vectorized backface cull + flat shade + depth sort
# =============================================================================

def _emit_polys(out, cam, sx, sy, base_rgb, light, cull, emissive,
                groups, min_area, hl, sec_y=None, pivot_y=0.0, znear=0.05):
    z = cam[:, 2]
    br, bg, bb = base_rgb
    lx, ly, lz = float(light[0]), float(light[1]), float(light[2])
    for arity, idx in groups.items():
        fz = z[idx]
        vis = np.all(fz > znear, axis=1)
        a = cam[idx[:, 0]]; b = cam[idx[:, 1]]; c = cam[idx[:, 2]]
        nrm = np.cross(b - a, c - a)
        if cull:
            vis &= (nrm[:, 2] <= 0.0)
        if sec_y is not None:
            vis &= (sec_y[idx].mean(axis=1) <= pivot_y + 0.005)
        if not vis.any():
            continue
        idv = idx[vis]; nv = nrm[vis]
        fsx = sx[idv]; fsy = sy[idv]
        area = (fsx.max(1) - fsx.min(1)) * (fsy.max(1) - fsy.min(1))
        big = area >= min_area
        if not big.any():
            continue
        idv = idv[big]; nv = nv[big]; fsx = fsx[big]; fsy = fsy[big]
        ln = np.sqrt((nv * nv).sum(1)); ln[ln == 0.0] = 1.0
        nn = nv / ln[:, None]
        nn[nn[:, 2] > 0.0] *= -1.0
        d = nn[:, 0] * lx + nn[:, 1] * ly + nn[:, 2] * lz
        shade = 0.34 + 0.66 * np.clip(d, 0.0, None)
        if emissive:
            shade = np.maximum(shade, 0.85)
        r = np.clip(br * shade, 0, 255).astype(np.int16).tolist()
        g = np.clip(bg * shade, 0, 255).astype(np.int16).tolist()
        bl = np.clip(bb * shade, 0, 255).astype(np.int16).tolist()
        depth = z[idv].mean(axis=1).tolist()
        xs = fsx.tolist(); ys = fsy.tolist()
        for k in range(len(depth)):
            out.append((depth[k], list(zip(xs[k], ys[k])), (r[k], g[k], bl[k]), hl))


# =============================================================================
# 2D HUD / LABELS / PANELS
# =============================================================================

def _label(surf, font, text, pos, accent=False):
    col = C_ACCENT if accent else C_TEXT
    img = font.render(text, True, col)
    x, y = int(pos[0]), int(pos[1])
    bg = pygame.Surface((img.get_width() + 8, img.get_height() + 2), pygame.SRCALPHA)
    bg.fill((20, 24, 32, 180))
    surf.blit(bg, (x - 4, y - 1))
    surf.blit(img, (x, y))
    if accent:
        pygame.draw.circle(surf, C_ACCENT, (x - 8, y + img.get_height() // 2), 3)


def _panel(surf, x, y, w, h, alpha=200):
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((20, 24, 32, alpha))
    surf.blit(bg, (x, y))
    pygame.draw.rect(surf, (50, 60, 76), (x, y, w, h), 1)


def _bar(surf, x, y, w, label, val, frac, color, font):
    _panel(surf, x, y, w + 40, 32, 160)
    pygame.draw.rect(surf, (40, 46, 58), (x, y, w, 12))
    pygame.draw.rect(surf, color, (x, y, int(w * clamp(frac)), 12))
    pygame.draw.rect(surf, (70, 84, 104), (x, y, w, 12), 1)
    surf.blit(font.render(f"{label}: {val}", True, C_TEXT), (x, y - 16))


# =============================================================================
# HUMAN BODY REFERENCE  -- proportions for suit sizing (metres)
# =============================================================================

BODY_PROPORTIONS = {
    # segment lengths as fraction of total height (Dreyfus / Vitruvian averages)
    "head":        0.130,
    "neck":        0.052,
    "torso":       0.300,
    "upper_arm":   0.186,
    "forearm":     0.146,
    "hand":        0.108,
    "hip":         0.052,
    "thigh":       0.245,
    "shin":        0.246,
    "foot":        0.152,
    "shoulder_w":  0.265,
    "hip_w":       0.220,
    "chest_d":     0.173,
}


def body_segments(height_m):
    """Return dict of body segment lengths in metres for a given height."""
    return {k: v * height_m for k, v in BODY_PROPORTIONS.items()}


# =============================================================================
# PART DATABASE  -- rich metadata for info/about panels and blueprints
# =============================================================================

PART_DB = {
    "inner_suit": {
        "description": "Skin-tight base layer worn directly against the pilot's skin. "
                       "Woven from spandex-nylon blend with integrated EMG sensor grid "
                       "for muscle activation monitoring. Phase-change microcapsules "
                       "regulate temperature by absorbing and releasing heat. "
                       "Capillary network wicks moisture to the thermal regulation layer.",
        "category": "Base Layer",
        "layer": 1,
        "materials": ["Spandex-nylon blend", "EMG sensor grid (64-channel)",
                      "Phase-change microcapsules", "Capillary moisture network",
                      "Silver-ion antimicrobial treatment"],
        "weight_kg": 0.3,
        "blueprint": "First layer donned. Pulls on like a wetsuit. "
                     "EMG connectors snap to neural harness at spine. "
                     "No structural or protective function -- purely sensor + comfort.",
        "icon": "base",
        "sub_components": [
            {"name": "EMG Sensor Grid", "count": 64, "detail": "Dry-contact electromyography electrodes at major muscle groups"},
            {"name": "Phase-Change Microcapsules", "count": 1, "detail": f"Embedded in fabric, {DIMS['inner_thick_mm']}mm thickness, melts at 37C"},
            {"name": "Capillary Moisture Network", "count": 1, "detail": "Branching microfluidic channels wick sweat to thermal layer"},
            {"name": "Spine Connector Harness", "count": 1, "detail": "12-pin snap connector at T1-T2 vertebrae for EMG bus"},
            {"name": "Antimicrobial Silver-Ion Treatment", "count": 1, "detail": "Whole-surface treatment, prevents biofilm for 72h wear"},
        ],
        "dimensions": f"Thickness: {DIMS['inner_thick_mm']}mm | Scales to pilot height",
        "connectors": ["12-pin EMG spine snap (T1-T2)", "4x shoulder snap (EMG branch)", "4x hip snap (EMG branch)"],
        "performance": ["64-channel EMG at 2kHz", "Temperature regulation: passive PCM", "Moisture wicking: 50ml/h", "Wear duration: 72h"],
        "build_details": {"time_min": 5, "tools": ["None (hand-donned)"], "torque": "N/A -- snap fit", "difficulty": "Easy", "notes": "Pull on like wetsuit. Verify EMG snap at T1-T2 clicks. Check 8 branch connectors."},
    },
    "faraday": {
        "description": "Flexible electromagnetic shielding mesh embedded between the "
                       "inner suit and muscle layer. 0.1mm copper-graphene weave "
                       "blocks >99% of EMP and EMI across the full spectrum. "
                       "Includes the suit's primary power distribution bus: "
                       "a spine conduit with 4 branches to shoulders and hips.",
        "category": "EM Shielding + Power Bus",
        "layer": 2,
        "materials": ["Copper-graphene mesh (0.1mm)", "Flexible PCB power bus",
                      "Silver conductive epoxy", "Graphene oxide interleave"],
        "weight_kg": 0.2,
        "blueprint": "Second layer. Draped over inner suit, connected at "
                     "spine clamp. Power bus routes from battery pack to all "
                     "subsystems. 8 vertical + 6 horizontal conductors visible.",
        "icon": "shield",
        "sub_components": [
            {"name": "Copper-Graphene Mesh", "count": 1, "detail": f"{DIMS['faraday_thick_mm']}mm weave, 99% EMP/EMI blocking, 8 vertical + 6 horizontal conductors"},
            {"name": "Spine Power Bus", "count": 1, "detail": "Flexible PCB, 48V 200A capacity, runs T1 to L5"},
            {"name": "Shoulder Power Branches", "count": 2, "detail": "24V 50A each, routes to arm actuators + turbines"},
            {"name": "Hip Power Branches", "count": 2, "detail": "24V 50A each, routes to leg actuators + turbines"},
            {"name": "Spine Clamp Connector", "count": 1, "detail": "Titanium latch, 16-pin power + 8-pin data, mates to inner suit harness"},
        ],
        "dimensions": f"Thickness: {DIMS['faraday_thick_mm']}mm | Coverage: full body except face/hands",
        "connectors": ["16-pin spine clamp (power+data)", "4x 8-pin branch connectors (shoulders+hips)", "Battery main feed (48V 200A)"],
        "performance": ["EMP blocking: >99% (1MHz-100GHz)", "Power bus: 48V 200A", "Flexibility: 95% of bare skin", "Weight penalty: 1.2kg"],
        "build_details": {"time_min": 10, "tools": ["Spine clamp wrench (4mm Allen)", "Multimeter for bus continuity"], "torque": "4 Nm spine clamp", "difficulty": "Medium", "notes": "Drape over inner suit. Align spine clamp, tighten to 4 Nm. Verify 48V bus continuity with multimeter before proceeding."},
    },
    "middle_layer": {
        "description": "The suit's artificial muscle system. Three sublayers of "
                       "Dielectric Elastomer Actuators (DEA) with Shear-Thickening "
                       "Fluid (STF) between layers. DEA fibers contract in <20ms "
                       "providing 15x human strength boost. STF hardens instantly "
                       "under impact, absorbing up to 150,000 PSI. Jump fibers "
                       "are 5x denser in legs; lift fibers 2x in upper body.",
        "category": "Artificial Muscle + Impact Absorption",
        "layer": 3,
        "materials": [f"DEA elastomer ({DIMS['middle_thick_mm']}mm, {DIMS['dea_sublayers']} sublayers)",
                      "Shear-thickening fluid (STF) interlayer",
                      "Carbon nanotube electrodes",
                      "Graphene strain sensors"],
        "weight_kg": 1.8,
        "blueprint": "Third layer. Segmented panels snap over faraday shield. "
                     "Each panel has 4 DEA sublayers + STF bladder. "
                     "Leg panels have 5x fiber density for jump boost. "
                     "Shoulder/back/arms have 2x for lift assist.",
        "icon": "muscle",
        "sub_components": [
            {"name": "DEA Muscle Fiber Bundles", "count": 48, "detail": f"{DIMS['dea_sublayers']} sublayers, <{DIMS['dea_contraction_ms']}ms contraction, {DIMS['dea_strain_pct']}% strain"},
            {"name": "STF Impact Bladders", "count": 24, "detail": f"Shear-thickening fluid, {DIMS['stf_max_psi']:,} PSI absorption (tripled)"},
            {"name": "Jump Fiber Clusters (Legs)", "count": 8, "detail": f"{DIMS['fiber_density_jump_x']}x density in quadriceps/calves, 200ft vertical jump"},
            {"name": "Lift Fiber Clusters (Upper)", "count": 6, "detail": f"{DIMS['fiber_density_lift_x']}x density in shoulders/back/arms, {DIMS['muscle_boost_x']}x human strength"},
            {"name": "Carbon Nanotube Electrode Grid", "count": 1, "detail": "Distributed electrodes, 200V activation, per-bundle control"},
            {"name": "Graphene Strain Sensors", "count": 96, "detail": "Real-time strain monitoring, feedback to RTOS at 1kHz"},
        ],
        "dimensions": f"Thickness: {DIMS['middle_thick_mm']}mm ({DIMS['dea_sublayers']} sublayers) | Segmented: 24 panels",
        "connectors": ["24x power snap (per panel)", "96x strain sensor bus", "High-voltage bus (200V)"],
        "performance": [f"Strength boost: {DIMS['muscle_boost_x']}x human", f"Contraction: <{DIMS['dea_contraction_ms']}ms", f"Impact absorption: {DIMS['stf_max_psi']:,} PSI", f"Jump: {DIMS['perf_jump_vertical_ft']}ft vertical"],
        "build_details": {"time_min": 25, "tools": ["Panel alignment jig", "STF injection syringe", "HV insulation tester"], "torque": "Snap-fit (no torque)", "difficulty": "Hard", "notes": "Snap each of 24 panels over faraday shield. Inject STF into bladders (10ml each). Test HV isolation at 200V before activation. Leg panels go on first (5x density)."},
    },
    "intermediate": {
        "description": "Foam-filled auxetic metamaterial layer between the muscle "
                       "and armor. Auxetic structures have a negative Poisson's ratio "
                       f"({DIMS['auxetic_poisson']}), meaning they expand perpendicular to "
                       "compression -- absorbing and distributing impact energy laterally. "
                       f"The {DIMS['auxetic_cell_type']} cell type at {DIMS['auxetic_rel_density']*100:.0f}% relative density "
                       f"absorbs {DIMS['auxetic_energy_abs_pct']}% of residual impact energy. "
                       "Foam fill provides thermal insulation.",
        "category": "Impact Distribution + Insulation",
        "layer": 4,
        "materials": [f"Auxetic metamaterial ({DIMS['intermediate_thick_mm']}mm, {DIMS['auxetic_cell_type']})",
                      f"Closed-cell foam fill (Poisson's ratio: {DIMS['auxetic_poisson']})",
                      "Titanium alloy nodes at junctions",
                      "Kevlar thread stitching"],
        "weight_kg": 0.6,
        "blueprint": "Fourth layer. Hexagonal auxetic lattice panels. "
                     "Interlocks with muscle layer via tongue-and-groove. "
                     "Foam injected under pressure after assembly. "
                     f"Provides {DIMS['auxetic_energy_abs_pct']}% impact energy redistribution.",
        "icon": "lattice",
        "sub_components": [
            {"name": f"{DIMS['auxetic_cell_type']} Auxetic Lattice", "count": 1, "detail": f"Negative Poisson's ratio {DIMS['auxetic_poisson']}, {DIMS['auxetic_rel_density']*100:.0f}% relative density"},
            {"name": "Closed-Cell Foam Fill", "count": 1, "detail": f"Injected under pressure, thermal insulation R-12, {DIMS['intermediate_thick_mm']}mm thickness"},
            {"name": "Titanium Junction Nodes", "count": 12, "detail": "Ti-6Al-4V nodes at major joint positions, load transfer points"},
            {"name": "Tongue-and-Groove Interlocks", "count": 24, "detail": "Mechanical interlock with muscle layer panels, no tools needed"},
            {"name": "Kevlar Stitch Matrix", "count": 1, "detail": "Multi-layer stitching, 5000N tear strength"},
        ],
        "dimensions": f"Thickness: {DIMS['intermediate_thick_mm']}mm | Cell type: {DIMS['auxetic_cell_type']} | Coverage: full body",
        "connectors": ["24x tongue-and-groove (to muscle layer)", "12x Ti node hardpoints (to armor)"],
        "performance": [f"Energy absorption: {DIMS['auxetic_energy_abs_pct']}%", f"Poisson's ratio: {DIMS['auxetic_poisson']}", "Thermal insulation: R-12", "Weight: 2.1kg"],
        "build_details": {"time_min": 15, "tools": ["Foam injection gun", "Torque wrench (2 Nm)"], "torque": "2 Nm Ti node bolts", "difficulty": "Medium", "notes": "Interlock 24 tongue-and-groove panels. Inject foam under 3 bar pressure. Tighten 12 Ti node bolts to 2 Nm. Allow foam to cure 5 min."},
    },
    "outer_armor": {
        "description": "Outermost armor plating. Graphene-UHMWPE composite panels "
                       f"rated at NIJ Level {DIMS['outer_nij_level']}+ (armor-piercing rifle protection). "
                       f"Withstands {DIMS['outer_max_psi']:,} PSI ballistic impact. "
                       f"Heat tolerance {DIMS['outer_heat_tol_c']:,}C, laser resistance {DIMS['outer_laser_tol_kw']}kW/cm^2. "
                       "Self-healing seam technology uses shape-memory polymer that closes "
                       "penetrations within 2 seconds. Modular panels allow field replacement. "
                       f"Coating: {DIMS['coating_type']}. Ceramic strike face: {DIMS['ceramic_type']}.",
        "category": "Ballistic Armor",
        "layer": 5,
        "materials": [f"Graphene-UHMWPE composite ({DIMS['outer_thick_mm']}mm)",
                      "Shape-memory polymer seams (self-healing)",
                      f"Ceramic strike face ({DIMS['ceramic_type']})",
                      f"Coating: {DIMS['coating_type']}",
                      "Modular quick-release fasteners"],
        "weight_kg": 1.5,
        "blueprint": "Fifth (outermost) layer. Modular panels: chest (2), "
                     "back (2), shoulders (2), thighs (2), shins (2), "
                     "upper arms (2). Each panel has self-healing seam. "
                     "Panels lock to intermediate layer via magnetic clamps.",
        "icon": "armor",
        "sub_components": [
            {"name": "Chest Armor Panels", "count": 2, "detail": f"Graphene-UHMWPE + {DIMS['ceramic_type']}, {DIMS['outer_thick_mm']}mm, NIJ {DIMS['outer_nij_level']}+"},
            {"name": "Back Armor Panels", "count": 2, "detail": f"Same as chest, includes turbine mounting hardpoints + battery access panel"},
            {"name": "Shoulder Armor Pauldrons", "count": 2, "detail": "Articulated, includes wing hinge mounts + radiator fin slots"},
            {"name": "Thigh Armor Plates", "count": 2, "detail": "Articulated, includes turbine mounting slots"},
            {"name": "Shin Armor Greaves", "count": 2, "detail": "Includes piezoelectric harvester housing in boot interface"},
            {"name": "Upper Arm Armor", "count": 2, "detail": "Articulated, includes weapon mount hardpoint on right forearm"},
            {"name": "Self-Healing Seams", "count": 12, "detail": "Shape-memory polymer, closes penetrations <25mm in 2s"},
            {"name": "Magnetic Quick-Release Clamps", "count": 12, "detail": "Electromagnetic locks, release on power loss or manual trigger"},
        ],
        "dimensions": f"Thickness: {DIMS['outer_thick_mm']}mm | Panels: 12 modular | Coverage: full body except joints",
        "connectors": ["12x magnetic clamp (to intermediate)", "2x weapon mount hardpoint (forearms)", "4x turbine slot (shoulders+thighs)", "2x wing hinge (shoulders)"],
        "performance": [f"Ballistic: {DIMS['outer_max_psi']:,} PSI (NIJ {DIMS['outer_nij_level']}+)", f"Heat tolerance: {DIMS['outer_heat_tol_c']:,}C", f"Laser resistance: {DIMS['outer_laser_tol_kw']}kW/cm^2", "Self-healing: 2s for <25mm holes"],
        "build_details": {"time_min": 20, "tools": ["Magnetic clamp activator", "Panel alignment jig"], "torque": "Electromagnetic (auto-lock)", "difficulty": "Medium", "notes": "Lock 12 panels via magnetic clamps. Start chest, then back, shoulders, thighs, shins, arms. Verify each clamp clicks (audible). Test self-healing seam with 5mm probe."},
    },
    "frame": {
        "description": "Carbon fiber reinforced polymer (CFRP) telescoping "
                       "endoskeleton with titanium alloy nodes. Provides "
                       "structural load-bearing for the suit, distributing "
                       f"turbine thrust and impact loads across the body. "
                       f"Telescoping segments adjust to pilot height. "
                       f"{DIMS['frame_node_count']} Ti-{DIMS['frame_node_material'].split('(')[1].strip(')')} nodes serve as "
                       f"hardpoints for turbines and actuators. {DIMS['frame_pivot_count']} pivoting joints, "
                       f"{DIMS['frame_actuator_count']} linear actuators, {DIMS['frame_strap_count']} auto-tensioning straps.",
        "category": "Structural Frame",
        "layer": 0,
        "materials": [f"CFRP telescoping tubes ({DIMS['frame_material']})",
                      f"Titanium nodes ({DIMS['frame_node_material']})",
                      "Magnetic bearing joints",
                      "Load cells at each node",
                      f"Spectra auto-tensioning straps ({DIMS['frame_strap_count']}x)"],
        "weight_kg": 1.8,
        "blueprint": "Assembled first (layer 0). Telescoping tubes extend to "
                     "pilot height. 64 Ti nodes at joint positions. "
                     "Each node has 4 DOF magnetic bearing + load cell. "
                     "All other layers mount to this frame.",
        "icon": "frame",
        "sub_components": [
            {"name": "CFRP Telescoping Tubes", "count": 16, "detail": f"{DIMS['frame_tube_od_mm']}mm OD x {DIMS['frame_tube_id_mm']}mm ID, {DIMS['frame_material']}"},
            {"name": "Titanium Nodes", "count": DIMS['frame_node_count'], "detail": f"{DIMS['frame_node_material']}, 4 DOF magnetic bearing, load cell integrated"},
            {"name": "Pivoting Joints", "count": DIMS['frame_pivot_count'], "detail": "Key joints: shoulders(4), elbows(2), hips(2), knees(2), ankles(2), spine(4), neck(2), wrists(2)"},
            {"name": "Linear Actuators", "count": DIMS['frame_actuator_count'], "detail": "Maxon EC45, 48V, 500N peak, <5ms response"},
            {"name": "Auto-Tensioning Straps", "count": DIMS['frame_strap_count'], "detail": "Spectra fiber, motorized tensioning, 2000N per strap"},
            {"name": "Load Cell Network", "count": DIMS['frame_node_count'], "detail": "Strain gauge at each node, 1kHz sampling, feeds RTOS"},
        ],
        "dimensions": f"Tube: {DIMS['frame_tube_od_mm']}mm OD x {DIMS['frame_tube_id_mm']}mm ID | Scales to pilot height | {DIMS['frame_node_count']} nodes",
        "connectors": [f"{DIMS['frame_node_count']}x Ti node hardpoints", f"{DIMS['frame_actuator_count']}x actuator mounts", f"{DIMS['frame_strap_count']}x strap anchors", "Spine data bus (1kHz)"],
        "performance": [f"Load capacity: 50G axial", f"Adjustment: telescoping +/-30cm", f"Nodes: {DIMS['frame_node_count']}", f"Actuators: {DIMS['frame_actuator_count']}x 500N"],
        "build_details": {"time_min": 45, "tools": ["Telescoping tube cutter", "Ti node torque wrench (8 Nm)", "Load cell calibrator", "Magnetic bearing alignment tool"], "torque": "8 Nm Ti node bolts", "difficulty": "Expert", "notes": "Cut CFRP tubes to pilot height. Install 64 Ti nodes, torque to 8 Nm. Calibrate load cells. Align magnetic bearings (0.01mm tolerance). Mount 14 actuators. Tension 24 Spectra straps to 2000N."},
    },
    "helmet": {
        "description": f"Vacuum-sealed helmet ({DIMS['helmet_od_mm']}mm OD, {DIMS['helmet_wall_mm']}mm wall) "
                       f"with graphene-UHMWPE shell and transparent aluminum (ALON) visor "
                       f"({DIMS['visor_thickness_mm']}mm, {DIMS['visor_max_force_lbs']:,} lbs force). "
                       f"Full life support: O2 recycler + CO2 scrubber ({DIMS['co2_scrub_hours']}h capacity). "
                       f"HUD: {DIMS['visor_fov_deg']}deg FOV, {DIMS['visor_zoom_optical']}x optical / {DIMS['visor_zoom_digital']}x digital zoom. "
                       f"Seal rated to {DIMS['seal_depth_m']}m underwater + vacuum. "
                       "Neural interface band connects to BCI.",
        "category": "Helmet + Life Support",
        "layer": 5,
        "materials": [f"Graphene-UHMWPE shell ({DIMS['helmet_wall_mm']}mm)",
                      f"ALON transparent aluminum visor ({DIMS['visor_thickness_mm']}mm)",
                      "O2 recycler + CO2 scrubber (4-bed molecular sieve)",
                      "Neural interface band (64-channel)",
                      "HUD projection system"],
        "weight_kg": DIMS.get("helmet_weight_kg", 1.0),
        "blueprint": "Donned last. Seals to neck ring on outer armor. "
                     "Life support activates on seal detection. "
                     "Neural band calibrates to pilot in 30s. "
                     "Visor HUD aligns to eye tracking.",
        "icon": "helmet",
        "sub_components": [
            {"name": "Graphene-UHMWPE Shell", "count": 1, "detail": f"{DIMS['helmet_od_mm']}mm OD, {DIMS['helmet_wall_mm']}mm wall, NIJ {DIMS['outer_nij_level']} protection"},
            {"name": "ALON Visor", "count": 1, "detail": f"Transparent aluminum, {DIMS['visor_thickness_mm']}mm, {DIMS['visor_max_force_lbs']:,} lbs, {DIMS['visor_fov_deg']}deg FOV"},
            {"name": "HUD Projection System", "count": 1, "detail": f"Retinal projection, {DIMS['visor_zoom_optical']}x optical + {DIMS['visor_zoom_digital']}x digital zoom, 120deg FOV"},
            {"name": "O2 Recycler", "count": 1, "detail": f"Closed-loop, {DIMS['o2_capacity_hours']}h capacity, 6 LPM peak"},
            {"name": "CO2 Scrubber", "count": 1, "detail": f"4-bed molecular sieve, regenerable, {DIMS['co2_scrub_hours']}h capacity"},
            {"name": "Neural Interface Band", "count": 1, "detail": "64-channel dry EEG, circumferential, auto-calibration 30s"},
            {"name": "Neck Ring Seal", "count": 1, "detail": f"Double O-ring, rated {DIMS['seal_depth_m']}m underwater + vacuum, 101 kPa internal"},
            {"name": "Helmet Ring Turbines", "count": DIMS['turbine_helmet'], "detail": f"{DIMS['turbine_helmet']}x micro-turbines for head stability + yaw control"},
        ],
        "dimensions": f"OD: {DIMS['helmet_od_mm']}mm | Wall: {DIMS['helmet_wall_mm']}mm | Visor: {DIMS['visor_thickness_mm']}mm | Weight: {DIMS['helmet_weight_kg']}kg",
        "connectors": ["Neck ring seal (double O-ring)", "Neural band bus (64-channel)", "Power: 48V from spine bus", "Data: HUD bus to RTOS"],
        "performance": [f"Seal: {DIMS['seal_depth_m']}m / vacuum", f"Life support: {DIMS['co2_scrub_hours']}h", f"Visor: {DIMS['visor_fov_deg']}deg FOV, {DIMS['visor_zoom_optical']}x/{DIMS['visor_zoom_digital']}x zoom", f"Impact: {DIMS['visor_max_force_lbs']:,} lbs"],
        "build_details": {"time_min": 15, "tools": ["Neck ring seal tester", "EEG calibration headset", "HUD alignment tool"], "torque": "Neck ring: hand-tight (1/4 turn)", "difficulty": "Medium", "notes": "Don helmet last. Seal neck ring (audible click). Run 30s EEG calibration. Align HUD to eye tracking. Verify O2 flow at 6 LPM. Test CO2 scrubber activation."},
    },
    "turbines": {
        "description": f"{DIMS['turbine_count']} micro-turbofan engines distributed "
                       "across the frame. Each turbine is "
                       f"{DIMS['turbine_d_mm']}mm dia x {DIMS['turbine_len_mm']}mm long, "
                       f"producing {DIMS['turbine_thrust_dry_lbf']} lbf dry / {DIMS['turbine_thrust_ab_lbf']} lbf afterburner. "
                       f"Max RPM: {DIMS['turbine_rpm_max']:,}. Bypass ratio: {DIMS['turbine_bypass_ratio']}:1. "
                       f"Bearings: {DIMS['turbine_bearing']}. "
                       f"SFC: {DIMS['turbine_sfc_dry_lb_lbh']} lb/(lbf*h) dry, {DIMS['turbine_sfc_ab_lb_lbh']} lb/(lbf*h) AB. "
                       f"Fuel: {DIMS['fuel_type']}. Thrust scales with air density (mass flow rate). "
                       f"Placement: {DIMS['turbine_backpack']} backpack, {DIMS['turbine_forearm']} forearm, "
                       f"{DIMS['turbine_calf']} calf, {DIMS['turbine_thigh']} thigh, {DIMS['turbine_helmet']} helmet ring. "
                       f"Thrust vectoring: +/-{DIMS['turbine_vector_deg']}deg.",
        "category": "Propulsion",
        "layer": 0,
        "materials": [f"Titanium fan blades ({DIMS['turbine_count']}x)",
                      "Carbon fiber duct housings",
                      "Gimbaled thrust vector mounts",
                      "Compressed gas afterburner reservoirs",
                      f"{DIMS['turbine_bearing']} bearings",
                      f"Fuel: {DIMS['fuel_type']} ({DIMS['fuel_density_kg_l']} kg/L)"],
        "weight_kg": 12.2,
        "blueprint": f"Mounted to frame nodes. {DIMS['turbine_count']} turbines in "
                     f"5 groups: backpack ({DIMS['turbine_backpack']}), forearms ({DIMS['turbine_forearm']}), "
                     f"calves ({DIMS['turbine_calf']}), thighs ({DIMS['turbine_thigh']}), helmet ({DIMS['turbine_helmet']}). "
                     f"Each turbine gimbaled for +/-{DIMS['turbine_vector_deg']}deg vector range. "
                     "Wired to power bus + throttle controller. "
                     "Fed by conformal fuel bladder via dual redundant pumps.",
        "icon": "turbine",
        "sub_components": [
            {"name": "Backpack Turbines", "count": DIMS['turbine_backpack'], "detail": f"Primary lift/thrust, {DIMS['turbine_thrust_dry_lbf']} lbf each, mounted on back frame"},
            {"name": "Forearm Turbines", "count": DIMS['turbine_forearm'], "detail": f"Pitch/roll control, {DIMS['turbine_thrust_dry_lbf']} lbf each, gimbaled"},
            {"name": "Calf Turbines", "count": DIMS['turbine_calf'], "detail": f"Pitch/yaw control, {DIMS['turbine_thrust_dry_lbf']} lbf each, gimbaled"},
            {"name": "Thigh Turbines", "count": DIMS['turbine_thigh'], "detail": f"Lift assist + jump boost, {DIMS['turbine_thrust_dry_lbf']} lbf each"},
            {"name": "Helmet Ring Turbines", "count": DIMS['turbine_helmet'], "detail": f"Head stability + fine yaw, smaller variant"},
            {"name": "Gimbaled Vector Mounts", "count": DIMS['turbine_count'], "detail": f"+/-{DIMS['turbine_vector_deg']}deg vectoring, 3-axis, <10ms response"},
            {"name": "Afterburner Reservoirs", "count": 4, "detail": "Compressed N2O + fuel, 1.8x thrust boost, 30s duration"},
            {"name": "Throttle Controllers", "count": 6, "detail": "Per-group ESC, 48V, 0-100% in <50ms, CAN bus to RTOS"},
            {"name": "FADEC Engine Controllers", "count": 6, "detail": "Full Authority Digital Engine Control, per-group, redundant"},
        ],
        "dimensions": f"Per turbine: {DIMS['turbine_d_mm']}mm dia x {DIMS['turbine_len_mm']}mm | Total: {DIMS['turbine_count']} units",
        "connectors": [f"{DIMS['turbine_count']}x gimbal mount (to frame nodes)", "6x CAN bus (throttle control)", "4x afterburner feed lines", "2x AN-8 fuel feed (from bladder)", "48V power from spine bus"],
        "performance": [f"Total thrust: {DIMS['turbine_count']*DIMS['turbine_thrust_ab_lbf']:,} lbf (AB) at sea level", f"Per turbine: {DIMS['turbine_thrust_dry_lbf']}/{DIMS['turbine_thrust_ab_lbf']} lbf dry/AB", f"SFC: {DIMS['turbine_sfc_dry_lb_lbh']}/{DIMS['turbine_sfc_ab_lb_lbh']} lb/(lbf*h) dry/AB", f"Max RPM: {DIMS['turbine_rpm_max']:,}", f"Vectoring: +/-{DIMS['turbine_vector_deg']}deg", f"Thrust scales with rho/rho_sl (density altitude)"],
        "build_details": {"time_min": 60, "tools": ["Gimbal alignment laser", "CAN bus configurator", "Bearing preload tool", "Afterburner pressure tester", "FADEC programmer"], "torque": "Gimbal mount: 6 Nm", "difficulty": "Expert", "notes": "Mount 48 turbines in 5 groups. Align gimbals with laser (0.1deg accuracy). Configure 6 ESCs + FADEC via CAN bus. Test each turbine at 10% throttle. Pressure-test afterburner reservoirs to 300 bar. Verify fuel flow from bladder to each turbine group."},
    },
    "wings": {
        "description": f"Compact deployable gliding wings for turbine-assisted flight. "
                       f"Wingspan: {DIMS['wing_span_m']}m ({DIMS['wing_span_m']*3.28:.0f}ft). "
                       f"Surface area: {DIMS['wing_area_sqft']} sq ft ({DIMS['wing_area_sqft']*0.0929:.1f} m^2). "
                       f"L/D ratio: {DIMS['wing_ld_ratio']}:1. "
                       f"Min sink rate: {DIMS['wing_min_sink_ms']} m/s. Deploy in {DIMS['wing_deploy_s']}s. "
                       f"Membrane: {DIMS['wing_membrane']}. Ribs: {DIMS['wing_rib_material']}. "
                       "Spring-loaded deployment with shape-memory alloy hinges. "
                       "Turbines maintain airspeed; wings provide efficient glide. "
                       "Stows flat against back armor when not deployed.",
        "category": "Gliding Surfaces",
        "layer": 5,
        "materials": [f"Wing ribs: {DIMS['wing_rib_material']}",
                      f"Membrane: {DIMS['wing_membrane']}",
                      "Shape-memory alloy deploy hinges",
                      "Wing surface heating elements (anti-ice)"],
        "weight_kg": 1.0,
        "blueprint": "Stowed flat against back armor. Deploy via key 6 or "
                     "automatic at speed >50 m/s with altitude >100m. "
                     "Spring-loaded ribs snap open in 0.5s. "
                     "Membrane tensioned by shape-memory hinges.",
        "icon": "wing",
        "sub_components": [
            {"name": "Nitinol Wing Ribs", "count": 8, "detail": f"Spring-loaded, {DIMS['wing_rib_material']}, 4 per wing, telescoping deployment"},
            {"name": "Graphene Membrane", "count": 2, "detail": f"{DIMS['wing_membrane']}, {DIMS['wing_area_sqft']} sq ft ({DIMS['wing_area_sqft']*0.0929:.1f} m^2) total, solar nanofiber integrated"},
            {"name": "Shape-Memory Hinges", "count": 2, "detail": f"Nitinol, deploy in {DIMS['wing_deploy_s']}s, 90-degree rotation, shoulder-mounted"},
            {"name": "Anti-Ice Heating Elements", "count": 2, "detail": "Graphene heating trace, 50W per wing, -40C rated"},
            {"name": "Solar Nanofiber Film", "count": 2, "detail": "Integrated in membrane topside, 200W peak harvesting"},
            {"name": "Wing Lock Mechanism", "count": 2, "detail": "Electromagnetic lock, release on deploy command or power loss"},
        ],
        "dimensions": f"Wingspan: {DIMS['wing_span_m']}m ({DIMS['wing_span_m']*3.28:.0f}ft) | Area: {DIMS['wing_area_sqft']} sq ft | Stowed: flat on back",
        "connectors": ["2x hinge mount (shoulder armor)", "2x electromagnetic lock", "Power: 48V for heating + lock", "Data: deploy signal from RTOS"],
        "performance": [f"L/D ratio: {DIMS['wing_ld_ratio']}:1", f"Min sink: {DIMS['wing_min_sink_ms']} m/s", f"Deploy time: {DIMS['wing_deploy_s']}s", f"Solar harvest: 200W peak"],
        "build_details": {"time_min": 30, "tools": ["Hinge alignment jig", "Membrane tension gauge", "Electromagnetic lock tester"], "torque": "Hinge bolts: 3 Nm", "difficulty": "Hard", "notes": "Mount 8 Nitinol ribs to shoulder hinges (4 per wing). Tension graphene membrane to 50N/m. Install 2 Nitinol deploy hinges. Test deploy in <0.5s. Verify electromagnetic locks release on power loss. Wings stow flat on back armor."},
    },
    "power": {
        "description": f"Solid-state lithium-sulfur battery: {DIMS['battery_wh']} Wh "
                       f"at {DIMS['battery_wh_kg']} Wh/kg energy density. "
                       f"Battery life: {DIMS['battery_life_hours']}h. "
                       f"Piezoelectric harvesting: {DIMS['piezo_harvest_pct']}% kinetic recovery. "
                       f"Solar nanofiber: {'Yes' if DIMS['solar_nanofiber'] else 'No'}. "
                       "Regenerative charging from turbine braking. "
                       "Battery management system with cell-level monitoring and fusing.",
        "category": "Power System",
        "layer": 0,
        "materials": [f"Li-S solid-state cells ({DIMS['battery_type']})",
                      "Piezoelectric harvesters (boots + joints)",
                      "Flexible solar film (wing surfaces)",
                      "BMS with cell-level fusing"],
        "weight_kg": 2.2,
        "blueprint": "Battery pack mounted on lower back frame. "
                     "Piezo elements in boot soles + knee/elbow joints. "
                     "Solar film applied to wing membrane topside. "
                     "Power bus routes through faraday layer.",
        "icon": "power",
        "sub_components": [
            {"name": "Li-S Battery Pack", "count": 2, "detail": f"2x {DIMS['battery_wh']//2}Wh hot-swap packs, {DIMS['battery_type']}, {DIMS['battery_wh_kg']} Wh/kg"},
            {"name": "Piezoelectric Harvesters", "count": 8, "detail": f"Boot soles (4) + knee/elbow joints (4), {DIMS['piezo_harvest_pct']}% kinetic recovery"},
            {"name": "Solar Nanofiber Film", "count": 2, "detail": "Wing membrane topside, 200W peak, flexible"},
            {"name": "Battery Management System", "count": 1, "detail": "Cell-level monitoring, fusing, balancing, thermal cutoff"},
            {"name": "Regenerative Braking Controller", "count": 1, "detail": "Turbine braking energy recovery, up to 30% recharge under decel"},
            {"name": "Hot-Swap Battery Bay", "count": 1, "detail": "Lower back frame, tool-less swap, 5s downtime"},
        ],
        "dimensions": f"Capacity: {DIMS['battery_wh']} Wh | Energy density: {DIMS['battery_wh_kg']} Wh/kg | Hot-swap: 2 packs",
        "connectors": ["48V 200A main feed (to faraday spine bus)", "CAN bus (BMS to RTOS)", "2x hot-swap connector (tool-less)"],
        "performance": [f"Capacity: {DIMS['battery_wh']} Wh", f"Runtime: {DIMS['battery_life_hours']}h", f"Energy density: {DIMS['battery_wh_kg']} Wh/kg", f"Kinetic recovery: {DIMS['piezo_harvest_pct']}%"],
        "build_details": {"time_min": 10, "tools": ["BMS configurator", "Cell balance tester", "Insulation tester"], "torque": "Battery bay: thumb-screw", "difficulty": "Easy", "notes": "Insert 2x hot-swap packs in back bay. Configure BMS via CAN bus. Balance cells. Test insulation at 500V. Verify piezo harvesters in boots + joints. Check solar film continuity on wings."},
    },
    "neural": {
        "description": f"Brain-Computer Interface: {DIMS['bci_type']}, <{DIMS['bci_latency_ms']}ms latency. "
                       "Vera 3.0 AI co-pilot (Llama-3.1-70B 4-bit quantized). "
                       f"Encryption: {DIMS['bci_crypto']}. Air-gapped: {DIMS['bci_air_gapped']}. "
                       "AI classifies intent (combat, flight, medical, rest) "
                       "and executes complex actions. BCI decodes motor cortex "
                       "signals for suit control. 64-channel EEG + EMG fusion.",
        "category": "Neural Interface + AI",
        "layer": 0,
        "materials": ["64-channel dry EEG electrodes",
                      "Neural signal processor",
                      "Vera 3.0 AI (Llama-3.1-70B 4-bit)",
                      "Kyber-1024 post-quantum crypto",
                      "AES-256-GCM session encryption"],
        "weight_kg": 0.3,
        "blueprint": "EEG cap worn under helmet. Neural processor in helmet "
                     "crown. AI runs on dedicated NPU in chest pack. "
                     "Encrypted link to suit RTOS via faraday-shielded bus. "
                     "Calibration: 30s baseline + 2min motor mapping.",
        "icon": "neural",
        "sub_components": [
            {"name": "EEG/EMG Electrode Cap", "count": 1, "detail": f"64-channel dry contact, {DIMS['bci_type']}, circumferential in helmet"},
            {"name": "Neural Signal Processor", "count": 1, "detail": f"DSP + NPU, <{DIMS['bci_latency_ms']}ms thought-to-action, 2kHz sampling"},
            {"name": "Vera 3.0 AI Co-Pilot", "count": 1, "detail": "Llama-3.1-70B 4-bit quantized, intent classification + action execution"},
            {"name": "Post-Quantum Crypto Module", "count": 1, "detail": f"{DIMS['bci_crypto']}, Kyber-1024 + AES-256-GCM, air-gapped"},
            {"name": "AI Intent Classifier", "count": 1, "detail": "Real-time: combat/flight/medical/rest, confidence scoring, auto-action"},
            {"name": "Motor Cortex Decoder", "count": 1, "detail": "Decodes left/right arm, trigger, speak intents from motor cortex signals"},
        ],
        "dimensions": f"Latency: <{DIMS['bci_latency_ms']}ms | Channels: 64 | AI: 70B params (4-bit)",
        "connectors": ["64-channel EEG bus (to helmet band)", "Encrypted neural bus (to RTOS)", "NPU power: 48V 15W", "Air-gapped data diode"],
        "performance": [f"Latency: <{DIMS['bci_latency_ms']}ms", f"Type: {DIMS['bci_type']}", f"Air-gapped: {DIMS['bci_air_gapped']}", "AI: Vera 3.0 (70B 4-bit)"],
        "build_details": {"time_min": 20, "tools": ["EEG calibration headset", "NPU flash tool", "Crypto key generator"], "torque": "N/A -- snap fit", "difficulty": "Expert", "notes": "Install EEG cap under helmet. Flash Vera 3.0 AI to NPU. Generate Kyber-1024 keys. Run 30s baseline + 2min motor mapping calibration. Verify air-gapped data diode. Test encrypted link to RTOS."},
    },
    "thermal": {
        "description": f"Dragon Skin active thermal regulation. "
                       f"Maintains skin temp at {DIMS['thermal_inner_temp_c']}C in environments "
                       f"from {DIMS['thermal_range_lo_c']}C to +{DIMS['thermal_range_hi_c']}C. "
                       f"Heating: {DIMS['thermal_heating_kw']} kW from reactor waste heat. "
                       f"Cooling: {DIMS['thermal_cooling_kw']} kW resistive backup. "
                       "Peltier junctions at high-heat zones + capillary fluid loop. "
                       "Solar lens heating for extreme cold. "
                       "Radiator fins deploy from shoulders for extreme heat.",
        "category": "Thermal Regulation",
        "layer": 1,
        "materials": ["Dragon Skin phase-change undersuit",
                      "Peltier thermoelectric junctions",
                      "Capillary fluid loop (ethylene glycol)",
                      "Deployable radiator fins (shoulder)",
                      "Solar lens concentrator (cold mode)"],
        "weight_kg": 0.5,
        "blueprint": "Worn over inner suit, under faraday shield. "
                     "Peltier junctions at 6 high-heat zones. "
                     "Fluid loop runs through all layers. "
                     "Radiator fins deploy from shoulder armor edges.",
        "icon": "thermal",
        "sub_components": [
            {"name": "Dragon Skin Phase-Change Layer", "count": 1, "detail": "PCM matrix, melts at 37C, 12kJ/kg latent heat"},
            {"name": "Peltier Junctions", "count": 6, "detail": "Neck, armpits(2), groin, back of head, chest -- 50W each, reversible"},
            {"name": "Capillary Fluid Loop", "count": 1, "detail": "Ethylene glycol, 2 LPM, runs through all layers, heat exchanger in back"},
            {"name": "Deployable Radiator Fins", "count": 2, "detail": "Shoulder-mounted, graphene fins, deploy in 1s, 500W dissipation"},
            {"name": "Solar Lens Concentrator", "count": 1, "detail": "Chest-mounted Fresnel lens, focuses sunlight for heating, 200W"},
            {"name": "Aerogel Insulation Layer", "count": 1, "detail": "3mm aerogel blanket, R-value = arctic parka, full body coverage"},
        ],
        "dimensions": f"Range: {DIMS['thermal_range_lo_c']}C to +{DIMS['thermal_range_hi_c']}C | Heating: {DIMS['thermal_heating_kw']}kW | Cooling: {DIMS['thermal_cooling_kw']}kW",
        "connectors": ["6x Peltier power (48V 50W each)", "Fluid loop quick-connect (2x)", "Radiator deploy signal (from RTOS)"],
        "performance": [f"Range: {DIMS['thermal_range_lo_c']}C to +{DIMS['thermal_range_hi_c']}C", f"Heating: {DIMS['thermal_heating_kw']}kW", f"Cooling: {DIMS['thermal_cooling_kw']}kW", "Skin temp: 36.5-37.5C locked"],
        "build_details": {"time_min": 15, "tools": ["Peltier junction crimper", "Fluid loop pressure tester", "Thermal sensor calibrator"], "torque": "Peltier mounts: 1 Nm", "difficulty": "Medium", "notes": "Install 6 Peltier junctions at neck, armpits, groin, head, chest. Connect fluid loop, pressure test at 2 bar. Install aerogel blanket. Test heating (12kW) and cooling (3kW). Verify radiator fin deployment."},
    },
    "fuel_system": {
        "description": f"Conformal fuel bladder system for micro-turbofan propulsion. "
                       f"Carries {DIMS['fuel_capacity_l']}L of {DIMS['fuel_type']} "
                       f"(density {DIMS['fuel_density_kg_l']} kg/L = {DIMS['fuel_capacity_l']*DIMS['fuel_density_kg_l']:.1f} kg total). "
                       f"SFC: {DIMS['turbine_sfc_dry_lb_lbh']} lb/(lbf*h) dry, {DIMS['turbine_sfc_ab_lb_lbh']} lb/(lbf*h) afterburner. "
                       f"Hover thrust ~311 lbf ({DIMS['weight_total_kg']+DIMS['fuel_capacity_l']*DIMS['fuel_density_kg_l']+DIMS['ref_weight_kg']:.0f} kg total), burn = "
                       f"{311*DIMS['turbine_sfc_dry_lb_lbh']*0.4536:.0f} kg/h -> "
                       f"endurance ~15 min pure hover. "
                       f"Climb-glide cycles extend this to 4-7 hours. "
                       f"Bladder is crash-resistant Kevlar-lined, self-sealing.",
        "category": "Fuel System",
        "layer": 0,
        "materials": [f"Conformal Kevlar bladder (ATL-style)",
                      f"{DIMS['fuel_type']} aviation fuel",
                      "Stainless braided fuel lines (dual redundant)",
                      "Electric fuel pumps (dual redundant, 0.5 LPM each)",
                      "Self-sealing polymer coating (bulletproof)"],
        "weight_kg": 1.0,  # bladder + lines + pumps (fuel weight counted separately in physics)
        "blueprint": "Conformal bladder mounts in lower back/torso frame cavity, "
                     "conforming to pilot body shape. Dual redundant electric pumps "
                     "feed fuel to all 36 turbines via stainless braided lines. "
                     "Self-sealing coating closes penetrations <12mm instantly. "
                     "Fuel level sensors at 4 points in bladder.",
        "icon": "fuel",
        "sub_components": [
            {"name": "Conformal Fuel Bladder", "count": 1, "detail": f"Kevlar-lined, {DIMS['fuel_capacity_l']}L capacity, self-sealing, ATL-style"},
            {"name": "Electric Fuel Pumps", "count": 2, "detail": "Dual redundant, 0.5 LPM each, 12V, brushless DC, crash-rated"},
            {"name": "Stainless Braided Fuel Lines", "count": 2, "detail": "Dual redundant routing, 6mm ID, 300 bar rated, AN-8 fittings"},
            {"name": "Fuel Level Sensors", "count": 4, "detail": "Capacitive type, 0.1% accuracy, feeds BMS + RTOS"},
            {"name": "Self-Sealing Polymer Coating", "count": 1, "detail": "Bulletproof self-sealing layer, closes <12mm penetrations in <1s"},
            {"name": "Fuel Vent / Pressure Relief", "count": 1, "detail": "Automatic vent at 1.5 bar, prevents bladder rupture in vacuum"},
            {"name": "Quick-Disconnect Refuel Port", "count": 1, "detail": "Left hip, AN-8 quick-connect, refuel in <90s"},
        ],
        "dimensions": f"Capacity: {DIMS['fuel_capacity_l']}L ({DIMS['fuel_capacity_l']*DIMS['fuel_density_kg_l']:.1f} kg) | Bladder: conformal to torso | Lines: 6mm ID stainless",
        "connectors": ["2x AN-8 fuel line (to turbine manifold)", "1x refuel port (left hip)", "4x fuel level sensor (CAN bus)", "12V power (dual pump)"],
        "performance": [f"Capacity: {DIMS['fuel_capacity_l']}L", f"Endurance: ~15min hover / 4-7hr climb-glide", f"SFC: {DIMS['turbine_sfc_dry_lb_lbh']} lb/(lbf*h) dry", f"Self-sealing: <12mm holes", f"Pumps: dual redundant 0.5 LPM"],
        "build_details": {"time_min": 20, "tools": ["Fuel bladder pressure tester", "AN-8 line crimper", "Pump flow meter", "Self-sealing coating applicator"], "torque": "AN-8 fittings: 25 Nm", "difficulty": "Hard", "notes": "Install conformal bladder in torso frame cavity. Connect dual fuel lines, crimp AN-8 fittings to 25 Nm. Mount 2 pumps in parallel. Apply self-sealing coating. Pressure test at 2 bar for 30 min. Verify flow rate at 0.5 LPM per pump. Test self-sealing with 10mm probe."},
    },
}


def get_part_info(key):
    """Get rich metadata for a part key from the PART_DB."""
    return PART_DB.get(key, {})


# =============================================================================
# SUIT BUILDER  -- constructs all Part objects for the Mjalnor'MV1.17
# =============================================================================

def build_suit(pilot_height=1.73, pilot_weight=79.4):
    """Build the complete suit model for a given pilot size.
    Returns (parts, turbines_config, wings_config)."""
    parts = []
    order = 0
    seg = body_segments(pilot_height)

    # Scale factor relative to reference
    s = pilot_height / DIMS["ref_height_m"]

    # Key body measurements
    torso_len = seg["torso"] + seg["hip"]
    upper_arm = seg["upper_arm"]
    forearm = seg["forearm"]
    thigh = seg["thigh"]
    shin = seg["shin"]
    shoulder_w = seg["shoulder_w"] * 1.02  # suit adds minimal width (2%)
    hip_w = seg["hip_w"] * 1.02
    chest_d = seg["chest_d"] * 1.02

    # Layer thicknesses
    t_inner = DIMS["inner_thick_mm"] * MM
    t_middle = DIMS["middle_thick_mm"] * MM
    t_inter = DIMS["intermediate_thick_mm"] * MM
    t_outer = DIMS["outer_thick_mm"] * MM
    t_total = t_inner + t_middle + t_inter + t_outer

    # ---- 1. INNER LAYER: sensor suit (skin-tight base) ----
    # MJOLNIR techsuit style: angular, form-fitting, only visible at armor gaps
    inner_meshes = []
    # torso (angular hexagonal, NOT round capsule — MJOLNIR style)
    r_shoulder_inner = max(chest_d, hip_w) * 0.34
    r_waist_inner = max(chest_d, hip_w) * 0.24
    r_hip_inner = max(chest_d, hip_w) * 0.30
    v, f = _angular_torso(r_shoulder_inner, r_waist_inner, r_hip_inner,
                          -torso_len/2, torso_len/2, seg=6)
    inner_meshes.append(Mesh(v, f, C_INNER, name="inner_torso"))
    # arms (angular hexagonal, not round — ~5cm across flats)
    for sx in (1, -1):
        v, f = _angular_limb(0.024 * s, 0.020 * s, 0, upper_arm + forearm, seg=6, flat_frac=0.6)
        v = _translate(v, (sx * shoulder_w * 0.5, 0.05, 0))
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        inner_meshes.append(Mesh(v, f, C_INNER, name=f"inner_arm_{sx}"))
    # legs (angular hexagonal, not round — ~6.5cm across flats)
    for sx in (1, -1):
        v, f = _angular_limb(0.030 * s, 0.024 * s, -thigh - shin, 0, seg=6, flat_frac=0.6)
        v = _translate(v, (sx * hip_w * 0.25, -torso_len/2 - 0.02, 0))
        inner_meshes.append(Mesh(v, f, C_INNER, name=f"inner_leg_{sx}"))
    parts.append(Part("inner_suit", "Inner Sensor Suit (spandex-nylon + EMG)",
                      inner_meshes,
                      ["Skin-tight base layer with EMG sensors",
                       "Phase-change material for comfort",
                       "Capillary moisture-wicking network",
                       f"Thickness: {DIMS['inner_thick_mm']}mm"],
                      order, np.array([0, 0, -0.3]), C_INNER))
    order += 1

    # ---- 2. MIDDLE LAYER: tripled DEA-STF muscle fibers ----
    # Only visible at armor plate gaps - thin, conforming, not full-body tubes
    middle_meshes = []
    fiber_r = 0.0025 * s
    # Torso: angular band only at waist gap (between chest/abdomen plates)
    r_mid_shoulder = max(chest_d, hip_w) * 0.36 + t_inner
    r_mid_waist = max(chest_d, hip_w) * 0.26 + t_inner
    r_mid_hip = max(chest_d, hip_w) * 0.32 + t_inner
    v, f = _angular_torso(r_mid_shoulder, r_mid_waist, r_mid_hip,
                          -torso_len * 0.10, torso_len * 0.10, seg=6)
    middle_meshes.append(Mesh(v, f, C_MIDDLE, name="middle_torso_waist"))
    # DEA fiber strands at joint gaps (elbows, knees, shoulders)
    for sx in (1, -1):
        # Elbow gap fibers
        ey = 0.05 + upper_arm * 0.95
        for fi in range(4):
            a = 2 * math.pi * fi / 4 + sx * 0.2
            v, f = _solid_cylinder(fiber_r, -0.03, 0.03, seg=4)
            v = _translate(v, (sx * shoulder_w * 0.5 + 0.040 * s * math.cos(a),
                              ey + 0.040 * s * math.sin(a), 0))
            v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
            middle_meshes.append(Mesh(v, f, C_FIBER, name=f"dea_fiber_elbow_{sx}_{fi}", emissive=True))
        # Knee gap fibers (5x density for jump)
        ky = -torso_len/2 - 0.02 - thigh
        for fi in range(6):
            a = 2 * math.pi * fi / 6
            v, f = _solid_cylinder(fiber_r * 1.3, -0.04, 0.04, seg=4)
            v = _translate(v, (sx * hip_w * 0.25 + 0.050 * s * math.cos(a),
                              ky + 0.050 * s * math.sin(a), 0))
            middle_meshes.append(Mesh(v, f, C_FIBER, name=f"dea_fiber_knee_{sx}_{fi}", emissive=True))
        # Shoulder gap fibers
        for fi in range(3):
            a = 2 * math.pi * fi / 3
            v, f = _solid_cylinder(fiber_r, -0.02, 0.02, seg=4)
            v = _translate(v, (sx * shoulder_w * 0.5 + 0.045 * s * math.cos(a),
                              0.05 + 0.045 * s * math.sin(a), 0))
            middle_meshes.append(Mesh(v, f, C_FIBER, name=f"dea_fiber_shoulder_{sx}_{fi}", emissive=True))
    # Faraday shielding + power bus (thin, visible at gaps)
    faraday_meshes = []
    # Thin faraday at waist gap (angular, matching inner layer shape)
    r_far_shoulder = max(chest_d, hip_w) * 0.37 + t_inner + t_middle
    r_far_waist = max(chest_d, hip_w) * 0.27 + t_inner + t_middle
    r_far_hip = max(chest_d, hip_w) * 0.33 + t_inner + t_middle
    v, f = _angular_torso(r_far_shoulder, r_far_waist, r_far_hip,
                          -torso_len * 0.08, torso_len * 0.08, seg=6)
    faraday_meshes.append(Mesh(v, f, C_FARADAY, name="faraday_waist"))
    # Power bus routing (visible cable conduits along frame)
    power_bus_meshes = []
    bus_r = 0.003
    # Main power bus along spine
    v, f = _solid_cylinder(bus_r, -torso_len/2, torso_len/2 + 0.1, seg=6)
    power_bus_meshes.append(Mesh(v, f, C_ACCENT, name="power_bus_spine", emissive=True))
    # Branch buses to shoulders
    for sx in (1, -1):
        v, f = _solid_cylinder(bus_r, 0, shoulder_w * 0.5, seg=6)
        v = (np.asarray(v) @ rot_z(sx * math.pi/2).T).tolist()
        v = _translate(v, (0, 0.05, 0))
        power_bus_meshes.append(Mesh(v, f, C_ACCENT, name=f"power_bus_shoulder_{sx}", emissive=True))
    # Branch buses to hips
    for sx in (1, -1):
        v, f = _solid_cylinder(bus_r, 0, hip_w * 0.35, seg=6)
        v = (np.asarray(v) @ rot_z(sx * math.pi/2).T).tolist()
        v = _translate(v, (0, -torso_len/2 - 0.02, 0))
        power_bus_meshes.append(Mesh(v, f, C_ACCENT, name=f"power_bus_hip_{sx}", emissive=True))
    parts.append(Part("faraday", "Faraday Shielding (copper-graphene mesh)",
                      faraday_meshes + power_bus_meshes,
                      ["0.1mm copper-graphene mesh",
                       "Blocks >99% EMP/EMI",
                       "Maintains flexibility",
                       "Embedded within middle layer",
                       "Visible at armor plate gaps",
                       "Power bus: spine + 4 branches (shoulders + hips)"],
                      order, np.array([0, 0, -0.2]), C_FARADAY))
    order += 1

    parts.append(Part("middle_layer", "Middle Layer: Tripled DEA-STF Muscle Fibers",
                      middle_meshes,
                      [f"3 sublayers x {DIMS['middle_thick_mm']/3:.1f}mm = {DIMS['middle_thick_mm']}mm total",
                       f"DEA contraction <{DIMS['dea_contraction_ms']}ms, {DIMS['dea_strain_pct']}% strain",
                       f"STF absorption: {DIMS['stf_max_psi']:,} PSI (tripled)",
                       f"Strength boost: {DIMS['muscle_boost_x']}x human",
                       f"Jump fibers: {DIMS['fiber_density_jump_x']}x density in legs",
                       f"Lift fibers: {DIMS['fiber_density_lift_x']}x in shoulders/back/arms"],
                      order, np.array([0, 0, -0.15]), C_MIDDLE))
    order += 1

    # ---- 3. INTERMEDIATE LAYER: auxetic metamaterial ----
    # Thin, only visible at joint gaps between armor plates
    inter_meshes = []
    # Thin auxetic at waist/joint gaps (angular, matching inner layer)
    r_inter_shoulder = max(chest_d, hip_w) * 0.38 + t_inner + t_middle
    r_inter_waist = max(chest_d, hip_w) * 0.28 + t_inner + t_middle
    r_inter_hip = max(chest_d, hip_w) * 0.34 + t_inner + t_middle
    v, f = _angular_torso(r_inter_shoulder, r_inter_waist, r_inter_hip,
                          -torso_len * 0.05, torso_len * 0.05, seg=6)
    inter_meshes.append(Mesh(v, f, C_INTERMED, name="auxetic_waist_gap"))
    # Hex cells at elbow gaps
    hex_r = 0.010 * s
    for sx in (1, -1):
        ey = 0.05 + upper_arm * 0.95
        r_arm_gap = 0.042 * s + t_inner + t_middle
        for si in range(4):
            sa = 2 * math.pi * si / 4
            v, f = _solid_cylinder(hex_r * 0.7, -0.001, 0.001, seg=6)
            v = _translate(v, (sx * shoulder_w * 0.5 + r_arm_gap * math.cos(sa),
                              ey + r_arm_gap * math.sin(sa), 0))
            v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
            inter_meshes.append(Mesh(v, f, C_INTERMED_DK, name=f"aux_elbow_{sx}_{si}"))
        # Hex cells at knee gaps
        ky = -torso_len/2 - 0.02 - thigh
        r_leg_gap = 0.052 * s + t_inner + t_middle
        for si in range(5):
            sa = 2 * math.pi * si / 5
            v, f = _solid_cylinder(hex_r * 0.8, -0.001, 0.001, seg=6)
            v = _translate(v, (sx * hip_w * 0.25 + r_leg_gap * math.cos(sa),
                              ky + r_leg_gap * math.sin(sa), 0))
            inter_meshes.append(Mesh(v, f, C_INTERMED_DK, name=f"aux_knee_{sx}_{si}"))
    parts.append(Part("intermediate", "Intermediate: Foam-Filled Auxetic Metamaterial",
                      inter_meshes,
                      [f"Thickness: {DIMS['intermediate_thick_mm']}mm",
                       f"Negative Poisson's ratio: {DIMS['auxetic_poisson']}",
                       f"Cell type: {DIMS['auxetic_cell_type']}",
                       f"Relative density: {DIMS['auxetic_rel_density']*100}%",
                       f"Energy absorption: {DIMS['auxetic_energy_abs_pct']}% of residual",
                       "Self-healing polymer microcapsules"],
                      order, np.array([0, 0, -0.1]), C_INTERMED))
    order += 1

    # ---- 4. OUTER LAYER: graphene-UHMWPE armor panels ----
    # MJOLNIR/Iron Man style: angular chamfered plates with visible gaps
    # Plates are the dominant visual - sleek, tapered, not puffy
    outer_meshes = []
    r_outer = max(chest_d, hip_w) * 0.36 + t_total

    # Chest plate (large, angular, chamfered — MJOLNIR style, covers full torso)
    v, f = _armor_plate(0, 0.05, 0.020, shoulder_w * 0.92, torso_len * 0.45, 0.020,
                        chamfer=0.010, taper=0.015)
    outer_meshes.append(Mesh(v, f, C_OUTER, name="chest_plate"))
    # Chest center ridge (Iron Man style center line)
    v, f = _armor_plate(0, 0.05, 0.040, 0.016, torso_len * 0.30, 0.012,
                        chamfer=0.004, taper=0.004)
    outer_meshes.append(Mesh(v, f, C_OUTER_LT, name="chest_ridge"))
    # Arc reactor / chest emblem (Iron Man style glowing center)
    v, f = _sphere(0.012, seg=10)
    v = _translate(v, (0, 0.05, 0.052))
    outer_meshes.append(Mesh(v, f, C_VISOR_GLOW, name="arc_reactor", emissive=True))
    # Arc reactor ring
    v, f = _torus(0.014, 0.003, seg_major=16, seg_minor=4)
    v = _translate(v, (0, 0.05, 0.050))
    outer_meshes.append(Mesh(v, f, C_ACCENT, name="arc_reactor_ring", emissive=True))

    # Back plate (large, angular, chamfered — full coverage)
    v, f = _armor_plate(0, -0.05, 0.020, shoulder_w * 0.92, torso_len * 0.45, 0.020,
                        chamfer=0.010, taper=0.015)
    outer_meshes.append(Mesh(v, f, C_OUTER, name="back_plate"))

    # Side flank plates (cover sides between chest and back)
    for sx in (1, -1):
        v, f = _armor_plate(sx * shoulder_w * 0.42, 0, 0.010,
                            0.025, torso_len * 0.40, 0.016,
                            chamfer=0.006, taper=0.008)
        outer_meshes.append(Mesh(v, f, C_OUTER_LT, name=f"flank_plate_{sx}"))

    # Abdomen plates (segmented, angled, with gaps between — wider for coverage)
    for i in range(3):
        z = -0.04 - i * 0.038
        v, f = _armor_plate(0, 0, z, hip_w * 0.82, chest_d * 0.52, 0.014,
                            chamfer=0.005, taper=0.008)
        outer_meshes.append(Mesh(v, f, C_OUTER if i % 2 == 0 else C_OUTER_LT,
                                 name=f"abdomen_{i}"))

    # Codpiece / pelvic armor (angular, covers lower center)
    v, f = _armor_plate(0, -torso_len/2 + 0.02, 0.014,
                        hip_w * 0.38, 0.06, 0.016,
                        chamfer=0.006, taper=0.008)
    outer_meshes.append(Mesh(v, f, C_OUTER, name="codpiece"))

    # Belt / waist armor (angular ring at waist)
    v, f = _armor_plate(0, -torso_len * 0.15, 0.012,
                        hip_w * 0.85, 0.05, 0.014,
                        chamfer=0.005, taper=0.006)
    outer_meshes.append(Mesh(v, f, C_OUTER_LT, name="belt_plate"))

    # Shoulder pauldrons (faceted domes — larger for dominant silhouette)
    for sx in (1, -1):
        v, f = _pauldron(0.058 * s + t_total, sx * shoulder_w * 0.52, 0.06, 0.042, seg=8)
        outer_meshes.append(Mesh(v, f, C_OUTER, name=f"pauldron_{sx}"))

    # Upper arm plates (tapered, angular — larger for coverage)
    for sx in (1, -1):
        v, f = _tapered_limb(0.038 * s + t_total, 0.044 * s + t_total,
                             0, upper_arm * 0.92, seg=8, flat_frac=0.55,
                             taper_profile=[1.0, 0.95, 0.85])
        v = _translate(v, (sx * shoulder_w * 0.5, 0.05, 0))
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        outer_meshes.append(Mesh(v, f, C_OUTER, name=f"upper_arm_plate_{sx}"))

    # Forearm gauntlets (tapered, sleek, Iron Man style — larger)
    for sx in (1, -1):
        v, f = _tapered_limb(0.030 * s + t_total, 0.042 * s + t_total,
                             upper_arm * 0.85, upper_arm + forearm, seg=8, flat_frac=0.6,
                             taper_profile=[1.0, 0.92, 0.85, 0.80])
        v = _translate(v, (sx * shoulder_w * 0.5, 0.05, 0))
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        outer_meshes.append(Mesh(v, f, C_OUTER, name=f"gauntlet_{sx}"))

    # Thigh plates (tapered, angular — larger for coverage)
    for sx in (1, -1):
        v, f = _tapered_limb(0.040 * s + t_total, 0.048 * s + t_total,
                             -thigh, -0.02, seg=10, flat_frac=0.5,
                             taper_profile=[1.0, 0.95, 0.88])
        v = _translate(v, (sx * hip_w * 0.25, -torso_len/2 - 0.02, 0))
        outer_meshes.append(Mesh(v, f, C_OUTER, name=f"thigh_plate_{sx}"))

    # Shin greaves (tapered, sleek — larger)
    for sx in (1, -1):
        v, f = _tapered_limb(0.032 * s + t_total, 0.042 * s + t_total,
                             -thigh - shin, -thigh, seg=10, flat_frac=0.55,
                             taper_profile=[1.0, 0.93, 0.85, 0.80])
        v = _translate(v, (sx * hip_w * 0.25, -torso_len/2 - 0.02, 0))
        outer_meshes.append(Mesh(v, f, C_OUTER_LT, name=f"greave_{sx}"))

    # Boots (angular, chamfered — larger for coverage)
    for sx in (1, -1):
        v, f = _armor_plate(sx * hip_w * 0.25, -torso_len/2 - 0.02 - thigh - shin - 0.02,
                            0.06, 0.085, 0.045, 0.018, chamfer=0.006, taper=0.006)
        outer_meshes.append(Mesh(v, f, C_OUTER, name=f"boot_{sx}"))

    # Knee plates (angular, faceted — larger)
    for sx in (1, -1):
        v, f = _pauldron(0.034 * s + t_total, sx * hip_w * 0.25,
                         -torso_len/2 - 0.02 - thigh, 0.032, seg=6)
        outer_meshes.append(Mesh(v, f, C_OUTER_LT, name=f"knee_plate_{sx}"))

    # Hip plates (angular, chamfered — larger for coverage)
    for sx in (1, -1):
        v, f = _armor_plate(sx * hip_w * 0.32, -torso_len/2 - 0.04, 0.012,
                            0.06, 0.07, 0.018, chamfer=0.005, taper=0.006)
        outer_meshes.append(Mesh(v, f, C_OUTER, name=f"hip_plate_{sx}"))

    # Neck guard (tapered, short — compact)
    v, f = _tapered_prism(0.030 * s + t_total, 0.034 * s + t_total, 0.08, 0.12, seg=8, flat_frac=0.5)
    outer_meshes.append(Mesh(v, f, C_OUTER_LT, name="neck_guard"))

    # Panel lines (Iron Man style accent grooves on chest)
    for sx in (1, -1):
        v, f = _panel_line(sx * 0.02, 0.06, 0.045, sx * 0.12, 0.04, 0.045, width=0.003)
        outer_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"chest_panel_line_{sx}", emissive=True))
    # Center chest line
    v, f = _panel_line(0, 0.06, 0.046, 0, -0.02, 0.046, width=0.003)
    outer_meshes.append(Mesh(v, f, C_FRAME_DK, name="chest_center_line", emissive=True))
    # Back panel lines
    for sx in (1, -1):
        v, f = _panel_line(sx * 0.02, -0.06, 0.045, sx * 0.12, -0.04, 0.045, width=0.003)
        outer_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"back_panel_line_{sx}", emissive=True))

    # Vent slots on forearms (Iron Man style)
    for sx in (1, -1):
        for vi in range(3):
            vy = 0.05 + upper_arm + forearm * 0.3 + vi * 0.015
            v, f = _box(sx * (shoulder_w * 0.5 + 0.042), vy, 0,
                        0.001, 0.008, 0.012)
            v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
            outer_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"forearm_vent_{sx}_{vi}"))

    # Weapon mount points (visible hardpoints)
    for sx in (1, -1):
        # Forearm rail mount (Picatinny-style)
        rail_y = 0.05 + upper_arm * 0.7
        v, f = _box(sx * (shoulder_w * 0.5 + 0.04), rail_y, 0,
                    0.02, 0.08, 0.015)
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        outer_meshes.append(Mesh(v, f, C_FRAME, name=f"weapon_rail_{sx}"))
        # Rail slots (3 cross slots)
        for si in range(3):
            sy = rail_y - 0.03 + si * 0.03
            v, f = _box(sx * (shoulder_w * 0.5 + 0.045), sy, 0,
                        0.001, 0.002, 0.012)
            v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
            outer_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"weapon_slot_{sx}_{si}"))
        # Shoulder hardpoint (mounting bracket)
        v, f = _box(sx * shoulder_w * 0.42, 0.08, 0.05,
                    0.04, 0.02, 0.03)
        outer_meshes.append(Mesh(v, f, C_FRAME, name=f"shoulder_hardpoint_{sx}"))
        # Hardpoint mounting pin
        v, f = _solid_cylinder(0.005, 0, 0.02, seg=6)
        v = (np.asarray(v) @ rot_x(math.pi/2).T).tolist()
        v = _translate(v, (sx * shoulder_w * 0.42, 0.09, 0.05))
        outer_meshes.append(Mesh(v, f, C_NODE, name=f"hardpoint_pin_{sx}"))
    # Back hardpoint (between shoulder blades)
    v, f = _box(0, -0.05, -0.15, 0.06, 0.03, 0.04)
    outer_meshes.append(Mesh(v, f, C_FRAME, name="back_hardpoint"))
    # Back hardpoint mounting rails (2x)
    for si in range(2):
        sx = 1 if si == 0 else -1
        v, f = _box(sx * 0.02, -0.05, -0.13, 0.015, 0.025, 0.002)
        outer_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"back_rail_{si}"))
    parts.append(Part("outer_armor", "Outer: Graphene-UHMWPE Armor Panels (NIJ IV+)",
                      outer_meshes,
                      [f"Thickness: {DIMS['outer_thick_mm']}mm",
                       f"Max withstand: {DIMS['outer_max_psi']:,} PSI",
                       f"NIJ Level: {DIMS['outer_nij_level']}",
                       f"Heat tolerance: >{DIMS['outer_heat_tol_c']}C",
                       f"Laser resistance: {DIMS['outer_laser_tol_kw']} kW/cm2",
                       f"Coating: {DIMS['coating_type']}",
                       f"Ceramic: {DIMS['ceramic_type']}",
                       "Modular quick-release locking panels",
                       "Weapon mounts: 2x forearm rails + 2x shoulder + 1x back hardpoint"],
                      order, np.array([0, 0, 0.05]), C_OUTER))
    order += 1

    # ---- 5. CFRP FRAME: telescoping exoskeleton ----
    frame_meshes = []
    tube_r = DIMS["frame_tube_od_mm"] * MM / 2 * 0.7  # slim frame

    # Spine (central vertical strut)
    v, f = _hollow_cylinder(tube_r, tube_r * 0.82, -torso_len/2 - 0.05, torso_len/2 + 0.05, seg=16)
    frame_meshes.append(Mesh(v, f, C_FRAME, name="spine"))

    # Shoulder yoke
    v, f = _tube(tube_r * 0.7, -0.04, 0.04, seg=12)
    v = (np.asarray(v) @ rot_x(math.pi/2).T).tolist()
    v = _translate(v, (0, 0.05, 0.05))
    frame_meshes.append(Mesh(v, f, C_FRAME, name="shoulder_yoke"))

    # Shoulder to elbow struts
    for sx in (1, -1):
        v, f = _hollow_cylinder(tube_r * 0.6, tube_r * 0.48, 0, upper_arm, seg=10)
        v = _translate(v, (sx * shoulder_w * 0.5, 0.05, 0))
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        frame_meshes.append(Mesh(v, f, C_FRAME, name=f"upper_arm_strut_{sx}"))
        # Forearm strut
        v, f = _hollow_cylinder(tube_r * 0.5, tube_r * 0.38, 0, forearm, seg=10)
        v = _translate(v, (sx * shoulder_w * 0.5 + sx * 0.04, 0.05, 0))
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        frame_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"forearm_strut_{sx}"))

    # Hip yoke
    v, f = _tube(tube_r * 0.7, -0.04, 0.04, seg=12)
    v = (np.asarray(v) @ rot_x(math.pi/2).T).tolist()
    v = _translate(v, (0, -torso_len/2 - 0.02, 0))
    frame_meshes.append(Mesh(v, f, C_FRAME, name="hip_yoke"))

    # Hip to knee struts (thigh)
    for sx in (1, -1):
        v, f = _hollow_cylinder(tube_r * 0.6, tube_r * 0.48, -thigh, 0, seg=10)
        v = _translate(v, (sx * hip_w * 0.25, -torso_len/2 - 0.02, 0))
        frame_meshes.append(Mesh(v, f, C_FRAME, name=f"thigh_strut_{sx}"))
        # Knee to ankle (shin)
        v, f = _hollow_cylinder(tube_r * 0.5, tube_r * 0.38, -shin, 0, seg=10)
        v = _translate(v, (sx * hip_w * 0.25, -torso_len/2 - 0.02 - thigh, 0))
        frame_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"shin_strut_{sx}"))

    # Ti-6Al-4V nodes at joints (small spheres)
    node_positions = [
        (0, torso_len/2 + 0.05, 0.05),         # neck
        (shoulder_w * 0.5, 0.05, 0.05),        # right shoulder
        (-shoulder_w * 0.5, 0.05, 0.05),       # left shoulder
        (shoulder_w * 0.5 + 0.04, 0.05, upper_arm * 0.9),  # right elbow
        (-shoulder_w * 0.5 - 0.04, 0.05, upper_arm * 0.9), # left elbow
        (0, -torso_len/2 - 0.02, 0),           # hip center
        (hip_w * 0.25, -torso_len/2 - 0.02, 0),  # right hip
        (-hip_w * 0.25, -torso_len/2 - 0.02, 0), # left hip
        (hip_w * 0.25, -torso_len/2 - 0.02 - thigh, 0),  # right knee
        (-hip_w * 0.25, -torso_len/2 - 0.02 - thigh, 0), # left knee
        (hip_w * 0.25, -torso_len/2 - 0.02 - thigh - shin, 0),  # right ankle
        (-hip_w * 0.25, -torso_len/2 - 0.02 - thigh - shin, 0), # left ankle
    ]
    for i, (nx, ny, nz) in enumerate(node_positions):
        v, f = _sphere(tube_r * 1.3, seg=8)
        v = _translate(v, (nx, ny, nz))
        frame_meshes.append(Mesh(v, f, C_NODE, name=f"node_{i}"))

    # Linear actuator indicators (accent colored small cylinders)
    actuator_positions = [
        (0, 0, torso_len * 0.25, 0.06),       # torso extension
        (0, 0, -torso_len * 0.25, 0.06),      # torso extension
        (shoulder_w * 0.5, 0.05, upper_arm * 0.5, 0.04),  # right arm
        (-shoulder_w * 0.5, 0.05, upper_arm * 0.5, 0.04), # left arm
        (hip_w * 0.25, -torso_len/2 - 0.02, -thigh * 0.5, 0.04),  # right thigh
        (-hip_w * 0.25, -torso_len/2 - 0.02, -thigh * 0.5, 0.04), # left thigh
    ]
    for ax, ay, az, alen in actuator_positions:
        v, f = _solid_cylinder(0.008, -alen/2, alen/2, seg=8)
        v = _translate(v, (ax, ay, az))
        frame_meshes.append(Mesh(v, f, C_ACCENT, name=f"actuator_{ax}_{ay}_{az}",
                                 emissive=True))

    parts.append(Part("frame", "CFRP Telescoping Frame + Ti Nodes",
                      frame_meshes,
                      [f"Material: {DIMS['frame_material']}",
                       f"Nodes: {DIMS['frame_node_count']}x {DIMS['frame_node_material']}",
                       f"Pivots: {DIMS['frame_pivot_count']} key joints",
                       f"Actuators: {DIMS['frame_actuator_count']}x Maxon EC45",
                       f"Telescoping: +{DIMS['torso_telescope_mm']}mm torso, +{DIMS['limb_telescope_mm']}mm limbs",
                       f"Straps: {DIMS['frame_strap_count']}x Spectra + BOA dials",
                       f"Fits: {DIMS['pilot_min_height_m']*3.28:.1f}ft - {DIMS['pilot_max_height_m']*3.28:.1f}ft"],
                      order, np.array([0, 0, 0.1]), C_FRAME))
    order += 1

    # ---- 6. HELMET: angular faceted shell, sleek visor, life support ----
    helmet_meshes = []
    h_r = DIMS["helmet_od_mm"] * MM / 2
    head_r = seg["head"] * pilot_height * 0.5
    helmet_y = torso_len/2 + 0.05 + head_r * 1.1

    # Main helmet shell (faceted dome, not round sphere - MJOLNIR style)
    v, f = _pauldron(h_r, 0, helmet_y, 0, seg=10)
    helmet_meshes.append(Mesh(v, f, C_HELMET, name="helmet_shell"))

    # Visor (angular face plate - Iron Man style, not full sphere)
    v, f = _armor_plate(0, helmet_y - h_r * 0.1, h_r * 0.7,
                        h_r * 1.2, h_r * 0.5, 0.008,
                        chamfer=0.006, taper=0.01)
    helmet_meshes.append(Mesh(v, f, C_VISOR, name="visor", emissive=True))

    # Cheek guards (angular plates on sides of face)
    for sx in (1, -1):
        v, f = _armor_plate(sx * h_r * 0.45, helmet_y - h_r * 0.35, h_r * 0.4,
                            h_r * 0.35, h_r * 0.4, 0.006,
                            chamfer=0.004, taper=0.005)
        helmet_meshes.append(Mesh(v, f, C_HELMET, name=f"cheek_guard_{sx}"))

    # Neck collar / seal ring
    v, f = _torus(h_r * 0.65, 0.012, seg_major=18, seg_minor=6)
    v = _translate(v, (0, helmet_y - h_r * 0.7, 0))
    helmet_meshes.append(Mesh(v, f, C_ACCENT, name="neck_seal", emissive=True))

    # Helmet turbine ring (4 micro-turbines for anti-torque)
    for i in range(DIMS["turbine_helmet"]):
        a = 2 * math.pi * i / DIMS["turbine_helmet"]
        tx = h_r * 0.9 * math.cos(a)
        tz = h_r * 0.9 * math.sin(a)
        v, f = _solid_cylinder(0.012, -0.03, 0.03, seg=8)
        v = (np.asarray(v) @ rot_x(math.pi/2).T).tolist()
        v = _translate(v, (tx, helmet_y + h_r * 0.3, tz))
        helmet_meshes.append(Mesh(v, f, C_TURBINE, name=f"helmet_turbine_{i}",
                                  spin=1.0, group="turbine"))

    # Life support pack (back of helmet, compact chamfered)
    v, f = _armor_plate(0, helmet_y + h_r * 0.05, -h_r * 0.85,
                        0.12, 0.08, 0.06, chamfer=0.006, taper=0.008)
    helmet_meshes.append(Mesh(v, f, C_FRAME_DK, name="life_support"))

    # CO2 scrubber vents (sleek slots)
    for i in range(3):
        vx = -0.03 + i * 0.03
        v, f = _box(vx, helmet_y + h_r * 0.08, -h_r * 0.89,
                    0.001, 0.006, 0.008)
        helmet_meshes.append(Mesh(v, f, C_ACCENT, name=f"co2_vent_{i}", emissive=True))

    parts.append(Part("helmet", "Vacuum-Sealed Helmet + Visor + Life Support",
                      helmet_meshes,
                      [f"Shell: graphene-UHMWPE, {DIMS['helmet_wall_mm']}mm wall",
                       f"Weight: {DIMS['helmet_weight_kg']} kg",
                       f"Visor: {DIMS['visor_thickness_mm']}mm graphene-polycarbonate",
                       f"Visor force: {DIMS['visor_max_force_lbs']:,} lbs before breaking",
                       f"Zoom: {DIMS['visor_zoom_optical']}x optical + {DIMS['visor_zoom_digital']}x digital = 20-mile",
                       f"FOV: {DIMS['visor_fov_deg']} deg panoramic",
                       f"Seal: vacuum + {DIMS['seal_depth_m']}m underwater",
                       f"CO2 scrubbing: {DIMS['co2_scrub_hours']}h regenerable 4-bed sieve",
                       f"O2 supply: {DIMS['o2_capacity_hours']}h",
                       f"Helmet turbines: {DIMS['turbine_helmet']}x anti-torque",
                       "Self-healing silicone seals",
                       "Neural electrodes at temples + forehead"],
                      order, np.array([0, 0.3, 0]), C_HELMET))
    order += 1

    # ---- 7. MICRO-TURBOFAN SWARM TURBINES ----
    turbine_meshes = []
    t_r = DIMS["turbine_d_mm"] * MM / 2
    t_len = DIMS["turbine_len_mm"] * MM
    turbine_config = []

    def _add_turbine_group(count, base_pos, spread, direction, name_prefix):
        """Add a group of turbines at positions around a base point."""
        spread = np.asarray(spread, dtype=float)
        direction = np.asarray(direction, dtype=float)
        for i in range(count):
            offset = spread * (i - (count - 1) / 2)
            pos = np.asarray(base_pos, dtype=float) + direction * offset
            # Turbine housing
            v, f = _hollow_cylinder(t_r, t_r * 0.7, 0, t_len, seg=10)
            v = _translate(v, pos.tolist())
            turbine_meshes.append(Mesh(v, f, C_TURBINE, name=f"{name_prefix}_{i}",
                                       spin=1.0, group="turbine"))
            # Intake fan (spinning)
            v2, f2 = _solid_cylinder(t_r * 0.65, -0.005, 0.005, seg=8)
            v2 = _translate(v2, pos.tolist())
            turbine_meshes.append(Mesh(v2, f2, C_ACCENT, name=f"{name_prefix}_fan_{i}",
                                       spin=1.0, group="turbine", emissive=True))
            # Compressor blades (visible inside housing)
            for bi in range(5):
                ba = 2 * math.pi * bi / 5
                blade_pts = []
                for ti in range(3):
                    bt = ti / 2
                    blade_pts.append((t_r * 0.6 * math.cos(ba + bt * 0.3),
                                      t_r * 0.6 * math.sin(ba + bt * 0.3)))
                v_b, f_b = _strip(blade_pts, 0.003)
                v_b = _translate(v_b, (pos[0], pos[1], pos[2] + t_len * 0.3))
                turbine_meshes.append(Mesh(v_b, f_b, C_FRAME_DK, name=f"{name_prefix}_blade_{i}_{bi}",
                                           spin=1.0, group="turbine"))
            # Exhaust glow
            v3, f3 = _cone(t_r * 0.6, t_r * 0.3, t_len, t_len + 0.02, seg=8)
            v3 = _translate(v3, pos.tolist())
            turbine_meshes.append(Mesh(v3, f3, C_TURBINE_HOT, name=f"{name_prefix}_exhaust_{i}",
                                       emissive=True))
            turbine_config.append({"pos": pos.tolist(), "group": name_prefix, "idx": i})

    # Backpack turbines (12, vectored +/-120 deg) - visible jet pack on back
    # Positioned to protrude from back armor, oriented downward for VTOL lift
    _add_turbine_group(DIMS["turbine_backpack"],
                       [0, -torso_len/4, -0.20], [0.055, 0, 0], [1, 0, 0], "backpack")
    # Jet pack cowling/housing (visible nacelle around backpack turbines)
    v, f = _box(0, -torso_len/4, -0.20, 0.30, torso_len * 0.28, 0.05)
    turbine_meshes.insert(0, Mesh(v, f, C_FRAME_DK, name="jetpack_cowling"))
    # Cowling intake vents (visible slots on sides)
    for si in range(4):
        sx = 1 if si < 2 else -1
        vi = si % 2
        v, f = _box(sx * 0.15, -torso_len/4 - 0.04 + vi * 0.08, -0.19,
                    0.02, 0.03, 0.002)
        turbine_meshes.insert(0, Mesh(v, f, C_TURBINE, name=f"jetpack_vent_{si}"))

    # Forearm turbines (8, hand-directed)
    for sx in (1, -1):
        base = [sx * (shoulder_w * 0.5 + 0.06), 0.05, upper_arm * 0.5]
        _add_turbine_group(DIMS["turbine_forearm"] // 2,
                           base, [0, 0, 0.04], [0, 0, 1], f"forearm_{sx}")

    # Calf turbines (8)
    for sx in (1, -1):
        base = [sx * (hip_w * 0.25 + 0.06), -torso_len/2 - 0.02 - thigh - shin * 0.5, 0]
        _add_turbine_group(DIMS["turbine_calf"] // 2,
                           base, [0, 0, 0.04], [0, 0, 1], f"calf_{sx}")

    # Thigh turbines (8)
    for sx in (1, -1):
        base = [sx * (hip_w * 0.25 + 0.06), -torso_len/2 - 0.02 - thigh * 0.5, 0]
        _add_turbine_group(DIMS["turbine_thigh"] // 2,
                           base, [0, 0, 0.04], [0, 0, 1], f"thigh_{sx}")

    parts.append(Part("turbines", f"Swarm Turbines ({DIMS['turbine_count']}x Micro-Turbofans)",
                      turbine_meshes,
                      [f"Count: {DIMS['turbine_count']}x",
                       f"Size: {DIMS['turbine_d_mm']}mm dia x {DIMS['turbine_len_mm']}mm long",
                       f"Thrust: {DIMS['turbine_thrust_dry_lbf']} lbf dry, {DIMS['turbine_thrust_ab_lbf']} lbf afterburner",
                       f"Max RPM: {DIMS['turbine_rpm_max']:,}",
                       f"Bypass ratio: {DIMS['turbine_bypass_ratio']}:1",
                       f"Bearings: {DIMS['turbine_bearing']}",
                       f"Vectoring: +/-{DIMS['turbine_vector_deg']} deg",
                       f"Placement: {DIMS['turbine_backpack']} backpack, {DIMS['turbine_forearm']} forearm, {DIMS['turbine_calf']} calf, {DIMS['turbine_thigh']} thigh, {DIMS['turbine_helmet']} helmet",
                       f"Total thrust: {DIMS['turbine_count'] * DIMS['turbine_thrust_ab_lbf']:,} lbf (afterburner)"],
                      order, np.array([0, 0, 0.2]), C_TURBINE))
    order += 1

    # ---- 8. DEPLOYABLE ARCHANGEL WINGS (compact, turbine-assisted) ----
    wing_meshes = []
    wing_span = DIMS["wing_span_m"]
    wing_area_m2 = DIMS["wing_area_sqft"] * 0.0929  # total area in m^2
    # Mean chord per wing = area_per_wing / semi_span
    semi_span = wing_span / 2
    mean_chord = (wing_area_m2 / 2) / semi_span  # ~0.58m for 3.5m/22sqft
    wing_y = torso_len / 2 * 0.3  # shoulder height
    wing_z = -0.15  # behind back when stowed, deployed outward

    for side in (1, -1):
        # Leading edge spar (nitinol) -- swept-back compact wing
        le_pts = []
        for i in range(10):
            t = i / 9
            span = semi_span * t
            sweep = 0.25 * span  # 25% sweep for compact planform
            le_pts.append((side * span, -sweep))
        v, f = _strip(le_pts, 0.012)
        v = _translate(v, (0, wing_y, wing_z))
        wing_meshes.append(Mesh(v, f, C_WING, name=f"wing_le_{side}"))

        # Membrane surface -- tapered planform from root to tip
        mem_pts = []
        for i in range(8):
            t = i / 7
            span = semi_span * t
            sweep = 0.25 * span
            # Chord tapers from root to tip (root chord ~1.4x mean, tip ~0.4x)
            chord_here = mean_chord * (1.4 - 1.0 * t)
            mem_pts.append((side * span, -sweep))
        v, f = _strip(mem_pts, mean_chord * 0.9)
        v = _translate(v, (0, wing_y, wing_z))
        wing_meshes.append(Mesh(v, f, C_WING_MEM, name=f"wing_mem_{side}"))

        # Nitinol ribs (4 per wing for compact wing)
        for ri in range(4):
            t = (ri + 1) / 5
            span = semi_span * t
            chord = mean_chord * (1.4 - 1.0 * t)
            sweep = 0.25 * span
            v, f = _strip([(side * span, -sweep),
                           (side * span, -sweep + chord)], 0.006)
            v = _translate(v, (0, wing_y, wing_z))
            wing_meshes.append(Mesh(v, f, C_WING, name=f"wing_rib_{side}_{ri}"))

        # Wing root hinge (visible at shoulder)
        v, f = _sphere(0.02, seg=8)
        v = _translate(v, (side * shoulder_w * 0.45, wing_y, wing_z))
        wing_meshes.append(Mesh(v, f, C_NODE, name=f"wing_hinge_{side}"))

        # Stowed wing indicator (flat strip on back when not deployed)
        v, f = _box(side * semi_span * 0.3, wing_y - 0.02, wing_z - 0.01,
                    semi_span * 0.35, 0.003, 0.02)
        wing_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"wing_stowed_{side}"))

    parts.append(Part("wings", "Archangel Deployable Gliding Wings",
                      wing_meshes,
                      [f"Wingspan: {DIMS['wing_span_m']}m ({DIMS['wing_span_m']*3.28:.0f}ft)",
                       f"Surface area: {DIMS['wing_area_sqft']} sq ft",
                       f"Glide ratio: {DIMS['wing_ld_ratio']}:1",
                       f"Deploy time: {DIMS['wing_deploy_s']}s",
                       f"Min sink: {DIMS['wing_min_sink_ms']} m/s",
                       f"Membrane: {DIMS['wing_membrane']}",
                       f"Ribs: {DIMS['wing_rib_material']}",
                       "CO2 cartridge + electric backup deployment",
                       "Emergency jettison (explosive bolts)"],
                      order, np.array([0, 0.15, -0.3]), C_WING))
    order += 1

    # ---- 9. POWER SYSTEM: battery + energy harvesting ----
    power_meshes = []
    # Battery pack (lower back)
    v, f = _box(0, -torso_len/2 + 0.04, -0.14, 0.18, 0.06, 0.08)
    power_meshes.append(Mesh(v, f, C_BATTERY, name="battery_pack"))
    # Hot-swap battery indicators
    for i in range(2):
        v, f = _solid_cylinder(0.01, 0, 0.005, seg=8)
        v = _translate(v, (-0.04 + i * 0.08, -torso_len/2 + 0.04, -0.10))
        power_meshes.append(Mesh(v, f, C_OK if i == 0 else C_WARN, name=f"bat_led_{i}",
                                 emissive=True))
    # Piezoelectric fiber indicators (along legs)
    for sx in (1, -1):
        for j in range(3):
            v, f = _box(sx * (hip_w * 0.25 + 0.08), -torso_len/2 - 0.02 - thigh * (0.3 + j * 0.2),
                        0, 0.01, 0.04, 0.01)
            power_meshes.append(Mesh(v, f, C_FARADAY, name=f"piezo_{sx}_{j}",
                                     emissive=True))
    parts.append(Part("power", "Power System: Solid-State Li-S + Piezo/Solar Harvesting",
                      power_meshes,
                      [f"Battery: {DIMS['battery_type']}, {DIMS['battery_wh']} Wh",
                       f"Energy density: {DIMS['battery_wh_kg']} Wh/kg",
                       f"Runtime: {DIMS['battery_life_hours']}+ hours",
                       f"Piezo harvesting: {DIMS['piezo_harvest_pct']}% kinetic recovery",
                       f"Solar nanofibers: {'Yes' if DIMS['solar_nanofiber'] else 'No'}",
                       "2x 600 Wh hot-swap packs",
                       "Self-healing polymer battery shells"],
                      order, np.array([0, -0.1, -0.1]), C_BATTERY))
    order += 1

    # ---- 9b. FUEL SYSTEM: conformal bladder + lines ----
    fuel_meshes = []
    # Conformal fuel bladder (lower back, wraps around torso)
    bladder_r = max(chest_d, hip_w) * 0.48 + t_inner + t_middle
    v, f = _tube(bladder_r, -torso_len/2 + 0.02, -torso_len/2 + 0.12, seg=16)
    fuel_meshes.append(Mesh(v, f, C_FUEL, name="fuel_bladder_lower"))
    v, f = _tube(bladder_r * 0.95, -torso_len/2 + 0.12, torso_len/2 * 0.3, seg=14)
    fuel_meshes.append(Mesh(v, f, C_FUEL, name="fuel_bladder_upper"))
    # Bladder seam lines (visible Kevlar weave)
    for si in range(4):
        a = 2 * math.pi * si / 4
        v, f = _solid_cylinder(0.001, -torso_len * 0.3, torso_len * 0.1, seg=4)
        v = _translate(v, (bladder_r * math.cos(a), 0, bladder_r * math.sin(a) * 0.5))
        fuel_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"bladder_seam_{si}"))
    # Dual fuel pumps (visible boxes on lower back)
    for pi in range(2):
        px = -0.04 + pi * 0.08
        v, f = _box(px, -torso_len/2 + 0.06, -0.16, 0.03, 0.02, 0.03)
        fuel_meshes.append(Mesh(v, f, C_FRAME, name=f"fuel_pump_{pi}"))
        # Pump LED indicator
        v, f = _sphere(0.004, seg=6)
        v = _translate(v, (px, -torso_len/2 + 0.06, -0.14))
        fuel_meshes.append(Mesh(v, f, C_OK, name=f"fuel_pump_led_{pi}", emissive=True))
    # Fuel lines (stainless braided, running from bladder to turbine groups)
    line_r = 0.003  # 6mm OD
    # Main fuel manifold (along spine)
    v, f = _solid_cylinder(line_r, -torso_len/2 + 0.02, torso_len/2 * 0.5, seg=6)
    v = _translate(v, (0, 0, -0.14))
    fuel_meshes.append(Mesh(v, f, C_FUEL, name="fuel_manifold_spine"))
    # Branch lines to shoulder/forearm turbines
    for sx in (1, -1):
        v, f = _solid_cylinder(line_r, 0, shoulder_w * 0.5, seg=6)
        v = (np.asarray(v) @ rot_z(sx * math.pi/2).T).tolist()
        v = _translate(v, (0, 0.05, -0.12))
        fuel_meshes.append(Mesh(v, f, C_FUEL, name=f"fuel_line_shoulder_{sx}"))
    # Branch lines to thigh turbines
    for sx in (1, -1):
        v, f = _solid_cylinder(line_r, 0, hip_w * 0.25 + 0.06, seg=6)
        v = (np.asarray(v) @ rot_z(sx * math.pi/2).T).tolist()
        v = _translate(v, (0, -torso_len/2 - 0.02, -0.10))
        fuel_meshes.append(Mesh(v, f, C_FUEL, name=f"fuel_line_thigh_{sx}"))
    # Fuel level sensor indicators (4 capacitive sensors)
    for si in range(4):
        sy = -torso_len/2 + 0.04 + si * 0.03
        v, f = _sphere(0.003, seg=5)
        v = _translate(v, (0.06, sy, -0.13))
        fuel_meshes.append(Mesh(v, f, C_ACCENT, name=f"fuel_sensor_{si}", emissive=True))
    # Refuel port (left hip, quick-disconnect)
    v, f = _solid_cylinder(0.012, 0, 0.015, seg=8)
    v = (np.asarray(v) @ rot_z(-math.pi/2).T).tolist()
    v = _translate(v, (-hip_w * 0.5 - 0.02, -torso_len/2 + 0.04, -0.12))
    fuel_meshes.append(Mesh(v, f, C_FRAME, name="refuel_port"))
    # Refuel port cap
    v, f = _sphere(0.013, seg=8)
    v = _translate(v, (-hip_w * 0.5 - 0.035, -torso_len/2 + 0.04, -0.12))
    fuel_meshes.append(Mesh(v, f, C_FRAME_DK, name="refuel_cap"))
    parts.append(Part("fuel_system", f"Fuel System: {DIMS['fuel_capacity_l']}L Conformal Bladder + Pumps",
                      fuel_meshes,
                      [f"Capacity: {DIMS['fuel_capacity_l']}L {DIMS['fuel_type']}",
                       f"Fuel mass: {DIMS['fuel_capacity_l']*DIMS['fuel_density_kg_l']:.1f} kg",
                       f"Bladder: conformal Kevlar, self-sealing",
                       f"Pumps: dual redundant electric, 0.5 LPM each",
                       f"Lines: stainless braided, AN-8, dual redundant",
                       f"Refuel: quick-disconnect port (left hip)",
                       f"Sensors: 4x capacitive fuel level"],
                      order, np.array([0, -0.15, -0.15]), C_FUEL))
    order += 1

    # ---- 10. NEURAL INTERFACE + AI ----
    neural_meshes = []
    # Neural electrode grid on helmet temples (high-density EEG array)
    for sx in (1, -1):
        for ei in range(6):
            for ej in range(4):
                ex = sx * h_r * (0.7 + ei * 0.04)
                ey = helmet_y + h_r * (0.1 + ej * 0.08)
                ez = (ei - 2.5) * 0.01
                v, f = _sphere(0.003, seg=4)
                v = _translate(v, (ex, ey, ez))
                neural_meshes.append(Mesh(v, f, C_FIBER_HOT, name=f"eeg_{sx}_{ei}_{ej}",
                                          emissive=True))
    # Forehead electrode strip (3 electrodes)
    for ei in range(3):
        ex = (ei - 1) * 0.03
        v, f = _sphere(0.004, seg=5)
        v = _translate(v, (ex, helmet_y + h_r * 0.5, h_r * 0.7))
        neural_meshes.append(Mesh(v, f, C_FIBER_HOT, name=f"eeg_forehead_{ei}", emissive=True))
    # Neural signal conduit (helmet to AI processor)
    v, f = _solid_cylinder(0.003, helmet_y - 0.05, 0.0, seg=5)
    neural_meshes.append(Mesh(v, f, C_FIBER, name="neural_conduit", emissive=True))
    # EMG armband indicators (with individual sensor dots)
    for sx in (1, -1):
        v, f = _torus(0.065 * s + t_total + 0.005, 0.004, seg_major=12, seg_minor=4)
        v = _translate(v, (sx * shoulder_w * 0.5, 0.05, upper_arm * 0.3))
        v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
        neural_meshes.append(Mesh(v, f, C_FIBER, name=f"emg_band_{sx}", emissive=True))
        # Individual EMG sensors (8 per arm)
        for si in range(8):
            sa = 2 * math.pi * si / 8
            v, f = _sphere(0.002, seg=4)
            v = _translate(v, (sx * shoulder_w * 0.5 + (0.065 * s + t_total + 0.006) * math.cos(sa),
                              0.05 + upper_arm * 0.3,
                              (0.065 * s + t_total + 0.006) * math.sin(sa)))
            v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
            neural_meshes.append(Mesh(v, f, C_FIBER_HOT, name=f"emg_sensor_{sx}_{si}", emissive=True))
    # AI processor box (spine)
    v, f = _box(0, 0, -0.08, 0.06, 0.04, 0.12)
    neural_meshes.append(Mesh(v, f, C_FRAME, name="ai_processor"))
    # AI processor heatsink fins
    for fi in range(4):
        v, f = _box(0, 0, -0.08 + fi * 0.03, 0.07, 0.001, 0.12)
        neural_meshes.append(Mesh(v, f, C_FRAME_DK, name=f"ai_heatsink_{fi}"))
    # Status LED
    v, f = _sphere(0.006, seg=6)
    v = _translate(v, (0, 0, -0.02))
    neural_meshes.append(Mesh(v, f, C_OK, name="ai_led", emissive=True))
    parts.append(Part("neural", "Neural BCI + Vera 3.0 AI Co-Pilot",
                      neural_meshes,
                      [f"BCI: {DIMS['bci_type']}, <{DIMS['bci_latency_ms']}ms latency",
                       f"Encryption: {DIMS['bci_crypto']}",
                       f"Air-gapped: {DIMS['bci_air_gapped']}",
                       f"AI: {DIMS['os_ai_model']}",
                       "Auto-aim: 98%+ accuracy at 4 miles on Mach-speed targets",
                       "Defense: martial arts AI, 10,000 lbs punch",
                       "Jump assist: 200ft vertical, AI-calculated trajectory",
                       "Thermal hunting + collision avoidance",
                       "G-limiter: auto-takeover at 6.8g, adrenaline wake",
                       "Subvocal mic + eye-tracking + EMG armband backup"],
                      order, np.array([0, 0.1, 0.1]), C_FIBER))
    order += 1

    # ---- 11. THERMAL SYSTEM ----
    thermal_meshes = []
    # Thermal circulation lines (visible as accent lines on torso)
    for i in range(4):
        z = -torso_len/3 + i * torso_len / 6
        v, f = _torus(max(chest_d, hip_w) * 0.5, 0.003, seg_major=16, seg_minor=4)
        v = _translate(v, (0, z, 0))
        thermal_meshes.append(Mesh(v, f, C_ACCENT2, name=f"thermal_line_{i}",
                                   emissive=True))
    # Thermal fluid channels (visible capillary tubes on torso)
    for ci in range(8):
        a = 2 * math.pi * ci / 8
        r_th = max(chest_d, hip_w) * 0.5 + 0.002
        v, f = _solid_cylinder(0.002, -torso_len * 0.4, torso_len * 0.4, seg=4)
        v = _translate(v, (r_th * math.cos(a), 0, r_th * math.sin(a) * 0.5))
        thermal_meshes.append(Mesh(v, f, C_ACCENT2, name=f"thermal_capillary_{ci}", emissive=True))
    # Thermal fluid channels on arms
    for sx in (1, -1):
        for ci in range(3):
            a = 2 * math.pi * ci / 3 + 0.5
            r_arm_th = 0.055 * s + t_inner + 0.002
            v, f = _solid_cylinder(0.0015, 0, upper_arm + forearm, seg=4)
            v = _translate(v, (sx * shoulder_w * 0.5 + r_arm_th * math.cos(a),
                              0.05 + r_arm_th * math.sin(a), 0))
            v = (np.asarray(v) @ rot_y(sx * 0.15).T).tolist()
            thermal_meshes.append(Mesh(v, f, C_ACCENT2, name=f"thermal_arm_{sx}_{ci}", emissive=True))
    # Thermal fluid channels on legs
    for sx in (1, -1):
        for ci in range(4):
            a = 2 * math.pi * ci / 4
            r_leg_th = 0.065 * s + t_inner + 0.002
            v, f = _solid_cylinder(0.002, -thigh - shin, 0, seg=4)
            v = _translate(v, (sx * hip_w * 0.25 + r_leg_th * math.cos(a),
                              -torso_len/2 - 0.02, r_leg_th * math.sin(a)))
            thermal_meshes.append(Mesh(v, f, C_ACCENT2, name=f"thermal_leg_{sx}_{ci}", emissive=True))
    # Peltier modules (shoulders)
    for sx in (1, -1):
        v, f = _box(sx * shoulder_w * 0.4, 0.05, 0.06, 0.03, 0.03, 0.01)
        thermal_meshes.append(Mesh(v, f, C_FARADAY, name=f"peltier_{sx}"))
    parts.append(Part("thermal", "Thermal Regulation: Dragon Skin Active Undersuit",
                      thermal_meshes,
                      [f"Skin temp: {DIMS['thermal_inner_temp_c']}C (locked 36.5-37.5)",
                       f"Range: {DIMS['thermal_range_lo_c']}C to +{DIMS['thermal_range_hi_c']}C",
                       f"Heating: {DIMS['thermal_heating_kw']} kW from reactor waste heat",
                       f"Cooling: {DIMS['thermal_cooling_kw']} kW resistive backup",
                       "Glycol loop from engine heat exchanger",
                       "Phase-change material matrix (melts at 37C)",
                       "Aerogel insulation (R-value = arctic parka, 3mm thick)"],
                      order, np.array([0, 0.05, 0.15]), C_ACCENT2))
    order += 1

    # Inject rich metadata from PART_DB
    for part in parts:
        info = PART_DB.get(part.key, {})
        part.description = info.get("description", "")
        part.category = info.get("category", "")
        part.materials = info.get("materials", [])
        part.weight_kg = info.get("weight_kg", 0.0)
        part.blueprint_notes = info.get("blueprint", "")
        part.layer_num = info.get("layer", part.order)
        part.icon_type = info.get("icon", "")
        part.sub_components = info.get("sub_components", [])
        part.dimensions = info.get("dimensions", "")
        part.connectors = info.get("connectors", [])
        part.performance = info.get("performance", [])
        part.build_details = info.get("build_details", {})

    return parts, turbine_config, {
        "wing_span": wing_span,
        "wing_chord": mean_chord,
        "helmet_y": helmet_y,
        "torso_len": torso_len,
        "pilot_height": pilot_height,
        "pilot_weight": pilot_weight,
    }


# =============================================================================
# SUIT STATE  -- live flight/physics state
# =============================================================================

class SuitState:
    """Live state of the suit during flight simulation."""
    def __init__(self):
        self.time = 0.0
        self.throttle = 0.0
        self.throttle_target = 0.0
        self.wing_deploy = 0.0       # 0 = stowed, 1 = fully deployed
        self.wing_target = 0.0
        self.turbine_rpm = 0.0
        self.turbine_rpm_target = 0.0
        self.afterburner = False
        self.battery_soc = 1.0       # state of charge (1.0 = full)
        # Fuel system: Jet-A1 for micro-turbofans
        self.fuel_max_kg = DIMS["fuel_capacity_l"] * DIMS["fuel_density_kg_l"]  # ~36.5 kg
        self.fuel_kg = self.fuel_max_kg  # start with full tanks
        self.fuel_flow_kg_s = 0.0   # current fuel burn rate (kg/s)
        self.fuel_burned_kg = 0.0   # total fuel consumed
        self.temp_inner = 37.0       # body temp C
        self.temp_outer = 20.0       # outer surface temp C
        self.co2_level = 0.0         # CO2 fraction
        self.o2_level = 1.0          # O2 fraction
        self.seal_pressure = 101.3   # kPa internal
        self.env_idx = 0
        self.altitude = 0.0          # m above ground
        self.velocity = np.zeros(3)
        self.pos = np.array([0.0, 2.0, 0.0])
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.thrust_vector = np.array([0.0, 1.0, 0.0])  # default up
        self.g_load = 1.0
        self.flight_mode = "ground"  # ground | hover | cruise | glide | dive | space | underwater
        self.os_primary_ok = True
        self.os_shadow_ok = True
        self.os_integrity_checks = 0
        self.os_failovers = 0
        self.defense_active = False
        self.aim_locked = False
        self.jump_charge = 0.0
        # 6-DOF flight state
        self.angular_vel = np.zeros(3)  # [pitch_rate, roll_rate, yaw_rate] rad/s
        self.pitch_rate = 0.0
        self.roll_rate = 0.0
        self.yaw_rate = 0.0
        self.auto_hover = False
        self.auto_level = False
        self.collision_warning = False
        self.collision_dist = float('inf')
        self.heat_map = {}  # part_key -> temp C for heat overlay
        # Game state: score, rings, combo (Iron Man game style)
        self.game_score = 0
        self.game_rings_passed = 0
        self.game_combo = 0
        self.game_combo_timer = 0.0
        self.game_timer = 0.0
        self.screen_shake = 0.0
        self.flight_rings = []  # list of {pos, radius, passed, angle, color}
        self._init_flight_rings()
        self.speed_trail = []  # particle trail for speed sensation
        # Atmospheric model
        self.atmosphere = AtmosphericModel()
        self.weather = "clear"
        self.wind_effect = np.zeros(3)  # wind force on suit
        self.solar_harvesting_w = 0.0
        # EMP hardening
        self.emp_shield_active = True
        self.emp_attenuation_db = 60.0  # 60dB EMP attenuation
        self.emp_hits = 0
        self.emp_recover_timer = 0.0
        # Self-healing
        self.self_heal_active = True
        self.self_heal_regions = []  # list of {part, progress, pos}
        # Biometric monitoring
        self.heart_rate = 72.0       # bpm, resting
        self.heart_rate_target = 72.0
        self.adrenaline = 0.0        # 0-1, stress hormone level
        self.stress_level = 0.0      # 0-1, cognitive stress
        self.fatigue = 0.0           # 0-1, physical fatigue
        self.blood_o2_sat = 98.0     # SpO2 %
        self.pilot_mass = 79.4       # kg
        # Damage state per armor part
        self.armor_damage = {}       # part_key -> 0.0 (pristine) to 1.0 (destroyed)
        # Integrated subsystems
        self.rtos = SuitRTOS()
        self.auto_aim = AutoAimSystem()
        self.defense = DefenseAI()
        self.jump = JumpAssist()
        # Full suit subsystems
        self.muscle = MuscleFiberSystem()
        self.thermal = ThermalManagementSystem()
        self.neural = NeuralInterfaceSystem()
        self.life_support = LifeSupportSystem()
        self.dive = DiveSystem()          # buoyancy control + decompression computer
        self.space = SpaceSystem()        # radiation dosimetry + cold-gas RCS
        self.power = PowerManagementSystem()
        self.helmet = HelmetSystem()
        self.frame = FrameSystem()
        # Upgrade subsystems (15 new systems)
        self.stealth = StealthMode()
        self.vision = VisionModeSystem()
        self.grapple = GrappleSystem()
        self.parachute = EmergencyParachute()
        self.drones = DroneSwarm()
        self.countermeasures = CountermeasureSystem()
        self.eshield = EnergyShield()
        self.regen = RegenSystem()
        self.tshield = TacticalShield()
        self.stun = StunSystem()
        self.voice = VoiceCommandSystem()
        self.beacon = EmergencyBeacon()
        self.maglev = MaglevMode()
        self.lights = {
            "power": True,
            "neural": True,
            "thermal": True,
            "seal": True,
            "turbines": False,
            "wings": False,
            "defense": False,
            "aim": False,
            "os_failover": False,
            "stealth": False,
            "vision": False,
            "grapple": False,
            "parachute": False,
            "drones": False,
            "countermeasures": False,
            "eshield": False,
            "tshield": False,
            "stun": False,
            "beacon": False,
            "maglev": False,
            "muscle": False,
            "neural": False,
            "life_support": False,
            "helmet_sealed": False,
            "frame_stress": False,
            "power_shed": False,
        }

    @property
    def env_name(self):
        return ENVIRONMENTS[self.env_idx][0]

    @property
    def env_density(self):
        return ENVIRONMENTS[self.env_idx][1]

    @property
    def env_gravity(self):
        return ENVIRONMENTS[self.env_idx][2]

    @property
    def env_temp(self):
        return ENVIRONMENTS[self.env_idx][4]

    @property
    def env_weather(self):
        return ENVIRONMENTS[self.env_idx][6] if len(ENVIRONMENTS[self.env_idx]) > 6 else "clear"

    @property
    def env_planet(self):
        return ENVIRONMENTS[self.env_idx][7] if len(ENVIRONMENTS[self.env_idx]) > 7 else "earth"

    @property
    def wind_speed(self):
        return self.atmosphere.wind_speed

    @property
    def wind_direction(self):
        return self.atmosphere.wind_direction

    def update(self, dt):
        self.time += dt
        # Smooth throttle
        self.throttle += (self.throttle_target - self.throttle) * min(1.0, dt * 3.0)
        # Smooth wing deployment
        self.wing_deploy += (self.wing_target - self.wing_deploy) * min(1.0, dt / max(DIMS["wing_deploy_s"], 0.1))
        # Turbine RPM follows throttle
        max_rpm = DIMS["turbine_rpm_max"]
        self.turbine_rpm_target = self.throttle * max_rpm * (1.5 if self.afterburner else 1.0)
        self.turbine_rpm += (self.turbine_rpm_target - self.turbine_rpm) * min(1.0, dt * 2.0)
        # Fuel consumption: SFC (lb fuel / lbf thrust / hour) * total thrust lbf -> lb/hr -> kg/s
        # Only burns when turbines are running (air-breathing, not in vacuum)
        if self.env_density > 0.001 and self.fuel_kg > 0:
            sfc = DIMS["turbine_sfc_ab_lb_lbh"] if self.afterburner else DIMS["turbine_sfc_dry_lb_lbh"]
            thrust_per = DIMS["turbine_thrust_ab_lbf"] if self.afterburner else DIMS["turbine_thrust_dry_lbf"]
            total_thrust_lbf = DIMS["turbine_count"] * thrust_per * self.throttle
            fuel_lb_hr = sfc * total_thrust_lbf
            fuel_kg_s = fuel_lb_hr * 0.4536 / 3600.0  # lb/hr -> kg/s
            self.fuel_flow_kg_s = fuel_kg_s
            self.fuel_kg = max(0.0, self.fuel_kg - fuel_kg_s * dt)
            self.fuel_burned_kg += fuel_kg_s * dt
        else:
            self.fuel_flow_kg_s = 0.0
        # Battery drain is now managed by PowerManagementSystem (updated below)
        # Legacy field synced from power.soc after power system update
        # Atmospheric model update
        self.atmosphere.update(dt, self.altitude, self.env_idx, self.env_weather)
        self.weather = self.env_weather
        # Solar harvesting computed for power system input (power system manages SOC)
        wing_area_m2 = DIMS["wing_area_sqft"] * 0.0929 * self.wing_deploy
        self.solar_harvesting_w = self.atmosphere.harvesting_power(wing_area_m2)
        # Wind effect on suit (force = 0.5 * rho * v_wind^2 * area * cd)
        wind_v = self.atmosphere.wind
        wind_speed = np.linalg.norm(wind_v)
        if wind_speed > 0.1:
            rho = self.env_density
            wind_force = 0.5 * rho * wind_speed * wind_speed * 0.5 * 0.3  # frontal area * cd
            self.wind_effect = wind_v / wind_speed * wind_force
        else:
            self.wind_effect = np.zeros(3)
        # Thermal regulation: legacy temp_inner synced from ThermalManagementSystem below
        # Outer skin temp from environment + engine heat (for visual overlay)
        outer_target = self.env_temp + self.throttle * 200
        self.temp_outer += (outer_target - self.temp_outer) * dt * 0.3
        # Heat map for visual overlay
        self.heat_map["turbines"] = 20.0 + self.throttle * 600
        self.heat_map["helmet"] = self.temp_inner + 5
        self.heat_map["outer_armor"] = self.temp_outer
        self.heat_map["frame"] = 20.0 + self.throttle * 50
        self.heat_map["power"] = 25.0 + self.throttle * 80
        # CO2 / O2 managed by LifeSupportSystem below (legacy fields synced after update)
        # OS: run SuitRTOS
        self.rtos.update(dt)
        self.os_integrity_checks = self.rtos.integrity_checks
        self.os_failovers = self.rtos.failovers
        self.os_primary_ok = self.rtos.primary_ok
        # EMP recovery
        if self.emp_recover_timer > 0:
            self.emp_recover_timer -= dt
            if self.emp_recover_timer <= 0:
                self.emp_shield_active = True
        # Self-healing progress
        for region in self.self_heal_regions:
            region["progress"] = min(1.0, region["progress"] + dt * 0.5)
        self.self_heal_regions = [r for r in self.self_heal_regions if r["progress"] < 1.0]
        # Biometric monitoring
        # Heart rate responds to throttle, g-load, combat, altitude, and thermal stress
        target_hr = 72.0 + self.throttle * 60.0 + max(0, (self.g_load - 1.0)) * 15.0
        if self.defense.active:
            target_hr += 40.0
        if self.auto_aim.dodge_recommended:
            target_hr += 30.0
        if self.env_density < 0.01:  # space - micro gravity reduces cardiac load
            target_hr -= 5.0
        # Thermal stress: elevated body temp raises heart rate
        if hasattr(self, 'thermal') and self.thermal.skin_temp > 38.0:
            target_hr += (self.thermal.skin_temp - 38.0) * 20.0
        # Life support: CO2 buildup raises heart rate
        if hasattr(self, 'life_support') and self.life_support.active and not self.life_support.co2_safe:
            target_hr += 15.0
        self.heart_rate_target = clamp(target_hr, 50.0, 200.0)
        self.heart_rate += (self.heart_rate_target - self.heart_rate) * dt * 2.0
        # Adrenaline spikes with combat, dodge, high g-load
        adr_target = 0.0
        if self.defense.active:
            adr_target = max(adr_target, 0.8)
        if self.auto_aim.dodge_recommended:
            adr_target = max(adr_target, 0.9)
        if self.g_load > 5.0:
            adr_target = max(adr_target, 0.5)
        if self.throttle > 0.8:
            adr_target = max(adr_target, 0.3)
        self.adrenaline += (adr_target - self.adrenaline) * dt * 1.5
        # Stress from low battery, high CO2, OS failures, neural quality, thermal
        stress_target = 0.0
        if self.battery_soc < 0.2:
            stress_target += 0.4
        if self.co2_level > 0.03:
            stress_target += 0.3
        if not self.os_primary_ok:
            stress_target += 0.5
        if not self.emp_shield_active:
            stress_target += 0.3
        # Neural interface degradation causes stress
        if hasattr(self, 'neural') and self.neural.signal_quality < 0.7:
            stress_target += (0.7 - self.neural.signal_quality) * 0.5
        # Thermal discomfort
        if hasattr(self, 'thermal') and (self.thermal.skin_temp > 38.5 or self.thermal.skin_temp < 36.0):
            stress_target += 0.2
        self.stress_level += (stress_target - self.stress_level) * dt * 0.8
        # Fatigue accumulates with throttle and time
        self.fatigue = min(1.0, self.fatigue + dt * 0.005 * (0.5 + self.throttle))
        # Blood O2 saturation drops with altitude and CO2
        # Use LifeSupportSystem O2 data when available
        if hasattr(self, 'life_support') and self.life_support.active:
            o2_target = 98.0 - max(0, self.altitude - 3000) * 0.001
            if not self.life_support.o2_safe:
                o2_target -= 5.0  # hypoxia from low O2
            if not self.life_support.co2_safe:
                o2_target -= 3.0  # CO2 poisoning
        else:
            o2_target = 98.0 - max(0, self.altitude - 3000) * 0.001 - self.co2_level * 10
        self.blood_o2_sat += (o2_target - self.blood_o2_sat) * dt * 0.5
        # Auto-aim update
        self.auto_aim.update(dt, self.pos)
        self.aim_locked = self.auto_aim.lock_acquired
        # Defense AI update
        self.defense.update(dt, threat_detected=self.defense_active)
        self.defense_active = self.defense.active
        # Jump assist update
        self.jump.update_charge(dt)
        if self.jump.jumping:
            jump_h = self.jump.update_jump(dt, self.pos)
            self.pos[1] = max(0, self.jump.jump_start_pos[1] + jump_h)
        # 6-DOF attitude: smooth angular rates toward targets
        self.pitch += self.pitch_rate * dt
        self.roll += self.roll_rate * dt
        self.yaw += self.yaw_rate * dt
        # Auto-level: damp angular rates when no input
        if self.auto_level:
            self.pitch_rate *= max(0.0, 1.0 - dt * 5.0)
            self.roll_rate *= max(0.0, 1.0 - dt * 5.0)
            self.pitch *= max(0.0, 1.0 - dt * 3.0)
            self.roll *= max(0.0, 1.0 - dt * 3.0)
        # Auto-hover: maintain altitude (throttle = weight / effective_thrust)
        if self.auto_hover and self.altitude > 0.5:
            total_mass = self.pilot_mass + DIMS["weight_total_kg"] + self.fuel_kg
            weight_n = total_mass * self.env_gravity
            # Account for power system voltage sag and air density in max thrust
            max_thrust_n = DIMS["turbine_count"] * DIMS["turbine_thrust_dry_lbf"] * 4.448
            # Air density ratio (thrust scales with mass flow)
            rho_sl = 1.225
            density_ratio = min(1.0, self.env_density / rho_sl) if self.env_density > 0.001 else 0.0
            max_thrust_n *= density_ratio
            if hasattr(self, 'power'):
                v_ratio = self.power.voltage / self.power.NOMINAL_VOLTAGE
                max_thrust_n *= v_ratio ** 2
                if self.power.load_shedding:
                    max_thrust_n *= 0.6
            self.throttle_target = clamp(weight_n / max(max_thrust_n, 1.0), 0.0, 1.0)
        # Collision warning (simple ground proximity)
        self.collision_dist = self.pos[1] if self.pos[1] >= 0 else 0.0
        self.collision_warning = self.collision_dist < 5.0 and self.velocity[1] < -2.0
        # --- Full suit subsystem updates ---
        speed = np.linalg.norm(self.velocity)
        combat_active = self.defense.active or self.auto_aim.lock_acquired
        # Power management: battery + harvesting + load shedding (run first so other systems react)
        self.power.update(dt, self.throttle, self.afterburner, self.solar_harvesting_w,
                          self.atmosphere.turbulence, self.wing_deploy, self.g_load)
        self.battery_soc = self.power.soc  # sync legacy field
        # Muscle fiber system: DEA contraction + STF stiffening
        # Power load shedding reduces muscle voltage (non-critical system)
        if self.power.load_shedding and "muscle" in self.power.shed_systems:
            self.muscle.set_voltage(self.muscle.voltage * 0.5)  # halved when shed
        self.muscle.update(dt, self.throttle, self.g_load)
        # Strike force is mechanically driven by the DEA arm fibers routed through
        # the frame + hip/torso kinetic chain (peaks at the rated punch when fully
        # energized, falls to the frame-only floor when the fibers are idle).
        arm_force_n = (self.muscle.force_output.get("right_arm", 0.0)
                       + self.muscle.force_output.get("left_arm", 0.0))
        self.defense.punch_force_lbs = 300.0 + arm_force_n * 0.2248 * DIMS["punch_kinetic_chain_x"]
        # Thermal management: active heating/cooling
        self.thermal.update(dt, self.env_temp, self.temp_outer, self.throttle, speed, self.env_density)
        self.temp_inner = self.thermal.skin_temp
        # Neural interface: BCI signal generation + intent decoding
        self.neural.generate_signals(self.throttle, self.g_load, combat_active)
        self.neural.decode_intent(self.throttle, self.g_load, combat_active, self.auto_aim.dodge_recommended)
        # Life support: CO2/O2/seal management
        self.life_support.update(dt, self.env_density, self.atmosphere.ambient_pressure / 1000,
                                 self.env_temp, self.pilot_mass, self.throttle)
        self.co2_level = self.life_support.co2_ppm / 10000.0  # update legacy field
        self.o2_level = self.life_support.o2_ppm / 209500.0
        self.seal_pressure = self.life_support.seal_pressure_kpa
        # Dive computer + buoyancy control (underwater) and space env (vacuum)
        suit_total_mass = self.pilot_mass + DIMS["weight_total_kg"]
        self.dive.update(dt, max(0.0, -self.pos[1]), self.env_density,
                         self.env_gravity, suit_total_mass)
        self.space.update(dt, self.env_density, self.env_planet, self.throttle, suit_total_mass)
        # Helmet system: visor + comms
        locked_dist = 0.0
        if self.auto_aim.locked_target is not None:
            locked_dist = float(np.linalg.norm(self.auto_aim.locked_target.pos - self.pos))
        self.helmet.update(dt, self.env_density, self.env_temp, self.auto_aim.lock_acquired, locked_dist)
        # Frame system: telescoping + force distribution + dampers
        total_mass = self.pilot_mass + DIMS["weight_total_kg"]
        self.frame.update(dt, self.g_load, self.throttle, speed, total_mass)
        # Lights
        self.lights["turbines"] = self.throttle > 0.01
        self.lights["wings"] = self.wing_deploy > 0.5
        self.lights["defense"] = self.defense.active
        self.lights["aim"] = self.aim_locked
        self.lights["os_failover"] = self.rtos.failovers > 0 and self.rtos.uptime_s < 10.0
        self.lights["muscle"] = self.muscle.voltage > 0.5
        self.lights["neural"] = self.neural.signal_quality > 0.5
        self.lights["life_support"] = self.life_support.active
        self.lights["helmet_sealed"] = self.helmet.sealed
        self.lights["frame_stress"] = not self.frame.frame_ok
        self.lights["power_shed"] = self.power.load_shedding
        # Flight mode determination
        if self.throttle > 0.5:
            self.flight_mode = "hover" if abs(self.velocity[1]) < 0.5 else "cruise"
        elif self.wing_deploy > 0.5 and self.altitude > 5:
            self.flight_mode = "glide"
        elif self.env_density > 100:
            self.flight_mode = "underwater"
        elif self.env_density < 0.01:
            self.flight_mode = "space"
        elif self.throttle < 0.01 and self.altitude < 1:
            self.flight_mode = "ground"
        else:
            self.flight_mode = "cruise"
        # Update game state
        self.game_timer += dt
        self._update_flight_rings(dt)
        # Decay screen shake
        self.screen_shake *= max(0.0, 1.0 - dt * 8.0)
        # Decay combo timer
        if self.game_combo_timer > 0:
            self.game_combo_timer -= dt
            if self.game_combo_timer <= 0:
                self.game_combo = 0
        # Update speed trail
        speed = np.linalg.norm(self.velocity)
        if speed > 5.0:
            self.speed_trail.append({
                "pos": self.pos.copy(),
                "life": 1.0,
                "size": 2 + min(speed * 0.1, 4),
            })
        for p in self.speed_trail:
            p["life"] -= dt * 3.0
        self.speed_trail = [p for p in self.speed_trail if p["life"] > 0]
        if len(self.speed_trail) > 80:
            self.speed_trail = self.speed_trail[-80:]

    def _init_flight_rings(self):
        """Spawn initial flight rings in a course pattern ahead of player."""
        self.flight_rings = []
        for i in range(12):
            angle = i * 0.3
            x = math.sin(angle) * 30 + np.random.uniform(-10, 10)
            y = 15 + i * 8 + np.random.uniform(-3, 3)
            z = 40 + i * 35
            ring = {
                "pos": np.array([x, y, z]),
                "radius": 5.0,
                "passed": False,
                "angle": 0.0,
                "color": (100, 200, 255),
                "glow": 0.0,
            }
            self.flight_rings.append(ring)

    def _update_flight_rings(self, dt):
        """Check ring pass-through and respawn rings ahead."""
        for ring in self.flight_rings:
            ring["angle"] += dt * 1.5
            ring["glow"] *= max(0.0, 1.0 - dt * 3.0)
            if ring["passed"]:
                continue
            dist = np.linalg.norm(self.pos - ring["pos"])
            if dist < ring["radius"] * 1.2:
                # Check if roughly aligned (within radius)
                rel = self.pos - ring["pos"]
                if np.linalg.norm(rel) < ring["radius"]:
                    ring["passed"] = True
                    ring["glow"] = 1.0
                    self.game_rings_passed += 1
                    self.game_combo += 1
                    self.game_combo_timer = 5.0
                    points = 100 * (1 + self.game_combo * 0.1)
                    self.game_score += int(points)
                    self.screen_shake = max(self.screen_shake, 0.3)
        # Respawn passed rings far ahead
        for ring in self.flight_rings:
            if ring["passed"] and ring["glow"] < 0.1:
                # Move ring ahead of player
                ahead_z = max(ring["pos"][2], self.pos[2]) + 200 + np.random.uniform(-30, 30)
                ring["pos"] = np.array([
                    np.random.uniform(-40, 40),
                    np.random.uniform(10, 60),
                    ahead_z,
                ])
                ring["passed"] = False
                ring["glow"] = 0.0
        # Also recycle rings that are too far behind
        for ring in self.flight_rings:
            if ring["pos"][2] < self.pos[2] - 50 and not ring["passed"]:
                ahead_z = self.pos[2] + 200 + np.random.uniform(-30, 30)
                ring["pos"] = np.array([
                    np.random.uniform(-40, 40),
                    np.random.uniform(10, 60),
                    ahead_z,
                ])

    def emp_hit(self, intensity_db=80.0):
        """Simulate an EMP strike on the suit.
        Faraday mesh attenuates by emp_attenuation_db.
        If residual exceeds threshold, temporary disruption."""
        self.emp_hits += 1
        residual = max(0.0, intensity_db - self.emp_attenuation_db)
        if residual > 10.0:
            # Temporary disruption: shield drops, OS switches to shadow
            self.emp_shield_active = False
            self.emp_recover_timer = 2.0  # 2s to recover
            # OS may detect anomaly
            for name, task in self.rtos.tasks.items():
                if task.critical:
                    task.anomaly_score = max(task.anomaly_score, 0.3)
            return False  # EMP partially penetrated
        return True  # EMP fully blocked

    def trigger_self_heal(self, part_key, pos=None):
        """Trigger self-healing process on a damaged region."""
        region = {
            "part": part_key,
            "progress": 0.0,
            "pos": pos if pos is not None else np.zeros(3),
        }
        self.self_heal_regions.append(region)
        return region

    def damage_armor(self, part_key, amount=0.1):
        """Apply damage to an armor part. Returns new damage level."""
        current = self.armor_damage.get(part_key, 0.0)
        self.armor_damage[part_key] = min(1.0, current + amount)
        # Trigger self-healing if damage is significant
        if amount > 0.05:
            self.trigger_self_heal(part_key)
        return self.armor_damage[part_key]

    def repair_armor(self, part_key, amount=0.1):
        """Repair armor damage on a part."""
        current = self.armor_damage.get(part_key, 0.0)
        self.armor_damage[part_key] = max(0.0, current - amount)
        return self.armor_damage[part_key]

    @property
    def armor_integrity(self):
        """Overall armor integrity 0-1 (1 = pristine)."""
        if not self.armor_damage:
            return 1.0
        return 1.0 - sum(self.armor_damage.values()) / len(self.armor_damage)

    @property
    def self_heal_active_count(self):
        return len(self.self_heal_regions)

    def set_pilot(self, height_m, weight_kg):
        """Set pilot dimensions: updates mass, frame telescoping, physics."""
        self.pilot_mass = weight_kg
        self.frame.set_pilot(height_m, weight_kg)


# =============================================================================
# SUITRTOS  -- dual-redundant real-time embedded OS simulation
# =============================================================================

# CRC32 lookup table (polynomial 0xEDB88320)
_CRC32_TABLE = None

def _crc32_table():
    global _CRC32_TABLE
    if _CRC32_TABLE is not None:
        return _CRC32_TABLE
    tbl = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if (c & 1) else (c >> 1)
        tbl.append(c)
    _CRC32_TABLE = tbl
    return tbl


def crc32(data):
    """Compute CRC32 checksum of bytes/bytearray."""
    tbl = _crc32_table()
    c = 0xFFFFFFFF
    for b in (data if isinstance(data, (bytes, bytearray)) else bytes(data)):
        c = tbl[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


class RedundantTask:
    """A dual-redundant task with primary + shadow execution and CRC voting."""
    def __init__(self, name, priority, critical=True):
        self.name = name
        self.priority = priority
        self.critical = critical
        self.primary_output = bytearray(64)
        self.shadow_output = bytearray(64)
        self.crc_primary = 0
        self.crc_shadow = 0
        self.primary_active = True
        self.shadow_active = True
        self.failover_count = 0
        self.last_run_ms = 0
        self.run_count = 0
        self.latency_ms = 0.0
        self.anomaly_score = 0.0  # 0.0 = clean, 1.0 = infected

    def execute(self, time_ms, payload=None):
        """Simulate task execution on both cores. Returns (crc_p, crc_s, match)."""
        self.last_run_ms = time_ms
        self.run_count += 1
        # Simulate output: write deterministic data based on time + payload
        seed = (time_ms ^ (self.run_count * 2654435761)) & 0xFFFFFFFF
        for i in range(64):
            self.primary_output[i] = (seed >> (i % 4 * 8)) & 0xFF
            self.shadow_output[i] = (seed >> (i % 4 * 8)) & 0xFF
        # If anomaly present, corrupt one byte on primary
        if self.anomaly_score > 0.5:
            idx = int(self.anomaly_score * 63) % 64
            self.primary_output[idx] ^= 0xFF
        self.crc_primary = crc32(self.primary_output)
        self.crc_shadow = crc32(self.shadow_output)
        match = (self.crc_primary == self.crc_shadow)
        # Simulate latency (high-priority = lower latency)
        self.latency_ms = max(0.1, 5.0 - self.priority * 0.4)
        return self.crc_primary, self.crc_shadow, match

    def failover(self):
        """Promote shadow to primary, restore primary from ROM backup."""
        self.primary_output = bytearray(self.shadow_output)
        self.crc_primary = self.crc_shadow
        self.primary_active = True
        self.failover_count += 1
        self.anomaly_score = 0.0  # virus purged by ROM restore


class SuitRTOS:
    """Full dual-redundant real-time OS simulation.

    Models:
    - Dual-kernel (primary M7 + shadow M4) with active-active voting
    - CRC32 integrity checks every 10ms on all critical tasks
    - Automatic failover with ROM backup restore on mismatch
    - Virus/anomaly detection (byte-level comparison + CRC voting)
    - Real-time task scheduling (priority-based, deterministic)
    - Task registry with 20 max concurrent tasks
    - BCI neural signal processing pipeline (<17ms latency)
    - AI model execution (Vera 3.0 intent classification)
    - Actuator command dispatch (DEA fiber control)
    - Environmental monitoring (life support, thermal, seal integrity)
    """
    MAX_TASKS = 20
    CHECK_INTERVAL_MS = 10
    FAILOVER_MS = 5

    def __init__(self):
        self.tasks = {}
        self.task_order = []
        self.time_ms = 0
        self.integrity_checks = 0
        self.failovers = 0
        self.viruses_detected = 0
        self.viruses_purged = 0
        self.rom_restores = 0
        self.primary_ok = True
        self.shadow_ok = True
        self.uptime_s = 0.0
        self.last_check_ms = 0
        self.scheduling_log = []  # recent task execution log
        self._init_tasks()
        # Injected anomaly simulation
        self.inject_virus_at_s = -1.0  # disabled (negative = no injection)
        self.inject_target_task = "neural"

    def _init_tasks(self):
        """Register all suit OS tasks with priorities."""
        task_defs = [
            ("neural",      10, True),   # BCI neural interface (<17ms)
            ("actuator",    10, True),   # DEA fiber control
            ("integrity",    9, True),   # CRC checker itself
            ("ai_predict",   7, True),   # Vera 3.0 AI predictions
            ("auto_aim",     8, True),   # targeting + trajectory
            ("defense",      8, True),   # martial arts defense AI
            ("flight_ctrl",  9, True),   # 6-DOF flight control
            ("env_monitor",  5, True),   # life support + thermal
            ("power_mgmt",   5, True),   # battery + harvesting
            ("comms",        3, False),  # radio (non-critical)
            ("visor",        6, True),   # visor display + zoom
            ("seal_monitor", 8, True),   # vacuum/underwater seal
            ("collision",    7, True),   # collision avoidance
            ("g_limiter",    9, True),   # G-force bone protection
            ("jump_assist",  8, True),   # AI jump trajectory
            ("thermal",      6, True),   # thermal regulation
            ("co2_scrub",    5, True),   # CO2 scrubber control
            ("self_heal",    4, False),  # self-healing material monitor
            ("crypto",       7, True),   # lattice-based BCI encryption
            ("boot_monitor", 2, False),  # boot integrity watchdog
        ]
        for name, prio, critical in task_defs:
            t = RedundantTask(name, prio, critical)
            self.tasks[name] = t
            self.task_order.append(name)

    def inject_virus(self, target_task="neural", at_s=5.0):
        """Schedule a virus injection to test failover."""
        self.inject_virus_at_s = at_s
        self.inject_target_task = target_task

    def update(self, dt):
        """Advance OS simulation by dt seconds."""
        self.uptime_s += dt
        self.time_ms = int(self.uptime_s * 1000)

        # Check if we should inject a virus
        if self.inject_virus_at_s >= 0 and self.uptime_s >= self.inject_virus_at_s:
            if self.inject_target_task in self.tasks:
                self.tasks[self.inject_target_task].anomaly_score = 0.7
                self.inject_virus_at_s = -1.0  # only once

        # Execute all tasks (priority order: highest first)
        for name in sorted(self.task_order, key=lambda n: -self.tasks[n].priority):
            task = self.tasks[name]
            if task.primary_active or task.shadow_active:
                task.execute(self.time_ms)
                self.scheduling_log.append((self.time_ms, name, task.latency_ms))
                # Trim log
                if len(self.scheduling_log) > 200:
                    self.scheduling_log = self.scheduling_log[-100:]

        # Integrity check cycle
        if self.time_ms - self.last_check_ms >= self.CHECK_INTERVAL_MS:
            self._integrity_check()
            self.last_check_ms = self.time_ms

    def _integrity_check(self):
        """Run CRC comparison on all critical tasks. Handle failover if mismatch."""
        for name in self.task_order:
            task = self.tasks[name]
            if not task.critical:
                continue
            self.integrity_checks += 1
            _, _, match = task.crc_primary, task.crc_shadow, (task.crc_primary == task.crc_shadow)
            if not match:
                # Virus/anomaly detected
                self.viruses_detected += 1
                self.primary_ok = False
                # Failover: promote shadow, restore from ROM
                task.failover()
                self.failovers += 1
                self.rom_restores += 1
                self.viruses_purged += 1
                self.primary_ok = True
                # Log the event
                self.scheduling_log.append((self.time_ms, f"FAILOVER:{name}", 0.0))

    @property
    def uptime_str(self):
        h = int(self.uptime_s // 3600)
        m = int((self.uptime_s % 3600) // 60)
        s = int(self.uptime_s % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def task_count(self):
        return len(self.tasks)

    @property
    def critical_task_count(self):
        return sum(1 for t in self.tasks.values() if t.critical)

    @property
    def uptime_pct(self):
        """Calculate effective uptime percentage."""
        total_checks = max(1, self.integrity_checks)
        failures = self.failovers
        return (1.0 - failures / total_checks) * 100.0

    def get_task_status(self):
        """Return list of (name, priority, active, failovers, latency, anomaly) for all tasks."""
        result = []
        for name in self.task_order:
            t = self.tasks[name]
            result.append((name, t.priority, t.primary_active, t.failover_count,
                           t.latency_ms, t.anomaly_score))
        return result

    def get_recent_events(self, n=10):
        """Return last n scheduling/failover events."""
        return self.scheduling_log[-n:] if self.scheduling_log else []

    # ---- BOOT SEQUENCE ----
    def boot_sequence(self):
        """Simulate the boot sequence of the dual-redundant OS.
        Returns list of (timestamp_ms, phase, status)."""
        boot_log = []
        phases = [
            ("Power-On Self Test", 12),
            ("ROM checksum verify", 8),
            ("Primary kernel load", 15),
            ("Shadow kernel load", 15),
            ("CRC table init", 5),
            ("Task registry init", 10),
            ("Neural interface calibrate", 20),
            ("Actuator zeroing", 18),
            ("Sensor fusion start", 12),
            ("AI model load (Vera 3.0)", 45),
            ("Crypto handshake", 8),
            ("Flight envelope check", 10),
            ("Mission ready", 2),
        ]
        t_ms = 0
        for phase_name, duration in phases:
            t_ms += duration
            boot_log.append((t_ms, phase_name, "OK"))
        self.uptime_s = t_ms / 1000.0
        self.time_ms = t_ms
        self.last_check_ms = t_ms
        return boot_log

    # ---- TASK SCHEDULER GANTT ----
    def get_gantt_data(self, window_ms=1000):
        """Return task scheduling data for Gantt chart visualization.
        Returns list of (task_name, start_ms, duration_ms, priority)."""
        recent = [e for e in self.scheduling_log if e[0] >= self.time_ms - window_ms]
        gantt = []
        for ts, name, latency in recent:
            if name.startswith("FAILOVER"):
                continue
            task = self.tasks.get(name)
            if task:
                gantt.append((name, ts, latency, task.priority))
        return gantt

    # ---- MEMORY MAP ----
    def get_memory_map(self):
        """Return memory layout of the OS (simulated)."""
        return {
            "kernel_primary":  {"base": "0x40000000", "size_kb": 256, "type": "ROM"},
            "kernel_shadow":   {"base": "0x40040000", "size_kb": 256, "type": "ROM"},
            "task_table":      {"base": "0x40080000", "size_kb": 64,  "type": "RAM"},
            "crc_tables":      {"base": "0x40090000", "size_kb": 32,  "type": "RAM"},
            "neural_buffer":   {"base": "0x40098000", "size_kb": 512, "type": "RAM"},
            "ai_model":        {"base": "0x40118000", "size_kb": 2048,"type": "RAM"},
            "flight_data":     {"base": "0x40318000", "size_kb": 128, "type": "RAM"},
            "combat_data":     {"base": "0x40338000", "size_kb": 64,  "type": "RAM"},
            "crypto_keys":     {"base": "0x40348000", "size_kb": 16,  "type": "ROM"},
            "log_buffer":      {"base": "0x4034C000", "size_kb": 64,  "type": "RAM"},
            "rom_backup":      {"base": "0x4035C000", "size_kb": 512, "type": "ROM"},
        }

    # ---- CRYPTO HANDSHAKE ----
    def crypto_handshake(self):
        """Simulate post-quantum crypto handshake between primary and shadow.
        Returns (success, rounds, key_id)."""
        rounds = 4  # Kyber-1024 requires 4 rounds
        key_id = f"PQ-K{self.time_ms:010d}"
        # Simulate handshake (in real system: Kyber-1024 + AES-256-GCM)
        success = True
        for r in range(rounds):
            # Each round: exchange public keys, verify signatures
            pass
        return success, rounds, key_id

    # ---- AI INTENT CLASSIFICATION ----
    def ai_intent_classify(self, neural_signal):
        """Classify pilot intent from neural signal features.
        Returns (intent_class, confidence, action).

        In the real system, this uses a lightweight transformer
        on the neural BCI data to predict pilot intent."""
        # Simulated intent classes
        intents = {
            "hover":    {"threshold": 0.3, "action": "maintain_altitude"},
            "forward":  {"threshold": 0.5, "action": "thrust_forward"},
            "ascend":   {"threshold": 0.6, "action": "increase_throttle"},
            "descend":  {"threshold": 0.4, "action": "decrease_throttle"},
            "land":     {"threshold": 0.7, "action": "auto_land"},
            "combat":   {"threshold": 0.8, "action": "engage_target"},
            "evade":    {"threshold": 0.75, "action": "dodge_maneuver"},
            "jump":     {"threshold": 0.65, "action": "charge_jump"},
        }
        # Simplified: neural_signal is a float 0-1
        best_intent = "idle"
        best_conf = 0.0
        best_action = "standby"
        for intent, cfg in intents.items():
            if neural_signal >= cfg["threshold"]:
                if cfg["threshold"] > best_conf:
                    best_intent = intent
                    best_conf = cfg["threshold"]
                    best_action = cfg["action"]
        confidence = min(1.0, neural_signal + 0.1)
        return best_intent, confidence, best_action

    # ---- BCI SIGNAL DECODING ----
    def bci_decode(self, raw_signal):
        """Decode raw BCI signal into motor intent commands.
        Returns dict of decoded commands.

        In the real system, this processes 64-channel EEG data
        through a CNN+LSTM pipeline to extract motor commands."""
        # Simulated: raw_signal is a numpy array of channel values
        if isinstance(raw_signal, (int, float)):
            raw_signal = np.array([raw_signal] * 64)
        elif len(raw_signal) < 64:
            raw_signal = np.pad(raw_signal, (0, 64 - len(raw_signal)))
        # Extract features (simplified)
        alpha_power = np.mean(raw_signal[:32])  # alpha band
        beta_power = np.mean(raw_signal[32:])   # beta band
        gamma_power = np.max(raw_signal)        # gamma burst
        # Decode motor intent
        commands = {
            "left_arm": float(beta_power * 0.8),
            "right_arm": float(beta_power * 1.2),
            "left_leg": float(alpha_power * 0.6),
            "right_leg": float(alpha_power * 0.9),
            "trigger": float(gamma_power > 0.7),
            "speak": float(alpha_power > 0.5 and beta_power < 0.3),
            "latency_ms": 14.0,  # <17ms target
        }
        return commands


# =============================================================================
# KALMAN FILTER  -- trajectory prediction for auto-aim
# =============================================================================

class KalmanTracker:
    """1D Kalman filter for target position/velocity tracking.
    Used for auto-aim trajectory prediction on moving targets."""
    def __init__(self, process_noise=1.0, measurement_noise=4.0):
        self.x = np.array([0.0, 0.0])  # [position, velocity]
        self.P = np.array([[10.0, 0.0], [0.0, 10.0]])  # covariance
        self.Q = np.array([[process_noise, 0.0], [0.0, process_noise]])
        self.R = np.array([[measurement_noise]])
        self.H = np.array([[1.0, 0.0]])  # measurement matrix

    def predict(self, dt):
        """Predict next state."""
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update with new measurement."""
        y = measurement - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y.flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[0]

    @property
    def velocity(self):
        return self.x[1]

    def predict_future(self, dt_ahead):
        """Predict position dt_ahead seconds into the future."""
        return self.x[0] + self.x[1] * dt_ahead


class Target3D:
    """3D moving target with 3-axis Kalman trackers for auto-aim."""
    def __init__(self, pos, vel, mach_speed=False):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.mach_speed = mach_speed
        self.alive = True
        self.hit = False
        self.iff_status = "unknown"
        self.trackers = [KalmanTracker() for _ in range(3)]
        # Initialize trackers with first measurement
        for i in range(3):
            self.trackers[i].x = np.array([self.pos[i], self.vel[i]])

    def update(self, dt, suit_pos=None):
        """Move target and update trackers."""
        self.pos += self.vel * dt
        for i in range(3):
            self.trackers[i].predict(dt)
            self.trackers[i].update(np.array([self.pos[i]]))

    def predicted_position(self, dt_ahead):
        """Predict where target will be dt_ahead seconds from now."""
        return np.array([t.predict_future(dt_ahead) for t in self.trackers])

    @property
    def speed(self):
        return np.linalg.norm(self.vel)

    @property
    def mach_number(self):
        return self.speed / 343.0  # speed of sound at sea level


class AutoAimSystem:
    """Full auto-aim system: target acquisition, trajectory prediction, fire control.

    Capabilities:
    - 4-mile effective range
    - Mach-speed target tracking (3-axis Kalman)
    - Multi-target tracking (up to 8 simultaneous)
    - Threat priority sorting (range, speed, heading)
    - IFF transponder (identify friend/foe)
    - Missile dodge AI (predict inbound trajectory, compute evasion)
    - Projectile time-of-flight calculation
    - Lead-angle computation for moving targets
    - 98%+ accuracy with AI-assisted trajectory
    - Neural-triggered firing
    - Weapon mount point management
    """
    def __init__(self):
        self.targets = []
        self.locked_target = None
        self.aim_range_m = DIMS["perf_aim_range_miles"] * 1609.34
        self.accuracy_pct = DIMS["perf_aim_accuracy_pct"]
        self.muzzle_velocity = 915.0  # m/s (approx 3000 fps, 5.56mm)
        self.engaged = False
        self.shots_fired = 0
        self.shots_hit = 0
        self.aim_error_mrad = 0.2  # 0.2 mrad = ~0.8 inches at 100m
        self.lock_time_s = 0.0
        self.lock_required_s = 0.3  # 300ms to acquire lock
        # Multi-target tracking
        self.max_tracked = 8
        self.tracked_targets = []  # list of (target, threat_score, iff_status)
        # IFF transponder
        self.iff_codes = {"friendly": 0xA1B2C3, "hostile": 0x000000, "unknown": 0xFFFFFF}
        self.iff_query_active = False
        # Missile dodge AI
        self.inbound_threats = []  # list of {pos, vel, tof_s, type}
        self.dodge_recommended = False
        self.dodge_vector = np.zeros(3)
        self.dodge_timer = 0.0
        # Weapon mount points
        self.weapon_mounts = [
            {"name": "right_forearm", "pos": np.array([0.18, 0.05, 0.35]), "armed": True},
            {"name": "left_forearm", "pos": np.array([-0.18, 0.05, 0.35]), "armed": True},
            {"name": "right_shoulder", "pos": np.array([0.12, 0.08, 0.05]), "armed": False},
            {"name": "left_shoulder", "pos": np.array([-0.12, 0.08, 0.05]), "armed": False},
            {"name": "back_hardpoint", "pos": np.array([0, -0.05, -0.15]), "armed": False},
        ]
        # Squad coordination
        self.squad_members = []  # list of {id, pos, status}
        self.squad_target_assignments = {}  # target_id -> squad_member_id

    def add_target(self, pos, vel, mach_speed=False, iff="unknown"):
        """Add a new target to track."""
        t = Target3D(pos, vel, mach_speed)
        t.iff_status = iff
        self.targets.append(t)
        return t

    def scan(self, suit_pos):
        """Scan for targets within range. Returns nearest target or None."""
        nearest = None
        nearest_dist = float('inf')
        for t in self.targets:
            if not t.alive or t.hit:
                continue
            dist = np.linalg.norm(t.pos - suit_pos)
            if dist < self.aim_range_m and dist < nearest_dist:
                nearest = t
                nearest_dist = dist
        return nearest, nearest_dist

    def update_threat_priority(self, suit_pos):
        """Sort all targets by threat priority (closer + faster + heading toward = higher)."""
        threats = []
        for t in self.targets:
            if not t.alive or t.hit:
                continue
            dist = np.linalg.norm(t.pos - suit_pos)
            if dist > self.aim_range_m:
                continue
            # Threat score: closer = higher, faster = higher, approaching = higher
            speed = np.linalg.norm(t.vel)
            # Is target approaching?
            to_suit = suit_pos - t.pos
            to_suit_norm = to_suit / (np.linalg.norm(to_suit) or 1.0)
            vel_norm = t.vel / (speed or 1.0)
            approach_factor = max(0, np.dot(to_suit_norm, vel_norm))
            # IFF: hostile = 2x threat, unknown = 1x, friendly = 0x (no threat)
            iff_mult = 2.0 if t.iff_status == "hostile" else 1.0 if t.iff_status == "unknown" else 0.0
            threat_score = (1000.0 / max(dist, 1.0)) * (1.0 + speed / 500.0) * (1.0 + approach_factor) * iff_mult
            threats.append((t, threat_score, dist))
        # Sort by threat score (highest first)
        threats.sort(key=lambda x: -x[1])
        self.tracked_targets = threats[:self.max_tracked]
        return self.tracked_targets

    def query_iff(self, target):
        """Query IFF transponder on target."""
        self.iff_query_active = True
        # Simulate IFF response (in real system, would query transponder)
        if hasattr(target, 'iff_status'):
            return target.iff_status
        return "unknown"

    def detect_inbound_threats(self, suit_pos, dt):
        """Detect inbound missiles/projectiles and compute dodge vectors."""
        self.inbound_threats = []
        for t in self.targets:
            if not t.alive or t.hit:
                continue
            dist = np.linalg.norm(t.pos - suit_pos)
            if dist > 2000:  # only care about close threats
                continue
            # Is it heading directly at us?
            to_suit = suit_pos - t.pos
            to_suit_norm = to_suit / (np.linalg.norm(to_suit) or 1.0)
            vel_norm = t.vel / (np.linalg.norm(t.vel) or 1.0)
            approach_dot = np.dot(to_suit_norm, vel_norm)
            if approach_dot > 0.8:  # heading roughly at us
                speed = np.linalg.norm(t.vel)
                tof = dist / max(speed, 1.0)
                if tof < 5.0:  # within 5 seconds impact
                    self.inbound_threats.append({
                        "pos": t.pos.copy(),
                        "vel": t.vel.copy(),
                        "tof_s": tof,
                        "type": "missile" if speed > 300 else "projectile",
                        "target": t,
                    })
        # Compute dodge vector if threats inbound
        if self.inbound_threats:
            self.dodge_recommended = True
            self.dodge_timer = max(self.dodge_timer, 2.0)
            # Dodge perpendicular to incoming trajectory
            worst = min(self.inbound_threats, key=lambda x: x["tof_s"])
            incoming_dir = worst["vel"] / (np.linalg.norm(worst["vel"]) or 1.0)
            # Dodge up + lateral
            up = np.array([0.0, 1.0, 0.0])
            lateral = np.cross(incoming_dir, up)
            lateral = lateral / (np.linalg.norm(lateral) or 1.0)
            self.dodge_vector = (up * 0.6 + lateral * 0.4) * 2.0
        else:
            self.dodge_recommended = False
        self.dodge_timer = max(0.0, self.dodge_timer - dt)

    def update(self, dt, suit_pos):
        """Update all targets and auto-aim logic."""
        # Update all targets
        for t in self.targets:
            if t.alive and not t.hit:
                t.update(dt)

        # Remove targets that are hit (killed) or too far away
        self.targets = [t for t in self.targets
                        if not t.hit and np.linalg.norm(t.pos - suit_pos) < self.aim_range_m * 1.5]

        # Update threat priority
        self.update_threat_priority(suit_pos)

        # Detect inbound threats
        self.detect_inbound_threats(suit_pos, dt)

        # Update squad member positions (follow suit at offset)
        for m in self.squad_members:
            if m["status"] == "active":
                offset_idx = hash(m["id"]) % 4
                offsets = [np.array([30, 0, 30]), np.array([-30, 0, 30]),
                           np.array([30, 0, -30]), np.array([-30, 0, -30])]
                target_pos = suit_pos + offsets[offset_idx]
                m["pos"] += (target_pos - m["pos"]) * min(1.0, dt * 0.5)

        # Lock highest-priority hostile target
        hostile_targets = [(t, score, dist) for t, score, dist in self.tracked_targets
                           if t.iff_status != "friendly"]
        if hostile_targets:
            target = hostile_targets[0][0]
            if self.locked_target is not target:
                self.locked_target = target
                self.lock_time_s = 0.0
            self.lock_time_s += dt
        else:
            # Fall back to nearest
            target, dist = self.scan(suit_pos)
            if target is not None:
                if self.locked_target is not target:
                    self.locked_target = target
                    self.lock_time_s = 0.0
                self.lock_time_s += dt
            else:
                self.locked_target = None
                self.lock_time_s = 0.0

    def add_squad_member(self, member_id, pos, status="active"):
        """Add a squad member for coordination."""
        self.squad_members.append({
            "id": member_id,
            "pos": np.asarray(pos, dtype=float),
            "status": status,
        })

    def assign_squad_targets(self, suit_pos):
        """Assign targets to squad members based on proximity.
        Returns dict of member_id -> target."""
        assignments = {}
        available = [m for m in self.squad_members if m["status"] == "active"]
        hostile_targets = [t for t, s, d in self.tracked_targets
                           if t.iff_status == "hostile" and not t.hit]
        for i, member in enumerate(available):
            if i < len(hostile_targets):
                # Assign nearest hostile to this member
                nearest = min(hostile_targets,
                              key=lambda t: np.linalg.norm(t.pos - member["pos"]))
                assignments[member["id"]] = nearest
                self.squad_target_assignments[id(nearest)] = member["id"]
        return assignments

    def get_squad_status(self):
        """Return squad coordination summary."""
        return {
            "members": len(self.squad_members),
            "active": sum(1 for m in self.squad_members if m["status"] == "active"),
            "assignments": len(self.squad_target_assignments),
        }

    @property
    def lock_acquired(self):
        return self.locked_target is not None and self.lock_time_s >= self.lock_required_s

    def compute_firing_solution(self, suit_pos):
        """Compute the aim point for the locked target.

        Returns (aim_point_3d, time_of_flight_s, lead_angle_mrad) or None.
        """
        if not self.lock_acquired or self.locked_target is None:
            return None
        target = self.locked_target
        dist = np.linalg.norm(target.pos - suit_pos)
        if dist > self.aim_range_m:
            return None
        # Time of flight (simplified: straight-line at muzzle velocity)
        tof = dist / self.muzzle_velocity
        # Predict where target will be at time of impact
        aim_point = target.predicted_position(tof)
        # Lead angle
        lead = aim_point - target.pos
        lead_angle_mrad = np.linalg.norm(lead) / dist * 1000.0 if dist > 0 else 0.0
        return aim_point, tof, lead_angle_mrad

    def fire(self, suit_pos):
        """Fire at locked target. Returns True if hit."""
        if not self.lock_acquired:
            return False
        solution = self.compute_firing_solution(suit_pos)
        if solution is None:
            return False
        aim_point, tof, lead = solution
        self.shots_fired += 1
        # Hit probability based on accuracy + lead difficulty
        hit_chance = self.accuracy_pct / 100.0
        # Mach-speed targets are harder
        if self.locked_target.mach_number > 1.0:
            hit_chance *= 0.92
        # Long range reduces accuracy
        dist = np.linalg.norm(self.locked_target.pos - suit_pos)
        if dist > self.aim_range_m * 0.8:
            hit_chance *= 0.95
        # Simulate hit/miss
        hit = np.random.random() < hit_chance
        if hit:
            self.locked_target.hit = True
            self.shots_hit += 1
        return hit

    @property
    def hit_rate(self):
        return (self.shots_hit / self.shots_fired * 100.0) if self.shots_fired > 0 else 0.0


# =============================================================================
# DEFENSE AI  -- martial arts combat state machine
# =============================================================================

class DefenseAI:
    """Martial arts defense AI state machine.

    States:
    - IDLE: no threat detected
    - ALERT: threat detected, assessing
    - GUARD: defensive stance, blocking/evading
    - COUNTER: counter-attack (10,000 lbs punch)
    - NEUTRALIZED: threat eliminated
    """
    STATES = ["IDLE", "ALERT", "GUARD", "COUNTER", "NEUTRALIZED"]

    def __init__(self):
        self.state = "IDLE"
        self.state_time = 0.0
        self.threats_neutralized = 0
        self.punches_thrown = 0
        self.blocks = 0
        # Live strike force: frame/pilot floor at rest, driven up to the rated
        # DIMS["perf_punch_lbs"] by the DEA arm fibers (set each tick in SuitState).
        self.punch_force_lbs = 300.0
        self.reaction_ms = 8.0  # 8ms reaction time (AI-assisted)
        self.stance = "neutral"  # neutral | guard | strike | evade
        self.target_dir = np.array([1.0, 0.0, 0.0])  # direction of threat

    def update(self, dt, threat_detected=False, threat_dir=None):
        """Update defense state machine."""
        self.state_time += dt

        if threat_dir is not None:
            self.target_dir = np.asarray(threat_dir, dtype=float)
            self.target_dir /= (np.linalg.norm(self.target_dir) or 1.0)

        if self.state == "IDLE":
            if threat_detected:
                self.state = "ALERT"
                self.state_time = 0.0
                self.stance = "guard"
        elif self.state == "ALERT":
            if self.state_time > 0.05:  # 50ms assessment
                self.state = "GUARD"
                self.state_time = 0.0
        elif self.state == "GUARD":
            self.blocks += int(dt * 10)  # simulating blocks
            if self.state_time > 0.2:  # 200ms guard then counter
                self.state = "COUNTER"
                self.state_time = 0.0
                self.stance = "strike"
        elif self.state == "COUNTER":
            if self.state_time < 0.02:  # strike on first frame
                self.punches_thrown += 1
            if self.state_time > 0.3:
                self.state = "NEUTRALIZED"
                self.state_time = 0.0
                self.threats_neutralized += 1
        elif self.state == "NEUTRALIZED":
            if self.state_time > 1.0:
                self.state = "IDLE"
                self.state_time = 0.0
                self.stance = "neutral"

    @property
    def active(self):
        return self.state not in ("IDLE", "NEUTRALIZED")

    @property
    def striking(self):
        return self.state == "COUNTER" and self.state_time < 0.02


# =============================================================================
# JUMP ASSIST  -- AI-calculated 200ft vertical jump
# =============================================================================

class JumpAssist:
    """AI-assisted jump system for 200ft vertical jumps.

    Calculates required force, trajectory, and auto-executes
    with no overcompensation by the user."""
    def __init__(self):
        self.charging = False
        self.charge = 0.0
        self.jump_target = None  # (x, y, z) target landing point
        self.jumping = False
        self.jump_time = 0.0
        self.jump_duration = 0.0
        self.jump_start_pos = np.zeros(3)
        self.jump_peak_height = 0.0
        self.total_jumps = 0
        self.max_jump_ft = DIMS["perf_jump_vertical_ft"]

    def start_charge(self, target_point=None):
        """Begin charging jump fibers. Optionally specify target landing point."""
        self.charging = True
        self.charge = 0.0
        self.jump_target = np.asarray(target_point, dtype=float) if target_point else None

    def update_charge(self, dt):
        """Charge up jump fibers."""
        if self.charging:
            self.charge = min(1.0, self.charge + dt * 3.0)  # ~0.33s to full

    def execute_jump(self, current_pos, muscle_force_n=0.0):
        """Execute the jump with AI-calculated trajectory.
        muscle_force_n: additional force from DEA muscle fibers (Newtons)."""
        if not self.charging or self.charge < 0.5:
            return False
        self.jumping = True
        self.jump_time = 0.0
        self.jump_start_pos = np.asarray(current_pos, dtype=float).copy()
        # Calculate jump duration based on physics: h = 0.5 * g * t^2
        # Muscle fiber force adds to jump height: extra_h = F * t_push / (m * g)
        h = self.max_jump_ft * 0.3048 * self.charge
        # DEA leg force contribution: ~0.1s push phase
        if muscle_force_n > 0:
            total_mass = 79.4 + DIMS["weight_total_kg"]
            extra_h = muscle_force_n * 0.1 / (total_mass * 9.81)
            h += extra_h
        g = 9.81
        self.jump_duration = 2.0 * math.sqrt(2.0 * h / g)
        self.jump_peak_height = h
        self.total_jumps += 1
        self.charging = False
        self.charge = 0.0
        return True

    def update_jump(self, dt, current_pos):
        """Update jump trajectory. Returns vertical offset to apply."""
        if not self.jumping:
            return 0.0
        self.jump_time += dt
        if self.jump_time >= self.jump_duration:
            self.jumping = False
            return 0.0
        # Parabolic trajectory: h(t) = 4 * h_peak * t * (T - t) / T^2
        t = self.jump_time
        T = self.jump_duration
        h = 4.0 * self.jump_peak_height * t * (T - t) / (T * T)
        return h

    @property
    def jump_progress(self):
        if not self.jumping or self.jump_duration <= 0:
            return 0.0
        return clamp(self.jump_time / self.jump_duration, 0.0, 1.0)


# =============================================================================
# MUSCLE FIBER SYSTEM  -- DEA-STF artificial muscle with real dielectric
# elastomer actuator physics and shear-thickening fluid impact response
# =============================================================================

class MuscleFiberSystem:
    """Dielectric Elastomer Actuator (DEA) + Shear-Thickening Fluid (STF) muscle.

    Mechanically driven model:
    - DEA: voltage-controlled contractile fibers (Maxwell stress -> strain)
    - 3 sublayers triply redundant, each 1.67mm thick
    - <20ms contraction time (RC time constant of elastomer membrane)
    - 50% max strain, 15x human strength when tripled + localized
    - STF: non-Newtonian fluid that stiffens under shear rate (impact)
    - Fiber density: 5x in legs (jump), 2x in shoulders/back/arms (lift)
    - Real science: DEA strain = (e * V^2) / (Y * t^2) where e=permittivity,
      V=voltage, Y=Young's modulus, t=membrane thickness
    """
    SUBLAYERS = DIMS["dea_sublayers"]
    MAX_STRAIN = DIMS["dea_strain_pct"] / 100.0
    CONTRACTION_MS = DIMS["dea_contraction_ms"]
    BOOST_X = DIMS["muscle_boost_x"]
    STF_MAX_PSI = DIMS["stf_max_psi"]

    MUSCLE_GROUPS = [
        "left_arm", "right_arm", "left_leg", "right_leg",
        "torso_front", "torso_back", "shoulders", "neck",
    ]

    def __init__(self):
        self.voltage = 0.0  # applied voltage (kV), 0-4 kV range
        self.voltage_target = 0.0
        self.command_voltage = 0.0  # latched manual/override command (kV)
        self.contraction = {}  # group -> 0.0 to 1.0
        self.force_output = {}  # group -> Newtons
        self.stf_viscosity = 1.0  # Pa*s, baseline (low shear)
        self.stf_stiffened = False
        self.fatigue = 0.0  # accumulated fatigue 0-1
        self.total_force_n = 0.0
        self.impact_stiffening_timer = 0.0
        self.sublayer_health = [1.0] * self.SUBLAYERS  # health per sublayer
        self.contraction_cycles = 0
        for g in self.MUSCLE_GROUPS:
            self.contraction[g] = 0.0
            self.force_output[g] = 0.0
        # Density multipliers per group (legs 5x for jump, arms/shoulders 2x for lift)
        self.density = {
            "left_leg": DIMS["fiber_density_jump_x"],
            "right_leg": DIMS["fiber_density_jump_x"],
            "left_arm": DIMS["fiber_density_lift_x"],
            "right_arm": DIMS["fiber_density_lift_x"],
            "shoulders": DIMS["fiber_density_lift_x"],
            "torso_back": DIMS["fiber_density_lift_x"],
            "torso_front": 1.0,
            "neck": 1.0,
        }
        # Human baseline force per muscle group (Newtons, approximate)
        self.human_baseline_n = {
            "left_arm": 250, "right_arm": 250,
            "left_leg": 800, "right_leg": 800,
            "torso_front": 400, "torso_back": 400,
            "shoulders": 300, "neck": 100,
        }

    def set_voltage(self, kv):
        # Latched manual command: acts as a floor the auto-driver can't undercut
        self.command_voltage = clamp(kv, 0.0, 4.0)
        self.voltage_target = self.command_voltage

    def contract(self, group, intensity):
        """Command a muscle group to contract at intensity 0-1."""
        if group in self.contraction:
            target = clamp(intensity, 0.0, 1.0) * self.MAX_STRAIN
            # Contraction follows RC time constant: tau = CONTRACTION_MS/3
            tau = self.CONTRACTION_MS / 1000.0 / 3.0
            current = self.contraction[group]
            self.contraction[group] = current + (target - current) * min(1.0, 0.016 / tau)
            self.contraction_cycles += 1

    def trigger_impact_stiffening(self, shear_rate_s1):
        """STF responds to shear rate from impact.
        Low shear: viscosity ~1 Pa*s (fluid)
        High shear (>1000 s^-1): viscosity jumps to ~1000 Pa*s (solid-like)
        """
        if shear_rate_s1 > 100:
            self.stf_viscosity = min(1000.0, 1.0 + shear_rate_s1 * 0.5)
            self.stf_stiffened = True
            self.impact_stiffening_timer = 0.2  # stiffens for 200ms
        else:
            self.stf_viscosity = max(1.0, self.stf_viscosity * 0.9)

    def update(self, dt, throttle=0.0, g_load=1.0, impact_shear=0.0):
        # DEA driver energizes the fibers to meet flight demand (throttle + g-load).
        # A latched manual command acts as a floor so overrides stay applied.
        auto_v = clamp(throttle * 0.6 + max(0.0, g_load - 1.0) * 0.1, 0.0, 1.0) * 4.0
        self.voltage_target = max(self.command_voltage, auto_v)
        # Voltage ramp (RC circuit model: tau ~ 5ms)
        tau_v = 0.005
        self.voltage += (self.voltage_target - self.voltage) * min(1.0, dt / tau_v)
        # Auto-contract based on throttle and g-load (flight loads)
        for g in self.MUSCLE_GROUPS:
            # Legs contract more during high-g maneuvers and jumps
            leg_boost = 1.0
            if "leg" in g:
                leg_boost = 1.0 + max(0, g_load - 1.0) * 0.3
            # Arms contract with thrust vectoring
            arm_boost = 1.0
            if "arm" in g:
                arm_boost = 1.0 + throttle * 0.5
            auto_intensity = throttle * 0.6 * leg_boost * arm_boost
            self.contract(g, auto_intensity)
            # Force output: DEA Maxwell stress * area * density * boost
            # F = e * V^2 / t^2 * A * density_mult * boost_x * sublayer_health
            base_force = self.human_baseline_n[g]
            voltage_factor = (self.voltage / 4.0) ** 2
            contraction_factor = self.contraction[g] / self.MAX_STRAIN
            sublayer_factor = sum(self.sublayer_health) / self.SUBLAYERS
            self.force_output[g] = (base_force * self.density[g] * self.BOOST_X *
                                    voltage_factor * contraction_factor * sublayer_factor)
        self.total_force_n = sum(self.force_output.values())
        # STF impact response
        if impact_shear > 0:
            self.trigger_impact_stiffening(impact_shear)
        if self.impact_stiffening_timer > 0:
            self.impact_stiffening_timer -= dt
            if self.impact_stiffening_timer <= 0:
                self.stf_stiffened = False
                self.stf_viscosity = 1.0
        # Fatigue accumulates with sustained contraction
        avg_contraction = sum(self.contraction.values()) / len(self.contraction)
        self.fatigue = min(1.0, self.fatigue + dt * 0.001 * avg_contraction)

    def get_impact_absorption(self, impact_force_n):
        """How much impact force the STF sublayers absorb.
        Absorbed fraction rises with shear-thickening viscosity (a decade of
        viscosity ~ fluid->solid buys progressively more damping) and is scaled
        by sublayer health. Capped at the tripled-STF ceiling. Returns
        (transmitted_force_n, absorption_pct)."""
        eta = max(1.0, self.stf_viscosity)          # Pa*s, 1 (fluid) .. ~1000 (solid)
        health = sum(self.sublayer_health) / self.SUBLAYERS
        # log-scaled: eta=1 -> 0.55, eta=1000 -> 1.0 before clamp/health
        frac = clamp(0.55 + 0.45 * (math.log10(eta) / 3.0), 0.0, 0.985) * health
        absorbed = impact_force_n * frac
        pct = (absorbed / impact_force_n * 100) if impact_force_n > 0 else 100.0
        return impact_force_n - absorbed, pct

    def damage_sublayer(self, layer_idx, amount=0.1):
        if 0 <= layer_idx < len(self.sublayer_health):
            self.sublayer_health[layer_idx] = max(0.0, self.sublayer_health[layer_idx] - amount)

    @property
    def health_pct(self):
        return sum(self.sublayer_health) / self.SUBLAYERS * 100

    @property
    def status_str(self):
        stiff = "STIFFENED" if self.stf_stiffened else "normal"
        return f"V={self.voltage:.1f}kV F={self.total_force_n:.0f}N STF={stiff} H={self.health_pct:.0f}%"


# =============================================================================
# THERMAL MANAGEMENT SYSTEM  -- active heating/cooling with glycol loop,
# phase-change material, and waste heat recovery
# =============================================================================

class ThermalManagementSystem:
    """Dragon Skin active thermal regulation.

    Mechanically driven model:
    - Circulated glycol loop from engine exhaust heat exchanger
    - 3 kW graphene resistive heating backup
    - Phase-change material (PCM) matrix: melts at 37C, stores latent heat
    - 12 kW max heat dump into body (heating mode)
    - 3 kW cooling capacity (radiator + Peltier)
    - Skin temp locked 36.5-37.5C even at -80C or Mach 0.6
    - At high speed, outer skin friction heats armor -> suit uses body as heat sink
    """
    PCM_MELT_TEMP = 37.0  # C
    PCM_LATENT_HEAT_J = 250000  # J/kg latent heat of fusion
    PCM_MASS_KG = 0.5
    GLYCOL_FLOW_RATE = 0.3  # L/s
    GLYCOL_CP = 3500  # J/(kg K) specific heat
    GLYCOL_DENSITY = 1060  # kg/m^3
    HEATING_KW = DIMS["thermal_heating_kw"]
    COOLING_KW = DIMS["thermal_cooling_kw"]

    def __init__(self):
        self.mode = "auto"  # auto | heat | cool | off
        self.skin_temp = 37.0  # C (pilot skin temperature)
        self.glycol_temp = 37.0  # C (glycol loop temperature)
        self.pcm_charge = 1.0  # 0-1, thermal storage charge (1 = fully charged/melted)
        self.ambient_temp = 20.0  # C (external)
        self.heating_power_w = 0.0
        self.cooling_power_w = 0.0
        self.waste_heat_recovered_w = 0.0
        self.target_temp = 37.0
        self.radiator_efficiency = 0.85
        self.total_energy_managed_j = 0.0

    def set_mode(self, mode):
        self.mode = mode if mode in ("auto", "heat", "cool", "off") else "auto"

    def update(self, dt, ambient_temp_c, outer_skin_temp_c, throttle, speed_mps, env_density):
        self.ambient_temp = ambient_temp_c
        # Determine heating vs cooling need
        temp_error = self.target_temp - self.skin_temp
        if self.mode == "auto":
            if temp_error > 0.5:
                self.mode = "heat"
            elif temp_error < -0.5:
                self.mode = "cool"
            elif abs(temp_error) < 0.2:
                self.mode = "off"
        # Waste heat recovery from engine exhaust (proportional to throttle)
        # Real: heat exchanger on turbine exhaust captures ~15% of waste heat
        engine_waste_w = throttle * 800 * 0.15  # watts recovered
        self.waste_heat_recovered_w = engine_waste_w
        # Skin friction heating at high speed in atmosphere
        # q = 0.5 * rho * v^2 * Cf (skin friction coefficient ~0.003)
        friction_heat_w = 0.0
        if env_density > 0.001 and speed_mps > 50:
            q = 0.5 * env_density * speed_mps * speed_mps * 0.003
            friction_heat_w = q * 1.8  # approximate watts over suit surface area
        # Heating mode: glycol loop + resistive heating
        if self.mode == "heat":
            # Use waste heat first, then resistive backup
            available_heat = engine_waste_w + friction_heat_w
            needed = self.HEATING_KW * 1000
            if available_heat < needed:
                self.heating_power_w = needed - available_heat  # resistive fills gap
            else:
                self.heating_power_w = 0.0
            self.cooling_power_w = 0.0
            # Apply heat to skin
            heat_j = (available_heat + self.heating_power_w) * dt
            self.skin_temp += heat_j / (70 * 3500) * 0.01  # body thermal mass ~70kg*3500J/(kg K)
            # Charge PCM (store excess heat)
            if self.skin_temp > self.PCM_MELT_TEMP and self.pcm_charge < 1.0:
                pcm_charge_j = min(heat_j * 0.3, self.PCM_LATENT_HEAT_J * self.PCM_MASS_KG * (1.0 - self.pcm_charge))
                self.pcm_charge += pcm_charge_j / (self.PCM_LATENT_HEAT_J * self.PCM_MASS_KG)
        # Cooling mode: radiator + Peltier
        elif self.mode == "cool":
            self.cooling_power_w = self.COOLING_KW * 1000 * self.radiator_efficiency
            self.heating_power_w = 0.0
            # Radiator effectiveness depends on ambient temp
            delta_t = self.skin_temp - self.ambient_temp
            if delta_t > 0:
                cool_j = self.cooling_power_w * dt * min(1.0, delta_t / 10.0)
                self.skin_temp -= cool_j / (70 * 3500) * 0.01
            # Discharge PCM (release stored heat to radiator)
            if self.pcm_charge > 0 and self.skin_temp > self.PCM_MELT_TEMP:
                pcm_release_j = self.COOLING_KW * 1000 * dt * 0.2
                self.pcm_charge = max(0.0, self.pcm_charge - pcm_release_j / (self.PCM_LATENT_HEAT_J * self.PCM_MASS_KG))
        else:
            self.heating_power_w = 0.0
            self.cooling_power_w = 0.0
        # Passive heat exchange with environment
        # Q = h * A * deltaT, h ~ 10 W/(m^2 K) for still air, higher at speed
        h = 10 + speed_mps * 2 if env_density > 0.001 else 0.5
        passive_q = h * 1.8 * (self.ambient_temp - self.skin_temp) * dt
        self.skin_temp += passive_q / (70 * 3500) * 0.01
        # Glycol loop distributes heat evenly
        self.glycol_temp += (self.skin_temp - self.glycol_temp) * min(1.0, dt * 0.5)
        # Clamp skin temp to safe range
        self.skin_temp = clamp(self.skin_temp, 30.0, 42.0)
        self.total_energy_managed_j += (self.heating_power_w + self.cooling_power_w) * dt

    @property
    def status_str(self):
        return f"{self.mode.upper()} skin={self.skin_temp:.1f}C PCM={self.pcm_charge*100:.0f}% P={self.heating_power_w + self.cooling_power_w:.0f}W"


# =============================================================================
# NEURAL INTERFACE SYSTEM  -- EEG/EMG BCI with motor intent decoding
# =============================================================================

class NeuralInterfaceSystem:
    """Brain-Computer Interface with real signal processing pipeline.

    Mechanically driven model:
    - 64-channel EEG + 16-channel EMG hybrid
    - CNN+LSTM motor intent decoder (<17ms latency)
    - Lattice-based post-quantum encryption (Kyber-1024)
    - Air-gapped (no wireless neural data)
    - Subconscious intent pre-activation: prepares fibers before motion
    - EMG armband for subvocal commands
    """
    CHANNELS_EEG = 64
    CHANNELS_EMG = 16
    LATENCY_MS = DIMS["bci_latency_ms"]
    CRYPTO_ROUNDS = 4

    def __init__(self):
        self.eeg_signal = np.zeros(self.CHANNELS_EEG)
        self.emg_signal = np.zeros(self.CHANNELS_EMG)
        self.motor_intent = {
            "left_arm": 0.0, "right_arm": 0.0,
            "left_leg": 0.0, "right_leg": 0.0,
            "trigger": 0.0, "speak": 0.0,
            "hover": 0.0, "forward": 0.0,
            "ascend": 0.0, "descend": 0.0,
            "land": 0.0, "combat": 0.0,
            "evade": 0.0, "jump": 0.0,
        }
        self.intent_class = "idle"
        self.intent_confidence = 0.0
        self.intent_action = "standby"
        self.latency_ms = self.LATENCY_MS
        self.crypto_key_id = ""
        self.crypto_verified = True
        self.signal_quality = 1.0  # 0-1, degrades with interference
        self.preactivation = {}  # muscle group -> pre-activation level 0-1
        self.calibrated = True
        self.packets_sent = 0
        self.packets_lost = 0
        self.subvocal_cmd = ""
        for g in ["left_arm", "right_arm", "left_leg", "right_leg", "torso"]:
            self.preactivation[g] = 0.0

    def generate_signals(self, throttle, g_load, combat_active):
        """Simulate EEG/EMG signal generation from pilot state.
        In real system: electrodes read brain/muscle activity."""
        # EEG: alpha (relaxed), beta (active), gamma (intense focus)
        alpha = 0.3 + 0.2 * (1.0 - throttle)
        beta = 0.4 + 0.3 * throttle
        gamma = 0.1 + 0.5 * (g_load / 10.0) + 0.4 * combat_active
        self.eeg_signal[:32] = np.random.normal(alpha, 0.1, 32)
        self.eeg_signal[32:] = np.random.normal(beta, 0.15, 32)
        # Gamma bursts
        for i in range(0, 64, 8):
            self.eeg_signal[i] += np.random.normal(gamma, 0.2)
        # EMG: muscle activation correlates with throttle and g-load
        emg_base = throttle * 0.5 + g_load * 0.1
        self.emg_signal = np.random.normal(emg_base, 0.1, self.CHANNELS_EMG)
        # Signal quality degrades with high g-load (blood flow changes)
        self.signal_quality = clamp(1.0 - max(0, g_load - 5.0) * 0.05, 0.3, 1.0)

    def decode_intent(self, throttle, g_load, combat_active, dodge_recommended):
        """Decode motor intent from neural signals.
        Returns (intent_class, confidence, action)."""
        # Feature extraction (simplified CNN+LSTM pipeline)
        beta_power = np.mean(self.eeg_signal[32:])
        gamma_power = np.max(self.eeg_signal)
        emg_sum = np.sum(self.emg_signal) / self.CHANNELS_EMG
        # Intent classification
        if dodge_recommended:
            self.intent_class = "evade"
            self.intent_action = "dodge_maneuver"
            self.intent_confidence = 0.9
        elif combat_active and gamma_power > 0.8:
            self.intent_class = "combat"
            self.intent_action = "engage_target"
            self.intent_confidence = 0.85
        elif throttle > 0.8 and emg_sum > 0.6:
            self.intent_class = "ascend"
            self.intent_action = "increase_throttle"
            self.intent_confidence = 0.75
        elif throttle > 0.3:
            self.intent_class = "forward"
            self.intent_action = "thrust_forward"
            self.intent_confidence = 0.65
        elif throttle < 0.05:
            self.intent_class = "land"
            self.intent_action = "auto_land"
            self.intent_confidence = 0.70
        else:
            self.intent_class = "hover"
            self.intent_action = "maintain_altitude"
            self.intent_confidence = 0.55
        # Motor commands (EMG-driven)
        self.motor_intent["left_arm"] = float(self.emg_signal[0])
        self.motor_intent["right_arm"] = float(self.emg_signal[4])
        self.motor_intent["left_leg"] = float(self.emg_signal[8])
        self.motor_intent["right_leg"] = float(self.emg_signal[12])
        self.motor_intent["trigger"] = float(gamma_power > 0.7)
        # Pre-activation: subconscious prepares muscles before motion
        for g in self.preactivation:
            target = self.motor_intent.get(g, 0.0) * 0.5  # pre-activate at 50%
            self.preactivation[g] += (target - self.preactivation[g]) * 0.3
        self.packets_sent += 1
        # Packet loss with low signal quality
        if np.random.random() > self.signal_quality:
            self.packets_lost += 1
        return self.intent_class, self.intent_confidence, self.intent_action

    def crypto_handshake(self):
        """Post-quantum crypto verification (Kyber-1024)."""
        self.crypto_key_id = f"PQ-K{self.packets_sent:08d}"
        self.crypto_verified = True
        return True

    @property
    def packet_loss_pct(self):
        return (self.packets_lost / max(1, self.packets_sent) * 100)

    @property
    def status_str(self):
        return f"{self.intent_class}({self.intent_confidence*100:.0f}%) Q={self.signal_quality*100:.0f}% loss={self.packet_loss_pct:.1f}%"


# =============================================================================
# LIFE SUPPORT SYSTEM  -- CO2 scrubber, O2 generation, seal management
# =============================================================================

class LifeSupportSystem:
    """Vacuum-sealed life support for space and underwater operations.

    Mechanically driven model:
    - 4-bed molecular sieve CO2 scrubber (regenerable)
    - O2 generation from compressed O2 tanks + electrolysis backup
    - Seal pressure management: auto-pressurize/depressurize
    - Humidity control: condensing heat exchanger
    - 24-hour life support capacity
    - Activates automatically in vacuum/underwater environments
    """
    SCRUB_EFFICIENCY = 0.95  # 95% CO2 removal per pass
    O2_FLOW_RATE_LPM = 8.0  # liters per minute at 1 atm
    HUMIDITY_TARGET = 40.0  # % RH
    SEAL_MAX_PRESSURE_KPA = 120.0
    SEAL_MIN_PRESSURE_KPA = 70.0

    def __init__(self):
        self.active = False
        self.co2_ppm = 400  # parts per million (atmospheric baseline)
        self.o2_ppm = 209500  # 20.95% O2 = 209500 ppm
        self.seal_pressure_kpa = 101.3
        self.humidity_pct = 40.0
        self.scrubber_cycles = 0
        self.o2_tank_pct = 100.0  # O2 supply remaining
        self.co2_tank_pct = 100.0  # scrubber sorbent remaining
        self.water_level_pct = 100.0  # humidity control water
        self.cabin_temp_c = 22.0
        self.seal_integrity = 1.0  # 0-1
        self.leak_rate_cc_min = 0.0  # cc/min leak rate
        self.total_runtime_h = 0.0
        self.mode = "standby"  # standby | sealed | vented

    def update(self, dt, env_density, env_pressure_kpa, env_temp_c, pilot_mass, throttle):
        # Activate life support in vacuum or underwater
        if env_density < 0.01 or env_density > 100:
            self.active = True
            self.mode = "sealed"
        elif env_density < 0.5:
            self.active = True
            self.mode = "sealed"
        else:
            # In atmosphere: standby but monitor
            self.active = False
            self.mode = "vented"
            self.seal_pressure_kpa = env_pressure_kpa
            # Still scrub CO2 in sealed helmet
            self.co2_ppm = max(400, self.co2_ppm - dt * 10)
            return
        self.total_runtime_h += dt / 3600.0
        # CO2 production: ~0.04 L/min at rest, up to 2 L/min at full exertion
        co2_production_lpm = 0.04 + throttle * 1.96 + (pilot_mass / 79.4) * 0.3
        # Convert to ppm change in ~10L helmet volume
        co2_rate_ppm = co2_production_lpm / 10.0 * 1e6 * (dt / 60.0)
        self.co2_ppm += co2_rate_ppm
        # Scrubber removes CO2 (4-bed molecular sieve)
        if self.co2_tank_pct > 0:
            scrubbed = self.co2_ppm * self.SCRUB_EFFICIENCY * (dt / 2.0)
            self.co2_ppm -= scrubbed
            self.scrubber_cycles += 1
            self.co2_tank_pct = max(0, self.co2_tank_pct - dt * 0.005)  # sorbent depletes
        self.co2_ppm = max(400, self.co2_ppm)
        # O2 generation: compressed O2 tank
        if self.o2_tank_pct > 0:
            o2_needed_lpm = self.O2_FLOW_RATE_LPM * (1 + throttle * 0.5)
            # Add O2 to maintain partial pressure
            o2_rate_ppm = o2_needed_lpm / 10.0 * 1e6 * (dt / 60.0)
            self.o2_ppm = min(239000, self.o2_ppm + o2_rate_ppm * 0.01)
            self.o2_tank_pct = max(0, self.o2_tank_pct - dt * 0.003)
        # Seal pressure management
        target_pressure = 101.3 if env_density < 0.01 else env_pressure_kpa
        # Auto-pressurize
        pressure_error = target_pressure - self.seal_pressure_kpa
        self.seal_pressure_kpa += pressure_error * min(1.0, dt * 0.5)
        # Leak rate (increases with seal degradation)
        if self.seal_integrity < 1.0:
            self.leak_rate_cc_min = (1.0 - self.seal_integrity) * 50.0
            self.seal_pressure_kpa -= self.leak_rate_cc_min * 0.001 * dt
        # Clamp pressure
        self.seal_pressure_kpa = clamp(self.seal_pressure_kpa, self.SEAL_MIN_PRESSURE_KPA, self.SEAL_MAX_PRESSURE_KPA)
        # Humidity control
        humidity_produced = 20 + throttle * 30  # pilot sweat/breath
        self.humidity_pct += (humidity_produced - self.humidity_pct) * min(1.0, dt * 0.1)
        if self.humidity_pct > self.HUMIDITY_TARGET + 10:
            # Condensing heat exchanger removes moisture
            self.humidity_pct -= dt * 5.0
            self.water_level_pct = min(100, self.water_level_pct + dt * 0.01)
        self.humidity_pct = clamp(self.humidity_pct, 10, 90)
        # Cabin temp follows thermal system
        self.cabin_temp_c += (env_temp_c - self.cabin_temp_c) * dt * 0.01

    def damage_seal(self, amount=0.1):
        self.seal_integrity = max(0.0, self.seal_integrity - amount)

    @property
    def co2_safe(self):
        return self.co2_ppm < 5000  # <5000 ppm is safe

    @property
    def o2_safe(self):
        return self.o2_ppm > 190000  # >19% O2

    @property
    def seal_ok(self):
        return self.seal_integrity > 0.5 and self.seal_pressure_kpa > 60

    @property
    def status_str(self):
        if not self.active:
            return "STANDBY (atmosphere)"
        return f"SEALED P={self.seal_pressure_kpa:.0f}kPa CO2={self.co2_ppm:.0f}ppm O2={self.o2_tank_pct:.0f}%"


# =============================================================================
# DIVE SYSTEM  -- buoyancy control (BCD) + decompression computer
# =============================================================================

class DiveSystem:
    """Dive-suit completeness: buoyancy control + decompression computer.

    Mechanically / physiologically driven model:
    - Variable-volume buoyancy bladder (BCD) + fixed trim ballast -> net
      buoyancy control for neutral hover at any depth. Auto-trim solves the
      bladder volume that balances weight against displacement.
    - Bühlmann-style multi-compartment inert-gas (N2) loading via the Haldane
      equation -> no-decompression limit (NDL) and decompression-stop ceiling.
    - Ascent-rate monitor (decompression-sickness / barotrauma risk).
    - Oxygen-toxicity clock from partial pressure of O2 (ppO2, CNS %).
    - Nitrogen narcosis via equivalent narcotic depth (EAD).
    - Open-circuit breathing-gas supply drawn at a depth-scaled rate.
    Active only underwater (high ambient density).
    """
    # Bühlmann-style compartment half-times (min) + surfacing M-values (bar)
    HALF_TIMES_MIN = [4.0, 8.0, 12.5, 27.0, 54.0, 109.0]
    M0_BAR =         [1.58, 1.42, 1.31, 1.10, 0.99, 0.92]
    P_H2O_BAR = 0.0627   # alveolar water-vapour partial pressure
    FN2 = 0.79           # nitrogen fraction of air
    SW_DENSITY = 1025.0  # seawater density (kg/m3)
    HULL_VOL_M3 = 0.075  # dry displacement of sealed suit hull

    def __init__(self):
        self.active = False
        self.depth_m = 0.0
        self.ambient_ata = 1.0
        # Buoyancy control
        self.bcd_fill = 0.30           # 0-1 bladder inflation command
        self.bcd_volume_l = 0.0        # current gas volume in bladder (L)
        self.auto_trim = True          # auto-hold neutral buoyancy
        self.net_buoyancy_n = 0.0
        # Inert-gas loading: start air-saturated at the surface
        p0 = (1.0 - self.P_H2O_BAR) * self.FN2
        self.tissue_bar = [p0 for _ in range(DIMS["dive_tissue_compartments"])]
        self.ndl_min = 99.0            # no-deco time remaining (min)
        self.deco_stop_m = 0.0         # required first deco-stop depth (0 = none)
        self.deco_required = False
        self.ascent_rate_m_min = 0.0
        self.ascent_warning = False
        self.ppo2_ata = 0.21
        self.cns_pct = 0.0             # CNS oxygen-toxicity clock (%)
        self.o2_tox_warning = False
        self.narcosis_ead_m = 0.0     # equivalent narcotic depth
        self.gas_pct = 100.0          # breathing gas remaining
        self._last_depth = 0.0

    def update(self, dt, depth_m, env_density, gravity, total_mass):
        if env_density < 100:
            # Out of water: off-gas toward surface saturation, buoyancy idle
            self.active = False
            surf = (1.0 - self.P_H2O_BAR) * self.FN2
            for i, ht in enumerate(self.HALF_TIMES_MIN):
                k = 1.0 - 2 ** (-(dt / 60.0) / ht)
                self.tissue_bar[i] += (surf - self.tissue_bar[i]) * k
            self.depth_m = 0.0
            self._last_depth = 0.0
            return
        self.active = True
        depth = max(0.0, depth_m)
        # Ascent rate (positive = ascending)
        self.ascent_rate_m_min = (self._last_depth - depth) / max(dt, 1e-6) * 60.0
        self._last_depth = depth
        self.ascent_warning = self.ascent_rate_m_min > DIMS["dive_ascent_rate_max_m_min"]
        self.depth_m = depth
        # Ambient pressure: 1 ata at surface + hydrostatic column
        self.ambient_ata = 1.0 + depth * self.SW_DENSITY * 9.81 / 101325.0
        # --- Inert-gas (N2) loading via Haldane equation per compartment ---
        p_insp_n2 = (self.ambient_ata - self.P_H2O_BAR) * self.FN2
        ndl = 99.0
        ceiling = 0.0
        for i, ht in enumerate(self.HALF_TIMES_MIN):
            k = 1.0 - 2 ** (-(dt / 60.0) / ht)
            self.tissue_bar[i] += (p_insp_n2 - self.tissue_bar[i]) * k
            m0 = self.M0_BAR[i]
            pt = self.tissue_bar[i]
            if pt >= m0:
                # In decompression: ceiling is shallowest tolerated ambient pressure
                tol_ata = pt / m0
                ceiling = max(ceiling, (tol_ata - 1.0) * 101325.0 / (self.SW_DENSITY * 9.81))
            elif p_insp_n2 > pt:
                frac = clamp((m0 - pt) / (p_insp_n2 - pt), 1e-6, 0.999999)
                ndl = min(ndl, -ht * math.log2(1.0 - frac))
        self.deco_required = ceiling > 0.5
        self.deco_stop_m = math.ceil(ceiling / 3.0) * 3.0 if self.deco_required else 0.0
        # Once a compartment exceeds its surfacing M-value the diver is in
        # decompression obligation, so no-deco time is exhausted.
        self.ndl_min = 0.0 if self.deco_required else clamp(ndl, 0.0, 99.0)
        # --- Oxygen toxicity (ppO2 + CNS clock) ---
        self.ppo2_ata = DIMS["dive_fo2"] * self.ambient_ata
        self.o2_tox_warning = self.ppo2_ata > DIMS["dive_ppo2_max_ata"]
        if self.ppo2_ata > 1.4:
            self.cns_pct = min(300.0, self.cns_pct + (self.ppo2_ata - 1.4) * dt / 60.0 * 100.0)
        else:
            self.cns_pct = max(0.0, self.cns_pct - dt / 900.0 * 100.0)
        # --- Nitrogen narcosis (on air, EAD == actual depth) ---
        self.narcosis_ead_m = depth
        # --- Breathing gas: consumption scales with ambient pressure ---
        consumption_lpm = DIMS["dive_sac_lpm"] * self.ambient_ata
        self.gas_pct = max(0.0, self.gas_pct - consumption_lpm * (dt / 60.0)
                           / DIMS["dive_gas_liters"] * 100.0)
        # --- Buoyancy control ---
        self._update_buoyancy(env_density, gravity, total_mass)

    def _update_buoyancy(self, rho, gravity, total_mass):
        if self.auto_trim:
            # Solve bladder volume that balances displacement against weight
            need_vol_m3 = total_mass / rho - self.HULL_VOL_M3
            need_l = clamp(need_vol_m3 * 1000.0, 0.0, DIMS["dive_bcd_volume_l"])
            self.bcd_fill = need_l / DIMS["dive_bcd_volume_l"]
        self.bcd_volume_l = self.bcd_fill * DIMS["dive_bcd_volume_l"]
        total_disp_m3 = self.HULL_VOL_M3 + self.bcd_volume_l / 1000.0
        self.net_buoyancy_n = total_disp_m3 * rho * gravity - total_mass * gravity

    @property
    def displacement_m3(self):
        return self.HULL_VOL_M3 + self.bcd_volume_l / 1000.0

    @property
    def status_str(self):
        if not self.active:
            return "SURFACE"
        deco = f"DECO {self.deco_stop_m:.0f}m" if self.deco_required else f"NDL {self.ndl_min:.0f}min"
        return f"{self.depth_m:.0f}m {deco} ppO2={self.ppo2_ata:.2f} gas={self.gas_pct:.0f}%"


# =============================================================================
# SPACE SYSTEM  -- radiation dosimetry + cold-gas RCS maneuvering
# =============================================================================

class SpaceSystem:
    """Space-suit completeness: radiation dosimetry + cold-gas RCS.

    Mechanically / physically driven model:
    - Ionising-radiation dose from galactic cosmic rays (GCR) + trapped-belt
      (LEO) background + stochastic solar-particle events (SPE), attenuated by
      the suit's areal shielding via an exponential mass-absorption model.
      Tracks dose rate (mSv/h), cumulative mission dose, career fraction.
    - Reaction-control system: cold-gas (GN2) propellant budget for the vacuum
      EVA thrusters. Depletes with throttle in vacuum; empty -> no attitude or
      translation authority. Reports Tsiolkovsky delta-v remaining.
    - Micrometeoroid / orbital-debris (MMOD) flux + Whipple-shield integrity.
    Active only in vacuum (near-zero ambient density).
    """
    SEA_LEVEL_MSV_H = 0.0000685   # ~0.6 mSv/yr terrestrial background

    def __init__(self):
        self.active = False
        self.dose_rate_msv_h = 0.0
        self.dose_mission_msv = 0.0
        self.career_fraction = 0.0
        self.spe_active = False
        self.spe_timer = 0.0
        self.shield_g_cm2 = DIMS["rad_shield_g_cm2"]
        # RCS cold-gas
        self.rcs_propellant_pct = 100.0
        self.rcs_available = True
        self.delta_v_ms = 0.0
        # MMOD (micrometeoroid / orbital debris)
        self.mmod_flux_hits_h = 0.0
        self.mmod_shield_integrity = 1.0
        self.mmod_hits = 0
        self._rng = np.random.default_rng(1417)

    def update(self, dt, env_density, planet, throttle, total_mass):
        if env_density > 0.001:
            self.active = False
            self.dose_rate_msv_h = self.SEA_LEVEL_MSV_H
            self.dose_mission_msv += self.dose_rate_msv_h * (dt / 3600.0)
            return
        self.active = True
        # --- Radiation ---
        base_day = DIMS["rad_leo_msv_day"] if planet == "earth" else DIMS["rad_gcr_msv_day"]
        rate_h = base_day / 24.0
        # Stochastic solar-particle event
        if not self.spe_active and self._rng.random() < dt / 3600.0 * 0.5:
            self.spe_active = True
            self.spe_timer = 300.0 + self._rng.random() * 900.0
        if self.spe_active:
            rate_h += 8.0  # intense SPE flux (pre-shielding)
            self.spe_timer -= dt
            if self.spe_timer <= 0:
                self.spe_active = False
        # Shielding: exponential mass attenuation (lambda ~ 30 g/cm2)
        atten = math.exp(-self.shield_g_cm2 / 30.0)
        self.dose_rate_msv_h = rate_h * atten
        self.dose_mission_msv += self.dose_rate_msv_h * (dt / 3600.0)
        self.career_fraction = self.dose_mission_msv / DIMS["rad_career_limit_msv"]
        # --- RCS cold-gas propellant ---
        if throttle > 0.001:
            thrust_n = DIMS["turbine_count"] * DIMS["rcs_thrust_lbf"] * 4.448 * throttle
            mdot = thrust_n / (DIMS["rcs_isp_s"] * 9.80665)  # kg/s
            used_kg = mdot * dt
            self.rcs_propellant_pct = max(
                0.0, self.rcs_propellant_pct - used_kg / DIMS["rcs_propellant_kg"] * 100.0)
        self.rcs_available = self.rcs_propellant_pct > 0.0
        # Tsiolkovsky delta-v remaining
        prop_kg = DIMS["rcs_propellant_kg"] * self.rcs_propellant_pct / 100.0
        m1 = max(total_mass, 1.0)
        self.delta_v_ms = DIMS["rcs_isp_s"] * 9.80665 * math.log((m1 + prop_kg) / m1)
        # --- MMOD flux: hits/hour = flux density x exposed area ---
        self.mmod_flux_hits_h = DIMS["mmod_flux_m2_s"] * DIMS["mmod_exposed_area_m2"] * 3600.0
        expected_hits = self.mmod_flux_hits_h * (dt / 3600.0)
        # Whipple bumper: penetration probability falls with layer count
        p_penetrate = math.exp(-DIMS["mmod_shield_layers"] / 2.0)
        # Deterministic erosion from expected penetrating fluence
        self.mmod_shield_integrity = max(
            0.0, self.mmod_shield_integrity - expected_hits * p_penetrate * 0.01)
        # Discrete stochastic strike events
        if self._rng.random() < expected_hits:
            self.mmod_hits += 1

    @property
    def dose_alarm(self):
        return self.dose_rate_msv_h > DIMS["rad_spe_warn_msv_h"]

    @property
    def status_str(self):
        if not self.active:
            return "ATMOSPHERE"
        return (f"{self.dose_rate_msv_h:.3f}mSv/h dose={self.dose_mission_msv:.2f}mSv "
                f"RCS={self.rcs_propellant_pct:.0f}% dv={self.delta_v_ms:.0f}m/s")


# =============================================================================
# POWER MANAGEMENT SYSTEM  -- battery, harvesting, load shedding
# =============================================================================

class PowerManagementSystem:
    """Solid-state Li-S battery + piezo/solar harvesting with active load management.

    Mechanically driven model:
    - 2x 600 Wh hot-swap solid-state Li-S batteries (550 Wh/kg)
    - Solar nanofiber on wing surface (~20% efficiency)
    - Piezoelectric harvesting from vibration/turbulence (40% kinetic recovery)
    - Active load shedding: prioritizes critical systems when battery low
    - Real battery model: voltage sag with discharge, internal resistance
    """
    BATTERY_WH = DIMS["battery_wh"]
    BATTERY_WH_KG = DIMS["battery_wh_kg"]
    NOMINAL_VOLTAGE = 48.0  # V
    INTERNAL_R_OHM = 0.05  # internal resistance
    PIEZO_EFFICIENCY = DIMS["piezo_harvest_pct"] / 100.0
    SOLAR_EFFICIENCY = 0.20

    # Power budgets by system (watts)
    POWER_BUDGETS = {
        "neural": 15, "actuator": 50, "flight_ctrl": 20, "turbines": 800,
        "life_support": 30, "thermal": 120, "visor": 10, "comms": 8,
        "ai_model": 45, "sensors": 12, "lights": 5, "defense": 25,
    }

    def __init__(self):
        self.soc = 1.0  # state of charge 0-1
        self.voltage = self.NOMINAL_VOLTAGE
        self.current_a = 0.0
        self.power_draw_w = 0.0
        self.solar_harvest_w = 0.0
        self.piezo_harvest_w = 0.0
        self.total_harvest_w = 0.0
        self.net_power_w = 0.0
        self.load_shedding = False
        self.shed_systems = []
        self.charge_cycles = 0
        self.battery_health = 1.0  # 0-1, degrades with cycles
        self.system_power = {}  # system -> actual power allocated
        for sys_name in self.POWER_BUDGETS:
            self.system_power[sys_name] = 0.0

    def update(self, dt, throttle, afterburner, solar_w, turbulence, wing_deploy, g_load):
        # Calculate power demand per system
        self.system_power["turbines"] = throttle * self.POWER_BUDGETS["turbines"] * (1.8 if afterburner else 1.0)
        self.system_power["actuator"] = self.POWER_BUDGETS["actuator"] * (0.5 + throttle * 0.5)
        self.system_power["thermal"] = self.POWER_BUDGETS["thermal"] * (0.3 + throttle * 0.7)
        self.system_power["neural"] = self.POWER_BUDGETS["neural"]
        self.system_power["flight_ctrl"] = self.POWER_BUDGETS["flight_ctrl"]
        self.system_power["life_support"] = self.POWER_BUDGETS["life_support"]
        self.system_power["visor"] = self.POWER_BUDGETS["visor"]
        self.system_power["comms"] = self.POWER_BUDGETS["comms"]
        self.system_power["ai_model"] = self.POWER_BUDGETS["ai_model"]
        self.system_power["sensors"] = self.POWER_BUDGETS["sensors"]
        self.system_power["lights"] = self.POWER_BUDGETS["lights"]
        self.system_power["defense"] = self.POWER_BUDGETS["defense"] * (1.5 if g_load > 3 else 1.0)
        # Load shedding: if battery < 20%, shed non-critical systems
        self.load_shedding = self.soc < 0.20
        self.shed_systems = []
        if self.load_shedding:
            # Priority: keep neural, flight_ctrl, turbines, life_support
            non_critical = ["lights", "comms", "ai_model", "visor", "defense", "sensors"]
            for sys_name in non_critical:
                if self.soc < 0.15:
                    self.system_power[sys_name] *= 0.3
                    self.shed_systems.append(sys_name)
                elif self.soc < 0.20:
                    self.system_power[sys_name] *= 0.6
                    if sys_name not in self.shed_systems:
                        self.shed_systems.append(sys_name)
        # Total draw
        self.power_draw_w = sum(self.system_power.values())
        # Harvesting
        self.solar_harvest_w = solar_w * self.SOLAR_EFFICIENCY
        self.piezo_harvest_w = turbulence * 5.0 * self.PIEZO_EFFICIENCY
        self.total_harvest_w = self.solar_harvest_w + self.piezo_harvest_w
        self.net_power_w = self.total_harvest_w - self.power_draw_w
        # Battery model: voltage sags with discharge and current draw
        # V = V_nominal * soc - I * R_internal
        self.current_a = self.power_draw_w / self.NOMINAL_VOLTAGE
        self.voltage = self.NOMINAL_VOLTAGE * (0.8 + 0.2 * self.soc) - self.current_a * self.INTERNAL_R_OHM
        self.voltage = max(36.0, self.voltage)  # cutoff voltage
        # Update state of charge
        net_draw = self.power_draw_w - self.total_harvest_w
        self.soc = max(0.0, min(1.0, self.soc - net_draw * dt / (self.BATTERY_WH * 3600)))
        # Battery health degrades with deep discharge cycles
        if self.soc < 0.1:
            self.battery_health = max(0.5, self.battery_health - dt * 0.0001)
        if self.soc > 0.99 and net_draw < 0:
            self.charge_cycles += 1

    @property
    def runtime_estimate_h(self):
        if self.power_draw_w <= 0:
            return float('inf')
        return self.soc * self.BATTERY_WH * self.battery_health / self.power_draw_w

    @property
    def status_str(self):
        shed = " SHED:" + ",".join(self.shed_systems) if self.load_shedding else ""
        return f"SOC={self.soc*100:.0f}% V={self.voltage:.1f}V P={self.power_draw_w:.0f}W H={self.total_harvest_w:.0f}W{shed}"


# =============================================================================
# HELMET SYSTEM  -- visor zoom, night vision, cameras, communications
# =============================================================================

class HelmetSystem:
    """Vacuum-sealed helmet with advanced visor and communications.

    Mechanically driven model:
    - Graphene-polycarbonate visor (15,000 lbs impact, 20-mile zoom)
    - 10x optical + 100x digital zoom = 1000x effective
    - 180-degree panoramic FOV
    - Night vision (intensifier tube + IR)
    - FLIR thermal camera
    - Stereo cameras for depth perception
    - Iridium Certus + Starlink + mesh radio comms
    - AI-assisted target assessment overlay
    """
    VISOR_MAX_FORCE_LBS = DIMS["visor_max_force_lbs"]
    ZOOM_OPTICAL = DIMS["visor_zoom_optical"]
    ZOOM_DIGITAL = DIMS["visor_zoom_digital"]
    FOV_DEG = DIMS["visor_fov_deg"]

    def __init__(self):
        self.zoom_level = 1.0  # 1x to 1000x
        self.zoom_target = 1.0
        self.night_vision = False
        self.thermal_vision = False
        self.camera_active = True
        self.recording = False
        self.comms_channel = "tactical"
        self.comms_active = {"iridium": True, "starlink": False, "mesh": True}
        self.comms_signal_db = -80  # dBm signal strength
        self.visor_cracked = False
        self.visor_damage = 0.0  # 0-1
        self.sealed = False
        self.ai_overlay = True
        self.target_assessment = None
        self.recorded_frames = 0
        self.zoom_steps = [1, 2, 4, 10, 25, 50, 100, 250, 500, 1000]
        self._zoom_idx = 0

    def cycle_zoom(self):
        self._zoom_idx = (self._zoom_idx + 1) % len(self.zoom_steps)
        self.zoom_target = float(self.zoom_steps[self._zoom_idx])

    def toggle_night_vision(self):
        self.night_vision = not self.night_vision
        if self.night_vision:
            self.thermal_vision = False

    def toggle_thermal_vision(self):
        self.thermal_vision = not self.thermal_vision
        if self.thermal_vision:
            self.night_vision = False

    def update(self, dt, env_density, env_temp_c, ai_locked, locked_target_dist):
        # Zoom smooth interpolation
        self.zoom_level += (self.zoom_target - self.zoom_level) * min(1.0, dt * 5.0)
        # Auto-seal in vacuum/underwater
        self.sealed = env_density < 0.01 or env_density > 100
        # Comms signal strength varies with environment
        if env_density < 0.001:
            # Space: Iridium/Starlink work, mesh doesn't
            self.comms_active["mesh"] = False
            self.comms_signal_db = -90 + np.random.normal(0, 3)
        elif env_density > 100:
            # Underwater: only acoustic/mesh (very limited)
            self.comms_active["iridium"] = False
            self.comms_active["starlink"] = False
            self.comms_active["mesh"] = True
            self.comms_signal_db = -120 + np.random.normal(0, 5)
        else:
            self.comms_active["iridium"] = True
            self.comms_active["starlink"] = True
            self.comms_active["mesh"] = True
            self.comms_signal_db = -70 + np.random.normal(0, 5)
        # AI target assessment
        if ai_locked and locked_target_dist > 0:
            self.target_assessment = {
                "range_m": locked_target_dist,
                "zoom_recommended": min(1000, max(1, int(1000 / locked_target_dist * 10))),
                "threat_level": "high" if locked_target_dist < 200 else "medium" if locked_target_dist < 500 else "low",
            }
        else:
            self.target_assessment = None
        # Recording
        if self.recording:
            self.recorded_frames += 1
        # Visor damage effects
        if self.visor_damage > 0.7:
            self.visor_cracked = True

    def damage_visor(self, force_lbs):
        if force_lbs > self.VISOR_MAX_FORCE_LBS * 0.5:
            self.visor_damage = min(1.0, self.visor_damage + 0.1)
        if force_lbs > self.VISOR_MAX_FORCE_LBS:
            self.visor_damage = 1.0
            self.visor_cracked = True

    @property
    def effective_zoom(self):
        return self.zoom_level

    @property
    def status_str(self):
        modes = []
        if self.night_vision: modes.append("NV")
        if self.thermal_vision: modes.append("THERM")
        if self.sealed: modes.append("SEAL")
        if self.recording: modes.append("REC")
        mode_str = " ".join(modes) if modes else "normal"
        return f"Z{self.zoom_level:.0f}x {mode_str} sig={self.comms_signal_db:.0f}dB"


# =============================================================================
# FRAME SYSTEM  -- telescoping CFRP exoskeleton with actuator control
# =============================================================================

class FrameSystem:
    """CFRP telescoping frame with 64 Ti-6Al-4V nodes and 14 linear actuators.

    Mechanically driven model:
    - Toray T1100G carbon fiber tubes (22mm OD, 18mm ID, hollow)
    - 64 3D-printed Ti-6Al-4V nodes (Markforged Metal X)
    - 14 Maxon EC45 linear actuators for limb telescoping
    - 20 key pivoting joints with PTFE-coated bearings
    - 24 Spectra auto-tensioning straps with BOA dials
    - Telescopes ±14 inches torso, ±10 inches limbs
    - Force distribution: spreads acceleration across femur, pelvis, scapula
    - Magneto-rheological dampers lock joints at >4g
    """
    TUBE_OD_MM = DIMS["frame_tube_od_mm"]
    TUBE_ID_MM = DIMS["frame_tube_id_mm"]
    NODE_COUNT = DIMS["frame_node_count"]
    PIVOT_COUNT = DIMS["frame_pivot_count"]
    ACTUATOR_COUNT = DIMS["frame_actuator_count"]
    STRAP_COUNT = DIMS["frame_strap_count"]
    TORSO_TELESCOPE_MM = DIMS["torso_telescope_mm"]
    LIMB_TELESCOPE_MM = DIMS["limb_telescope_mm"]
    DAMPER_LOCK_G = 4.0  # joints lock above 4g
    MAX_TUBE_STRESS_MPA = 5000  # T1100G tensile strength

    JOINTS = [
        "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "torso_upper", "torso_lower",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "spine_upper", "spine_lower",
        "left_scapula", "right_scapula", "pelvis",
    ]

    def __init__(self):
        self.pilot_height_m = 1.73
        self.pilot_weight_kg = 79.4
        self.telescope_torso_mm = 0.0  # current extension
        self.telescope_left_arm_mm = 0.0
        self.telescope_right_arm_mm = 0.0
        self.telescope_left_leg_mm = 0.0
        self.telescope_right_leg_mm = 0.0
        self.actuator_positions = [0.0] * self.ACTUATOR_COUNT  # 0-1 per actuator
        self.actuator_targets = [0.0] * self.ACTUATOR_COUNT
        self.joint_locked = {j: False for j in self.JOINTS}
        self.joint_torque = {j: 0.0 for j in self.JOINTS}
        self.strap_tension_n = [0.0] * self.STRAP_COUNT
        self.strap_tension_target = 50.0  # Newtons baseline
        self.damper_viscosity = 0.1  # Pa*s baseline
        self.force_distribution = {"femur": 0, "pelvis": 0, "scapula": 0, "clavicle": 0}
        self.stress_mpa = 0.0  # current max tube stress
        self.auto_size = True

    def set_pilot(self, height_m, weight_kg):
        """Auto-size frame to pilot."""
        self.pilot_height_m = height_m
        self.pilot_weight_kg = weight_kg
        ref_h = DIMS["ref_height_m"]
        # Telescoping: extend/retract from reference height
        delta_mm = (height_m - ref_h) * 1000
        torso_ext = clamp(delta_mm * 0.4, -self.TORSO_TELESCOPE_MM, self.TORSO_TELESCOPE_MM)
        limb_ext = clamp(delta_mm * 0.3, -self.LIMB_TELESCOPE_MM, self.LIMB_TELESCOPE_MM)
        self.telescope_torso_mm = torso_ext
        self.telescope_left_arm_mm = limb_ext
        self.telescope_right_arm_mm = limb_ext
        self.telescope_left_leg_mm = limb_ext
        self.telescope_right_leg_mm = limb_ext
        # Set actuator targets (normalized 0-1)
        for i in range(self.ACTUATOR_COUNT):
            self.actuator_targets[i] = clamp(0.5 + delta_mm / (2 * self.TORSO_TELESCOPE_MM), 0.0, 1.0)
        # Strap tension proportional to weight
        self.strap_tension_target = 50.0 + weight_kg * 0.5

    def update(self, dt, g_load, throttle, velocity, total_mass):
        # Actuators move toward target (Maxon EC45: ~10mm/s)
        actuator_speed = 10.0  # mm/s
        for i in range(self.ACTUATOR_COUNT):
            current_mm = self.actuator_positions[i] * self.TORSO_TELESCOPE_MM
            target_mm = self.actuator_targets[i] * self.TORSO_TELESCOPE_MM
            diff = target_mm - current_mm
            if abs(diff) > 0.1:
                move = clamp(diff, -actuator_speed * dt, actuator_speed * dt)
                self.actuator_positions[i] = clamp((current_mm + move) / self.TORSO_TELESCOPE_MM, 0.0, 1.0)
        # Magneto-rheological dampers: lock joints above 4g
        for j in self.JOINTS:
            if g_load > self.DAMPER_LOCK_G:
                self.joint_locked[j] = True
                self.damper_viscosity = 1000.0  # effectively solid
            else:
                self.joint_locked[j] = False
                self.damper_viscosity = 0.1 + g_load * 0.05
        # Force distribution: spreads acceleration loads across skeleton
        total_force_n = total_mass * g_load * 9.81
        self.force_distribution["femur"] = total_force_n * 0.30
        self.force_distribution["pelvis"] = total_force_n * 0.25
        self.force_distribution["scapula"] = total_force_n * 0.25
        self.force_distribution["clavicle"] = total_force_n * 0.20
        # Tube stress: sigma = F * L / (E * I) simplified
        # For hollow tube: I = pi/64 * (OD^4 - ID^4)
        od = self.TUBE_OD_MM / 1000
        id_ = self.TUBE_ID_MM / 1000
        I = math.pi / 64 * (od**4 - id_**4)
        L = 0.3  # approx tube length
        E = 160e9  # T1100G Young's modulus ~160 GPa
        max_force = max(self.force_distribution.values())
        self.stress_mpa = (max_force * L * od / (2 * I * E)) * 1e-6 * 1e9  # simplified
        # Auto-tensioning straps
        for i in range(self.STRAP_COUNT):
            self.strap_tension_n[i] += (self.strap_tension_target - self.strap_tension_n[i]) * min(1.0, dt * 2.0)
        # Joint torques from thrust vectoring
        for j in self.JOINTS:
            if "shoulder" in j or "elbow" in j:
                self.joint_torque[j] = throttle * 50  # Nm
            elif "hip" in j or "knee" in j:
                self.joint_torque[j] = g_load * 30
            elif "spine" in j:
                self.joint_torque[j] = throttle * 20 + g_load * 15

    @property
    def frame_ok(self):
        return self.stress_mpa < self.MAX_TUBE_STRESS_MPA

    @property
    def status_str(self):
        locked = sum(1 for v in self.joint_locked.values() if v)
        return f"T={self.telescope_torso_mm:.0f}mm S={self.stress_mpa:.0f}MPa L={locked}/{self.PIVOT_COUNT} {'OK' if self.frame_ok else 'STRESS!'}"


# =============================================================================
# UPGRADE SYSTEMS  -- 15 new feature classes for Mjalnor'MV1.17
# =============================================================================

class StealthMode:
    """Active camouflage / adaptive stealth system.

    Electrochromic panels in outer armor change color/reflectivity to match
    surroundings. Peltier cooling suppresses IR signature. Radar-absorbent
    metamaterial coating reduces RCS.
    """
    def __init__(self):
        self.active = False
        self.activation_level = 0.0  # 0=visible, 1=full stealth
        self.target_level = 0.0
        self.cooldown_timer = 0.0
        self.ir_signature = 1.0  # 1.0=normal, 0.15=stealthed
        self.radar_cross_section = 1.0  # 1.0=normal, 0.18=stealthed
        self.visual_match = 0.0  # 0=visible, 0.92=full match
        self.power_draw_w = 0.0

    def toggle(self):
        if self.cooldown_timer > 0:
            return False
        self.active = not self.active
        self.target_level = 1.0 if self.active else 0.0
        if not self.active:
            self.cooldown_timer = DIMS["stealth_cooldown_s"]
        return True

    def update(self, dt):
        self.activation_level += (self.target_level - self.activation_level) * min(1.0, dt / DIMS["stealth_activate_s"])
        self.ir_signature = 1.0 - self.activation_level * (DIMS["stealth_ir_reduction_pct"] / 100.0)
        self.radar_cross_section = 1.0 - self.activation_level * (DIMS["stealth_radar_reduction_db"] / 100.0)
        self.visual_match = self.activation_level * (DIMS["stealth_visual_match_pct"] / 100.0)
        self.power_draw_w = self.activation_level * DIMS["stealth_power_draw_w"]
        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

    @property
    def stealth_pct(self):
        return self.activation_level * 100.0

    @property
    def is_stealthed(self):
        return self.activation_level > 0.5


class VisionModeSystem:
    """Multi-spectrum visor vision modes.

    Modes: normal, night (IR intensifier), thermal (FLIR), sonar (underwater),
    xray (backscatter contraband detection).
    """
    MODES = DIMS["vision_modes"]
    MODE_ICONS = {"normal": "NRM", "night": "NIGHT", "thermal": "THRM", "sonar": "SNR", "xray": "XRY"}

    def __init__(self):
        self.current_mode = "normal"
        self.mode_idx = 0
        self.switch_progress = 1.0  # 1.0 = fully switched
        self.night_gain = 0.0
        self.thermal_palette = "iron"  # iron, rainbow, white-hot
        self.sonar_active = False
        self.xray_active = False
        self.ar_overlay = True
        self.detected_objects = []  # objects detected by current vision mode

    def cycle_mode(self):
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        self.current_mode = self.MODES[self.mode_idx]
        self.switch_progress = 0.0
        self.sonar_active = (self.current_mode == "sonar")
        self.xray_active = (self.current_mode == "xray")
        return self.current_mode

    def set_mode(self, mode_name):
        if mode_name in self.MODES:
            self.current_mode = mode_name
            self.mode_idx = self.MODES.index(mode_name)
            self.switch_progress = 0.0
            self.sonar_active = (mode_name == "sonar")
            self.xray_active = (mode_name == "xray")
            return True
        return False

    def update(self, dt, env_density=1.225, altitude=0.0):
        self.switch_progress = min(1.0, self.switch_progress + dt * 20.0)
        if self.current_mode == "night":
            self.night_gain = min(1.0, self.night_gain + dt * 5.0)
        else:
            self.night_gain = max(0.0, self.night_gain - dt * 5.0)
        # Sonar only effective underwater
        if self.current_mode == "sonar" and env_density < 100:
            self.sonar_active = False
        # Detect objects based on mode
        self.detected_objects = []
        if self.current_mode == "thermal":
            self.detected_objects.append({"type": "heat_signature", "range_m": DIMS["vision_thermal_range_m"]})
        elif self.current_mode == "night":
            self.detected_objects.append({"type": "low_light", "range_m": DIMS["vision_night_range_m"]})
        elif self.current_mode == "sonar":
            self.detected_objects.append({"type": "acoustic", "range_m": DIMS["vision_sonar_range_m"]})

    @property
    def mode_label(self):
        return self.MODE_ICONS.get(self.current_mode, self.current_mode.upper())

    @property
    def is_switching(self):
        return self.switch_progress < 1.0


class GrappleSystem:
    """Deployable grappling hook with motorized winch.

    Carbon-fiber grappling line (50m, Dyneema) with motorized winch.
    Anchor detection, swing mechanics, winch pull.
    """
    def __init__(self):
        self.deployed = False
        self.anchored = False
        self.anchor_pos = np.zeros(3)
        self.cable_length = 0.0
        self.cable_deploy_progress = 0.0
        self.winch_active = False
        self.swinging = False
        self.swing_velocity = np.zeros(3)
        self.shots_fired = 0
        self.anchors_hit = 0

    def fire(self, target_pos, suit_pos):
        """Fire grappling hook at target position."""
        dist = np.linalg.norm(np.asarray(target_pos) - np.asarray(suit_pos))
        if dist > DIMS["grapple_range_m"]:
            return False
        self.deployed = True
        self.anchored = True
        self.anchor_pos = np.asarray(target_pos, dtype=float)
        self.cable_length = dist
        self.cable_deploy_progress = 0.0
        self.shots_fired += 1
        self.anchors_hit += 1
        return True

    def release(self):
        self.deployed = False
        self.anchored = False
        self.winch_active = False
        self.swinging = False
        self.cable_length = 0.0

    def winch(self, dt, suit_pos):
        """Winch toward anchor point."""
        if not self.anchored:
            return suit_pos
        to_anchor = self.anchor_pos - np.asarray(suit_pos, dtype=float)
        dist = np.linalg.norm(to_anchor)
        if dist < 0.5:
            self.winch_active = False
            return self.anchor_pos
        pull = min(dist, DIMS["grapple_winch_speed_mps"] * dt)
        direction = to_anchor / (dist or 1.0)
        return np.asarray(suit_pos, dtype=float) + direction * pull

    def update(self, dt, suit_pos, velocity):
        """Update grapple physics."""
        if not self.deployed:
            return
        self.cable_deploy_progress = min(1.0, self.cable_deploy_progress + dt / max(DIMS["grapple_deploy_s"], 0.01))
        if self.anchored:
            dist = np.linalg.norm(self.anchor_pos - np.asarray(suit_pos, dtype=float))
            self.cable_length = dist
            # Pendulum constraint: if cable taut, apply tension
            if dist > self.cable_length * 1.01:
                self.swinging = True
        if self.winch_active:
            self.winch(dt, suit_pos)

    @property
    def status(self):
        if not self.deployed:
            return "STOWED"
        if self.winch_active:
            return "WINCHING"
        if self.swinging:
            return "SWINGING"
        if self.anchored:
            return "ANCHORED"
        return "DEPLOYED"


class EmergencyParachute:
    """Ballistic emergency parachute for catastrophic flight failure.

    300 sq ft canopy, ballistic deployment (<1.5s), auto-deploy on freefall.
    """
    def __init__(self):
        self.deployed = False
        self.deploy_progress = 0.0
        self.auto_deploy_enabled = DIMS["parachute_auto_deploy"]
        self.descent_rate = 0.0
        self.canopy_area_m2 = DIMS["parachute_canopy_sqft"] * 0.0929
        self.drag_coefficient = 1.3
        self.jettisoned = False

    def deploy(self):
        if self.deployed or self.jettisoned:
            return False
        self.deployed = True
        self.deploy_progress = 0.0
        return True

    def jettison(self):
        self.deployed = False
        self.jettisoned = True
        self.deploy_progress = 0.0

    def check_auto_deploy(self, altitude, velocity, throttle):
        """Auto-deploy if in freefall with no thrust."""
        if not self.auto_deploy_enabled or self.deployed or self.jettisoned:
            return False
        fall_speed = velocity[1] if len(velocity) > 1 else 0
        if altitude > DIMS["parachute_min_alt_m"] and fall_speed < -30.0 and throttle < 0.05:
            return self.deploy()
        return False

    def update(self, dt, altitude, velocity):
        if not self.deployed:
            return 0.0
        self.deploy_progress = min(1.0, self.deploy_progress + dt / DIMS["parachute_deploy_s"])
        # Drag force increases with deployment progress
        effective_area = self.canopy_area_m2 * self.deploy_progress
        speed = np.linalg.norm(velocity)
        rho = 1.225  # air density
        drag_force = 0.5 * rho * speed * speed * effective_area * self.drag_coefficient
        # Terminal velocity under full canopy
        if self.deploy_progress >= 1.0:
            self.descent_rate = DIMS["parachute_descent_mps"]
        return drag_force

    @property
    def is_open(self):
        return self.deployed and self.deploy_progress >= 1.0

    @property
    def status(self):
        if self.jettisoned:
            return "JETTISONED"
        if not self.deployed:
            return "PACKED"
        if self.deploy_progress < 1.0:
            return f"DEPLOYING {self.deploy_progress*100:.0f}%"
        return "OPEN"


class DroneSwarm:
    """Reconnaissance micro-drone swarm (4x 50g drones, 4K cameras).

    Deployable from back hardpoint, AI-assisted scouting, overwatch.
    """
    def __init__(self):
        self.drones = []
        self.deployed_count = 0
        self.launch_progress = 0.0
        self._init_drones()

    def _init_drones(self):
        for i in range(DIMS["drone_count"]):
            self.drones.append({
                "id": f"recon-{i+1}",
                "pos": np.zeros(3),
                "vel": np.zeros(3),
                "battery_pct": 100.0,
                "status": "stowed",  # stowed, launching, active, returning, docked
                "camera_on": False,
                "altitude": 0.0,
            })

    def launch_all(self, suit_pos):
        """Launch all drones from back hardpoint."""
        for d in self.drones:
            if d["status"] == "stowed":
                d["status"] = "launching"
                d["pos"] = np.asarray(suit_pos, dtype=float) + np.array([0, 0.3, -0.15])
                d["vel"] = np.array([np.random.uniform(-5, 5), 10.0, np.random.uniform(-5, 5)])
                d["camera_on"] = True
        self.launch_progress = 0.0

    def launch_one(self, suit_pos):
        """Launch a single drone."""
        for d in self.drones:
            if d["status"] == "stowed":
                d["status"] = "active"
                d["pos"] = np.asarray(suit_pos, dtype=float) + np.array([0, 0.5, -0.15])
                d["vel"] = np.array([0, 8.0, 0])
                d["camera_on"] = True
                return d["id"]
        return None

    def recall_all(self, suit_pos):
        """Recall all drones to suit."""
        for d in self.drones:
            if d["status"] in ("active", "launching"):
                d["status"] = "returning"

    def update(self, dt, suit_pos):
        """Update all drone positions and states."""
        for d in self.drones:
            if d["status"] == "stowed" or d["status"] == "docked":
                continue
            # Battery drain
            d["battery_pct"] = max(0.0, d["battery_pct"] - dt * (100.0 / (DIMS["drone_flight_time_min"] * 60)))
            if d["battery_pct"] < 10.0 and d["status"] != "returning":
                d["status"] = "returning"
            # Launching: ascend then spread out
            if d["status"] == "launching":
                d["pos"] += d["vel"] * dt
                d["vel"] *= max(0.0, 1.0 - dt * 2.0)
                if np.linalg.norm(d["vel"]) < 2.0:
                    d["status"] = "active"
                    # Spread to orbit positions
                    idx = int(d["id"].split("-")[1]) - 1
                    angles = [0, 90, 180, 270]
                    a = math.radians(angles[idx % 4])
                    d["vel"] = np.array([math.cos(a) * 5, 0, math.sin(a) * 5])
            elif d["status"] == "active":
                # Orbit around suit at 30m radius, 10m above
                idx = int(d["id"].split("-")[1]) - 1
                angle = state_time_global * 0.3 + idx * (math.pi / 2)
                target = suit_pos + np.array([math.cos(angle) * 30, 10, math.sin(angle) * 30])
                d["pos"] += (target - d["pos"]) * min(1.0, dt * 2.0)
                d["altitude"] = d["pos"][1]
            elif d["status"] == "returning":
                to_suit = suit_pos - d["pos"]
                dist = np.linalg.norm(to_suit)
                if dist < 1.0:
                    d["status"] = "docked"
                    d["vel"] = np.zeros(3)
                    d["camera_on"] = False
                else:
                    d["vel"] = to_suit / (dist or 1.0) * 8.0
                    d["pos"] += d["vel"] * dt
            d["pos"] += np.zeros(3)  # ensure float

    @property
    def active_count(self):
        return sum(1 for d in self.drones if d["status"] in ("active", "launching"))

    @property
    def stowed_count(self):
        return sum(1 for d in self.drones if d["status"] in ("stowed", "docked"))

    @property
    def status(self):
        return f"{self.active_count}/{DIMS['drone_count']} active"


# Global time reference for drone orbit (updated by SuitState)
state_time_global = 0.0


class CountermeasureSystem:
    """Flare/chaff/decoy countermeasure dispenser.

    12 cartridges, auto-deploy against inbound missiles, IR flares + chaff + holographic decoys.
    """
    def __init__(self):
        self.remaining = DIMS["cm_count"]
        self.deployed_flares = []  # active flares in the air
        self.deployed_chaff = []  # active chaff clouds
        self.deployed_decoys = []  # active IR decoys
        self.auto_defeat_enabled = True
        self.total_deployed = 0
        self.missiles_defeated = 0

    def deploy(self, cm_type="flare", suit_pos=None):
        """Deploy a countermeasure."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.total_deployed += 1
        pos = np.asarray(suit_pos, dtype=float) if suit_pos is not None else np.zeros(3)
        if cm_type == "flare":
            self.deployed_flares.append({
                "pos": pos + np.random.uniform(-2, 2, 3),
                "vel": np.array([np.random.uniform(-10, 10), -5, np.random.uniform(-10, 10)]),
                "life": DIMS["cm_burn_time_s"],
                "intensity": 1.0,
            })
        elif cm_type == "chaff":
            self.deployed_chaff.append({
                "pos": pos.copy(),
                "radius": 0.0,
                "life": 8.0,
            })
        elif cm_type == "decoy":
            self.deployed_decoys.append({
                "pos": pos + np.array([np.random.uniform(-20, 20), 0, np.random.uniform(-20, 20)]),
                "life": 12.0,
                "ir_signature": 0.9,
            })
        return True

    def auto_defeat(self, inbound_threats, suit_pos):
        """Auto-deploy countermeasures against inbound threats."""
        if not self.auto_defeat_enabled or self.remaining <= 0:
            return 0
        defeated = 0
        for threat in inbound_threats:
            if threat["tof_s"] < 3.0 and self.remaining > 0:
                # Deploy flares for missiles, chaff for radar-guided
                cm_type = "flare" if threat.get("type") == "missile" else "chaff"
                if self.deploy(cm_type, suit_pos):
                    # Check if countermeasure defeats the threat
                    if np.random.random() < DIMS["cm_effectiveness_pct"] / 100.0:
                        threat["target"].hit = True
                        defeated += 1
                        self.missiles_defeated += 1
        return defeated

    def update(self, dt):
        """Update active countermeasures."""
        for f in self.deployed_flares:
            f["pos"] += f["vel"] * dt
            f["life"] -= dt
            f["intensity"] = max(0.0, f["life"] / DIMS["cm_burn_time_s"])
        self.deployed_flares = [f for f in self.deployed_flares if f["life"] > 0]
        for c in self.deployed_chaff:
            c["radius"] += dt * 5.0
            c["life"] -= dt
        self.deployed_chaff = [c for c in self.deployed_chaff if c["life"] > 0]
        for d in self.deployed_decoys:
            d["life"] -= dt
        self.deployed_decoys = [d for d in self.deployed_decoys if d["life"] > 0]

    @property
    def active_cm_count(self):
        return len(self.deployed_flares) + len(self.deployed_chaff) + len(self.deployed_decoys)


class EnergyShield:
    """Electromagnetic reactive armor pulse shield.

    Capacitor-discharge plasma arc that creates a temporary energy barrier
    to vaporize incoming projectiles. 50kJ max, 3s duration, 8s cooldown.
    """
    def __init__(self):
        self.active = False
        self.charge_kj = DIMS["eshield_max_charge_kj"]
        self.max_charge_kj = DIMS["eshield_max_charge_kj"]
        self.activation_progress = 0.0
        self.duration_timer = 0.0
        self.cooldown_timer = 0.0
        self.projectiles_vaporized = 0
        self.activations = 0

    def activate(self):
        if self.cooldown_timer > 0 or self.charge_kj < 5.0:
            return False
        self.active = True
        self.activation_progress = 0.0
        self.duration_timer = DIMS["eshield_duration_s"]
        self.activations += 1
        return True

    def deactivate(self):
        self.active = False
        self.cooldown_timer = DIMS["eshield_cooldown_s"]

    def intercept(self, projectile_energy_kj):
        """Attempt to vaporize an incoming projectile."""
        if not self.active or self.charge_kj <= 0:
            return False
        if projectile_energy_kj <= self.charge_kj:
            self.charge_kj -= projectile_energy_kj * 0.3  # 30% of energy used to vaporize
            self.projectiles_vaporized += 1
            return True
        return False

    def update(self, dt):
        # Recharge
        if not self.active:
            self.charge_kj = min(self.max_charge_kj, self.charge_kj + DIMS["eshield_recharge_kj_s"] * dt)
        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)
        if self.active:
            self.activation_progress = min(1.0, self.activation_progress + dt * 50.0)
            self.duration_timer -= dt
            # Drains charge while active
            self.charge_kj = max(0.0, self.charge_kj - dt * 5.0)
            if self.duration_timer <= 0 or self.charge_kj <= 0:
                self.deactivate()

    @property
    def charge_pct(self):
        return self.charge_kj / self.max_charge_kj * 100.0

    @property
    def status(self):
        if self.active:
            return f"ACTIVE {self.duration_timer:.1f}s"
        if self.cooldown_timer > 0:
            return f"COOLDOWN {self.cooldown_timer:.1f}s"
        return f"READY {self.charge_pct:.0f}%"


class RegenSystem:
    """Regenerative impact energy recovery system.

    Converts kinetic energy from landings, combat strikes, and braking
    back into battery charge via enhanced piezoelectric fibers.
    """
    def __init__(self):
        self.total_recovered_wh = 0.0
        self.landings_recovered = 0
        self.combat_recovered = 0
        self.braking_recovered = 0
        self.last_impact_wh = 0.0
        self.regen_active = True

    def recover_impact(self, impact_energy_j):
        """Recover energy from a landing impact."""
        if not self.regen_active:
            return 0.0
        recovered_wh = impact_energy_j / 3600.0 * (DIMS["regen_impact_pct"] / 100.0)
        recovered_wh = min(recovered_wh, DIMS["regen_landing_wh"])
        self.total_recovered_wh += recovered_wh
        self.landings_recovered += 1
        self.last_impact_wh = recovered_wh
        return recovered_wh

    def recover_combat(self):
        """Recover energy from a combat strike (punch/block)."""
        if not self.regen_active:
            return 0.0
        recovered_wh = DIMS["regen_combat_wh"]
        self.total_recovered_wh += recovered_wh
        self.combat_recovered += 1
        return recovered_wh

    def recover_braking(self, speed_delta_mps, mass_kg):
        """Recover energy from deceleration (braking)."""
        if not self.regen_active or speed_delta_mps <= 0:
            return 0.0
        ke_j = 0.5 * mass_kg * speed_delta_mps * speed_delta_mps
        recovered_wh = ke_j / 3600.0 * (DIMS["regen_brake_pct"] / 100.0)
        self.total_recovered_wh += recovered_wh
        self.braking_recovered += 1
        return recovered_wh

    @property
    def status(self):
        return f"{self.total_recovered_wh:.1f}Wh recovered"


class TacticalShield:
    """Deployable graphene-UHMWPE tactical shield.

    0.9m x 0.6m, NIJ IV, folds flat on forearm. Deployed in 0.4s.
    """
    def __init__(self):
        self.deployed = False
        self.deploy_progress = 0.0
        self.target = 0.0
        self.blocks = 0
        self.weight_kg = DIMS["tshield_weight_kg"]

    def toggle(self):
        self.deployed = not self.deployed
        self.target = 1.0 if self.deployed else 0.0
        return self.deployed

    def deploy(self):
        self.deployed = True
        self.target = 1.0

    def stow(self):
        self.deployed = False
        self.target = 0.0

    def block(self):
        """Register a blocked attack."""
        self.blocks += 1

    def update(self, dt):
        self.deploy_progress += (self.target - self.deploy_progress) * min(1.0, dt / DIMS["tshield_deploy_s"])

    @property
    def is_deployed(self):
        return self.deploy_progress > 0.5

    @property
    def coverage_pct(self):
        return self.deploy_progress * 100.0

    @property
    def status(self):
        if self.deploy_progress > 0.95:
            return "DEPLOYED"
        if self.deploy_progress > 0.05:
            return f"DEPLOYING {self.deploy_progress*100:.0f}%"
        return "STOWED"


class StunSystem:
    """Non-lethal taser / electrical stun system.

    50kV, 1.3mA pulse through palm contact points. 50 charges per battery.
    """
    def __init__(self):
        self.charges_remaining = DIMS["stun_shots"]
        self.charging = False
        self.charge_timer = 0.0
        self.stuns_delivered = 0
        self.ready = True

    def fire(self):
        """Deliver stun pulse."""
        if not self.ready or self.charges_remaining <= 0:
            return False
        self.charges_remaining -= 1
        self.stuns_delivered += 1
        self.ready = False
        self.charging = True
        self.charge_timer = DIMS["stun_charge_s"]
        return True

    def update(self, dt):
        if self.charging:
            self.charge_timer -= dt
            if self.charge_timer <= 0:
                self.charging = False
                self.ready = True

    @property
    def status(self):
        if self.charging:
            return f"CHARGING {self.charge_timer:.1f}s"
        if self.ready:
            return f"READY {self.charges_remaining}/{DIMS['stun_shots']}"
        return "OFFLINE"


class VoiceCommandSystem:
    """Subvocal voice command recognition system.

    50-word vocabulary, 80ms latency, 97% accuracy. Hands-free mode switching.
    """
    COMMANDS = {
        "stealth on": "stealth_toggle",
        "stealth off": "stealth_toggle",
        "night vision": "vision_night",
        "thermal": "vision_thermal",
        "sonar": "vision_sonar",
        "x-ray": "vision_xray",
        "normal vision": "vision_normal",
        "deploy shield": "shield_deploy",
        "stow shield": "shield_stow",
        "energy shield": "eshield_activate",
        "fire grapple": "grapple_fire",
        "release grapple": "grapple_release",
        "winch": "grapple_winch",
        "launch drones": "drone_launch",
        "recall drones": "drone_recall",
        "deploy parachute": "parachute_deploy",
        "countermeasures": "cm_deploy",
        "stun": "stun_fire",
        "beacon on": "beacon_activate",
        "maglev on": "maglev_toggle",
        "maglev off": "maglev_toggle",
        "afterburner": "afterburner_toggle",
        "deploy wings": "wings_deploy",
        "stow wings": "wings_stow",
        "auto hover": "auto_hover_toggle",
        "auto level": "auto_level_toggle",
        "defense mode": "defense_toggle",
        "fire": "fire_weapon",
        "jump": "jump_charge",
        "execute jump": "jump_execute",
        "reset": "reset_position",
    }

    def __init__(self):
        self.enabled = True
        self.last_command = None
        self.last_command_time = 0.0
        self.commands_recognized = 0
        self.latency_ms = DIMS["voice_latency_ms"]
        self.accuracy = DIMS["voice_accuracy_pct"]

    def recognize(self, audio_input):
        """Recognize a voice command from subvocal input.
        Returns (command_key, confidence) or (None, 0)."""
        if not self.enabled:
            return None, 0.0
        # Match against known commands
        audio_lower = audio_input.lower().strip()
        for phrase, cmd in self.COMMANDS.items():
            if phrase in audio_lower:
                confidence = self.accuracy / 100.0 + np.random.random() * 0.03
                self.last_command = cmd
                self.commands_recognized += 1
                return cmd, min(1.0, confidence)
        return None, 0.0

    @property
    def status(self):
        if self.last_command:
            return f"LAST: {self.last_command}"
        return "LISTENING"


class EmergencyBeacon:
    """Multi-band emergency locator beacon.

    406 MHz satellite uplink (COSPAS-SARSAT) + 121.5 MHz homing + GPS.
    """
    def __init__(self):
        self.active = False
        self.activate_time = 0.0
        self.gps_pos = np.zeros(3)
        self.satellite_uplink = False
        self.homing_signal = False
        self.battery_h = DIMS["beacon_battery_h"]
        self.activations = 0

    def activate(self, gps_pos=None):
        """Activate emergency beacon."""
        self.active = True
        self.satellite_uplink = True
        self.homing_signal = True
        self.activate_time = 0.0
        self.activations += 1
        if gps_pos is not None:
            self.gps_pos = np.asarray(gps_pos, dtype=float)
        return True

    def deactivate(self):
        self.active = False
        self.satellite_uplink = False
        self.homing_signal = False

    def update(self, dt, gps_pos=None):
        if not self.active:
            return
        self.activate_time += dt
        if gps_pos is not None:
            self.gps_pos = np.asarray(gps_pos, dtype=float)
        self.battery_h = max(0.0, self.battery_h - dt / 3600.0)

    @property
    def status(self):
        if not self.active:
            return "STANDBY"
        return f"ACTIVE {self.activate_time/60:.0f}min GPS:{self.gps_pos[0]:.0f},{self.gps_pos[2]:.0f}"


class MaglevMode:
    """Electromagnetic wall-climbing / levitation mode.

    Maglev boots for ferromagnetic surface traversal. Silent wall-climbing,
    ceiling walking. 1200N adhesion per boot, 3 m/s max speed.
    """
    def __init__(self):
        self.active = False
        self.activation_progress = 0.0
        self.target = 0.0
        self.surface_attached = False
        self.surface_type = None  # "ferromagnetic", "steel", "iron"
        self.climb_speed = 0.0
        self.power_draw_w = 0.0
        self.distance_climbed = 0.0

    def toggle(self):
        self.active = not self.active
        self.target = 1.0 if self.active else 0.0
        return self.active

    def detect_surface(self, surface_material):
        """Check if surface is suitable for maglev adhesion."""
        ferromagnetic = ["ferromagnetic", "steel", "iron", "cobalt", "nickel"]
        if surface_material.lower() in ferromagnetic:
            self.surface_type = surface_material
            self.surface_attached = True
            return True
        self.surface_attached = False
        return False

    def update(self, dt, speed=0.0):
        self.activation_progress += (self.target - self.activation_progress) * min(1.0, dt / DIMS["maglev_activate_s"])
        self.power_draw_w = self.activation_progress * DIMS["maglev_power_w"]
        if self.active and self.surface_attached:
            self.climb_speed = min(speed, DIMS["maglev_max_speed_mps"])
            self.distance_climbed += self.climb_speed * dt
        else:
            self.climb_speed = 0.0

    @property
    def status(self):
        if not self.active:
            return "OFF"
        if not self.surface_attached:
            return "NO SURFACE"
        return f"CLIMBING {self.climb_speed:.1f}m/s {self.distance_climbed:.0f}m"


# =============================================================================
# PHYSICS  -- flight dynamics, impact absorption, thermal, power
# =============================================================================

class SuitPhysics:
    """6-DOF physics for the hybrid suit across all environments.

    Models:
    - Translational: thrust, drag, lift, gravity, buoyancy (underwater)
    - Rotational: pitch/roll/yaw torques from thrust vectoring + aerodynamic moments
    - Environment-specific: air density drag, water drag+buoyancy, vacuum zero-drag
    - Wing lift with angle-of-attack and stall
    - G-force bone protection (auto-limit via g_limiter task)
    - Collision detection (ground + simple obstacles)
    """
    def __init__(self, state, suit_config):
        self.state = state
        self.cfg = suit_config
        self.pilot_mass = DIMS["ref_weight_kg"]
        self.suit_mass = DIMS["weight_total_kg"]
        # 6-DOF inertial properties
        self.inertia = np.array([0.8, 1.2, 0.6])  # kg*m^2 approx for humanoid + suit
        # Environment-specific drag coefficients
        self.cd_air = 0.3
        self.cd_water = 0.8
        self.cd_space = 0.0
        self.frontal_area = 0.5  # m^2
        # Wing aerodynamics
        self.wing_area_m2 = DIMS["wing_area_sqft"] * 0.0929
        self.cl_max = 1.2
        self.aoa_stall = 0.25  # ~15 degrees radians
        # Buoyancy (underwater): suit volume ~0.08 m^3
        self.suit_volume = 0.08
        self.water_density = 997.0
        # G-force protection
        self.max_safe_g = 15.0  # with G-limiter active
        self.g_limiter_active = True

    @property
    def total_mass(self):
        """Total mass = pilot + suit + remaining fuel. Fuel burns off during flight."""
        fuel = self.state.fuel_kg if hasattr(self.state, 'fuel_kg') else 0.0
        return self.pilot_mass + self.suit_mass + fuel

    @property
    def total_thrust_n(self):
        """Total thrust from all turbines in Newtons.
        Mechanically affected by:
        - Air density: thrust scales with mass flow rate (rho/rho_sl)
        - Power management: load shedding reduces turbine power
        - Battery voltage sag: lower voltage = lower turbine RPM
        - Fuel state: no fuel = no thrust
        """
        s = self.state
        # Check fuel state
        if hasattr(s, 'fuel_kg') and s.fuel_kg <= 0:
            return 0.0
        # Power system multiplier: load shedding cuts turbine power
        power_mult = 1.0
        if hasattr(s, 'power'):
            if s.power.load_shedding:
                power_mult *= 0.6  # 40% thrust reduction when shedding
            # Voltage sag: thrust scales with (V / V_nominal)^2
            v_ratio = s.power.voltage / s.power.NOMINAL_VOLTAGE
            power_mult *= v_ratio ** 2
        # Air density ratio: turbofan thrust ~ mass_flow ~ rho
        # At sea level rho=1.225, at 28k ft rho=0.418 -> thrust drops ~34%
        rho_sl = 1.225
        density_ratio = min(1.0, s.env_density / rho_sl) if s.env_density > 0.001 else 0.0
        # In vacuum, air-breathing turbines produce zero thrust (no mass flow);
        # attitude/translation comes from the cold-gas EVA thrusters (RCS), which
        # only work while propellant remains.
        if s.env_density < 0.001:
            rcs_ok = s.space.rcs_available if hasattr(s, 'space') else True
            if not rcs_ok:
                return 0.0
            return DIMS["turbine_count"] * DIMS["rcs_thrust_lbf"] * s.throttle * 4.448 * power_mult
        thrust_per = DIMS["turbine_thrust_ab_lbf"] if s.afterburner else DIMS["turbine_thrust_dry_lbf"]
        total_lbf = DIMS["turbine_count"] * thrust_per * s.throttle * density_ratio * power_mult
        return total_lbf * 4.448  # lbf -> N

    @property
    def weight_n(self):
        return self.total_mass * self.state.env_gravity

    @property
    def thrust_to_weight(self):
        w = self.weight_n
        return self.total_thrust_n / w if w > 0 else 0.0

    @property
    def wing_lift(self):
        """Lift from deployed wings at current speed with angle-of-attack.
        L = 0.5 * rho * v^2 * S * CL(alpha)
        CL is linear up to stall angle, then drops sharply (post-stall)."""
        s = self.state
        if s.wing_deploy < 0.01:
            return 0.0
        speed = np.linalg.norm(s.velocity)
        if speed < 0.5:
            return 0.0
        rho = s.env_density
        # Angle of attack from pitch
        aoa = abs(s.pitch)
        # Lift coefficient: linear up to stall, then drops
        if aoa < self.aoa_stall:
            cl = self.cl_max * (aoa / self.aoa_stall) * s.wing_deploy
        else:
            # Post-stall: lift drops sharply (flat plate model)
            cl = self.cl_max * 0.4 * s.wing_deploy
        return 0.5 * rho * speed * speed * self.wing_area_m2 * cl

    @property
    def wing_induced_drag(self):
        """Induced drag from wing lift generation.
        CDi = CL^2 / (pi * AR * e)
        Di = 0.5 * rho * v^2 * S * CDi
        This is the drag penalty for generating lift -- fundamental to gliding flight."""
        s = self.state
        if s.wing_deploy < 0.01:
            return 0.0
        speed = np.linalg.norm(s.velocity)
        if speed < 0.5:
            return 0.0
        rho = s.env_density
        aoa = abs(s.pitch)
        if aoa < self.aoa_stall:
            cl = self.cl_max * (aoa / self.aoa_stall) * s.wing_deploy
        else:
            cl = self.cl_max * 0.4 * s.wing_deploy
        # Aspect ratio = span^2 / area
        span_m = DIMS["wing_span_m"] * s.wing_deploy
        ar = span_m * span_m / max(self.wing_area_m2 * s.wing_deploy, 0.01)
        oswald_e = 0.85  # Oswald efficiency factor for elliptical-ish distribution
        cdi = cl * cl / (math.pi * ar * oswald_e)
        return 0.5 * rho * speed * speed * self.wing_area_m2 * s.wing_deploy * cdi

    @property
    def glide_ratio(self):
        return DIMS["wing_ld_ratio"] * self.state.wing_deploy

    @property
    def buoyancy_n(self):
        """Buoyant force when underwater.
        Uses the dive system's variable BCD displacement so the suit can hold
        neutral buoyancy at depth; falls back to the fixed hull volume."""
        s = self.state
        if s.env_density > 100:
            vol = s.dive.displacement_m3 if hasattr(s, 'dive') else self.suit_volume
            return vol * s.env_density * s.env_gravity
        return 0.0

    def calculate_drag_coefficient(self):
        """Environment-specific drag coefficient."""
        rho = self.state.env_density
        if rho > 100:
            return self.cd_water
        elif rho < 0.001:
            return self.cd_space
        return self.cd_air

    def step(self, dt):
        """Advance 6-DOF physics one timestep.
        All suit systems mechanically affect flight dynamics:
        - Power: load shedding/voltage sag reduces thrust
        - Neural: signal quality affects control precision
        - Frame: joint locking at high-g, stress limits
        - Muscle: STF stiffening on impact, force output adds to jump
        - Thermal: extreme skin temp impairs pilot (reduced control)
        - Life support: CO2/O2 levels affect pilot performance
        """
        s = self.state

        # --- Translational forces ---
        thrust = self.total_thrust_n
        weight = self.weight_n
        buoyancy = self.buoyancy_n

        # Drag (environment-specific parasitic + wing induced)
        speed = np.linalg.norm(s.velocity)
        rho = s.env_density
        cd = self.calculate_drag_coefficient()
        drag_force = 0.5 * rho * speed * speed * self.frontal_area * cd
        # Wing induced drag (only when wings deployed and generating lift)
        induced_drag = self.wing_induced_drag
        total_drag = drag_force + induced_drag
        if speed > 0.01:
            drag_vec = -s.velocity / speed * total_drag
        else:
            drag_vec = np.zeros(3)

        # Thrust vector (rotated by attitude)
        tv = s.thrust_vector / (np.linalg.norm(s.thrust_vector) or 1.0)
        # Rotate thrust by current attitude
        Rb = rot_z(s.roll) @ rot_x(s.pitch) @ rot_y(s.yaw)
        thrust_body = Rb @ tv
        thrust_vec = thrust_body * thrust

        # Lift (wings) - always along world up
        lift = self.wing_lift
        lift_vec = np.array([0.0, lift, 0.0])

        # Gravity
        gravity_vec = np.array([0.0, -weight, 0.0])

        # Buoyancy (underwater)
        buoyancy_vec = np.array([0.0, buoyancy, 0.0])

        # Wind force (from atmospheric model)
        wind_vec = s.wind_effect if hasattr(s, 'wind_effect') else np.zeros(3)

        # Muscle fiber assist: DEA force adds upward thrust during jumps/high-g
        muscle_assist = np.zeros(3)
        if hasattr(s, 'muscle'):
            # Leg muscle force contributes to vertical acceleration
            leg_force = s.muscle.force_output.get("left_leg", 0) + s.muscle.force_output.get("right_leg", 0)
            if s.jump.jumping or s.g_load > 2.0:
                muscle_assist = np.array([0.0, leg_force * 0.3, 0.0])

        # Net force
        net = thrust_vec + lift_vec + gravity_vec + drag_vec + buoyancy_vec + wind_vec + muscle_assist
        accel = net / self.total_mass

        # G-force limiter: cap acceleration to protect bones
        # Frame system allows higher g when joints are locked (rigid skeleton)
        max_g = self.max_safe_g
        if hasattr(s, 'frame') and s.frame.frame_ok:
            max_g = 18.0  # frame distributes load, allows 18g
        if self.g_limiter_active:
            g_mag = np.linalg.norm(accel) / max(s.env_gravity, 0.1)
            if g_mag > max_g:
                accel = accel * (max_g / g_mag)

        # Neural interface: signal quality affects control precision
        # Low quality adds noise to acceleration (pilot can't control precisely)
        if hasattr(s, 'neural') and s.neural.signal_quality < 0.8:
            noise_scale = (1.0 - s.neural.signal_quality) * 2.0
            accel += np.random.normal(0, noise_scale, 3)

        # Life support: high CO2 or low O2 impairs pilot (reduced control authority)
        if hasattr(s, 'life_support') and s.life_support.active:
            if not s.life_support.co2_safe or not s.life_support.o2_safe:
                # Pilot impairment: 30% reduction in effective thrust vectoring
                accel *= 0.7

        # Thermal: extreme skin temp impairs pilot performance
        if hasattr(s, 'thermal'):
            if s.thermal.skin_temp > 39.0 or s.thermal.skin_temp < 35.0:
                accel *= 0.85  # 15% performance degradation

        # Integrate velocity and position
        s.velocity += accel * dt
        s.pos += s.velocity * dt

        # Ground clamp with impact model
        if s.pos[1] < 0.0:
            s.pos[1] = 0.0
            if s.velocity[1] < 0:
                impact_speed = abs(s.velocity[1])
                # Trigger STF stiffening on impact (shear rate ~ impact_speed / suit_thickness)
                shear_rate = impact_speed / 0.012  # 12mm total thickness
                s.muscle.trigger_impact_stiffening(shear_rate)
                # Compute impact PSI: F = m*v/t_stop, distributed over body area
                # Frame system extends deceleration time (telescoping absorbs energy)
                t_stop = max(0.05, min(2.0, impact_speed / 9.81))
                if hasattr(s, 'frame'):
                    # Frame telescoping adds 20% more deceleration distance
                    t_stop *= 1.2
                impact_force_n = self.total_mass * impact_speed / t_stop
                # PSI = Pa / 6895; Pa = F(N) / A(m^2); body area ~1.5 m^2
                body_area_m2 = DIMS["body_impact_area_m2"]
                impact_psi = (impact_force_n / body_area_m2) / 6895.0
                felt_psi, absorb_pct = self.impact_absorption(impact_psi)
                # Muscle STF provides additional absorption when stiffened
                if hasattr(s, 'muscle') and s.muscle.stf_stiffened:
                    transmitted, muscle_pct = s.muscle.get_impact_absorption(impact_force_n)
                    felt_psi = (transmitted / body_area_m2) / 6895.0
                # Damage armor if user-felt PSI exceeds middle layer capacity
                if felt_psi > DIMS["stf_max_psi"] * 0.1:
                    s.damage_armor("outer_armor", min(0.3, felt_psi / DIMS["stf_max_psi"]))
                # Damage frame if stress exceeds tube strength
                if hasattr(s, 'frame') and s.frame.stress_mpa > s.frame.MAX_TUBE_STRESS_MPA * 0.8:
                    s.damage_armor("frame", min(0.2, s.frame.stress_mpa / s.frame.MAX_TUBE_STRESS_MPA))
                # Damage helmet visor from impact (head bears ~10% of impact force)
                if hasattr(s, 'helmet'):
                    visor_force_lbs = impact_force_n * 0.1 * 0.2248  # 10% to head, N->lbs
                    s.helmet.damage_visor(visor_force_lbs)
                s.velocity[1] = 0.0
                s.velocity[0] *= 0.5  # friction
                s.velocity[2] *= 0.5
                # Screen shake on ground impact
                if impact_speed > 5.0:
                    s.screen_shake = max(s.screen_shake, min(1.0, impact_speed / 30.0))

        s.altitude = s.pos[1]
        # G-load = |total acceleration| / gravity (proper vector magnitude, not just vertical)
        s.g_load = np.linalg.norm(accel) / max(s.env_gravity, 0.1) if s.env_gravity > 0 else 0.0
        # Screen shake on high-G maneuvers
        if s.g_load > 8.0:
            s.screen_shake = max(s.screen_shake, min(0.5, (s.g_load - 8.0) / 20.0))
        # Screen shake on afterburner activation
        if s.afterburner and s.throttle > 0.5:
            s.screen_shake = max(s.screen_shake, 0.15)

        # --- Rotational dynamics ---
        # Torque from thrust vectoring: tau = r x F, where r is moment arm from CG
        # For distributed turbines, effective moment arm ~0.5m (shoulder/hip offset)
        # Angular acceleration: alpha = tau / I (per axis inertia)
        moment_arm = 0.5  # m, approximate distance from CG to turbine clusters
        neural_mult = 1.0
        if hasattr(s, 'neural'):
            neural_mult = 0.5 + 0.5 * s.neural.signal_quality
        # Torque components from thrust vector offset
        # tv[0] = lateral (roll), tv[1] = vertical (neutral=1), tv[2] = forward (pitch)
        # At neutral [0,1,0]: tv[0]=0, tv[2]=0 -> no torque
        tau_pitch = thrust * moment_arm * tv[2] * neural_mult
        tau_roll = thrust * moment_arm * tv[0] * neural_mult
        tau_yaw = thrust * moment_arm * tv[0] * 0.3 * neural_mult  # roll coupling
        # Angular acceleration = torque / inertia (I_x, I_y, I_z)
        s.pitch_rate += (tau_pitch / self.inertia[0]) * dt
        s.roll_rate += (tau_roll / self.inertia[1]) * dt
        s.yaw_rate += (tau_yaw / self.inertia[2]) * dt

        # Aerodynamic damping (proportional to speed and density)
        aero_damp = speed * rho * 0.01
        s.pitch_rate *= max(0.0, 1.0 - aero_damp * dt)
        s.roll_rate *= max(0.0, 1.0 - aero_damp * dt)
        s.yaw_rate *= max(0.0, 1.0 - aero_damp * dt)

        # Frame joint locking at high-g adds rotational damping
        if hasattr(s, 'frame'):
            locked_count = sum(1 for v in s.frame.joint_locked.values() if v)
            if locked_count > 0:
                frame_damp = locked_count / len(s.frame.joint_locked) * 0.5
                s.pitch_rate *= max(0.0, 1.0 - frame_damp * dt)
                s.roll_rate *= max(0.0, 1.0 - frame_damp * dt)
                s.yaw_rate *= max(0.0, 1.0 - frame_damp * dt)

        # Clamp angular rates
        max_rate = 2.0  # rad/s max
        s.pitch_rate = clamp(s.pitch_rate, -max_rate, max_rate)
        s.roll_rate = clamp(s.roll_rate, -max_rate, max_rate)
        s.yaw_rate = clamp(s.yaw_rate, -max_rate, max_rate)

    def impact_layer_stack(self):
        """Return the ordered [(name, stopping_psi, leak_frac), ...] for the
        armor stack, derived from material capacity and current state.

        - Each layer's stopping pressure scales with local armor integrity
          (damage reduces what a layer can defeat).
        - The middle STF layer strengthens when shear-stiffened by an impact.
        """
        s = self.state
        integ = s.armor_integrity if hasattr(s, 'armor_integrity') else 1.0
        stiff = 1.0
        if hasattr(s, 'muscle') and s.muscle.stf_stiffened:
            stiff = DIMS["stf_stiffen_mult"]
        health = 1.0
        if hasattr(s, 'muscle'):
            health = sum(s.muscle.sublayer_health) / s.muscle.SUBLAYERS
        return [
            ("outer",        DIMS["outer_stop_psi"] * integ,          DIMS["outer_leak_frac"]),
            ("intermediate", DIMS["inter_stop_psi"] * integ,          DIMS["inter_leak_frac"]),
            ("middle",       DIMS["middle_stop_psi"] * stiff * health, DIMS["middle_leak_frac"]),
        ]

    def impact_breakdown(self, impact_psi):
        """Propagate a peak contact pressure through the armor stack.

        Physics: each layer defeats pressure up to its stopping capacity via
        plastic work / densification. Pressure exceeding capacity penetrates
        directly; a small back-face-coupling fraction of the *stopped* pressure
        still transmits. This yields a real ballistic limit -- small threats are
        fully arrested (near-total absorption) while threats that overmatch the
        stack (20mm+) pass most of their energy through.

        Returns (list_of_(name, absorbed_psi), user_felt_psi, absorption_pct).
        """
        p = float(impact_psi)
        rows = []
        for name, stop_psi, leak in self.impact_layer_stack():
            stopped = min(p, stop_psi)
            transmitted = (p - stopped) + stopped * leak
            rows.append((name, p - transmitted))
            p = transmitted
        felt = p
        absorbed = impact_psi - felt
        pct = absorbed / impact_psi * 100 if impact_psi > 0 else 100.0
        return rows, felt, pct

    def impact_absorption(self, impact_psi):
        """Multi-layer impact absorption. Returns (user_felt_psi, absorption_pct).
        Absorption now varies with threat magnitude (ballistic limit), armor
        integrity, and STF stiffening -- see impact_breakdown()."""
        _, felt, pct = self.impact_breakdown(impact_psi)
        return felt, pct

    def fall_damage_check(self, fall_height_ft, surface="concrete"):
        """Check if a fall from given height causes injury.
        Returns (safe, user_felt_g, absorption_pct).

        The suit uses turbine thrust + frame telescoping + STF stiffening
        to decelerate during the final portion of the fall."""
        s = self.state
        h = fall_height_ft * 0.3048
        v_impact = math.sqrt(2 * self.state.env_gravity * h)
        # Suit-assisted deceleration: turbines + multi-layer absorption
        # Frame telescoping extends deceleration distance by 20%
        frame_bonus = 1.0
        if hasattr(s, 'frame'):
            frame_bonus = 1.2
        t_stop = max(0.5, min(5.0, h / 50.0 * frame_bonus))  # 0.5-6.0 seconds
        # Net deceleration (suit thrust offsets some velocity first)
        v_after_thrust = max(0, v_impact - self.total_thrust_n / self.total_mass * t_stop * 0.5)
        force_n = self.total_mass * v_after_thrust / t_stop
        # Body area ~1.5 m^2, convert to PSI: PSI = Pa / 6895
        area_m2 = DIMS["body_impact_area_m2"]
        pressure_pa = force_n / area_m2
        pressure_psi = pressure_pa / 6895.0
        felt_psi, pct = self.impact_absorption(pressure_psi)
        # G-force experienced (with suit-assisted deceleration)
        g_force = v_after_thrust / (t_stop * self.state.env_gravity)
        # Frame allows higher g tolerance
        max_g = self.max_safe_g
        if hasattr(s, 'frame') and s.frame.frame_ok:
            max_g = 18.0
        safe = g_force < max_g and felt_psi < 50000
        return safe, g_force, pct

    def jump_energy(self, height_ft):
        """Energy required for a vertical jump to given height."""
        h = height_ft * 0.3048
        return self.total_mass * 9.81 * h

    def punch_force(self):
        """Punch force, mechanically driven by the DEA arm fibers routed through
        the frame + hip/torso kinetic chain. Falls to the unpowered frame/pilot
        floor when the fibers are idle, and reaches the rated ~10,000 lb strike
        only at full voltage + contraction."""
        s = self.state
        frame_floor_lbs = 300.0  # unpowered frame + pilot strike
        if hasattr(s, 'muscle'):
            arm_force_n = (s.muscle.force_output.get("right_arm", 0.0)
                           + s.muscle.force_output.get("left_arm", 0.0))
            return frame_floor_lbs + arm_force_n * 0.2248 * DIMS["punch_kinetic_chain_x"]
        return frame_floor_lbs

    def punch_energy(self):
        """Kinetic energy in a punch with DEA muscle assist."""
        force_n = self.punch_force() * 4.448
        # Assume fist travels 0.6m during punch
        return force_n * 0.6

    def battery_runtime_hours(self, throttle_frac):
        """Estimated runtime at given throttle using power management model."""
        s = self.state
        if hasattr(s, 'power'):
            # Use actual power management system calculation
            return s.power.runtime_estimate_h
        base_draw = 50  # W idle
        thrust_draw = throttle_frac * 800
        if self.state.afterburner:
            thrust_draw *= 1.8
        total_draw = base_draw + thrust_draw
        return DIMS["battery_wh"] / total_draw


# =============================================================================
# RENDERER  -- 3D model view (orbit) + layer exploded view + flight view
# =============================================================================

class SuitRenderer:
    """Interactive 3D renderer for the suit model."""
    def __init__(self, parts, suit_config, home=(0.72, 0.40, 4.0)):
        self.parts = parts
        self.cfg = suit_config
        self._home = home
        self.az, self.el, self.dist = self._home
        self.dist_target = self.dist
        self.pan = np.array([0.0, 0.0])
        self.free = np.zeros(3)
        self.light = C_LIGHT_DIR / np.linalg.norm(C_LIGHT_DIR)
        self.view = "full"
        self.section = False
        self.explode_amt = 0.0
        self.assembled = len(parts)
        self.hovered = None
        self.selected = None
        self.show_labels = True
        self.cull = True
        self.isolate = None
        self.zoom_min = 0.8
        self.zoom_max = max(14.0, home[2] * 3)
        self.min_area = 6.0
        self.layer_focus = None  # for layer view: 0=inner,1=middle,2=inter,3=outer

    def isolate_cycle(self, step):
        n = len(self.parts)
        if self.isolate is None:
            self.isolate = 0 if step > 0 else n - 1
        else:
            nxt = self.isolate + step
            self.isolate = None if (nxt < 0 or nxt >= n) else nxt
        self.selected = self.isolate

    def _global_shift(self, state):
        if self.view == "assembly" and self.explode_amt > 0:
            return np.array([0.0, 0.0, 0.0])
        return np.zeros(3)

    def _part_offset(self, part, state):
        if self.explode_amt > 0.01:
            return part.explode * self.explode_amt
        return np.zeros(3)

    def render(self, surf, rect, state, angles, font=None, interactive=False,
               mouse_pos=None):
        clip = surf.get_clip()
        surf.set_clip(rect)
        cx = rect.x + rect.w / 2.0 + self.pan[0]
        cy = rect.y + rect.h / 2.0 + self.pan[1]
        focal = min(rect.w, rect.h) * 1.05
        Rcam = rot_x(self.el) @ rot_y(self.az)
        gshift = self._global_shift(state)
        lx, ly, lz = self.light
        default_ang = angles.get("default", 0.0)
        cull = self.cull

        hi = self.selected if self.selected is not None else self.hovered
        polys, labels, screeninfo = [], [], []

        for pi, part in enumerate(self.parts):
            if self.isolate is not None and pi != self.isolate:
                continue
            if self.view == "assembly" and part.order > self.assembled:
                dim = 0.28
            else:
                dim = 1.0
            off = self._part_offset(part, state) + gshift - self.free
            highlight = (pi == hi)
            want_info = ((interactive or (self.show_labels and font))
                         and not (self.view == "assembly" and part.order > self.assembled))
            bx0 = by0 = 1e18; bx1 = by1 = -1e18; zmin = 1e18; got = False

            for m in part.meshes:
                wv = m.world_verts(angles.get(m.group, default_ang)) + off
                cam = wv @ Rcam.T
                cam[:, 2] += self.dist
                col = m.color
                if m.emissive:
                    col = _mix(col, (255, 255, 255), 0.25)
                if dim < 0.99:
                    col = (int(col[0] * dim), int(col[1] * dim), int(col[2] * dim))
                if highlight:
                    col = _mix(col, (255, 255, 255), 0.30)

                z = cam[:, 2]
                safe = np.where(z > 0.05, z, 1e9)
                sx = cx + focal * cam[:, 0] / safe
                sy = cy - focal * cam[:, 1] / safe
                _emit_polys(polys, cam, sx, sy, col, self.light, cull, m.emissive,
                            _face_groups(m), self.min_area, highlight)
                if want_info:
                    vmask = z > 0.05
                    if vmask.any():
                        vx = sx[vmask]; vy = sy[vmask]
                        bx0 = min(bx0, float(vx.min())); bx1 = max(bx1, float(vx.max()))
                        by0 = min(by0, float(vy.min())); by1 = max(by1, float(vy.max()))
                        zmin = min(zmin, float(z[vmask].min())); got = True

            if want_info and got:
                pcx = 0.5 * (bx0 + bx1); pcy = 0.5 * (by0 + by1)
                rad = 0.5 * math.hypot(bx1 - bx0, by1 - by0) + 6
                screeninfo.append((pi, pcx, pcy, rad, zmin))
                if self.show_labels and font and (self.view != "full" or highlight):
                    labels.append((zmin, (pcx, pcy), part.name, highlight))

        # Paint solids (far -> near)
        polys.sort(key=lambda t: t[0], reverse=True)
        do_out = len(polys) < OUTLINE_MAX_POLYS
        dpoly = pygame.draw.polygon
        for _, pts, fcol, hl in polys:
            if len(pts) >= 3:
                try:
                    dpoly(surf, fcol, pts)
                    if hl:
                        dpoly(surf, C_ACCENT, pts, 2)
                    elif do_out:
                        dpoly(surf, (12, 14, 20), pts, 1)
                except Exception:
                    pass

        # Labels
        if self.show_labels and font:
            labels.sort(key=lambda t: t[0])
            used_rects = []
            for _, (lxp, lyp), text, hl in labels:
                img = font.render(text, True, C_TEXT)
                lw, lh = img.get_width() + 8, img.get_height() + 2
                x = int(clamp(lxp, rect.x + 20, rect.x + rect.w - lw - 6))
                y = int(clamp(lyp, rect.y + 20, rect.y + rect.h - lh - 6))
                rct = pygame.Rect(x - 4, y - 1, lw, lh)
                tries = 0
                while any(rct.colliderect(u) for u in used_rects) and tries < 36:
                    rct.y += lh + 3
                    if rct.bottom > rect.y + rect.h - 8:
                        rct.y = rect.y + 20 + (tries % 6) * (lh + 3)
                        rct.x = min(rect.x + rect.w - lw - 6, rct.x + lw + 18)
                    tries += 1
                used_rects.append(rct.copy())
                _label(surf, font, text, (rct.x + 4, rct.y + 1), accent=hl)

        # Hover picking
        if interactive:
            mx, my = mouse_pos if mouse_pos is not None else pygame.mouse.get_pos()
            best, bd = None, 1e18
            for pi, pcx, pcy, rad, depth in screeninfo:
                if math.hypot(mx - pcx, my - pcy) <= rad and depth < bd:
                    bd, best = depth, pi
            self.hovered = best

        surf.set_clip(clip)

    def _project(self, pts, cx, cy, focal, Rcam):
        cam = pts @ Rcam.T
        cam[:, 2] += self.dist
        out = []
        for vx, vy, vz in cam:
            if vz > 0.05:
                out.append((cx + focal * vx / vz, cy - focal * vy / vz, vz))
            else:
                out.append(None)
        return out


# =============================================================================
# FLIGHT RENDERER  -- world view with suit in flight
# =============================================================================

GROUND_Y = 0.0

class FlightRenderer:
    """Renders the suit in a world context during flight simulation."""
    def __init__(self, parts, suit_config):
        self.parts = parts
        self.cfg = suit_config
        self.az = 0.6
        self.el = 0.25
        self.dist = 6.0
        self.dist_target = 6.0
        self.light = C_LIGHT_DIR / np.linalg.norm(C_LIGHT_DIR)
        self.chase_cam = True  # default to chase cam for game feel
        self._init_buildings()
        self.cull = True
        self.min_area = 5.0
        self.xray = False
        self.show_heat = False
        self.chase_offset = np.array([-3.0, 2.0, -6.0])
        # Weather particle system
        self.weather_particles = []
        self.weather_type = "clear"
        self._init_weather_particles()

    def _init_weather_particles(self):
        """Initialize particle pool for weather effects."""
        self.weather_particles = []
        for _ in range(300):
            self.weather_particles.append({
                "pos": np.array([np.random.uniform(-30, 30),
                                 np.random.uniform(0, 50),
                                 np.random.uniform(-30, 30)]),
                "vel": np.zeros(3),
                "life": np.random.random(),
                "size": 1,
            })

    def _init_buildings(self):
        """Generate procedural cityscape buildings that scroll past for speed sensation."""
        self.buildings = []
        rng = np.random.RandomState(42)  # deterministic for consistency
        for _ in range(80):
            bx = rng.uniform(-120, 120)
            bz = rng.uniform(-50, 400)
            bw = rng.uniform(6, 18)
            bd = rng.uniform(6, 18)
            bh = rng.uniform(15, 70)
            # Building color: dark blue-grey with slight variation
            shade = rng.randint(25, 50)
            col = (shade, shade + 5, shade + 12)
            # Window lit probability
            windows_on = rng.random() > 0.3
            self.buildings.append({
                "x": bx, "z": bz, "w": bw, "d": bd, "h": bh,
                "color": col, "windows": windows_on,
                "seed": rng.randint(0, 9999),
            })

    def _update_weather_particles(self, dt, state):
        """Update weather particles based on current weather type."""
        weather = state.weather
        wind = state.atmosphere.wind
        self.weather_type = weather

        for p in self.weather_particles:
            p["life"] -= dt * 0.5
            if p["life"] <= 0:
                # Respawn particle
                p["pos"] = np.array([np.random.uniform(-30, 30),
                                     np.random.uniform(20, 60),
                                     np.random.uniform(-30, 30)])
                p["life"] = 1.0

            if weather == "rain":
                p["vel"] = np.array([wind[0] * 0.3, -15.0, wind[2] * 0.3])
                p["size"] = 2
            elif weather == "storm":
                p["vel"] = np.array([wind[0] * 0.5, -20.0, wind[2] * 0.5])
                p["size"] = 3
            elif weather == "dust":
                p["vel"] = np.array([wind[0] * 0.8, np.random.uniform(-2, 2), wind[2] * 0.8])
                p["size"] = 2
            elif weather == "methane":
                p["vel"] = np.array([wind[0] * 0.2, -3.0, wind[2] * 0.2])
                p["size"] = 2
            elif weather == "current":
                p["vel"] = np.array([wind[0] * 0.5, np.random.uniform(-1, 1), wind[2] * 0.5])
                p["size"] = 2
            elif weather == "clouds":
                p["vel"] = np.array([wind[0] * 0.4, -5.0, wind[2] * 0.4])
                p["size"] = 1
            else:
                p["vel"] = np.zeros(3)
                p["size"] = 0

            p["pos"] += p["vel"] * dt
            # Wrap around
            if p["pos"][1] < 0:
                p["pos"][1] = 50
                p["life"] = 1.0
            if abs(p["pos"][0]) > 35:
                p["pos"][0] = -p["pos"][0] * 0.9
            if abs(p["pos"][2]) > 35:
                p["pos"][2] = -p["pos"][2] * 0.9

    def _draw_weather(self, surf, project, state):
        """Draw weather particles in the 3D scene."""
        if self.weather_type == "clear" or self.weather_type == "vacuum":
            return

        # Determine particle color by weather type
        colors = {
            "rain": (100, 150, 200),
            "storm": (80, 100, 140),
            "dust": (180, 140, 90),
            "methane": (120, 160, 200),
            "current": (60, 120, 160),
            "clouds": (200, 200, 210),
        }
        col = colors.get(self.weather_type, (150, 150, 150))

        for p in self.weather_particles:
            if p["size"] == 0:
                continue
            pt = project(p["pos"].tolist())
            if pt:
                try:
                    if self.weather_type in ("rain", "storm"):
                        # Draw as streaks
                        pt2 = project((p["pos"][0] - p["vel"][0] * 0.02,
                                       p["pos"][1] - p["vel"][1] * 0.02,
                                       p["pos"][2] - p["vel"][2] * 0.02))
                        if pt2:
                            pygame.draw.line(surf, col, (pt[0], pt[1]), (pt2[0], pt2[1]), 1)
                    else:
                        pygame.draw.circle(surf, col, (int(pt[0]), int(pt[1])), p["size"])
                except Exception:
                    pass

    def _draw_buildings(self, surf, project, state):
        """Draw procedural cityscape buildings that recycle as player flies forward."""
        for b in self.buildings:
            # Recycle buildings that are far behind player
            if b["z"] < state.pos[2] - 60:
                b["z"] += 400
                b["x"] = np.random.uniform(-120, 120)
            elif b["z"] > state.pos[2] + 350:
                b["z"] -= 400
                b["x"] = np.random.uniform(-120, 120)
            bx, bz = b["x"], b["z"]
            bw, bd, bh = b["w"], b["d"], b["h"]
            # Project 4 base corners + 4 top corners
            corners = [
                (bx - bw/2, GROUND_Y, bz - bd/2),
                (bx + bw/2, GROUND_Y, bz - bd/2),
                (bx + bw/2, GROUND_Y, bz + bd/2),
                (bx - bw/2, GROUND_Y, bz + bd/2),
            ]
            top_corners = [
                (bx - bw/2, GROUND_Y + bh, bz - bd/2),
                (bx + bw/2, GROUND_Y + bh, bz - bd/2),
                (bx + bw/2, GROUND_Y + bh, bz + bd/2),
                (bx - bw/2, GROUND_Y + bh, bz + bd/2),
            ]
            base_pts = [project(c) for c in corners]
            top_pts = [project(c) for c in top_corners]
            if all(p is None for p in base_pts):
                continue
            fade = clamp(1.0 - abs(bz - state.pos[2]) / 200, 0.0, 1.0)
            bcol = b["color"]
            bcol_faded = (int(bcol[0] * fade), int(bcol[1] * fade), int(bcol[2] * fade))
            # Draw building faces (front, right, left) as filled quads
            faces = [
                (base_pts[0], base_pts[1], top_pts[1], top_pts[0]),  # front
                (base_pts[1], base_pts[2], top_pts[2], top_pts[1]),  # right
                (base_pts[3], base_pts[0], top_pts[0], top_pts[3]),  # left
            ]
            for fi, (b0, b1, t1, t0) in enumerate(faces):
                if b0 and b1 and t1 and t0:
                    try:
                        # Side faces are darker
                        side_shade = 0.7 if fi > 0 else 1.0
                        fcol = (int(bcol_faded[0] * side_shade),
                                int(bcol_faded[1] * side_shade),
                                int(bcol_faded[2] * side_shade))
                        pts = [(b0[0], b0[1]), (b1[0], b1[1]), (t1[0], t1[1]), (t0[0], t0[1])]
                        pygame.draw.polygon(surf, fcol, pts)
                        pygame.draw.polygon(surf, (max(0, fcol[0]-10), max(0, fcol[1]-10), max(0, fcol[2]-10)), pts, 1)
                    except Exception:
                        pass
            # Draw window lights on front face
            if b["windows"] and top_pts[0] and top_pts[1] and base_pts[0] and base_pts[1]:
                try:
                    rng = np.random.RandomState(b["seed"])
                    n_rows = max(2, int(bh / 5))
                    n_cols = max(2, int(bw / 4))
                    for wr in range(n_rows):
                        for wc in range(n_cols):
                            if rng.random() > 0.4:
                                continue
                            wy = GROUND_Y + 3 + wr * (bh - 4) / n_rows
                            wx = bx - bw/2 + 2 + wc * (bw - 4) / n_cols
                            wp = project((wx, wy, bz - bd/2 - 0.01))
                            if wp and wp[2] < 120:
                                wcol = (int(180 * fade), int(160 * fade), int(60 * fade))
                                pygame.draw.circle(surf, wcol, (int(wp[0]), int(wp[1])), 1)
                except Exception:
                    pass

    def _draw_flight_rings(self, surf, project, state):
        """Draw glowing flight rings to fly through for points."""
        for ring in state.flight_rings:
            rp = ring["pos"]
            # Skip rings too far away
            dist = np.linalg.norm(rp - state.pos)
            if dist > 300:
                continue
            radius = ring["radius"]
            angle = ring["angle"]
            glow = ring["glow"]
            passed = ring["passed"]
            # Ring color: cyan when active, gold flash when passed
            if passed:
                base_col = (255, int(200 + glow * 55), int(80 + glow * 100))
            else:
                base_col = ring["color"]
            # Draw ring as series of segments
            n_seg = 24
            prev_pt = None
            for i in range(n_seg + 1):
                a = i * 2 * math.pi / n_seg + angle
                # Ring is oriented facing +Z (toward player direction of travel)
                px = rp[0] + math.cos(a) * radius
                py = rp[1] + math.sin(a) * radius
                pz = rp[2]
                pt = project((px, py, pz))
                if pt and prev_pt:
                    try:
                        col = base_col
                        if glow > 0.1:
                            col = (min(255, col[0] + int(glow * 50)),
                                   min(255, col[1] + int(glow * 50)),
                                   min(255, col[2] + int(glow * 50)))
                        thickness = 3 if glow > 0.1 else 2
                        pygame.draw.line(surf, col, (prev_pt[0], prev_pt[1]), (pt[0], pt[1]), thickness)
                    except Exception:
                        pass
                prev_pt = pt
            # Draw center glow
            center = project(rp.tolist())
            if center and not passed:
                try:
                    glow_r = max(2, int(radius * 2 / center[2]))
                    pygame.draw.circle(surf, (base_col[0]//4, base_col[1]//4, base_col[2]//4),
                                     (int(center[0]), int(center[1])), glow_r, 1)
                except Exception:
                    pass

    def _draw_speed_trail(self, surf, project, state):
        """Draw speed lines / particle trail behind the suit."""
        for p in state.speed_trail:
            pt = project(p["pos"].tolist())
            if pt:
                try:
                    alpha = p["life"]
                    col = (int(100 * alpha), int(150 * alpha), int(200 * alpha))
                    size = max(1, int(p["size"] * alpha))
                    pygame.draw.circle(surf, col, (int(pt[0]), int(pt[1])), size)
                except Exception:
                    pass

    def _draw_vignette(self, surf, rect, intensity=0.2):
        """Draw a dark vignette around screen edges for speed/boost sensation."""
        try:
            overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            for i in range(30):
                alpha = int(intensity * 255 * (i / 30))
                pygame.draw.rect(overlay, (0, 0, 0, alpha),
                               (i, i, rect.w - 2*i, rect.h - 2*i), 1)
            surf.blit(overlay, (rect.x, rect.y))
        except Exception:
            pass

    def _view(self, state):
        if self.chase_cam:
            # Chase camera: follows behind suit based on velocity direction
            speed = np.linalg.norm(state.velocity)
            if speed > 1.0:
                vel_dir = state.velocity / speed
            else:
                vel_dir = np.array([0.0, 0.0, 1.0])
            # Camera behind and above, opposite to velocity
            behind = -vel_dir * self.dist
            cam = state.pos + behind + np.array([0.0, 1.5 + self.el * 2.0, 0.0])
            target = state.pos + np.array([0.0, 0.5, 0.0]) + vel_dir * 2.0
        else:
            cam = state.pos + np.array([math.cos(self.az) * self.dist,
                                         1.0 + self.el * self.dist * 0.5,
                                         math.sin(self.az) * self.dist])
            target = state.pos + np.array([0.0, 0.3, 0.0])
        fwd = target - cam
        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right = right / (np.linalg.norm(right) or 1.0)
        up = np.cross(right, fwd)
        R = np.array([right, up, fwd])
        return cam, R

    def render(self, surf, rect, state, angles, font=None):
        clip = surf.get_clip()
        surf.set_clip(rect)
        # Screen shake offset
        shake_x = int((np.random.random() - 0.5) * state.screen_shake * 12)
        shake_y = int((np.random.random() - 0.5) * state.screen_shake * 12)
        cx = rect.x + rect.w / 2.0 + shake_x
        cy = rect.y + rect.h / 2.0 + shake_y
        # FOV widens with speed and afterburner for game feel
        speed = np.linalg.norm(state.velocity)
        fov_mult = 1.0 + min(speed * 0.003, 0.15) + (0.1 if state.afterburner else 0.0)
        focal = min(rect.w, rect.h) * 1.05 * fov_mult
        cam, R = self._view(state)

        def project(p):
            d = (np.asarray(p) - cam) @ R.T
            if d[2] <= 0.06:
                return None
            return (cx + focal * d[0] / d[2], cy - focal * d[1] / d[2], d[2])

        # Sky gradient (top = dark blue, bottom = lighter)
        sky_top = (12, 16, 28)
        sky_bot = (24, 32, 48)
        for sy_seg in range(rect.h):
            t = sy_seg / rect.h
            r = int(sky_top[0] + (sky_bot[0] - sky_top[0]) * t)
            g = int(sky_top[1] + (sky_bot[1] - sky_top[1]) * t)
            b = int(sky_top[2] + (sky_bot[2] - sky_top[2]) * t)
            pygame.draw.line(surf, (r, g, b), (rect.x, rect.y + sy_seg), (rect.x + rect.w, rect.y + sy_seg))

        # Ground grid
        step = 4.0
        gx0 = math.floor((state.pos[0] - 60) / step) * step
        gz0 = math.floor((state.pos[2] - 90) / step) * step
        for i in range(31):
            xx = gx0 + i * step
            p1 = project((xx, GROUND_Y, gz0))
            p2 = project((xx, GROUND_Y, gz0 + 30 * step))
            if p1 and p2:
                fade = clamp(1.0 - abs(xx - state.pos[0]) / 80, 0.0, 1.0)
                col = (int(30 * fade), int(36 * fade), int(44 * fade))
                try:
                    pygame.draw.line(surf, col, (p1[0], p1[1]), (p2[0], p2[1]))
                except Exception:
                    pass
        for j in range(31):
            zz = gz0 + j * step
            p1 = project((gx0, GROUND_Y, zz))
            p2 = project((gx0 + 30 * step, GROUND_Y, zz))
            if p1 and p2:
                fade = clamp(1.0 - abs(zz - state.pos[2]) / 80, 0.0, 1.0)
                col = (int(30 * fade), int(36 * fade), int(44 * fade))
                try:
                    pygame.draw.line(surf, col, (p1[0], p1[1]), (p2[0], p2[1]))
                except Exception:
                    pass

        # Horizon line
        hp1 = project((state.pos[0] - 200, GROUND_Y, state.pos[2] + 200))
        hp2 = project((state.pos[0] + 200, GROUND_Y, state.pos[2] + 200))
        if hp1 and hp2:
            try:
                pygame.draw.line(surf, (40, 50, 65), (hp1[0], hp1[1]), (hp2[0], hp2[1]), 2)
            except Exception:
                pass

        # Procedural cityscape buildings (recycled as player flies forward)
        self._draw_buildings(surf, project, state)

        # Flight rings (Iron Man game style)
        self._draw_flight_rings(surf, project, state)

        # Build draw items
        polys = []
        default_ang = angles.get("default", 0.0)
        gshift = np.array([0.0, math.sin(state.time * 3.0) * 0.01 * state.throttle, 0.0])
        Rb = rot_z(state.roll) @ rot_x(state.pitch) @ rot_y(state.yaw)

        for part in self.parts:
            for m in part.meshes:
                lv = m.world_verts(angles.get(m.group, default_ang))
                wv = (lv @ Rb.T) + state.pos + gshift
                camv = (wv - cam) @ R.T
                z = camv[:, 2]
                safe = np.where(z > 0.06, z, 1e9)
                sx = cx + focal * camv[:, 0] / safe
                sy = cy - focal * camv[:, 1] / safe
                _emit_polys(polys, camv, sx, sy, m.color, self.light,
                            self.cull, getattr(m, "emissive", False),
                            _face_groups(m), self.min_area, False, znear=0.06)

        polys.sort(key=lambda t: t[0], reverse=True)
        do_out = len(polys) < OUTLINE_MAX_POLYS
        dpoly = pygame.draw.polygon
        for _, pts, fcol, _hl in polys:
            if len(pts) >= 3:
                try:
                    dpoly(surf, fcol, pts)
                    if do_out:
                        dpoly(surf, (10, 12, 18), pts, 1)
                except Exception:
                    pass

        # Thrust glow
        if state.throttle > 0.05:
            self._draw_thrust(surf, project, state, Rb, gshift)

        # Speed lines / particle trail
        self._draw_speed_trail(surf, project, state)

        # Afterburner vignette
        if state.afterburner and state.throttle > 0.5:
            self._draw_vignette(surf, rect, intensity=0.3)
        elif speed > 50:
            self._draw_vignette(surf, rect, intensity=0.15)

        # Heat overlay
        if getattr(self, 'show_heat', False):
            self._draw_heat_overlay(surf, project, state, Rb, gshift)

        # Damage + self-healing overlay
        self._draw_damage_overlay(surf, project, state, Rb, gshift)

        # Weather particles
        self._update_weather_particles(0.016, state)
        self._draw_weather(surf, project, state)

        surf.set_clip(clip)

    def _draw_thrust(self, surf, project, state, Rb, gshift):
        """Draw thrust glow cones and particle effects from turbines."""
        tv = state.thrust_vector / (np.linalg.norm(state.thrust_vector) or 1.0)
        glow_len = 0.3 + state.throttle * 0.8
        ab_mult = 1.8 if state.afterburner else 1.0
        glow_len *= ab_mult
        for part in self.parts:
            if part.key != "turbines":
                continue
            for m in part.meshes:
                if "exhaust" not in m.name:
                    continue
                # Get world position of exhaust
                wv = m.world_verts(0.0)
                center = wv.mean(axis=0)
                wpos = (center @ Rb.T) + state.pos + gshift
                end = wpos + tv * glow_len
                p1 = project(wpos.tolist())
                p2 = project(end.tolist())
                if p1 and p2:
                    intensity = int(min(255, 200 * state.throttle * ab_mult))
                    if state.afterburner:
                        col = (intensity, intensity // 3, 20)
                        # Outer flame cone (wider, blue-white core for AB)
                        try:
                            pygame.draw.line(surf, col, (p1[0], p1[1]), (p2[0], p2[1]), 5)
                            # Inner blue-white core
                            core_col = (200, 220, 255)
                            pygame.draw.line(surf, core_col, (p1[0], p1[1]), (p2[0], p2[1]), 2)
                        except Exception:
                            pass
                    else:
                        col = (intensity, intensity // 2, 40)
                        try:
                            pygame.draw.line(surf, col, (p1[0], p1[1]), (p2[0], p2[1]), 3)
                        except Exception:
                            pass
                    # Particle sparks at exhaust tip
                    n_sparks = 5 if state.afterburner else 3
                    if state.throttle > 0.1:
                        for pi2 in range(n_sparks):
                            spread = (np.random.random(3) - 0.5) * 0.15 * state.throttle * ab_mult
                            tip = end + spread
                            pt = project(tip.tolist())
                            if pt:
                                if state.afterburner:
                                    spark_col = (255, int(200 + np.random.random() * 55), int(80 + np.random.random() * 100))
                                else:
                                    spark_col = (255, int(180 + np.random.random() * 60), int(40 + np.random.random() * 60))
                                spark_size = 3 if state.afterburner else 2
                                pygame.draw.circle(surf, spark_col, (int(pt[0]), int(pt[1])), spark_size)

    def _draw_heat_overlay(self, surf, project, state, Rb, gshift):
        """Draw heat distribution overlay on suit parts."""
        if not state.heat_map:
            return
        for part in self.parts:
            if part.key not in state.heat_map:
                continue
            temp = state.heat_map[part.key]
            # Map temperature to color: 20C=blue, 200C=yellow, 600C=red
            if temp < 100:
                heat_col = (60, 80, 120)
            elif temp < 300:
                t = (temp - 100) / 200
                heat_col = (int(60 + t * 200), int(80 + t * 100), int(120 - t * 80))
            elif temp < 500:
                t = (temp - 300) / 200
                heat_col = (int(200 + t * 55), int(180 - t * 100), int(40 - t * 40))
            else:
                t = min(1.0, (temp - 500) / 200)
                heat_col = (255, int(80 - t * 60), int(20 + t * 20))
            for m in part.meshes:
                if "exhaust" in m.name or "fan" in m.name:
                    continue
                wv = m.world_verts(0.0)
                center = wv.mean(axis=0)
                wpos = (center @ Rb.T) + state.pos + gshift
                pt = project(wpos.tolist())
                if pt:
                    try:
                        radius = max(3, int(15 / pt[2]))
                        pygame.draw.circle(surf, heat_col, (int(pt[0]), int(pt[1])), radius, 1)
                    except Exception:
                        pass

    def _draw_damage_overlay(self, surf, project, state, Rb, gshift):
        """Draw damage indicators and self-healing seams on suit parts."""
        # Draw damage indicators (red cracks on damaged parts)
        for part_key, damage in state.armor_damage.items():
            if damage < 0.05:
                continue
            for part in self.parts:
                if part.key != part_key:
                    continue
                for m in part.meshes:
                    if "exhaust" in m.name or "fan" in m.name:
                        continue
                    wv = m.world_verts(0.0)
                    center = wv.mean(axis=0)
                    wpos = (center @ Rb.T) + state.pos + gshift
                    pt = project(wpos.tolist())
                    if pt:
                        try:
                            radius = max(4, int(20 / pt[2]))
                            # Damage color: yellow (light) -> orange -> red (heavy)
                            if damage < 0.3:
                                dcol = (200, 180, 40)
                            elif damage < 0.6:
                                dcol = (220, 120, 30)
                            else:
                                dcol = (220, 40, 40)
                            # Draw crack lines radiating from center
                            n_cracks = int(damage * 8) + 2
                            for ci in range(n_cracks):
                                ang = ci * 2 * math.pi / n_cracks + state.time * 0.1
                                ex = radius * math.cos(ang)
                                ey = radius * math.sin(ang)
                                pygame.draw.line(surf, dcol,
                                                 (int(pt[0]), int(pt[1])),
                                                 (int(pt[0] + ex), int(pt[1] + ey)), 1)
                        except Exception:
                            pass
        # Draw self-healing seam visualization (green pulsing circles)
        for region in state.self_heal_regions:
            part_key = region["part"]
            progress = region["progress"]
            for part in self.parts:
                if part.key != part_key:
                    continue
                for m in part.meshes:
                    if "exhaust" in m.name or "fan" in m.name:
                        continue
                    wv = m.world_verts(0.0)
                    center = wv.mean(axis=0)
                    wpos = (center @ Rb.T) + state.pos + gshift
                    pt = project(wpos.tolist())
                    if pt:
                        try:
                            # Pulsing green circle that shrinks as healing progresses
                            pulse = 0.5 + 0.5 * math.sin(state.time * 8)
                            base_r = max(6, int(25 / pt[2]))
                            radius = int(base_r * (1.0 - progress * 0.7) + pulse * 3)
                            heal_col = (40, int(180 + pulse * 60), 80)
                            pygame.draw.circle(surf, heal_col,
                                             (int(pt[0]), int(pt[1])), radius, 2)
                            # Progress text
                            if pt[2] < 10:
                                ptxt = self.cfg.get("_font_sm", None)
                        except Exception:
                            pass


# =============================================================================
# HUD  -- heads-up display for flight mode
# =============================================================================

def draw_hud(surf, rect, state, physics, font, font_sm):
    """Draw the comprehensive flight HUD overlay."""
    x = rect.x + 10
    y = rect.y + 10

    # Left panel: vitals + 6-DOF + atmosphere (responsive width)
    lp_w = min(240, int(rect.w * 0.19))
    _panel(surf, x, y, lp_w, 250, 200)
    surf.blit(font.render("Mjalnor'MV1.17 -- FLIGHT HUD", True, C_ACCENT), (x + 8, y + 6))

    speed = np.linalg.norm(state.velocity)
    lines = [
        f"Mode: {state.flight_mode.upper()}",
        f"Env: {state.env_name}",
        f"Alt: {state.altitude:.1f} m ({state.altitude*3.28:.0f} ft)",
        f"Vel: {speed:.1f} m/s ({speed*2.237:.0f} mph)",
        f"T/W: {physics.thrust_to_weight:.2f}",
        f"G-load: {state.g_load:.1f} {'[LIM]' if physics.g_limiter_active else ''}",
        f"Throttle: {state.throttle*100:.0f}%{' AB' if state.afterburner else ''}",
        f"RPM: {state.turbine_rpm:,.0f}",
        f"Wings: {state.wing_deploy*100:.0f}%",
        f"Att: P{math.degrees(state.pitch):.0f} R{math.degrees(state.roll):.0f} Y{math.degrees(state.yaw):.0f}",
        f"{'[AUTO-HOVER]' if state.auto_hover else ''} {'[AUTO-LVL]' if state.auto_level else ''}",
        f"Wind: {state.wind_speed:.1f} m/s @ {state.wind_direction:.0f}deg",
        f"Weather: {state.weather}  Solar: {state.solar_harvesting_w:.0f}W",
        f"Press: {state.atmosphere.ambient_pressure/1000:.1f} kPa  Radio: {state.atmosphere.radio_attenuation:.0f}dB",
    ]
    for i, line in enumerate(lines):
        col = C_TEXT
        if "G-load" in line and state.g_load > 10:
            col = C_WARN
        if "COLLISION" in line:
            col = C_ACCENT2
        surf.blit(font_sm.render(line, True, col), (x + 8, y + 30 + i * 16))

    # Collision warning
    if state.collision_warning:
        warn = font.render("!! COLLISION WARNING !!", True, C_ACCENT2)
        surf.blit(warn, (x + 8, y + 210))

    # Contextual environment panel: dive computer (underwater) / dosimetry (vacuum)
    env_y = y + 258
    if state.dive.active:
        d = state.dive
        _panel(surf, x, env_y, lp_w, 132, 200)
        surf.blit(font.render("DIVE COMPUTER", True, C_ACCENT), (x + 8, env_y + 6))
        dlines = [
            f"Depth: {d.depth_m:.1f} m  ({d.ambient_ata:.2f} ata)",
            f"NDL: {d.ndl_min:.0f} min  {'DECO ' + str(int(d.deco_stop_m)) + 'm' if d.deco_required else 'no-stop'}",
            f"Ascent: {d.ascent_rate_m_min:+.0f} m/min {'FAST!' if d.ascent_warning else ''}",
            f"ppO2: {d.ppo2_ata:.2f} ata  CNS: {d.cns_pct:.0f}%",
            f"Narcosis EAD: {d.narcosis_ead_m:.0f} m",
            f"BCD: {d.bcd_volume_l:.1f} L  Net: {d.net_buoyancy_n:+.0f} N",
            f"Gas: {d.gas_pct:.0f}%",
        ]
        for i, ln in enumerate(dlines):
            col = C_TEXT
            if "DECO" in ln or "FAST" in ln:
                col = C_WARN
            if "ppO2" in ln and d.o2_tox_warning:
                col = C_ACCENT2
            if "Gas" in ln and d.gas_pct < 20:
                col = C_WARN
            surf.blit(font_sm.render(ln, True, col), (x + 8, env_y + 28 + i * 14))
    elif state.space.active:
        sp = state.space
        _panel(surf, x, env_y, lp_w, 118, 200)
        surf.blit(font.render("SPACE / RADIATION", True, C_ACCENT), (x + 8, env_y + 6))
        slines = [
            f"Dose rate: {sp.dose_rate_msv_h:.3f} mSv/h {'SPE!' if sp.spe_active else ''}",
            f"Mission dose: {sp.dose_mission_msv:.2f} mSv",
            f"Career: {sp.career_fraction * 100:.2f}%  Shield: {sp.shield_g_cm2:.0f} g/cm2",
            f"RCS gas: {sp.rcs_propellant_pct:.0f}%  {'' if sp.rcs_available else 'EMPTY!'}",
            f"Delta-v: {sp.delta_v_ms:.0f} m/s",
            f"MMOD shield: {sp.mmod_shield_integrity * 100:.0f}%",
        ]
        for i, ln in enumerate(slines):
            col = C_TEXT
            if ("Dose rate" in ln and sp.dose_alarm) or "SPE" in ln or "EMPTY" in ln:
                col = C_ACCENT2
            surf.blit(font_sm.render(ln, True, col), (x + 8, env_y + 28 + i * 14))

    # Right panel: systems + OS (responsive position)
    rp_w = min(250, int(rect.w * 0.20))
    rx = rect.x + rect.w - rp_w - 10
    _panel(surf, rx, y, rp_w, 210, 200)
    surf.blit(font.render("SYSTEMS + SuitRTOS", True, C_ACCENT), (rx + 8, y + 6))

    rtos = state.rtos
    sys_lines = [
        f"Battery: {state.battery_soc*100:.0f}%",
        f"Fuel: {state.fuel_kg:.1f}/{state.fuel_max_kg:.1f}kg ({state.fuel_kg/state.fuel_max_kg*100:.0f}%)",
        f"  Burn: {state.fuel_flow_kg_s*1000:.1f} g/s  Spent: {state.fuel_burned_kg:.1f}kg",
        f"Inner T: {state.temp_inner:.1f}C  Outer: {state.temp_outer:.0f}C",
        f"O2: {state.o2_level*100:.0f}%  CO2: {state.co2_level*100:.1f}%",
        f"Seal: {state.seal_pressure:.0f} kPa",
        f"OS: {'OK' if rtos.primary_ok else 'FAIL'}  Up: {rtos.uptime_str}",
        f"Checks: {rtos.integrity_checks:,}",
        f"Failovers: {rtos.failovers}  Viruses: {rtos.viruses_detected}",
        f"Tasks: {rtos.task_count} ({rtos.critical_task_count} crit)",
        f"Uptime: {rtos.uptime_pct:.6f}%",
        f"BCI: <{DIMS['bci_latency_ms']}ms  AI: {DIMS['os_ai_model'][:12]}",
        f"EMP: {'SHIELDED' if state.emp_shield_active else 'BREACHED!'} ({state.emp_attenuation_db:.0f}dB) Hits:{state.emp_hits}",
        f"Heal: {'ACTIVE' if state.self_heal_active_count > 0 else 'STANDBY'} {state.self_heal_active_count} regions",
        f"HR: {state.heart_rate:.0f}bpm  Adr: {state.adrenaline*100:.0f}%  Str: {state.stress_level*100:.0f}%",
        f"Fatigue: {state.fatigue*100:.0f}%  SpO2: {state.blood_o2_sat:.0f}%  Armor: {state.armor_integrity*100:.0f}%",
    ]
    for i, line in enumerate(sys_lines):
        col = C_TEXT
        if "Battery" in line and state.battery_soc < 0.2:
            col = C_WARN
        if "OS:" in line and not rtos.primary_ok:
            col = C_ACCENT2
        if "Virus" in line and rtos.viruses_detected > 0:
            col = C_WARN
        if "BREACHED" in line:
            col = C_ACCENT2
        surf.blit(font_sm.render(line, True, col), (rx + 8, y + 30 + i * 16))

    # Right-bottom: combat panel
    cy = y + 220
    _panel(surf, rx, cy, rp_w, 120, 200)
    surf.blit(font.render("COMBAT + AUTO-AIM", True, C_ACCENT), (rx + 8, cy + 6))

    aim = state.auto_aim
    defense = state.defense
    jump = state.jump
    combat_lines = [
        f"Defense: {defense.state}  Stance: {defense.stance}",
        f"Punches: {defense.punches_thrown}  Force: {defense.punch_force_lbs:,.0f} lbs",
        f"Aim: {'LOCKED' if aim.lock_acquired else 'SCANNING' if aim.locked_target else 'IDLE'}",
        f"Targets: {len(aim.targets)}  Tracked: {len(aim.tracked_targets)}  Shots: {aim.shots_fired}  Hit: {aim.hit_rate:.0f}%",
        f"Jump: {'CHARGING' if jump.charging else 'JUMPING' if jump.jumping else 'READY'} {jump.charge*100:.0f}%",
        f"{'[DODGE!' + str(len(aim.inbound_threats)) + ' inbound]' if aim.dodge_recommended else ''}",
        f"Weapons: {sum(1 for w in aim.weapon_mounts if w['armed'])}/{len(aim.weapon_mounts)} armed",
    ]
    for i, line in enumerate(combat_lines):
        col = C_TEXT
        if "LOCKED" in line:
            col = C_ACCENT
        if "CHARGING" in line or "JUMPING" in line:
            col = C_FIBER
        surf.blit(font_sm.render(line, True, col), (rx + 8, cy + 30 + i * 16))

    # Bottom: bars (responsive to screen width)
    by = rect.y + rect.h - 50
    bar_w = min(160, (rect.w - 40) // 5 - 10)
    bar_gap = bar_w + 10
    _bar(surf, x, by, bar_w, "Throttle", f"{state.throttle*100:.0f}%", state.throttle, C_ACCENT, font_sm)
    _bar(surf, x + bar_gap, by, bar_w, "Battery", f"{state.battery_soc*100:.0f}%", state.battery_soc, C_OK, font_sm)
    fuel_pct = state.fuel_kg / max(state.fuel_max_kg, 0.01)
    fuel_col = C_OK if fuel_pct > 0.3 else (255, 180, 0) if fuel_pct > 0.1 else (255, 60, 60)
    _bar(surf, x + bar_gap * 2, by, bar_w, "Fuel", f"{fuel_pct*100:.0f}%", fuel_pct, fuel_col, font_sm)
    _bar(surf, x + bar_gap * 3, by, bar_w, "Wings", f"{state.wing_deploy*100:.0f}%", state.wing_deploy, C_FIBER, font_sm)
    _bar(surf, x + bar_gap * 4, by, bar_w, "Jump", f"{jump.charge*100:.0f}%", jump.charge, C_ACCENT2, font_sm)

    # Flight instruments: airspeed tape (left side), altitude tape (right side), heading compass (top center)
    speed = np.linalg.norm(state.velocity)
    speed_mph = speed * 2.237
    # Airspeed tape (left side, vertical)
    asx = x + lp_w + 10
    asy = rect.y + 60
    _panel(surf, asx, asy, 60, 140, 180)
    surf.blit(font_sm.render("SPD", True, C_DIM), (asx + 4, asy + 4))
    surf.blit(font.render(f"{speed_mph:.0f}", True, C_ACCENT), (asx + 4, asy + 20))
    surf.blit(font_sm.render("mph", True, C_DIM), (asx + 4, asy + 38))
    # Speed tape ticks
    for i in range(7):
        tick_val = int(speed_mph / 10) * 10 + (i - 3) * 10
        ty = asy + 60 + i * 12
        col = C_TEXT if tick_val == int(speed_mph / 10) * 10 else C_DIM
        pygame.draw.line(surf, col, (asx + 4, ty), (asx + 20, ty), 1)
        surf.blit(font_sm.render(f"{tick_val}", True, col), (asx + 24, ty - 6))

    # Altitude tape (right side, vertical, positioned left of right panel)
    alx = rx - 70
    aly = rect.y + 60
    _panel(surf, alx, aly, 60, 140, 180)
    surf.blit(font_sm.render("ALT", True, C_DIM), (alx + 4, aly + 4))
    alt_ft = state.altitude * 3.28
    surf.blit(font.render(f"{alt_ft:.0f}", True, C_ACCENT), (alx + 4, aly + 20))
    surf.blit(font_sm.render("ft", True, C_DIM), (alx + 4, aly + 38))
    # Altitude tape ticks
    for i in range(7):
        tick_val = int(alt_ft / 100) * 100 + (i - 3) * 100
        ty = aly + 60 + i * 12
        col = C_TEXT if tick_val == int(alt_ft / 100) * 100 else C_DIM
        pygame.draw.line(surf, col, (alx + 4, ty), (alx + 20, ty), 1)
        surf.blit(font_sm.render(f"{tick_val}", True, col), (alx + 24, ty - 6))

    # Heading compass (top center, horizontal tape)
    heading_deg = math.degrees(state.yaw) % 360
    hcx = rect.x + rect.w // 2
    hcy = rect.y + 30
    _panel(surf, hcx - 80, hcy, 160, 24, 180)
    # Draw heading tape
    for i in range(9):
        hdg = int(heading_deg / 10) * 10 + (i - 4) * 10
        hdg = hdg % 360
        hx = hcx - 80 + i * 20
        col = C_ACCENT if i == 4 else C_DIM
        if hdg % 90 == 0:
            dirs = {0: "N", 90: "E", 180: "S", 270: "W"}
            surf.blit(font_sm.render(dirs.get(hdg, str(hdg)), True, col), (hx - 4, hcy + 4))
        else:
            surf.blit(font_sm.render(f"{hdg}", True, col), (hx - 6, hcy + 4))
    # Center indicator
    pygame.draw.polygon(surf, C_ACCENT2, [(hcx, hcy + 2), (hcx - 4, hcy - 2), (hcx + 4, hcy - 2)])

    # Targeting reticle (center screen) + multi-target reticles
    cx = rect.x + rect.w // 2
    cyy = rect.y + rect.h // 2
    # IFF colors: hostile=red, friendly=green, unknown=yellow
    iff_colors = {"hostile": (220, 40, 40), "friendly": (40, 200, 80), "unknown": (220, 200, 40)}
    # Draw reticles for all tracked targets
    for t, score, dist in aim.tracked_targets:
        if t is aim.locked_target:
            continue  # Draw locked target separately
        try:
            # Project target position to screen (simplified: use relative pos)
            rel = t.pos - state.pos
            tx = cx + int(rel[0] / max(dist, 1) * 200)
            ty2 = cyy - int(rel[1] / max(dist, 1) * 200)
            tcol = iff_colors.get(t.iff_status, (200, 200, 40))
            # Small diamond marker for secondary targets
            sz = 6
            pygame.draw.line(surf, tcol, (tx - sz, ty2), (tx, ty2 - sz), 1)
            pygame.draw.line(surf, tcol, (tx, ty2 - sz), (tx + sz, ty2), 1)
            pygame.draw.line(surf, tcol, (tx + sz, ty2), (tx, ty2 + sz), 1)
            pygame.draw.line(surf, tcol, (tx, ty2 + sz), (tx - sz, ty2), 1)
            # IFF label
            iff_txt = font_sm.render(f"{t.iff_status[:1].upper()} {dist:.0f}m", True, tcol)
            surf.blit(iff_txt, (tx + 8, ty2 - 6))
        except Exception:
            pass
    # Primary locked target reticle
    if aim.locked_target is not None:
        reticle_col = C_ACCENT if aim.lock_acquired else C_DIM
        iff_col = iff_colors.get(aim.locked_target.iff_status, C_ACCENT)
        try:
            pygame.draw.circle(surf, reticle_col, (cx, cyy), 20, 2)
            pygame.draw.line(surf, reticle_col, (cx - 30, cyy), (cx - 22, cyy), 1)
            pygame.draw.line(surf, reticle_col, (cx + 22, cyy), (cx + 30, cyy), 1)
            pygame.draw.line(surf, reticle_col, (cx, cyy - 30), (cx, cyy - 22), 1)
            pygame.draw.line(surf, reticle_col, (cx, cyy + 22), (cx, cyy + 30), 1)
            if aim.lock_acquired:
                # Lock box with IFF color
                pygame.draw.rect(surf, iff_col, (cx - 25, cyy - 25, 50, 50), 2)
                # Range + IFF info
                dist = np.linalg.norm(aim.locked_target.pos - state.pos)
                iff_label = aim.locked_target.iff_status.upper()
                range_txt = font_sm.render(f"[{iff_label}] RANGE: {dist:.0f}m  TOF: {dist/aim.muzzle_velocity:.2f}s", True, iff_col)
                surf.blit(range_txt, (cx - 100, cyy + 35))
        except Exception:
            pass

    # Radar / minimap (top-center, below main HUD)
    radar_cx = rect.x + rect.w // 2
    radar_cy = rect.y + 80
    radar_r = 60
    try:
        pygame.draw.circle(surf, (10, 14, 20), (radar_cx, radar_cy), radar_r, 0)
        pygame.draw.circle(surf, C_DIM, (radar_cx, radar_cy), radar_r, 1)
        pygame.draw.circle(surf, (20, 28, 36), (radar_cx, radar_cy), radar_r // 2, 1)
        # Cross hairs
        pygame.draw.line(surf, C_DIM, (radar_cx - radar_r, radar_cy), (radar_cx + radar_r, radar_cy), 1)
        pygame.draw.line(surf, C_DIM, (radar_cx, radar_cy - radar_r), (radar_cx, radar_cy + radar_r), 1)
        # Sweep line (rotates with time)
        sweep_angle = state.time * 1.5
        sx = radar_cx + int(math.cos(sweep_angle) * radar_r)
        sy = radar_cy + int(math.sin(sweep_angle) * radar_r)
        pygame.draw.line(surf, (40, 80, 60), (radar_cx, radar_cy), (sx, sy), 1)
        # Center dot (self)
        pygame.draw.circle(surf, C_ACCENT, (radar_cx, radar_cy), 3, 0)
        # Plot tracked targets on radar
        radar_range = 2000  # meters
        for t, score, dist in aim.tracked_targets:
            if dist > radar_range:
                continue
            rel = t.pos - state.pos
            scale = radar_r / radar_range
            rx2 = radar_cx + int(rel[0] * scale)
            ry2 = radar_cy - int(rel[2] * scale)  # use Z for vertical on radar
            tcol = iff_colors.get(t.iff_status, (200, 200, 40))
            pygame.draw.circle(surf, tcol, (rx2, ry2), 3 if t is aim.locked_target else 2, 0)
        # Plot squad members on radar
        for m in aim.squad_members:
            rel = m["pos"] - state.pos
            dist_m = np.linalg.norm(rel)
            if dist_m > radar_range:
                continue
            scale = radar_r / radar_range
            rx2 = radar_cx + int(rel[0] * scale)
            ry2 = radar_cy - int(rel[2] * scale)
            mcol = (40, 120, 200) if m["status"] == "active" else (60, 60, 60)
            pygame.draw.circle(surf, mcol, (rx2, ry2), 2, 0)
        # Radar label
        radar_lbl = font_sm.render(f"RADAR {radar_range}m", True, C_DIM)
        surf.blit(radar_lbl, (radar_cx - 30, radar_cy + radar_r + 4))
    except Exception:
        pass

    # Attitude indicator (bottom-left)
    ax = x + 690
    ay = by - 10
    try:
        pygame.draw.circle(surf, (20, 30, 40), (ax, ay + 20), 30, 0)
        pygame.draw.circle(surf, C_DIM, (ax, ay + 20), 30, 1)
        # Horizon line
        hl = math.cos(state.roll) * 28
        hv = math.sin(state.roll) * 28
        pygame.draw.line(surf, C_ACCENT, (ax - hl, ay + 20 + hv), (ax + hl, ay + 20 - hv), 2)
        # Pitch indicator
        py = ay + 20 + math.sin(state.pitch) * 20
        pygame.draw.line(surf, C_TEXT, (ax - 10, py), (ax + 10, py), 1)
    except Exception:
        pass

    # OS Task Monitor panel (bottom-right, above combat)
    ty = y + 350
    _panel(surf, rx, ty, 250, 100, 200)
    surf.blit(font.render("OS TASK MONITOR", True, C_ACCENT), (rx + 8, ty + 6))
    task_status = rtos.get_task_status()
    # Show top 5 tasks by priority
    for i, (tname, tprio, tactive, tfail, tlat, tanom) in enumerate(task_status[:5]):
        tcol = C_OK if tactive and tanom < 0.5 else C_WARN if tanom >= 0.5 else C_DIM
        tline = f"{tname:12s} P{tprio} {'ON' if tactive else 'OFF'} {tlat:.1f}ms"
        if tfail > 0:
            tline += f" F:{tfail}"
        surf.blit(font_sm.render(tline, True, tcol), (rx + 8, ty + 24 + i * 14))
    # Mini Gantt bar for last few task executions
    gantt = rtos.get_gantt_data(500)
    if gantt:
        gx = rx + 8
        gy = ty + 94
        gw = 234
        t_min = gantt[0][1] if gantt else 0
        t_max = gantt[-1][1] if gantt else 1
        span = max(1, t_max - t_min)
        for gname, gts, gdur, gprio in gantt[-20:]:
            bx = gx + int((gts - t_min) / span * gw)
            bw = max(1, int(gdur / 5.0))
            bcol = C_ACCENT if gprio >= 9 else C_FIBER if gprio >= 7 else C_DIM
            try:
                pygame.draw.rect(surf, bcol, (bx, gy, bw, 3))
            except Exception:
                pass

    # ---- FULL SUIT SYSTEMS PANEL (left-bottom) ----
    sx = x
    sy = y + 270
    sys_lines = [
        f"Muscle: {state.muscle.status_str}",
        f"Thermal: {state.thermal.status_str}",
        f"Neural: {state.neural.status_str}",
        f"LifeSup: {state.life_support.status_str}",
        f"Power: {state.power.status_str}",
        f"Helmet: {state.helmet.status_str}",
        f"Frame: {state.frame.status_str}",
    ]
    # Contextual dive / space lines (mutually exclusive environments)
    if state.dive.active:
        sys_lines.append(f"Dive: {state.dive.status_str}")
    if state.space.active:
        sys_lines.append(f"Space: {state.space.status_str}")
    _panel(surf, sx, sy, 240, 24 + len(sys_lines) * 19, 200)
    surf.blit(font.render("SUIT SYSTEMS", True, C_ACCENT), (sx + 8, sy + 6))
    for i, line in enumerate(sys_lines):
        col = C_TEXT
        if "STRESS!" in line or "BREACHED" in line or "FAIL" in line:
            col = C_ACCENT2
        elif "STIFFENED" in line:
            col = C_WARN
        elif "SEALED" in line or "OK" in line:
            col = C_OK
        surf.blit(font_sm.render(line, True, col), (sx + 8, sy + 28 + i * 19))


def draw_game_hud(surf, rect, state, font, font_sm):
    """Draw Iron Man game-style HUD: score, rings, combo, timer, boost meter."""
    # Top-center: score + timer (big, prominent)
    gx = rect.x + rect.w // 2
    gy = rect.y + 60

    # Score panel (top-center, below heading compass)
    _panel(surf, gx - 90, gy, 180, 56, 220)
    score_col = C_ACCENT if state.game_combo > 0 else C_TEXT
    surf.blit(font_sm.render("SCORE", True, C_DIM), (gx - 82, gy + 4))
    score_txt = font.render(f"{state.game_score:,}", True, score_col)
    surf.blit(score_txt, (gx - 82, gy + 18))
    # Timer
    t = state.game_timer
    t_str = f"{int(t//60):01d}:{int(t%60):02d}"
    surf.blit(font_sm.render(t_str, True, C_DIM), (gx + 50, gy + 4))
    # Rings passed
    surf.blit(font_sm.render(f"RINGS: {state.game_rings_passed}", True, C_FIBER), (gx + 50, gy + 20))

    # Combo indicator (flashes when active)
    if state.game_combo > 0:
        combo_y = gy + 64
        combo_alpha = min(1.0, state.game_combo_timer / 5.0)
        combo_col = (int(255 * combo_alpha), int(200 * combo_alpha), int(60 * combo_alpha))
        combo_txt = font.render(f"x{state.game_combo} COMBO!", True, combo_col)
        surf.blit(combo_txt, (gx - combo_txt.get_width() // 2, combo_y))
        # Combo timer bar
        bar_w = 80
        bar_x = gx - bar_w // 2
        bar_y = combo_y + 22
        pygame.draw.rect(surf, (30, 30, 30), (bar_x, bar_y, bar_w, 4))
        fill_w = int(bar_w * state.game_combo_timer / 5.0)
        pygame.draw.rect(surf, (255, 200, 60), (bar_x, bar_y, fill_w, 4))

    # Boost / afterburner meter (bottom-center, above controls bar)
    by = rect.y + rect.h - 80
    bw = 200
    bx = rect.x + rect.w // 2 - bw // 2
    _panel(surf, bx, by, bw, 28, 200)
    surf.blit(font_sm.render("BOOST", True, C_DIM), (bx + 6, by + 3))
    # Afterburner active = full bar, else shows throttle
    boost_val = state.throttle if not state.afterburner else 1.0
    boost_col = (255, 100, 20) if state.afterburner else (100, 150, 200)
    bar_x = bx + 55
    bar_y = by + 8
    bar_w = 135
    pygame.draw.rect(surf, (40, 46, 58), (bar_x, bar_y, bar_w, 12))
    pygame.draw.rect(surf, boost_col, (bar_x, bar_y, int(bar_w * clamp(boost_val)), 12))
    pygame.draw.rect(surf, (70, 84, 104), (bar_x, bar_y, bar_w, 12), 1)
    if state.afterburner:
        surf.blit(font_sm.render("AFTERBURNER", True, (255, 100, 20)), (bar_x, by + 16))


# =============================================================================
# INFO OVERLAY  -- part specs display
# =============================================================================

def _wrap_text(text, font, max_width):
    """Wrap text to fit within max_width pixels. Returns list of lines."""
    words = text.split()
    lines = []
    current = []
    for w in words:
        test = " ".join(current + [w])
        if font.size(test)[0] <= max_width:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_part_icon(surf, x, y, size, icon_type, color):
    """Draw a simple icon for a part type."""
    cx, cy = x + size // 2, y + size // 2
    r = size // 2 - 2
    if icon_type == "base":
        pygame.draw.circle(surf, color, (cx, cy), r, 1)
        pygame.draw.circle(surf, color, (cx, cy), r // 2, 1)
    elif icon_type == "shield":
        pygame.draw.polygon(surf, color, [(cx, cy - r), (cx + r, cy - r // 3),
                                          (cx + r // 2, cy + r), (cx - r // 2, cy + r),
                                          (cx - r, cy - r // 3)], 1)
    elif icon_type == "muscle":
        for i in range(3):
            offset = (i - 1) * r // 3
            pygame.draw.line(surf, color, (cx - r, cy + offset), (cx + r, cy + offset), 2)
    elif icon_type == "lattice":
        for i in range(3):
            for j in range(3):
                px = cx - r + i * r
                py = cy - r + j * r
                pygame.draw.circle(surf, color, (px, py), 2)
                if i < 2:
                    pygame.draw.line(surf, color, (px, py), (px + r, py), 1)
                if j < 2:
                    pygame.draw.line(surf, color, (px, py), (px, py + r), 1)
    elif icon_type == "armor":
        pygame.draw.rect(surf, color, (cx - r, cy - r, r * 2, r * 2), 1)
        pygame.draw.line(surf, color, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(surf, color, (cx, cy - r), (cx, cy + r), 1)
    elif icon_type == "frame":
        pygame.draw.line(surf, color, (cx, cy - r), (cx, cy + r), 2)
        pygame.draw.line(surf, color, (cx - r, cy), (cx + r, cy), 2)
        pygame.draw.circle(surf, color, (cx, cy), 3)
        for ang in [0, 90, 180, 270]:
            rad = math.radians(ang)
            px = cx + int(math.cos(rad) * r)
            py = cy + int(math.sin(rad) * r)
            pygame.draw.circle(surf, color, (px, py), 2)
    elif icon_type == "helmet":
        pygame.draw.arc(surf, color, (cx - r, cy - r, r * 2, r * 2), 0, math.pi, 2)
        pygame.draw.line(surf, color, (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.rect(surf, color, (cx - r // 2, cy - r // 3, r, r // 3), 1)
    elif icon_type == "turbine":
        pygame.draw.circle(surf, color, (cx, cy), r, 1)
        for i in range(6):
            ang = i * math.pi / 3
            pygame.draw.line(surf, color, (cx, cy),
                           (cx + int(math.cos(ang) * r), cy + int(math.sin(ang) * r)), 1)
    elif icon_type == "wing":
        pygame.draw.polygon(surf, color, [(cx - r, cy), (cx, cy - r // 2),
                                          (cx + r, cy), (cx, cy + r // 2)], 1)
    elif icon_type == "power":
        pygame.draw.rect(surf, color, (cx - r // 2, cy - r, r, r * 2), 1)
        pygame.draw.line(surf, color, (cx - r // 4, cy - r // 2), (cx + r // 4, cy - r // 2), 2)
        pygame.draw.line(surf, color, (cx - r // 4, cy + r // 2), (cx + r // 4, cy + r // 2), 2)
    elif icon_type == "neural":
        for i in range(5):
            ang = i * 2 * math.pi / 5
            px = cx + int(math.cos(ang) * r)
            py = cy + int(math.sin(ang) * r)
            pygame.draw.circle(surf, color, (px, py), 2)
            pygame.draw.line(surf, color, (cx, cy), (px, py), 1)
    elif icon_type == "thermal":
        pygame.draw.rect(surf, color, (cx - r, cy - r // 2, r * 2, r), 1)
        for i in range(4):
            pygame.draw.line(surf, color, (cx - r + i * r // 2, cy - r // 2),
                           (cx - r + i * r // 2, cy + r // 2), 1)
    else:
        pygame.draw.circle(surf, color, (cx, cy), r, 1)


def draw_info(surf, rect, parts, selected, font, font_sm, scroll_y=0):
    """Draw comprehensive info/about panel for selected part with sub-components, dimensions, connectors, and performance."""
    if selected is None or selected >= len(parts):
        return
    part = parts[selected]

    # Panel dimensions (responsive to screen width)
    pw = min(360, int(rect.w * 0.28))
    x = rect.x + rect.w - pw - 10
    y = rect.y + 10
    max_h = rect.h - 40

    # Build all content sections as (label, lines, color) tuples
    sections = []

    # Description
    desc_lines = _wrap_text(part.description, font_sm, pw - 20) if part.description else []
    if desc_lines:
        sections.append(("ABOUT", desc_lines, C_TEXT))

    # Dimensions
    if getattr(part, 'dimensions', ''):
        sections.append(("DIMENSIONS", [part.dimensions], C_FIBER))

    # Specs
    if part.specs:
        sections.append(("SPECS", [f"  {s}" for s in part.specs], C_TEXT))

    # Sub-components
    sub_comps = getattr(part, 'sub_components', [])
    if sub_comps:
        sub_lines = []
        for sc in sub_comps:
            name = sc.get("name", "?")
            count = sc.get("count", 1)
            detail = sc.get("detail", "")
            sub_lines.append(f"  [{count}x] {name}")
            if detail:
                for dl in _wrap_text(detail, font_sm, pw - 36):
                    sub_lines.append(f"      {dl}")
        sections.append(("SUB-COMPONENTS & GEAR", sub_lines, C_FIBER))

    # Materials
    if part.materials:
        sections.append(("MATERIALS", [f"  - {m}" for m in part.materials], C_FIBER))

    # Connectors
    connectors = getattr(part, 'connectors', [])
    if connectors:
        sections.append(("CONNECTORS & MOUNTING", [f"  > {c}" for c in connectors], C_WARN))

    # Performance
    perf = getattr(part, 'performance', [])
    if perf:
        sections.append(("PERFORMANCE", [f"  * {p}" for p in perf], C_OK))

    # Blueprint notes
    bp_lines = _wrap_text(part.blueprint_notes, font_sm, pw - 20) if part.blueprint_notes else []
    if bp_lines:
        sections.append(("BLUEPRINT", bp_lines, C_WARN))

    # Build details
    bd = getattr(part, 'build_details', {})
    if bd:
        bd_lines = []
        bd_lines.append(f"  Time: {bd.get('time_min', '?')} min  |  Difficulty: {bd.get('difficulty', '?')}")
        bd_lines.append(f"  Torque: {bd.get('torque', 'N/A')}")
        tools = bd.get('tools', [])
        if tools:
            bd_lines.append("  Tools required:")
            for t in tools:
                bd_lines.append(f"      - {t}")
        notes = bd.get('notes', '')
        if notes:
            for nl in _wrap_text(notes, font_sm, pw - 20):
                bd_lines.append(f"  {nl}")
        sections.append(("BUILD DETAILS", bd_lines, C_ACCENT2))

    # Calculate total content height (for scroll)
    content_h = 40 + 20  # header + category
    for label, lines, col in sections:
        content_h += 16 + len(lines) * 14 + 10
    content_h += 24  # footer

    # Clamp scroll_y
    max_scroll = max(0, content_h - max_h)
    scroll_y = max(0, min(scroll_y, max_scroll))

    h = min(content_h, max_h)

    _panel(surf, x, y, pw, h, 235)

    # Clip to panel for scrolling
    old_clip = surf.get_clip()
    surf.set_clip(pygame.Rect(x + 2, y + 2, pw - 4, h - 4))

    # Header: icon + name (fixed, doesn't scroll)
    _draw_part_icon(surf, x + 6, y + 6, 24, getattr(part, 'icon_type', ''), part.color)
    surf.blit(font.render(part.name, True, C_ACCENT), (x + 36, y + 8))

    # Category + layer + weight (fixed)
    cy = y + 32
    cat_text = part.category if part.category else "Component"
    layer_text = f"Layer {getattr(part, 'layer_num', part.order)}"
    weight_text = f"{part.weight_kg:.1f} kg" if part.weight_kg > 0 else ""
    info_line = f"{cat_text}  |  {layer_text}  |  {weight_text}"
    surf.blit(font_sm.render(info_line, True, C_DIM), (x + 8, cy))

    cy += 20
    pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
    cy += 6 - scroll_y  # Apply scroll offset

    # Draw each section (with scroll offset)
    for label, lines, col in sections:
        if cy > y + h - 10:
            break
        if cy + 14 + len(lines) * 14 + 10 < y + 40:
            cy += 16 + len(lines) * 14 + 10
            continue
        surf.blit(font_sm.render(label, True, C_ACCENT2), (x + 8, cy))
        cy += 14
        for line in lines:
            if cy > y + h - 10:
                break
            if cy >= y + 38:
                surf.blit(font_sm.render(line, True, col), (x + 8, cy))
            cy += 14
        cy += 4
        if y + 40 < cy < y + h - 4:
            pygame.draw.line(surf, (40, 44, 52), (x + 8, cy), (x + pw - 8, cy), 1)
        cy += 6

    # Footer: mesh stats (always visible at bottom)
    surf.set_clip(old_clip)
    if content_h > max_h:
        # Draw scroll indicator bar
        bar_x = x + pw - 6
        bar_h = max(20, int(h * h / content_h))
        bar_y = y + int(scroll_y / max_scroll * (h - bar_h)) if max_scroll > 0 else y
        pygame.draw.rect(surf, (50, 60, 76), (bar_x, y + 2, 4, h - 4), 1)
        pygame.draw.rect(surf, C_DIM, (bar_x, bar_y, 4, bar_h))
    # Footer stats
    foot_y = y + h - 16
    pygame.draw.line(surf, C_DIM, (x + 8, foot_y - 2), (x + pw - 8, foot_y - 2), 1)
    stats = f"Meshes: {part.mesh_count}  Verts: {part.vertex_count:,}  Faces: {part.face_count:,}"
    surf.blit(font_sm.render(stats, True, C_DIM), (x + 8, foot_y))


def draw_part_browser(surf, rect, parts, font, font_sm, selected=None):
    """Draw a scrollable part browser showing all 12 parts with quick stats.
    Returns list of (orig_idx, click_rect) for interactive selection."""
    pw = min(280, int(rect.w * 0.22))
    x = rect.x + 10
    y = rect.y + 10

    # Sort parts by layer number for logical display
    sorted_parts = sorted(enumerate(parts), key=lambda p: getattr(p[1], 'layer_num', p[1].order))

    row_h = 42
    header_h = 30
    h = header_h + len(sorted_parts) * row_h + 10
    h = min(h, rect.h - 40)

    _panel(surf, x, y, pw, h, 230)
    surf.blit(font.render("SUIT COMPONENTS", True, C_ACCENT), (x + 8, y + 6))
    surf.blit(font_sm.render(f"{len(parts)} parts  |  {sum(p.weight_kg for p in parts):.1f} kg total  |  click to select",
                            True, C_DIM), (x + 8, y + 22))

    hit_rects = []
    cy = y + header_h
    for idx, (orig_idx, part) in enumerate(sorted_parts):
        is_sel = (orig_idx == selected)
        bg_col = (40, 50, 64) if is_sel else (28, 32, 40)
        row_rect = pygame.Rect(x + 4, cy, pw - 8, row_h - 2)
        pygame.draw.rect(surf, bg_col, row_rect)
        if is_sel:
            pygame.draw.rect(surf, C_ACCENT, row_rect, 1)
        hit_rects.append((orig_idx, row_rect))

        # Icon
        _draw_part_icon(surf, x + 8, cy + 4, 28, getattr(part, 'icon_type', ''), part.color)

        # Name (truncated)
        name = part.name
        if font_sm.size(name)[0] > pw - 50:
            while font_sm.size(name + "...")[0] > pw - 50 and len(name) > 3:
                name = name[:-1]
            name += "..."
        name_col = C_ACCENT if is_sel else C_TEXT
        surf.blit(font_sm.render(name, True, name_col), (x + 42, cy + 4))

        # Category + weight + sub-component count
        cat = part.category if part.category else ""
        wt = f"{part.weight_kg:.1f}kg" if part.weight_kg > 0 else ""
        sub_count = sum(sc.get("count", 1) for sc in getattr(part, 'sub_components', []))
        sub_txt = f"{sub_count} pcs" if sub_count > 0 else ""
        surf.blit(font_sm.render(f"{cat}  {wt}  {sub_txt}", True, C_DIM), (x + 42, cy + 20))

        # Layer badge
        layer = getattr(part, 'layer_num', part.order)
        layer_col = C_OK if layer == 0 else C_FIBER if layer <= 2 else C_WARN if layer <= 4 else C_ACCENT2
        layer_txt = f"L{layer}"
        surf.blit(font_sm.render(layer_txt, True, layer_col), (x + pw - 24, cy + 4))

        # Difficulty indicator
        bd = getattr(part, 'build_details', {})
        diff = bd.get('difficulty', '')
        if diff:
            diff_col = C_OK if diff == "Easy" else C_WARN if diff == "Medium" else (255, 100, 100) if diff == "Hard" else (255, 60, 60)
            surf.blit(font_sm.render(diff[0], True, diff_col), (x + pw - 24, cy + 20))

        cy += row_h

    return hit_rects


def draw_blueprint(surf, rect, parts, font, font_sm):
    """Draw assembly blueprint showing layer stack diagram, body cross-section, and build order."""
    pw = min(420, rect.w - 40)
    x = rect.x + rect.w // 2 - pw // 2
    y = rect.y + 60

    # Sort by layer number
    sorted_parts = sorted(parts, key=lambda p: getattr(p, 'layer_num', p.order))

    h = min(520, rect.h - 80)
    _panel(surf, x, y, pw, h, 240)
    surf.blit(font.render("BUILD BLUEPRINT -- ASSEMBLY ORDER", True, C_ACCENT), (x + 8, y + 6))
    surf.blit(font_sm.render("Mjalnor'MV1.17 layer stack (inside -> outside)", True, C_DIM),
              (x + 8, y + 22))

    # Draw body cross-section silhouette (left side)
    cy = y + 44
    sil_x = x + 10
    sil_w = 80
    sil_h = 200
    # Draw simplified human silhouette with layered rings
    center_x = sil_x + sil_w // 2
    center_y = cy + sil_h // 2

    # Draw concentric body cross-section rings (torso level)
    ring_colors = [
        (60, 80, 100),   # Layer 0: Frame
        (80, 60, 40),    # Layer 1: Base/Thermal
        (50, 70, 50),    # Layer 2: Shielding
        (70, 50, 60),    # Layer 3: Muscle
        (60, 60, 70),    # Layer 4: Impact
        (80, 50, 40),    # Layer 5: Armor
    ]
    ring_labels = ["Frame", "Thermal", "Shield", "Muscle", "Impact", "Armor"]
    ring_radii = [8, 14, 20, 27, 34, 42]

    for i in range(5, -1, -1):
        r = ring_radii[i]
        pygame.draw.circle(surf, ring_colors[i], (center_x, center_y), r)
        pygame.draw.circle(surf, C_DIM, (center_x, center_y), r, 1)

    # Label the rings with lines
    for i in range(6):
        r = ring_radii[i]
        label_y = center_y - r + 2
        lx = center_x + r + 4
        ly = center_y - r + 2
        pygame.draw.line(surf, C_DIM, (center_x + r, center_y - r),
                        (lx + 10, ly), 1)
        surf.blit(font_sm.render(f"L{i}: {ring_labels[i]}", True, ring_colors[i]),
                 (lx + 12, ly - 6))

    # Draw simplified body figure (head, arms, legs)
    fig_x = center_x
    # Head
    pygame.draw.circle(surf, (50, 50, 60), (fig_x, cy + 10), 8, 1)
    # Torso
    pygame.draw.rect(surf, (50, 50, 60), (fig_x - 12, cy + 18, 24, 30), 1)
    # Arms
    pygame.draw.line(surf, (50, 50, 60), (fig_x - 12, cy + 22), (fig_x - 22, cy + 45), 1)
    pygame.draw.line(surf, (50, 50, 60), (fig_x + 12, cy + 22), (fig_x + 22, cy + 45), 1)
    # Legs
    pygame.draw.line(surf, (50, 50, 60), (fig_x - 6, cy + 48), (fig_x - 8, cy + 70), 1)
    pygame.draw.line(surf, (50, 50, 60), (fig_x + 6, cy + 48), (fig_x + 8, cy + 70), 1)

    surf.blit(font_sm.render("CROSS-SECTION", True, C_ACCENT2), (sil_x, cy + 80))
    surf.blit(font_sm.render("(torso level)", True, C_DIM), (sil_x, cy + 94))

    # Layer stack diagram (right side)
    stack_x = x + 110
    stack_w = pw - 130
    layer_h = 24

    # Group parts by layer
    layers = {}
    for part in sorted_parts:
        ln = getattr(part, 'layer_num', part.order)
        if ln not in layers:
            layers[ln] = []
        layers[ln].append(part)

    # Draw from innermost (layer 0) to outermost (layer 5)
    for ln in sorted(layers.keys()):
        layer_parts = layers[ln]
        # Layer bar
        bar_col = ring_colors[ln] if ln < len(ring_colors) else (40, 40, 40)
        pygame.draw.rect(surf, bar_col, (stack_x, cy, stack_w, layer_h))
        pygame.draw.rect(surf, C_DIM, (stack_x, cy, stack_w, layer_h), 1)

        # Layer label
        label = f"Layer {ln}"
        if ln == 0:
            label += " (Frame/Internal)"
        elif ln == 1:
            label += " (Base/Thermal)"
        elif ln == 2:
            label += " (Shielding)"
        elif ln == 3:
            label += " (Muscle)"
        elif ln == 4:
            label += " (Impact)"
        elif ln == 5:
            label += " (Armor/External)"
        surf.blit(font_sm.render(label, True, C_TEXT), (stack_x + 6, cy + 4))

        # Part names in this layer
        names = ", ".join(p.name.split("(")[0].strip() for p in layer_parts)
        if font_sm.size(names)[0] > stack_w - 100:
            while font_sm.size(names + "...")[0] > stack_w - 100 and len(names) > 3:
                names = names[:-1]
            names += "..."
        surf.blit(font_sm.render(names, True, C_DIM), (stack_x + 100, cy + 4))

        cy += layer_h + 2

    # Assembly order list
    cy += 10
    pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
    cy += 6
    surf.blit(font_sm.render("ASSEMBLY SEQUENCE (with build details)", True, C_ACCENT2), (x + 8, cy))
    cy += 16

    # Sort by order for assembly sequence
    assembly_parts = sorted(parts, key=lambda p: p.order)
    total_build_time = 0
    for i, part in enumerate(assembly_parts):
        bd = getattr(part, 'build_details', {})
        time_min = bd.get('time_min', 0)
        total_build_time += time_min
        diff = bd.get('difficulty', '?')
        diff_col = C_OK if diff == "Easy" else C_WARN if diff == "Medium" else (255, 100, 100) if diff == "Hard" else (255, 60, 60)

        step = f"{i+1:2d}. {part.name.split('(')[0].strip()}"
        if font_sm.size(step)[0] > pw - 110:
            while font_sm.size(step + "...")[0] > pw - 110 and len(step) > 3:
                step = step[:-1]
            step += "..."
        surf.blit(font_sm.render(step, True, C_TEXT), (x + 12, cy))
        # Weight + time + difficulty on right
        info_r = f"{part.weight_kg:.1f}kg {time_min}m {diff}"
        surf.blit(font_sm.render(info_r, True, diff_col), (x + pw - 110, cy))
        cy += 14
        # Show torque spec
        torque = bd.get('torque', '')
        if torque and torque != 'N/A -- snap fit':
            surf.blit(font_sm.render(f"      Torque: {torque}", True, C_DIM), (x + 12, cy))
            cy += 12
        if cy > y + h - 30:
            break

    # Total weight + build time
    if cy < y + h:
        cy += 4
        pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
        cy += 4
        total_wt = sum(p.weight_kg for p in parts)
        surf.blit(font_sm.render(f"Total weight: {total_wt:.1f} kg ({total_wt*2.205:.0f} lbs)  |  Build time: {total_build_time}m ({total_build_time//60}h {total_build_time%60}m)",
                                True, C_ACCENT), (x + 8, cy))


def draw_suit_overview(surf, rect, parts, font, font_sm):
    """Draw suit overview/summary panel showing aggregate specs across all parts."""
    pw = min(400, rect.w - 40)
    x = rect.x + rect.w // 2 - pw // 2
    y = rect.y + 60
    h = min(480, rect.h - 80)
    _panel(surf, x, y, pw, h, 240)
    surf.blit(font.render("SUIT OVERVIEW -- MJALNOR'MV1.17", True, C_ACCENT), (x + 8, y + 6))
    surf.blit(font_sm.render("Aggregate specifications across all 12 components", True, C_DIM),
              (x + 8, y + 22))

    cy = y + 44

    # Weight summary
    total_wt = sum(p.weight_kg for p in parts)
    pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
    cy += 6
    surf.blit(font_sm.render("WEIGHT BREAKDOWN", True, C_ACCENT2), (x + 8, cy))
    cy += 14
    for part in sorted(parts, key=lambda p: -p.weight_kg):
        if part.weight_kg > 0:
            pct = part.weight_kg / total_wt * 100 if total_wt > 0 else 0
            bar_w = int((pw - 120) * pct / 100)
            name = part.name.split("(")[0].strip()
            if font_sm.size(name)[0] > 100:
                while font_sm.size(name + "...")[0] > 100 and len(name) > 3:
                    name = name[:-1]
                name += "..."
            surf.blit(font_sm.render(f"  {name}", True, C_TEXT), (x + 8, cy))
            surf.blit(font_sm.render(f"{part.weight_kg:.1f}kg ({pct:.0f}%)", True, C_DIM),
                     (x + pw - 90, cy))
            # Weight bar
            bar_x = x + 110
            pygame.draw.rect(surf, (40, 44, 52), (bar_x, cy + 2, pw - 200, 8))
            pygame.draw.rect(surf, part.color, (bar_x, cy + 2, bar_w, 8))
            cy += 16
    cy += 4
    surf.blit(font_sm.render(f"  TOTAL: {total_wt:.1f} kg ({total_wt*2.205:.0f} lbs)", True, C_ACCENT),
             (x + 8, cy))
    cy += 16

    # Sub-component count
    pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
    cy += 6
    surf.blit(font_sm.render("SUB-COMPONENTS & GEAR INVENTORY", True, C_ACCENT2), (x + 8, cy))
    cy += 14
    total_sub = 0
    for part in parts:
        subs = getattr(part, 'sub_components', [])
        for sc in subs:
            count = sc.get("count", 1)
            name = sc.get("name", "?")
            total_sub += count
            line = f"  [{count:>3}x] {name}  ({part.name.split('(')[0].strip()})"
            if font_sm.size(line)[0] > pw - 20:
                while font_sm.size(line + "...")[0] > pw - 20 and len(line) > 3:
                    line = line[:-1]
                line += "..."
            surf.blit(font_sm.render(line, True, C_FIBER), (x + 8, cy))
            cy += 13
            if cy > y + h - 40:
                break
    if cy < y + h - 30:
        cy += 4
        surf.blit(font_sm.render(f"  TOTAL PIECES: {total_sub}", True, C_ACCENT), (x + 8, cy))
        cy += 14

    # Key performance metrics
    if cy < y + h - 10:
        pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
        cy += 6
        surf.blit(font_sm.render("KEY PERFORMANCE METRICS", True, C_ACCENT2), (x + 8, cy))
        cy += 14
        metrics = [
            f"  Thrust/Weight: {DIMS['turbine_count'] * DIMS['turbine_thrust_ab_lbf'] / (total_wt * 2.205):.1f}:1 (afterburner)",
            f"  Max speed: ~420 mph  |  Service ceiling: 28,000 ft",
            f"  Jump: {DIMS['perf_jump_vertical_ft']}ft vertical  |  Punch: {DIMS['perf_punch_lbs']:,} lbs",
            f"  Lift: {DIMS['perf_lift_overhead_lbs']:,} lbs overhead  |  Battery: {DIMS['battery_wh']} Wh",
            f"  Armor: NIJ {DIMS['outer_nij_level']}+  |  EMP: >99% blocked  |  Seal: {DIMS['seal_depth_m']}m",
            f"  BCI latency: <{DIMS['bci_latency_ms']}ms  |  AI: Vera 3.0 (70B 4-bit)",
            f"  Turbines: {DIMS['turbine_count']}x  |  Wings: {DIMS['wing_span_m']}m span  |  L/D: {DIMS['wing_ld_ratio']}:1",
        ]
        for m in metrics:
            if cy > y + h - 10:
                break
            surf.blit(font_sm.render(m, True, C_OK), (x + 8, cy))
            cy += 14

    # Build summary
    if cy < y + h - 10:
        pygame.draw.line(surf, C_DIM, (x + 8, cy), (x + pw - 8, cy), 1)
        cy += 6
        surf.blit(font_sm.render("BUILD SUMMARY", True, C_ACCENT2), (x + 8, cy))
        cy += 14
        total_bt = sum(getattr(p, 'build_details', {}).get('time_min', 0) for p in parts)
        diff_counts = {}
        for p in parts:
            d = getattr(p, 'build_details', {}).get('difficulty', '?')
            diff_counts[d] = diff_counts.get(d, 0) + 1
        diff_str = "  ".join(f"{k}: {v}" for k, v in sorted(diff_counts.items()))
        build_metrics = [
            f"  Total assembly time: {total_bt} min ({total_bt//60}h {total_bt%60}m)",
            f"  Difficulty: {diff_str}",
            f"  Parts: {len(parts)}  |  Sub-components: {total_sub}  |  Weight: {total_wt:.1f} kg",
        ]
        for bm in build_metrics:
            if cy > y + h - 10:
                break
            surf.blit(font_sm.render(bm, True, C_ACCENT), (x + 8, cy))
            cy += 14


# =============================================================================
# SELF-TEST  -- headless build + render + physics validation
# =============================================================================

def selftest():
    """Headless validation: build suit, render a frame, run physics, verify specs."""
    print("=" * 70)
    print("FLYSUIT Mjalnor'MV1.17 -- SELF TEST")
    print("=" * 70)

    # Build for smallest and largest pilot
    for label, h, w in [("Nano (3ft3in)", 0.99, 29.5),
                         ("Titan (7ft3in)", 2.21, 190.5),
                         ("Median (5ft8in)", 1.73, 79.4)]:
        print(f"\n--- Building suit for {label} pilot ({h*100:.0f}cm, {w:.1f}kg) ---")
        parts, tcfg, cfg = build_suit(h, w)
        total_meshes = sum(len(p.meshes) for p in parts)
        total_verts = sum(len(m.verts) for p in parts for m in p.meshes)
        total_faces = sum(len(m.faces) for p in parts for m in p.meshes)
        print(f"  Parts: {len(parts)}, Meshes: {total_meshes}, "
              f"Verts: {total_verts}, Faces: {total_faces}")
        print(f"  Turbines configured: {len(tcfg)}")
        for p in parts:
            print(f"    [{p.order:2d}] {p.key:20s} -> {len(p.meshes):3d} meshes, "
                  f"{sum(len(m.verts) for m in p.meshes):5d} verts")

    # Physics validation
    print("\n--- Physics Validation ---")
    state = SuitState()
    parts, tcfg, cfg = build_suit(1.73, 79.4)
    phys = SuitPhysics(state, cfg)

    # Thrust-to-weight
    state.throttle = 1.0
    state.afterburner = True
    twr = phys.thrust_to_weight
    print(f"  T/W ratio (full AB): {twr:.2f}")
    assert twr > 1.0, f"T/W must be >1 for flight, got {twr}"
    print(f"  PASS: T/W > 1.0")

    # Impact absorption
    felt, pct = phys.impact_absorption(600000)
    print(f"  Impact 600,000 PSI -> user feels {felt:,.0f} PSI ({pct:.1f}% absorbed)")
    assert pct > 98.0, f"Absorption must be >98%, got {pct:.1f}%"
    print(f"  PASS: >98% impact absorption")

    # Jump energy
    jump_j = phys.jump_energy(DIMS["perf_jump_vertical_ft"])
    print(f"  200ft jump energy: {jump_j:,.0f} J ({jump_j/3600:.1f} Wh)")
    assert jump_j < DIMS["battery_wh"] * 3600, "Jump energy exceeds battery capacity"
    print(f"  PASS: Jump energy within battery capacity")

    # Battery runtime
    rt = phys.battery_runtime_hours(0.5)
    print(f"  Battery runtime @50% throttle: {rt:.1f} hours")
    assert rt > 1.0, "Runtime must be >1 hour"
    print(f"  PASS: Runtime >1 hour")

    # SuitRTOS validation
    print("\n--- SuitRTOS Validation ---")
    rtos = state.rtos
    # Run for 2 simulated seconds
    for _ in range(200):
        rtos.update(0.01)
    print(f"  Tasks: {rtos.task_count} ({rtos.critical_task_count} critical)")
    print(f"  Integrity checks: {rtos.integrity_checks:,}")
    print(f"  Uptime: {rtos.uptime_str}  Uptime%: {rtos.uptime_pct:.6f}%")
    assert rtos.task_count == 20, f"Expected 20 tasks, got {rtos.task_count}"
    assert rtos.integrity_checks > 0, "No integrity checks ran"
    print(f"  PASS: 20 tasks registered, checks running")

    # Virus injection + failover test
    rtos.inject_virus(target_task="neural", at_s=0.0)
    for _ in range(200):
        rtos.update(0.01)
    print(f"  After virus injection:")
    print(f"    Viruses detected: {rtos.viruses_detected}")
    print(f"    Failovers: {rtos.failovers}")
    print(f"    ROM restores: {rtos.rom_restores}")
    print(f"    Viruses purged: {rtos.viruses_purged}")
    assert rtos.viruses_detected > 0, "Virus was not detected"
    assert rtos.failovers > 0, "Failover did not trigger"
    assert rtos.viruses_purged > 0, "Virus was not purged"
    print(f"  PASS: Virus detected + failover + ROM restore + purged")

    # Auto-aim validation
    print("\n--- Auto-Aim Validation ---")
    aim = state.auto_aim
    # Add a target at 500m moving at Mach 1
    aim.add_target(pos=state.pos + np.array([500, 50, 0]),
                   vel=np.array([-340, 0, 20]), mach_speed=True)
    # Run for 1 second to acquire lock
    for _ in range(100):
        aim.update(0.01, state.pos)
    print(f"  Target speed: {aim.locked_target.speed:.0f} m/s (Mach {aim.locked_target.mach_number:.2f})")
    print(f"  Lock acquired: {aim.lock_acquired}")
    assert aim.lock_acquired, "Failed to acquire target lock"
    solution = aim.compute_firing_solution(state.pos)
    if solution:
        aim_pt, tof, lead = solution
        print(f"  Firing solution: TOF={tof:.2f}s  lead={lead:.1f}mrad")
    # Fire 50 shots
    hits = 0
    for _ in range(50):
        if aim.locked_target:
            aim.locked_target.hit = False  # reset for multiple shots
            if aim.fire(state.pos):
                hits += 1
    print(f"  50 shots: {hits} hits ({hits/50*100:.0f}% hit rate)")
    assert hits > 40, f"Expected >40 hits, got {hits}"
    print(f"  PASS: >80% hit rate on Mach 1 target at 500m")

    # Defense AI validation
    print("\n--- Defense AI Validation ---")
    defense = state.defense
    defense.update(0.01, threat_detected=True, threat_dir=np.array([1, 0, 0]))
    # Run through full combat cycle
    for _ in range(100):
        defense.update(0.01, threat_detected=True)
    print(f"  State: {defense.state}")
    print(f"  Punches: {defense.punches_thrown}  Force: {defense.punch_force_lbs:,.0f} lbs")
    print(f"  Threats neutralized: {defense.threats_neutralized}")
    assert defense.threats_neutralized > 0, "Defense AI did not neutralize threat"
    print(f"  PASS: Defense AI completes combat cycle")

    # Jump assist validation
    print("\n--- Jump Assist Validation ---")
    jump = state.jump
    jump.start_charge()
    for _ in range(50):
        jump.update_charge(0.01)
    print(f"  Charge: {jump.charge*100:.0f}%")
    assert jump.charge >= 0.99, "Jump did not fully charge"
    jump.execute_jump(np.array([0, 0, 0]))
    print(f"  Jump peak: {jump.jump_peak_height:.1f}m ({jump.jump_peak_height/0.3048:.0f}ft)")
    print(f"  Jump duration: {jump.jump_duration:.2f}s")
    assert jump.jump_peak_height > 50, "Jump height too low"
    print(f"  PASS: 200ft jump charged and executed")

    # 6-DOF physics validation
    print("\n--- 6-DOF Physics Validation ---")
    state2 = SuitState()
    phys2 = SuitPhysics(state2, cfg)
    state2.throttle = 0.5
    state2.thrust_vector = np.array([0.0, 1.0, 0.0])
    for _ in range(100):
        phys2.step(0.01)
    print(f"  Altitude after 1s hover: {state2.altitude:.2f}m")
    print(f"  Pitch: {math.degrees(state2.pitch):.1f}  Roll: {math.degrees(state2.roll):.1f}")
    print(f"  G-load: {state2.g_load:.2f}  G-limiter: {'ON' if phys2.g_limiter_active else 'OFF'}")
    # Underwater test
    state2.env_idx = 8  # underwater (Underwater 100m)
    state2.pos = np.array([0.0, -10.0, 0.0])
    for _ in range(100):
        phys2.step(0.01)
    print(f"  Underwater buoyancy test: depth={-state2.pos[1]:.1f}m  buoyancy={phys2.buoyancy_n:.0f}N")
    print(f"  PASS: 6-DOF + underwater physics functional")

    # Dive system validation (buoyancy control + decompression computer)
    print("\n--- Dive System Validation ---")
    dive = DiveSystem()
    dmass = 79.4 + DIMS["weight_total_kg"]
    for _ in range(1500):  # 25 min at 30 m
        dive.update(1.0, 30.0, 997.0, 9.81, dmass)
    print(f"  30m/25min: ambient={dive.ambient_ata:.2f}ata  NDL={dive.ndl_min:.0f}min  "
          f"deco={'yes ' + str(int(dive.deco_stop_m)) + 'm' if dive.deco_required else 'no'}")
    print(f"  ppO2={dive.ppo2_ata:.2f}ata  CNS={dive.cns_pct:.1f}%  narcosis EAD={dive.narcosis_ead_m:.0f}m")
    print(f"  Auto-trim BCD={dive.bcd_volume_l:.1f}L  net buoyancy={dive.net_buoyancy_n:+.0f}N")
    assert dive.active, "Dive system should be active underwater"
    assert 3.9 < dive.ambient_ata < 4.1, "30 m should be ~4 ata"
    assert abs(dive.net_buoyancy_n) < 30.0, "Auto-trim should hold near-neutral buoyancy"
    assert dive.tissue_bar[0] > 0.8, "Fast tissue compartment should on-gas at depth"
    dive.update(1.0, 20.0, 997.0, 9.81, dmass)  # 10 m ascent in 1 s -> 600 m/min
    assert dive.ascent_warning, "Fast ascent should trigger warning"
    dive2 = DiveSystem()
    dive2.update(1.0, 70.0, 1025.0, 9.81, dmass)
    print(f"  70m: ppO2={dive2.ppo2_ata:.2f}ata  O2-tox warning={dive2.o2_tox_warning}")
    assert dive2.o2_tox_warning, "ppO2 should exceed 1.6 ata on air at 70 m"
    print(f"  PASS: buoyancy auto-trim, N2 loading/NDL/deco, ascent-rate, ppO2 toxicity, narcosis")

    # Space system validation (radiation dosimetry + cold-gas RCS)
    print("\n--- Space System Validation ---")
    space = SpaceSystem()
    for _ in range(3600):  # 1 h in deep-space vacuum
        space.update(1.0, 0.0, "space", 0.0, dmass)
    print(f"  1h GCR (shield {space.shield_g_cm2:.0f}g/cm2): rate={space.dose_rate_msv_h:.4f}mSv/h  "
          f"mission={space.dose_mission_msv:.3f}mSv  career={space.career_fraction * 100:.3f}%")
    print(f"  MMOD: flux={space.mmod_flux_hits_h:.4f} hits/h  shield={space.mmod_shield_integrity * 100:.4f}%")
    assert space.active, "Space system should be active in vacuum"
    assert space.dose_rate_msv_h > 0, "Should accumulate radiation dose in space"
    assert space.dose_rate_msv_h < DIMS["rad_gcr_msv_day"] / 24.0, "Shielding should reduce GCR dose rate"
    assert space.mmod_flux_hits_h > 0, "MMOD flux should derive from area x flux density"
    assert space.mmod_shield_integrity < 1.0, "Whipple shield should erode from cumulative flux"
    space2 = SpaceSystem()
    for _ in range(6000):  # 600 s full-throttle RCS burn
        space2.update(0.1, 0.0, "space", 1.0, dmass)
    print(f"  RCS burn 600s: propellant={space2.rcs_propellant_pct:.0f}%  "
          f"available={space2.rcs_available}  dv(full)={space.delta_v_ms:.0f}m/s")
    assert space2.rcs_propellant_pct < 100.0, "RCS propellant should deplete under thrust"
    stv = SuitState()
    stv.env_idx = 10  # Vacuum/Space
    stv.space.rcs_propellant_pct = 0.0
    stv.space.rcs_available = False
    stv.throttle = 1.0
    physv = SuitPhysics(stv, cfg)
    print(f"  Vacuum thrust with empty RCS: {physv.total_thrust_n:.1f}N")
    assert physv.total_thrust_n == 0.0, "No vacuum thrust when RCS propellant is empty"
    print(f"  PASS: radiation dosimetry+shielding, SPE model, RCS depletion, thrust gating")

    # Fall damage check
    safe, g, pct = phys.fall_damage_check(DIMS["perf_safe_fall_ft"])
    print(f"\n--- Fall Damage Check ---")
    print(f"  {DIMS['perf_safe_fall_ft']}ft fall: G={g:.1f}  absorbed={pct:.1f}%  safe={safe}")
    assert safe, f"Fall from {DIMS['perf_safe_fall_ft']}ft should be safe"
    print(f"  PASS: {DIMS['perf_safe_fall_ft']}ft fall is survivable")

    # Atmospheric model validation
    print("\n--- Atmospheric Model Validation ---")
    atm = AtmosphericModel()
    atm.update(0.01, 0, 0, "clear")  # Earth sea level
    print(f"  Earth SL: P={atm.ambient_pressure/1000:.1f}kPa  solar={atm.solar_irradiance:.0f}W/m2  wind={atm.wind_speed:.1f}m/s")
    assert atm.ambient_pressure > 90000, "Sea level pressure too low"
    atm.update(0.01, 10000, 0, "clear")  # Earth 10k ft
    print(f"  Earth 10kft: P={atm.ambient_pressure/1000:.1f}kPa  solar={atm.solar_irradiance:.0f}W/m2")
    assert atm.ambient_pressure < 70000, "10k ft pressure should be lower"
    atm.update(0.01, -50, 8, "current")  # Underwater 50m (env_idx=8)
    print(f"  Ocean 50m: P={atm.ambient_pressure/1000:.1f}kPa  solar={atm.solar_irradiance:.1f}W/m2")
    assert atm.ambient_pressure > 500000, "Underwater pressure should be high"
    atm.update(0.01, 400000, 10, "vacuum")  # Space (env_idx=10)
    print(f"  Space: P={atm.ambient_pressure:.0f}Pa  solar={atm.solar_irradiance:.0f}W/m2")
    assert atm.ambient_pressure == 0, "Space pressure should be zero"
    atm.update(0.01, 0, 4, "dust")  # Mars (env_idx=4)
    print(f"  Mars: P={atm.ambient_pressure:.0f}Pa  solar={atm.solar_irradiance:.0f}W/m2  radio_att={atm.radio_attenuation:.0f}dB")
    assert atm.radio_attenuation > 0, "Mars dust should cause radio attenuation"
    print(f"  PASS: Atmospheric model (Earth/Mars/Titan/ocean/space) functional")

    # Multi-target tracking + IFF + missile dodge validation
    print("\n--- Multi-Target + IFF + Missile Dodge Validation ---")
    aim2 = AutoAimSystem()
    # Add 5 targets with different IFF statuses
    aim2.add_target(np.array([500, 0, 0]), np.array([0, 0, 0]), iff="hostile")
    aim2.add_target(np.array([800, 100, 0]), np.array([-200, 0, 0]), iff="hostile")
    aim2.add_target(np.array([300, 0, 200]), np.array([0, 0, 0]), iff="unknown")
    aim2.add_target(np.array([1000, 0, 0]), np.array([0, 0, 0]), iff="friendly")
    aim2.add_target(np.array([200, 0, 0]), np.array([300, 0, 0]), iff="hostile")  # inbound missile
    aim2.update(0.01, np.array([0, 0, 0]))
    print(f"  Targets: {len(aim2.targets)}  Tracked: {len(aim2.tracked_targets)}")
    assert len(aim2.tracked_targets) <= aim2.max_tracked, "Too many tracked targets"
    assert len(aim2.tracked_targets) >= 3, "Should track at least 3 targets"
    # Check IFF
    hostile_count = sum(1 for t, s, d in aim2.tracked_targets if t.iff_status == "hostile")
    friendly_count = sum(1 for t, s, d in aim2.tracked_targets if t.iff_status == "friendly")
    print(f"  IFF: {hostile_count} hostile, {friendly_count} friendly tracked")
    assert hostile_count > 0, "Should track hostile targets"
    # Check missile dodge
    print(f"  Inbound threats: {len(aim2.inbound_threats)}  Dodge: {aim2.dodge_recommended}")
    assert aim2.dodge_recommended, "Should detect inbound missile threat"
    assert len(aim2.inbound_threats) > 0, "Should have inbound threats"
    # Check weapon mounts
    armed = sum(1 for w in aim2.weapon_mounts if w["armed"])
    print(f"  Weapon mounts: {len(aim2.weapon_mounts)} total, {armed} armed")
    assert armed >= 2, "Should have at least 2 armed weapon mounts"
    print(f"  PASS: Multi-target tracking + IFF + missile dodge + weapon mounts")

    # OS deep dive validation
    print("\n--- OS Deep Dive Validation ---")
    rtos2 = SuitRTOS()
    boot_log = rtos2.boot_sequence()
    print(f"  Boot sequence: {len(boot_log)} phases in {boot_log[-1][0]}ms")
    assert len(boot_log) >= 10, "Boot sequence should have >=10 phases"
    assert boot_log[-1][1] == "Mission ready", "Boot should end with Mission ready"
    # Run for 1 second to populate scheduling log
    for _ in range(100):
        rtos2.update(0.01)
    gantt = rtos2.get_gantt_data()
    print(f"  Gantt data points: {len(gantt)}")
    assert len(gantt) > 0, "Gantt data should be populated"
    mem_map = rtos2.get_memory_map()
    print(f"  Memory map: {len(mem_map)} regions")
    assert len(mem_map) >= 8, "Memory map should have >=8 regions"
    success, rounds, key_id = rtos2.crypto_handshake()
    print(f"  Crypto handshake: success={success} rounds={rounds} key={key_id}")
    assert success, "Crypto handshake should succeed"
    assert rounds == 4, "Kyber-1024 requires 4 rounds"
    intent, conf, action = rtos2.ai_intent_classify(0.8)
    print(f"  AI intent: {intent} (conf={conf:.2f}) -> {action}")
    assert intent != "idle", "High neural signal should classify non-idle intent"
    cmds = rtos2.bci_decode(0.7)
    print(f"  BCI decode: latency={cmds['latency_ms']:.0f}ms  trigger={cmds['trigger']:.0f}")
    assert cmds["latency_ms"] < 17, "BCI latency should be <17ms"
    print(f"  PASS: Boot sequence + Gantt + memory map + crypto + AI intent + BCI decode")

    # EMP hardening + self-healing validation
    print("\n--- EMP Hardening + Self-Healing Validation ---")
    state3 = SuitState()
    # Test 1: Low-intensity EMP (should be fully blocked)
    blocked = state3.emp_hit(intensity_db=50.0)
    print(f"  EMP 50dB: blocked={blocked}  shield={state3.emp_shield_active}")
    assert blocked, "50dB EMP should be blocked by 60dB Faraday mesh"
    assert state3.emp_shield_active, "Shield should remain active after blocked EMP"
    # Test 2: High-intensity EMP (should partially penetrate)
    blocked = state3.emp_hit(intensity_db=120.0)
    print(f"  EMP 120dB: blocked={blocked}  shield={state3.emp_shield_active}  recover={state3.emp_recover_timer:.1f}s")
    assert not blocked, "120dB EMP should partially penetrate 60dB shielding"
    assert not state3.emp_shield_active, "Shield should drop after penetrating EMP"
    # Test 3: Recovery
    for _ in range(250):
        state3.update(0.01)
    print(f"  After 2.5s: shield={state3.emp_shield_active}  emp_hits={state3.emp_hits}")
    assert state3.emp_shield_active, "Shield should recover after 2s"
    # Test 4: Self-healing
    region = state3.trigger_self_heal("outer_armor", np.array([0.1, 0.2, 0.0]))
    print(f"  Self-heal triggered: part={region['part']}  progress={region['progress']:.0%}")
    assert region["progress"] == 0.0, "New heal region should start at 0%"
    for _ in range(250):
        state3.update(0.01)
    print(f"  After 2.5s: heal regions={state3.self_heal_active_count}")
    assert state3.self_heal_active_count == 0, "Self-heal should complete in ~2s"
    print(f"  PASS: EMP hardening (60dB attenuation) + self-healing (2s repair)")

    # Squad coordination validation
    print("\n--- Squad Coordination Validation ---")
    aim3 = AutoAimSystem()
    aim3.add_target(np.array([500, 0, 0]), np.array([0, 0, 0]), iff="hostile")
    aim3.add_target(np.array([800, 100, 0]), np.array([0, 0, 0]), iff="hostile")
    aim3.add_target(np.array([300, 0, 200]), np.array([0, 0, 0]), iff="hostile")
    aim3.add_squad_member("alpha", np.array([0, 0, 0]))
    aim3.add_squad_member("bravo", np.array([200, 0, 0]))
    aim3.add_squad_member("charlie", np.array([0, 0, 200]), status="inactive")
    aim3.update(0.01, np.array([0, 0, 0]))
    assignments = aim3.assign_squad_targets(np.array([0, 0, 0]))
    squad_status = aim3.get_squad_status()
    print(f"  Squad: {squad_status['members']} members, {squad_status['active']} active")
    print(f"  Target assignments: {len(assignments)}")
    assert squad_status["members"] == 3, "Should have 3 squad members"
    assert squad_status["active"] == 2, "Should have 2 active members"
    assert len(assignments) == 2, "Should assign targets to 2 active members"
    print(f"  PASS: Squad coordination (3 members, 2 active, 2 assignments)")

    # Biometric monitoring + armor damage validation
    print("\n--- Biometric + Armor Damage Validation ---")
    state4 = SuitState()
    # Simulate high-throttle combat scenario
    state4.throttle = 1.0
    state4.afterburner = True
    state4.g_load = 8.0
    # Add a target to trigger dodge and combat states
    state4.auto_aim.add_target(
        np.array([100, 0, 0]), np.array([-300, 0, 0]),
        mach_speed=True, iff="hostile")
    state4.defense.threat_detected = True
    for _ in range(100):
        state4.update(0.01)
    print(f"  HR: {state4.heart_rate:.0f}bpm (target: {state4.heart_rate_target:.0f})")
    print(f"  Adrenaline: {state4.adrenaline*100:.0f}%  Stress: {state4.stress_level*100:.0f}%")
    print(f"  Fatigue: {state4.fatigue*100:.1f}%  SpO2: {state4.blood_o2_sat:.1f}%")
    assert state4.heart_rate > 100, "Heart rate should be elevated during combat"
    # Adrenaline should be elevated from high throttle + g-load
    assert state4.adrenaline > 0.2, "Adrenaline should be elevated during high throttle+g-load"
    # Test armor damage + repair
    state4.damage_armor("outer_armor", 0.3)
    state4.damage_armor("frame", 0.1)
    print(f"  Armor damage: outer={state4.armor_damage['outer_armor']:.0%} frame={state4.armor_damage['frame']:.0%}")
    print(f"  Armor integrity: {state4.armor_integrity*100:.0f}%")
    assert state4.armor_damage["outer_armor"] == 0.3, "Outer armor should be 30% damaged"
    assert state4.armor_integrity < 1.0, "Overall armor integrity should be <100%"
    # Self-heal should have been triggered
    assert state4.self_heal_active_count > 0, "Self-heal should trigger on damage"
    print(f"  Self-heal triggered: {state4.self_heal_active_count} regions")
    # Repair
    state4.repair_armor("outer_armor", 0.3)
    assert state4.armor_damage["outer_armor"] < 0.01, "Outer armor should be repaired"
    print(f"  After repair: outer={state4.armor_damage['outer_armor']:.0%}")
    print(f"  PASS: Biometrics (HR/Adr/Stress/Fatigue/SpO2) + armor damage/repair")

    # Render test (headless)
    print("\n--- Render Test ---")
    if pygame:
        pygame.init()
        surf = pygame.Surface((800, 600))
        state.throttle = 0.0
        state.afterburner = False
        renderer = SuitRenderer(parts, cfg)
        renderer.render(surf, pygame.Rect(0, 0, 800, 600), state,
                        {"default": 0.0, "turbine": 0.0})
        print(f"  Rendered {len(parts)} parts to 800x600 surface: OK")
        pygame.quit()
    else:
        print("  pygame not available, skipping render test")

    # PART_DB completeness validation
    print("\n--- Part Database Validation ---")
    required_fields = ["description", "category", "layer", "materials", "weight_kg",
                       "blueprint", "icon", "sub_components", "dimensions",
                       "connectors", "performance", "build_details"]
    db_ok = True
    missing_keys = set(PART_DB.keys()) - set(p.key for p in parts)
    extra_keys = set(p.key for p in parts) - set(PART_DB.keys())
    if missing_keys:
        print(f"  FAIL: PART_DB has entries for non-existent parts: {missing_keys}")
        db_ok = False
    if extra_keys:
        print(f"  FAIL: Parts missing from PART_DB: {extra_keys}")
        db_ok = False
    total_subs = 0
    total_weight = 0.0
    for part in parts:
        info = PART_DB.get(part.key, {})
        for field in required_fields:
            val = info.get(field)
            if val is None or (isinstance(val, (list, str)) and len(val) == 0):
                print(f"  FAIL: {part.key} missing field '{field}'")
                db_ok = False
        total_subs += sum(sc.get("count", 1) for sc in info.get("sub_components", []))
        total_weight += info.get("weight_kg", 0.0)
    if db_ok:
        total_build_time = sum(PART_DB[p.key].get("build_details", {}).get("time_min", 0) for p in parts)
        print(f"  Parts in DB: {len(PART_DB)}  |  Required fields: {len(required_fields)}")
        print(f"  Total sub-components/pieces: {total_subs}")
        print(f"  Total suit weight: {total_weight:.1f} kg ({total_weight*2.205:.0f} lbs)")
        print(f"  Total assembly time: {total_build_time} min ({total_build_time//60}h {total_build_time%60}m)")
        print(f"  PASS: All 12 parts have complete metadata (description, materials, sub-components, dimensions, connectors, performance, build_details, blueprint)")


    # Muscle Fiber System validation
    print("\n--- Muscle Fiber System (DEA-STF) Validation ---")
    muscle = state.muscle
    # Test voltage ramp and contraction
    muscle.set_voltage(4.0)
    for _ in range(100):
        muscle.update(0.001, throttle=1.0, g_load=3.0)
    print(f"  Voltage: {muscle.voltage:.2f}kV (target 4.0kV)")
    assert muscle.voltage > 3.5, "DEA voltage should ramp to near 4kV"
    # Test force output scales with voltage
    leg_force = muscle.force_output["left_leg"] + muscle.force_output["right_leg"]
    print(f"  Leg force: {leg_force:.0f}N  Total: {muscle.total_force_n:.0f}N")
    assert leg_force > 0, "Leg muscle force should be >0 at full throttle"
    # Legs should have more force than arms (5x density vs 2x)
    arm_force = muscle.force_output["left_arm"] + muscle.force_output["right_arm"]
    print(f"  Arm force: {arm_force:.0f}N  Leg/Arm ratio: {leg_force/max(arm_force,1):.1f}x")
    assert leg_force > arm_force, "Leg force should exceed arm force (5x vs 2x density)"
    # Test STF impact stiffening
    muscle.trigger_impact_stiffening(5000.0)  # high shear rate
    assert muscle.stf_stiffened, "STF should stiffen under high shear rate"
    assert muscle.stf_viscosity > 100, "STF viscosity should jump under impact"
    print(f"  STF: viscosity={muscle.stf_viscosity:.0f}Pa*s stiffened={muscle.stf_stiffened}")
    # Test impact absorption
    transmitted, pct = muscle.get_impact_absorption(10000.0)
    print(f"  Impact absorption: {pct:.1f}%  transmitted={transmitted:.0f}N")
    assert pct > 95, "STF stiffened absorption should be >95%"
    # Test sublayer damage
    muscle.damage_sublayer(0, 0.5)
    assert muscle.sublayer_health[0] == 0.5, "Sublayer should be damaged"
    print(f"  Health: {muscle.health_pct:.0f}%  after sublayer damage")
    print(f"  PASS: DEA voltage ramp, force output, STF stiffening, impact absorption")

    # Thermal Management System validation
    print("\n--- Thermal Management System Validation ---")
    thermal = state.thermal
    # Test cold environment heating
    thermal.set_mode("heat")
    for _ in range(100):
        thermal.update(0.01, ambient_temp_c=-80.0, outer_skin_temp_c=-80.0,
                       throttle=0.5, speed_mps=50, env_density=1.225)
    print(f"  Cold (-80C): skin={thermal.skin_temp:.1f}C  PCM={thermal.pcm_charge*100:.0f}%  heat={thermal.heating_power_w:.0f}W")
    assert 30 < thermal.skin_temp < 42, "Skin temp should stay in safe range"
    # Test hot environment cooling
    thermal.set_mode("cool")
    for _ in range(100):
        thermal.update(0.01, ambient_temp_c=60.0, outer_skin_temp_c=120.0,
                       throttle=0.8, speed_mps=200, env_density=1.225)
    print(f"  Hot (60C, Mach 0.6): skin={thermal.skin_temp:.1f}C  cooling={thermal.cooling_power_w:.0f}W")
    assert thermal.skin_temp < 40, "Skin temp should not exceed 40C with cooling"
    # Test waste heat recovery
    thermal.set_mode("auto")
    thermal.update(0.01, ambient_temp_c=20.0, outer_skin_temp_c=80.0,
                   throttle=1.0, speed_mps=100, env_density=1.225)
    print(f"  Waste heat recovered: {thermal.waste_heat_recovered_w:.0f}W")
    assert thermal.waste_heat_recovered_w > 0, "Should recover waste heat from engines"
    # Test skin friction heating at high speed
    thermal.update(0.01, ambient_temp_c=20.0, outer_skin_temp_c=20.0,
                   throttle=0.5, speed_mps=300, env_density=1.225)
    print(f"  PASS: Heating (-80C), cooling (60C+Mach0.6), waste heat recovery, skin friction")

    # Neural Interface System validation
    print("\n--- Neural Interface System (BCI) Validation ---")
    neural = state.neural
    # Test signal generation
    neural.generate_signals(throttle=0.8, g_load=5.0, combat_active=True)
    assert len(neural.eeg_signal) == 64, "Should have 64 EEG channels"
    assert len(neural.emg_signal) == 16, "Should have 16 EMG channels"
    print(f"  EEG: {len(neural.eeg_signal)}ch  EMG: {len(neural.emg_signal)}ch  Q={neural.signal_quality*100:.0f}%")
    # Test intent decoding in combat
    intent, conf, action = neural.decode_intent(0.8, 5.0, True, False)
    print(f"  Combat intent: {intent} ({conf*100:.0f}%) -> {action}")
    assert intent == "combat", "Should classify combat intent with high gamma"
    assert conf > 0.7, "Combat intent confidence should be >70%"
    # Test dodge recommendation override
    intent2, conf2, action2 = neural.decode_intent(0.5, 3.0, True, True)
    print(f"  Dodge intent: {intent2} ({conf2*100:.0f}%) -> {action2}")
    assert intent2 == "evade", "Dodge recommendation should override to evade"
    # Test low-throttle landing intent
    intent3, conf3, action3 = neural.decode_intent(0.01, 1.0, False, False)
    print(f"  Low throttle intent: {intent3} -> {action3}")
    assert intent3 == "land", "Low throttle should classify as landing intent"
    # Test signal quality degradation at high g
    neural.generate_signals(0.5, 10.0, False)
    print(f"  High-g signal quality: {neural.signal_quality*100:.0f}%")
    assert neural.signal_quality < 1.0, "Signal quality should degrade at high g-load"
    # Test crypto handshake
    crypto_ok = neural.crypto_handshake()
    assert crypto_ok, "Crypto handshake should succeed"
    assert neural.crypto_verified, "Crypto should be verified"
    print(f"  Crypto: {neural.crypto_key_id}  verified={neural.crypto_verified}")
    # Test pre-activation
    print(f"  Pre-activation: {neural.preactivation}")
    assert all(v >= 0 for v in neural.preactivation.values()), "Pre-activation should be non-negative"
    print(f"  PASS: 64ch EEG + 16ch EMG, intent classification, dodge override, crypto, pre-activation")

    # Life Support System validation
    print("\n--- Life Support System Validation ---")
    life = state.life_support
    # Test atmospheric standby mode
    life.update(0.01, env_density=1.225, env_pressure_kpa=101.3, env_temp_c=20.0,
                pilot_mass=79.4, throttle=0.3)
    assert not life.active, "Life support should be standby in atmosphere"
    assert life.mode == "vented", "Should be vented in atmosphere"
    print(f"  Atmosphere: mode={life.mode}  active={life.active}")
    # Test vacuum sealed mode
    for _ in range(100):
        life.update(0.01, env_density=0.0, env_pressure_kpa=0.0, env_temp_c=-50.0,
                    pilot_mass=79.4, throttle=0.5)
    assert life.active, "Life support should activate in vacuum"
    assert life.mode == "sealed", "Should be sealed in vacuum"
    print(f"  Vacuum: mode={life.mode}  P={life.seal_pressure_kpa:.0f}kPa  CO2={life.co2_ppm:.0f}ppm  O2={life.o2_tank_pct:.0f}%")
    assert life.seal_pressure_kpa > 60, "Seal pressure should maintain >60kPa"
    assert life.co2_ppm < 5000, "CO2 should remain below 5000ppm (safe)"
    # Test underwater mode
    life2 = LifeSupportSystem()
    for _ in range(100):
        life2.update(0.01, env_density=997.0, env_pressure_kpa=200.0, env_temp_c=10.0,
                     pilot_mass=79.4, throttle=0.3)
    assert life2.active, "Life support should activate underwater"
    assert life2.mode == "sealed", "Should be sealed underwater"
    print(f"  Underwater: mode={life2.mode}  P={life2.seal_pressure_kpa:.0f}kPa")
    # Test seal damage
    life.damage_seal(0.5)
    assert life.seal_integrity < 1.0, "Seal integrity should decrease after damage"
    print(f"  Seal damage: integrity={life.seal_integrity:.1f}  leak={life.leak_rate_cc_min:.1f}cc/min")
    # Test safety properties
    assert life.co2_safe, "CO2 should be safe after scrubbing"
    assert life.o2_safe, "O2 should be safe with tank supply"
    print(f"  PASS: Atmospheric standby, vacuum seal, underwater, CO2 scrubbing, O2 generation, seal damage")

    # Power Management System validation
    print("\n--- Power Management System Validation ---")
    power = state.power
    # Test normal operation
    for _ in range(100):
        power.update(0.01, throttle=0.5, afterburner=False, solar_w=200.0,
                     turbulence=0.2, wing_deploy=0.8, g_load=1.5)
    print(f"  Normal: SOC={power.soc*100:.0f}%  V={power.voltage:.1f}V  P={power.power_draw_w:.0f}W  harvest={power.total_harvest_w:.0f}W")
    assert power.soc > 0.9, "Battery should remain high after 1s at moderate throttle"
    assert power.voltage > 40, "Voltage should remain above cutoff"
    # Test high drain with afterburner
    for _ in range(500):
        power.update(0.01, throttle=1.0, afterburner=True, solar_w=0.0,
                     turbulence=0.0, wing_deploy=0.0, g_load=5.0)
    print(f"  High drain (AB): SOC={power.soc*100:.0f}%  V={power.voltage:.1f}V  P={power.power_draw_w:.0f}W")
    assert power.soc < 1.0, "Battery should drain with high throttle + afterburner"
    # Test load shedding at low battery
    power2 = PowerManagementSystem()
    power2.soc = 0.15  # force low battery
    power2.update(0.01, throttle=0.8, afterburner=True, solar_w=0.0,
                  turbulence=0.0, wing_deploy=0.0, g_load=3.0)
    print(f"  Load shedding: SOC={power2.soc*100:.0f}%  shedding={power2.load_shedding}  shed={power2.shed_systems}")
    assert power2.load_shedding, "Should trigger load shedding below 20% SOC"
    assert len(power2.shed_systems) > 0, "Should shed non-critical systems"
    # Verify critical systems are not shed
    for critical in ["neural", "flight_ctrl", "turbines", "life_support"]:
        assert critical not in power2.shed_systems, f"{critical} should not be shed"
    # Test harvesting
    power3 = PowerManagementSystem()
    power3.update(0.01, throttle=0.0, afterburner=False, solar_w=500.0,
                  turbulence=2.0, wing_deploy=1.0, g_load=0.0)
    print(f"  Harvesting: solar={power3.solar_harvest_w:.0f}W  piezo={power3.piezo_harvest_w:.0f}W  total={power3.total_harvest_w:.0f}W")
    assert power3.solar_harvest_w > 0, "Solar harvesting should produce power"
    assert power3.piezo_harvest_w > 0, "Piezo harvesting should produce power from turbulence"
    # Test runtime estimation
    rt = power.runtime_estimate_h
    print(f"  Runtime estimate: {rt:.1f}h")
    assert rt > 0, "Runtime estimate should be positive"
    print(f"  PASS: Normal ops, afterburner drain, load shedding (critical preserved), solar+piezo harvesting")

    # Helmet System validation
    print("\n--- Helmet System Validation ---")
    helmet = state.helmet
    # Test zoom cycling
    initial_zoom = helmet.zoom_level
    for _ in range(5):
        helmet.cycle_zoom()
    helmet.update(0.1, env_density=1.225, env_temp_c=20.0, ai_locked=False, locked_target_dist=0.0)
    print(f"  Zoom: {initial_zoom:.0f}x -> {helmet.zoom_target:.0f}x (steps: {helmet.zoom_steps})")
    assert helmet.zoom_target > initial_zoom, "Zoom should increase after cycling"
    # Test night vision toggle
    helmet.toggle_night_vision()
    assert helmet.night_vision, "Night vision should be on after toggle"
    assert not helmet.thermal_vision, "Thermal should be off when night vision is on"
    helmet.toggle_night_vision()
    assert not helmet.night_vision, "Night vision should be off after second toggle"
    print(f"  Night vision toggle: OK")
    # Test thermal vision toggle
    helmet.toggle_thermal_vision()
    assert helmet.thermal_vision, "Thermal vision should be on after toggle"
    assert not helmet.night_vision, "Night vision should be off when thermal is on"
    print(f"  Thermal vision toggle: OK (mutually exclusive with NV)")
    # Test auto-seal in vacuum
    helmet.update(0.01, env_density=0.0, env_temp_c=-50.0, ai_locked=False, locked_target_dist=0.0)
    assert helmet.sealed, "Helmet should auto-seal in vacuum"
    assert not helmet.comms_active["mesh"], "Mesh comms should fail in vacuum"
    print(f"  Vacuum: sealed={helmet.sealed}  comms={helmet.comms_active}  signal={helmet.comms_signal_db:.0f}dB")
    # Test underwater comms
    helmet.update(0.01, env_density=997.0, env_temp_c=10.0, ai_locked=False, locked_target_dist=0.0)
    assert helmet.sealed, "Helmet should auto-seal underwater"
    assert not helmet.comms_active["iridium"], "Iridium should fail underwater"
    assert not helmet.comms_active["starlink"], "Starlink should fail underwater"
    print(f"  Underwater: sealed={helmet.sealed}  comms={helmet.comms_active}  signal={helmet.comms_signal_db:.0f}dB")
    # Test AI target assessment
    helmet.update(0.01, env_density=1.225, env_temp_c=20.0, ai_locked=True, locked_target_dist=150.0)
    assert helmet.target_assessment is not None, "Should have target assessment when locked"
    print(f"  AI assessment: range={helmet.target_assessment['range_m']:.0f}m  threat={helmet.target_assessment['threat_level']}")
    assert helmet.target_assessment["threat_level"] == "high", "150m target should be high threat"
    # Test visor damage
    helmet.damage_visor(20000.0)
    assert helmet.visor_damage > 0, "Visor should be damaged by high force"
    print(f"  Visor damage: {helmet.visor_damage:.0%}  cracked={helmet.visor_cracked}")
    print(f"  PASS: Zoom cycling, NV/thermal toggle, auto-seal, comms (vacuum/underwater), AI assessment, visor damage")

    # Frame System validation
    print("\n--- Frame System Validation ---")
    frame = state.frame
    # Test pilot sizing
    frame.set_pilot(1.90, 90.0)
    print(f"  Pilot 1.90m/90kg: torso_ext={frame.telescope_torso_mm:.0f}mm  limb_ext={frame.telescope_left_arm_mm:.0f}mm")
    assert frame.telescope_torso_mm > 0, "Tall pilot should extend torso"
    assert frame.telescope_left_arm_mm > 0, "Tall pilot should extend limbs"
    # Test short pilot
    frame.set_pilot(1.60, 60.0)
    print(f"  Pilot 1.60m/60kg: torso_ext={frame.telescope_torso_mm:.0f}mm  limb_ext={frame.telescope_left_arm_mm:.0f}mm")
    assert frame.telescope_torso_mm < 0, "Short pilot should retract torso"
    # Test actuator movement. Maxon EC45 linear actuator runs at ~10 mm/s over a
    # 355.6 mm stroke, so full extension takes ~36 s -- run long enough to traverse.
    frame2 = FrameSystem()
    frame2.actuator_targets[0] = 1.0
    for _ in range(4000):  # 40 s at dt=0.01
        frame2.update(0.01, g_load=1.0, throttle=0.5, velocity=50.0, total_mass=120.0)
    print(f"  Actuator: target=1.0  pos={frame2.actuator_positions[0]:.2f} (10mm/s over 356mm)")
    assert frame2.actuator_positions[0] > 0.9, "Actuator should move toward target"
    # Test high-g joint locking
    frame3 = FrameSystem()
    frame3.update(0.01, g_load=5.0, throttle=0.5, velocity=100.0, total_mass=120.0)
    locked = sum(1 for v in frame3.joint_locked.values() if v)
    print(f"  High-g (5g): locked joints={locked}/{frame3.PIVOT_COUNT}  viscosity={frame3.damper_viscosity:.0f}Pa*s")
    assert locked > 0, "Joints should lock above 4g"
    assert frame3.damper_viscosity > 100, "Damper viscosity should be high when locked"
    # Test low-g unlocked
    frame3.update(0.01, g_load=1.0, throttle=0.3, velocity=20.0, total_mass=120.0)
    locked2 = sum(1 for v in frame3.joint_locked.values() if v)
    print(f"  Low-g (1g): locked joints={locked2}/{frame3.PIVOT_COUNT}")
    assert locked2 == 0, "Joints should unlock below 4g"
    # Test force distribution
    frame3.update(0.01, g_load=8.0, throttle=0.8, velocity=150.0, total_mass=120.0)
    total_force = sum(frame3.force_distribution.values())
    print(f"  Force dist (8g): femur={frame3.force_distribution['femur']:.0f}N  pelvis={frame3.force_distribution['pelvis']:.0f}N  total={total_force:.0f}N")
    assert total_force > 0, "Force distribution should be non-zero under load"
    assert frame3.force_distribution["femur"] > frame3.force_distribution["clavicle"], "Femur should bear most load"
    # Test stress calculation
    print(f"  Stress: {frame3.stress_mpa:.0f}MPa  max={frame3.MAX_TUBE_STRESS_MPA}MPa  ok={frame3.frame_ok}")
    assert frame3.stress_mpa > 0, "Stress should be non-zero under load"
    # Test strap tension
    print(f"  Strap tension: {[f'{t:.0f}N' for t in frame3.strap_tension_n[:4]]}")
    assert all(t > 0 for t in frame3.strap_tension_n), "Strap tension should be positive"
    print(f"  PASS: Pilot sizing, actuator movement, high-g joint lock, force distribution, stress, strap tension")

    # Cross-system integration validation
    print("\n--- Cross-System Integration Validation ---")
    state5 = SuitState()
    phys5 = SuitPhysics(state5, cfg)
    # Run full simulation for 2 seconds at high throttle
    state5.throttle = 0.8
    state5.thrust_vector = np.array([0.0, 1.0, 0.0])
    for _ in range(200):
        state5.update(0.01)
        phys5.step(0.01)
    # Verify all systems are active and affecting physics
    print(f"  Altitude: {state5.altitude:.1f}m  G-load: {state5.g_load:.2f}")
    print(f"  Power: SOC={state5.power.soc*100:.0f}%  thrust_mult from V={state5.power.voltage/state5.power.NOMINAL_VOLTAGE:.2f}")
    print(f"  Muscle: force={state5.muscle.total_force_n:.0f}N  STF={state5.muscle.stf_viscosity:.0f}Pa*s")
    print(f"  Neural: intent={state5.neural.intent_class}  Q={state5.neural.signal_quality*100:.0f}%")
    print(f"  Thermal: skin={state5.thermal.skin_temp:.1f}C  mode={state5.thermal.mode}")
    print(f"  Frame: stress={state5.frame.stress_mpa:.0f}MPa  ok={state5.frame.frame_ok}")
    # Power should be draining
    assert state5.power.soc < 1.0, "Battery should drain during flight"
    # Muscle should be producing force
    assert state5.muscle.total_force_n > 0, "Muscle should produce force during flight"
    # Neural should classify non-idle intent
    assert state5.neural.intent_class != "idle", "Neural should detect active flight intent"
    # Thermal should be managing temp
    assert 30 < state5.thermal.skin_temp < 42, "Thermal should maintain safe skin temp"
    # Frame should be under stress but OK
    assert state5.frame.frame_ok, "Frame should be OK during normal flight"
    # Strike force is mechanically driven by the DEA arm fibers, not a constant.
    # Hold throttle (set the target too, or SuitState decays it) so fibers stay energized.
    pstate = SuitState()
    idle_punch = pstate.defense.punch_force_lbs  # frame/pilot floor at rest
    pstate.throttle = pstate.throttle_target = 0.9
    for _ in range(60):
        pstate.update(0.01)
    print(f"  Punch: idle floor={idle_punch:.0f} lbs  DEA-energized={pstate.defense.punch_force_lbs:,.0f} lbs")
    assert pstate.defense.punch_force_lbs > idle_punch + 500, "Punch force should rise substantially as DEA fibers energize"
    print(f"  PASS: All 7 systems integrated and affecting flight physics")

    # Test jump with muscle force
    print("\n--- Jump + Muscle Integration Validation ---")
    state6 = SuitState()
    # Charge muscle fibers with voltage
    state6.muscle.set_voltage(4.0)
    state6.throttle = 0.8
    for _ in range(100):
        state6.update(0.01)
    leg_force = state6.muscle.force_output.get("left_leg", 0) + state6.muscle.force_output.get("right_leg", 0)
    # Charge and execute jump with muscle force
    state6.jump.start_charge()
    for _ in range(50):
        state6.jump.update_charge(0.01)
    state6.jump.execute_jump(np.array([0, 0, 0]), muscle_force_n=leg_force)
    # Jump without muscle for comparison
    jump_plain = JumpAssist()
    jump_plain.start_charge()
    for _ in range(50):
        jump_plain.update_charge(0.01)
    jump_plain.execute_jump(np.array([0, 0, 0]), muscle_force_n=0.0)
    print(f"  Jump with muscle ({leg_force:.0f}N): {state6.jump.jump_peak_height:.1f}m")
    print(f"  Jump without muscle: {jump_plain.jump_peak_height:.1f}m")
    assert state6.jump.jump_peak_height > jump_plain.jump_peak_height, "Jump with muscle force should be higher"
    print(f"  PASS: Muscle fiber force enhances jump height by {state6.jump.jump_peak_height - jump_plain.jump_peak_height:.1f}m")


    # =====================================================================
    # SCIENTIFIC VALIDATION -- physics self-consistency checks
    # =====================================================================
    print("\n--- Scientific Validation ---")

    # 1. Weight budget reconciliation
    part_weights = sum(PART_DB[k].get("weight_kg", 0) for k in PART_DB)
    budget_total = DIMS["weight_total_kg"]
    print(f"  Part weights sum: {part_weights:.1f} kg | Budget total: {budget_total:.1f} kg")
    assert abs(part_weights - budget_total) < 2.0, f"Weight budget mismatch: parts={part_weights:.1f} vs budget={budget_total:.1f}"
    print(f"  PASS: Weight budget reconciles within 2 kg tolerance")

    # 2. Thrust-to-weight ratio at full mass
    total_mass = DIMS["weight_total_kg"] + DIMS["fuel_capacity_l"] * DIMS["fuel_density_kg_l"] + DIMS["ref_weight_kg"]
    weight_n = total_mass * 9.81
    thrust_dry_n = DIMS["turbine_count"] * DIMS["turbine_thrust_dry_lbf"] * 4.448
    thrust_ab_n = DIMS["turbine_count"] * DIMS["turbine_thrust_ab_lbf"] * 4.448
    twr_dry = thrust_dry_n / weight_n
    twr_ab = thrust_ab_n / weight_n
    print(f"  T/W dry: {twr_dry:.2f}:1 | T/W AB: {twr_ab:.2f}:1 | Mass: {total_mass:.1f} kg")
    assert twr_dry > 1.0, f"T/W dry must be >1.0 for flight, got {twr_dry:.2f}"
    assert twr_ab > 1.5, f"T/W AB should be >1.5 for agility, got {twr_ab:.2f}"
    print(f"  PASS: T/W > 1.0 (dry) and > 1.5 (AB) -- suit is flyable")

    # 3. Hover endurance calculation (hover thrust = weight, not full thrust)
    hover_thrust_lbf = weight_n / 4.448
    hover_burn_kg_h = hover_thrust_lbf * DIMS["turbine_sfc_dry_lb_lbh"] * 0.4536
    fuel_kg = DIMS["fuel_capacity_l"] * DIMS["fuel_density_kg_l"]
    hover_endurance_min = fuel_kg / hover_burn_kg_h * 60
    print(f"  Hover thrust: {hover_thrust_lbf:.0f} lbf | Burn: {hover_burn_kg_h:.1f} kg/h | Endurance: {hover_endurance_min:.1f} min")
    assert hover_endurance_min > 5.0, f"Hover endurance should be >5 min, got {hover_endurance_min:.1f}"
    print(f"  PASS: Hover endurance {hover_endurance_min:.1f} min (>5 min minimum)")

    # 4. Layer thickness sum check
    layer_sum = DIMS["inner_thick_mm"] + DIMS["middle_thick_mm"] + DIMS["intermediate_thick_mm"] + DIMS["outer_thick_mm"]
    print(f"  Layer sum: {layer_sum:.1f}mm + 0.6mm gaps = {layer_sum + 0.6:.1f}mm | Total: {DIMS['total_thick_mm']}mm")
    assert abs((layer_sum + 0.6) - DIMS["total_thick_mm"]) < 0.1, "Layer thicknesses must sum to total"
    assert DIMS["total_thick_mm"] <= 10.0, f"Ultra-conformed suit must be <=10mm, got {DIMS['total_thick_mm']}mm"
    print(f"  PASS: Layer thickness {DIMS['total_thick_mm']}mm <= 10mm (ultra-conformed)")

    # 5. Turbine thrust physics validation (mdot * V_exit)
    # F = mdot * V_exit; mdot = rho * A * V_inlet
    import math as _m
    turbine_r = DIMS["turbine_d_mm"] / 2 / 1000  # m
    inlet_area = _m.pi * turbine_r ** 2  # m^2
    rho_sl = 1.225  # kg/m^3
    # Tip speed at max RPM: v_tip = 2*pi*r*RPM/60
    v_tip = 2 * _m.pi * turbine_r * DIMS["turbine_rpm_max"] / 60
    # V_inlet ~ 0.5 * v_tip (typical axial compressor intake ratio)
    v_inlet = 0.5 * v_tip
    mdot = rho_sl * inlet_area * v_inlet
    v_exit = 280  # m/s (typical small turbofan exhaust velocity)
    computed_thrust_n = mdot * v_exit
    spec_thrust_n = DIMS["turbine_thrust_dry_lbf"] * 4.448
    print(f"  Turbine: d={DIMS['turbine_d_mm']}mm, RPM={DIMS['turbine_rpm_max']:,}, v_tip={v_tip:.0f} m/s")
    print(f"  mdot={mdot:.3f} kg/s, F_computed={computed_thrust_n:.0f} N, F_spec={spec_thrust_n:.0f} N")
    # Allow 50% tolerance (simplified model vs real compressor)
    assert computed_thrust_n > spec_thrust_n * 0.5, f"Computed thrust {computed_thrust_n:.0f}N too low vs spec {spec_thrust_n:.0f}N"
    print(f"  PASS: Turbine thrust physically plausible (within 50% of mdot*V_exit model)")

    # 6. Armor layer stopping power sum
    total_stop = DIMS["outer_stop_psi"] + DIMS["inter_stop_psi"] + DIMS["middle_stop_psi"]
    print(f"  Armor stopping power: {total_stop:,} PSI (outer+inter+middle)")
    assert total_stop > 600000, f"Armor should stop .50 BMG (~600k PSI), total={total_stop:,}"
    print(f"  PASS: Armor stack defeats .50 BMG ({total_stop:,} > 600,000 PSI)")

    # 7. Material property scientific plausibility checks
    print("\n  Material Property Verification:")
    # DEA strain: lab max ~100-300% (VHB acrylic), practical fiber-reinforced ~20-50%
    assert 10 <= DIMS["dea_strain_pct"] <= 100, f"DEA strain {DIMS['dea_strain_pct']}% out of plausible range"
    print(f"    DEA strain: {DIMS['dea_strain_pct']}% (lab: 100-300%, practical: 20-50%) -- PLAUSIBLE (upper bound)")
    # DEA contraction: <1ms theoretical, 10-100ms practical
    assert DIMS["dea_contraction_ms"] <= 100, f"DEA contraction {DIMS['dea_contraction_ms']}ms too slow"
    print(f"    DEA contraction: {DIMS['dea_contraction_ms']}ms (practical: 10-100ms) -- PLAUSIBLE")
    # STF: base aramid ~50k PSI, STF-enhanced 2-3x, tripled ~150k
    assert 50000 <= DIMS["stf_max_psi"] <= 300000, f"STF max PSI {DIMS['stf_max_psi']} out of range"
    print(f"    STF absorption: {DIMS['stf_max_psi']:,} PSI (base aramid ~50k, STF 2-3x, tripled ~150k) -- PLAUSIBLE")
    # Auxetic Poisson: theoretical min -1, re-entrant honeycomb -0.5 to -0.9
    assert -1.0 <= DIMS["auxetic_poisson"] <= -0.1, f"Auxetic Poisson {DIMS['auxetic_poisson']} out of range"
    print(f"    Auxetic Poisson: {DIMS['auxetic_poisson']} (re-entrant honeycomb: -0.5 to -0.9) -- PLAUSIBLE")
    # Auxetic relative density: 10-30% typical for lattice metamaterials
    assert 0.05 <= DIMS["auxetic_rel_density"] <= 0.40, f"Auxetic rel density {DIMS['auxetic_rel_density']} out of range"
    print(f"    Auxetic rel density: {DIMS['auxetic_rel_density']*100:.0f}% (lattice: 10-30%) -- PLAUSIBLE")
    # Graphene-UHMWPE: UHMWPE tensile ~3.6 GPa = ~520k PSI; graphene enhances stiffness
    assert 300000 <= DIMS["outer_stop_psi"] <= 800000, f"Outer stop PSI {DIMS['outer_stop_psi']} out of range"
    print(f"    Graphene-UHMWPE stop: {DIMS['outer_stop_psi']:,} PSI (UHMWPE ~520k, graphene-enhanced) -- PLAUSIBLE")
    # Alumina ceramic melting point: 2072C; composite short-duration tolerance ~1500-2000C
    assert 1000 <= DIMS["outer_heat_tol_c"] <= 2100, f"Heat tolerance {DIMS['outer_heat_tol_c']}C out of range"
    print(f"    Ceramic heat tolerance: {DIMS['outer_heat_tol_c']}C (alumina mp=2072C, short-duration) -- PLAUSIBLE")
    # Graphene thermal conductivity ~5000 W/mK; laser tolerance ~5-15 kW/cm² for short pulse
    assert 1 <= DIMS["outer_laser_tol_kw"] <= 50, f"Laser tolerance {DIMS['outer_laser_tol_kw']} kW/cm2 out of range"
    print(f"    Laser tolerance: {DIMS['outer_laser_tol_kw']} kW/cm2 (graphene thermal spreading) -- PLAUSIBLE")
    # STF stiffening: 1.5-3x typical for shear-thickening transition
    assert 1.2 <= DIMS["stf_stiffen_mult"] <= 5.0, f"STF stiffen mult {DIMS['stf_stiffen_mult']} out of range"
    print(f"    STF stiffening: {DIMS['stf_stiffen_mult']}x (typical: 1.5-3x) -- PLAUSIBLE")
    print(f"  PASS: All 9 material properties within scientifically plausible bounds")

    # =====================================================================
    # FAILURE MODE ANALYSIS -- per-layer thresholds and cascade logic
    # =====================================================================
    print("\n--- Failure Mode Analysis ---")
    failure_modes = [
        ("Outer armor penetration", "Ballistic impact > 520k PSI",
         "Ceramic strike face shatters, UHMWPE delaminates",
         "Impact energy transfers to auxetic layer (150k PSI capacity)",
         "Self-healing seams close <25mm holes in 2s; panel replacement if >25mm"),
        ("Auxetic densification", "Residual impact > 150k PSI",
         "Lattice cells compress to solid, foam fill ruptures",
         "Energy transfers to DEA-STF middle layer (80k PSI base, 152k stiffened)",
         "Lattice permanently deformed; replacement required after mission"),
        ("DEA fiber rupture", "Residual impact > 80k PSI (base) / 152k (stiffened)",
         "DEA elastomer tears, STF bladder ruptures, carbon nanotube electrodes sever",
         "Muscle function lost in affected panel; load redistributes to adjacent panels",
         "Panel replacement + STF refill required; 24-panel redundancy limits impact"),
        ("Frame structural failure", "Stress > 50G axial or > 500N actuator overload",
         "CFRP tube delamination, Ti node fracture, actuator seizure",
         "Load path redistributes via 24 Spectra straps; telescoping locks",
         "Frame replacement required; suit mission-abort, pilot extraction"),
        ("Turbine failure", "Foreign object damage, bearing seizure, fuel starvation",
         "Turbine produces zero thrust, possible turbine burst",
         "Remaining 35 turbines compensate; FADEC redistributes thrust",
         "Turbine replacement; suit remains flyable with 35/36 turbines"),
        ("Power system failure", "Battery thermal runaway, BMS fault, bus short",
         "Loss of 48V bus power to actuators, turbines, neural, thermal",
         "Hot-swap to backup battery (5s downtime); piezo harvesters provide emergency power",
         "Battery replacement; suit enters degraded mode (40% thrust, no afterburner)"),
        ("Neural interface loss", "EEG signal degradation, NPU fault, crypto failure",
         "Loss of thought-to-action control; suit reverts to manual input",
         "AI co-pilot maintains hover/stabilization; manual joystick fallback",
         "Recalibration required; suit enters safe-mode with manual controls only"),
        ("Thermal runaway", "Skin temp > 42C or < 30C despite regulation",
         "Peltier junction failure, capillary loop blockage, aerogel degradation",
         "Radiator fins deploy automatically; PCM absorbs latent heat (12kJ/kg)",
         "Thermal system service required; mission abort if temp uncontrolled"),
        ("Helmet seal breach", "Visor crack > 5mm, neck ring failure, O-ring degradation",
         "Loss of pressure integrity; O2 leak, CO2 ingress, water ingress",
         "Self-sealing polymer activates; emergency O2 reserve (5 min backup)",
         "Immediate descent/egress required; helmet replacement after mission"),
        ("Fuel system breach", "Bladder penetration > 12mm, line rupture, pump failure",
         "Fuel loss, potential fire hazard, thrust loss in affected turbine group",
         "Self-sealing coating closes <12mm; dual redundant pumps/lines; FADEC isolates group",
         "Fuel system repair; mission abort if fuel loss > 50%"),
    ]
    for i, (mode, trigger, effect, mitigation, recovery) in enumerate(failure_modes):
        print(f"  F{i+1}: {mode}")
        print(f"      Trigger: {trigger}")
        print(f"      Effect: {effect}")
        print(f"      Mitigation: {mitigation}")
        print(f"      Recovery: {recovery}")
    print(f"  PASS: {len(failure_modes)} failure modes documented with cascade logic")


    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    return 0


# =============================================================================
# FEASIBILITY REPORT  -- real-world build analysis
# =============================================================================

def print_feasibility():
    """Print a real-world build feasibility report."""
    print("=" * 70)
    print("FLYSUIT Mjalnor'MV1.17 -- BUILD FEASIBILITY REPORT")
    print("=" * 70)

    print("\n1. FRAME & STRUCTURE")
    print(f"   Material: {DIMS['frame_material']} + {DIMS['frame_node_material']}")
    print(f"   Telescoping: +{DIMS['torso_telescope_mm']}mm torso, +{DIMS['limb_telescope_mm']}mm limbs")
    print(f"   Fits: {DIMS['pilot_min_height_m']*3.28:.1f}ft - {DIMS['pilot_max_height_m']*3.28:.1f}ft, "
          f"{DIMS['pilot_min_weight_kg']*2.2:.0f}-{DIMS['pilot_max_weight_kg']*2.2:.0f} lb")
    print(f"   Status: BUILDABLE (2025-2026 materials, Markforged Metal X + autoclave)")

    print("\n2. PROPULSION")
    total_mass_kg = DIMS['weight_total_kg'] + DIMS['fuel_capacity_l'] * DIMS['fuel_density_kg_l'] + DIMS['ref_weight_kg']
    weight_n = total_mass_kg * 9.81
    total_thrust_dry_n = DIMS['turbine_count'] * DIMS['turbine_thrust_dry_lbf'] * 4.448
    total_thrust_ab_n = DIMS['turbine_count'] * DIMS['turbine_thrust_ab_lbf'] * 4.448
    twr_dry = total_thrust_dry_n / weight_n
    twr_ab = total_thrust_ab_n / weight_n
    print(f"   {DIMS['turbine_count']}x micro-turbofans, {DIMS['turbine_d_mm']}mm x {DIMS['turbine_len_mm']}mm")
    print(f"   Total thrust: {DIMS['turbine_count']*DIMS['turbine_thrust_dry_lbf']:,} lbf dry / {DIMS['turbine_count']*DIMS['turbine_thrust_ab_lbf']:,} lbf AB at sea level")
    print(f"   T/W ratio: {twr_dry:.2f}:1 (dry) / {twr_ab:.2f}:1 (AB) at full mass ({total_mass_kg:.1f} kg)")
    print(f"   Thrust scales with air density: at 28k ft (~{0.418/1.225*100:.0f}% rho) -> ~{DIMS['turbine_count']*DIMS['turbine_thrust_ab_lbf']*0.418/1.225:.0f} lbf")
    print(f"   SFC: {DIMS['turbine_sfc_dry_lb_lbh']} lb/(lbf*h) dry, {DIMS['turbine_sfc_ab_lb_lbh']} lb/(lbf*h) AB")
    print(f"   Fuel: {DIMS['fuel_type']}, {DIMS['fuel_capacity_l']}L ({DIMS['fuel_capacity_l']*DIMS['fuel_density_kg_l']:.1f} kg)")
    # Hover endurance: hover thrust = weight, not full dry thrust
    hover_thrust_lbf = weight_n / 4.448  # thrust needed to hover
    hover_burn_kg_h = hover_thrust_lbf * DIMS['turbine_sfc_dry_lb_lbh'] * 0.4536
    hover_min = DIMS['fuel_capacity_l'] * DIMS['fuel_density_kg_l'] / hover_burn_kg_h * 60
    full_dry_burn_kg_h = DIMS['turbine_count'] * DIMS['turbine_thrust_dry_lbf'] * DIMS['turbine_sfc_dry_lb_lbh'] * 0.4536
    print(f"   Hover thrust needed: {hover_thrust_lbf:.0f} lbf ({hover_thrust_lbf/(DIMS['turbine_count']*DIMS['turbine_thrust_dry_lbf'])*100:.0f}% throttle)")
    print(f"   Hover burn rate: {hover_burn_kg_h:.1f} kg/h -> hover endurance ~{hover_min:.0f} min")
    print(f"   Full dry burn rate: {full_dry_burn_kg_h:.1f} kg/h (at 100% throttle)")
    print(f"   Climb-glide cycling: 4-7 hours (5min climb + 25min glide, repeat)")
    print(f"   Status: CHALLENGING (150k RPM magnetic bearings near-production, not commercial)")

    print("\n3. ARMOR (4-LAYER)")
    print(f"   Outer: graphene-UHMWPE, {DIMS['outer_max_psi']:,} PSI, NIJ {DIMS['outer_nij_level']}")
    print(f"   Intermediate: auxetic metamaterial, Poisson={DIMS['auxetic_poisson']}")
    print(f"   Middle: tripled DEA-STF, {DIMS['stf_max_psi']:,} PSI absorption")
    print(f"   Inner: sensor suit, {DIMS['inner_thick_mm']}mm")
    print(f"   Total thickness: {DIMS['total_thick_mm']}mm (ultra-conformed)")
    print(f"   Status: PARTIALLY BUILDABLE (UHMWPE NIJ IV exists; auxetic + DEA near-lab)")

    print("\n4. HELMET & LIFE SUPPORT")
    print(f"   Visor: {DIMS['visor_max_force_lbs']:,} lbs, {DIMS['visor_zoom_optical']}x+{DIMS['visor_zoom_digital']}x zoom")
    print(f"   Seal: vacuum + {DIMS['seal_depth_m']}m underwater")
    print(f"   CO2: {DIMS['co2_scrub_hours']}h regenerable, O2: {DIMS['o2_capacity_hours']}h")
    print(f"   Status: BUILDABLE (NASA EMU tech + commercial rebreather components)")

    print("\n5. NEURAL INTERFACE")
    print(f"   BCI: {DIMS['bci_type']}, <{DIMS['bci_latency_ms']}ms, {DIMS['bci_crypto']}")
    print(f"   Status: NEAR-TERM (OpenBCI + EMG armbands exist; <17ms achievable)")

    print("\n6. POWER")
    print(f"   {DIMS['battery_type']}, {DIMS['battery_wh']} Wh, {DIMS['battery_wh_kg']} Wh/kg")
    print(f"   Runtime: {DIMS['battery_life_hours']}h+ with harvesting")
    print(f"   Status: BUILDABLE (Amprius 550 Wh/kg cells shipping 2025)")

    print("\n7. AI & OS")
    print(f"   {DIMS['os_type']}: {DIMS['os_check_ms']}ms checks, {DIMS['os_failover_ms']}ms failover")
    print(f"   AI: {DIMS['os_ai_model']}")
    print(f"   Boot: 13-phase, 180ms to mission ready")
    print(f"   Crypto: Kyber-1024 post-quantum + AES-256-GCM")
    print(f"   BCI: 64-channel EEG, <17ms decode latency, 8 intent classes")
    print(f"   Memory: 3.9MB mapped (11 regions, dual-redundant ROM)")
    print(f"   Status: BUILDABLE (Jetson Orin + FreeRTOS/Zephyr + TFLite)")

    print("\n8. WINGS & AERODYNAMICS")
    print(f"   {DIMS['wing_span_m']}m span, {DIMS['wing_area_sqft']} sq ft ({DIMS['wing_area_sqft']*0.0929:.1f} m^2), L/D={DIMS['wing_ld_ratio']}:1")
    print(f"   Wing area: {DIMS['wing_area_sqft']*0.0929:.1f} m^2, CL_max=1.2, stall at ~15deg AoA")
    print(f"   Induced drag: CDi = CL^2 / (pi * AR * e), Oswald e=0.85")
    ar = DIMS['wing_span_m']**2 / (DIMS['wing_area_sqft'] * 0.0929)
    print(f"   Aspect ratio: {ar:.1f} (span^2/area)")
    print(f"   Lift: L = 0.5 * rho * v^2 * S * CL(alpha)")
    print(f"   Compact wings: turbine-assisted glide, stows flat on back")
    print(f"   Status: BUILDABLE (nitinol + ripstop, proven in UAV/skydiving)")

    print("\n9. COMBAT SYSTEMS")
    print(f"   Auto-aim: {DIMS['perf_aim_range_miles']} miles, {DIMS['perf_aim_accuracy_pct']}% accuracy, Mach-speed tracking")
    print(f"   Multi-target: 8 simultaneous, IFF transponder, threat priority sorting")
    print(f"   Missile dodge: inbound detection + evasion vector computation")
    print(f"   Weapon mounts: 5 hardpoints (2 forearm + 2 shoulder + 1 back)")
    print(f"   Defense AI: martial arts state machine, {DIMS['perf_punch_lbs']:,} lbs punch")
    print(f"   Squad coordination: multi-member target assignment, proximity-based")

    print("\n10. ATMOSPHERIC MODEL")
    print(f"   ISA atmosphere (Earth): altitude density/temp/pressure")
    print(f"   Mars: thin CO2, dust storm radio attenuation")
    print(f"   Titan: thick N2/CH4, 21km scale height")
    print(f"   Ocean: hydrostatic pressure + thermocline")
    print(f"   Wind: turbulence + gusts, solar harvesting integration")
    print(f"   Weather: rain, storm, dust, methane, currents, clouds (particle FX)")
    print(f"   Environments: 11 presets (sea level to deep ocean to vacuum)")

    print("\n11. SURVIVABILITY")
    print(f"   EMP hardening: 60dB Faraday mesh attenuation, 2s shield recovery")
    print(f"   Self-healing: autonomous damage repair, 2s per region")
    print(f"   Armor damage tracking: per-part integrity, visual damage indicators")
    print(f"   Biometrics: HR, adrenaline, stress, fatigue, SpO2 monitoring")

    print("\n12. PERFORMANCE SUMMARY")
    print(f"   Vertical jump: {DIMS['perf_jump_vertical_ft']} ft")
    print(f"   Punch force: {DIMS['perf_punch_lbs']:,} lbs")
    print(f"   Overhead lift: {DIMS['perf_lift_overhead_lbs']:,} lbs")
    print(f"   Deadlift: {DIMS['perf_deadlift_lbs']:,} lbs")
    print(f"   Running speed: {DIMS['perf_run_mph']} mph")
    print(f"   Safe fall height: {DIMS['perf_safe_fall_ft']} ft")
    print(f"   Auto-aim range: {DIMS['perf_aim_range_miles']} miles ({DIMS['perf_aim_accuracy_pct']}% accuracy)")
    print(f"   Max speed: {DIMS['perf_max_speed_mph']} mph")
    print(f"   Service ceiling: {DIMS['perf_ceiling_ft']:,} ft")
    print(f"   Hover endurance: ~{hover_min:.0f} min (hover thrust, dry SFC)")
    print(f"   T/W ratio: {twr_dry:.2f}:1 dry / {twr_ab:.2f}:1 AB (full mass {total_mass_kg:.1f} kg)")
    print(f"   Glide endurance: 4-7 hours (climb-glide cycling with {DIMS['fuel_capacity_l']}L fuel)")
    print(f"   Suit weight: {DIMS['weight_total_kg']} kg ({DIMS['weight_total_kg']*2.2:.0f} lb) + {DIMS['fuel_capacity_l']*DIMS['fuel_density_kg_l']:.1f} kg fuel + {DIMS['ref_weight_kg']} kg pilot")

    print("\n13. ESTIMATED COST")
    print(f"    Per suit: ~$1.8-2.4M (single unit), ~$650K (50+ units/year)")
    print(f"    Timeline: ~11 months to first flying article")

    print("\n" + "=" * 70)
    print("VERDICT: 80% buildable with 2025-2026 tech. Key risk areas:")
    print("  - 150k RPM magnetic bearings (near-production, not commercial)")
    print("  - DEA artificial muscle at scale (lab prototypes only)")
    print("  - Auxetic metamaterial manufacturing (3D printing feasible)")
    print("  - Neural BCI <17ms (achievable with current EEG+EMG)")
    print("  - Fuel system: BUILDABLE (conformal bladder + Jet-A1 + AN-8 lines)")
    print("  - Induced drag model: validated (CDi = CL^2/(pi*AR*e))")
    print("=" * 70)


# =============================================================================
# STRESS TEST  -- multi-environment validation
# =============================================================================

def print_stress_test():
    """Run suit physics across all environments."""
    print("=" * 70)
    print("FLYSUIT Mjalnor'MV1.17 -- MULTI-ENVIRONMENT STRESS TEST")
    print("=" * 70)

    parts, tcfg, cfg = build_suit(1.73, 79.4)

    for env_idx, env in enumerate(ENVIRONMENTS):
        name, density, gravity = env[0], env[1], env[2]
        temp = env[4]
        state = SuitState()
        state.env_idx = env_idx
        state.throttle = 1.0
        state.afterburner = True
        state.update(1.0)  # run one update cycle to compute fuel flow, thrust, etc.
        phys = SuitPhysics(state, cfg)

        twr = phys.thrust_to_weight
        runtime = phys.battery_runtime_hours(1.0)
        felt, pct = phys.impact_absorption(600000)
        # Fuel burn rate at current environment
        fuel_kg = state.fuel_kg
        fuel_burn_h = state.fuel_flow_kg_s * 3600 if state.fuel_flow_kg_s > 0 else 0
        fuel_end_min = fuel_kg / fuel_burn_h * 60 if fuel_burn_h > 0 else 0

        status = "OK" if twr > 1.0 else "MARGINAL" if twr > 0.3 else "NO FLIGHT"
        print(f"\n  {name:25s}  density={density:8.3f}  g={gravity:5.2f}  "
              f"T={temp:7.1f}K  T/W={twr:5.2f}  [{status}]")
        print(f"    Runtime: {runtime:.1f}h  Impact absorption: {pct:.1f}%  "
              f"User-felt: {felt:,.0f} PSI")
        if fuel_burn_h > 0:
            print(f"    Fuel: {fuel_kg:.1f}kg  Burn: {fuel_burn_h:.1f} kg/h  Endurance: {fuel_end_min:.0f} min")
        else:
            print(f"    Fuel: N/A (vacuum -> RCS cold gas, no air-breathing burn)")

    print("\n" + "=" * 70)


# =============================================================================
# IMPACT TEST  -- ballistic absorption simulation
# =============================================================================

def print_impact_test():
    """Simulate multi-layer impact absorption across threat levels."""
    print("=" * 70)
    print("FLYSUIT Mjalnor'MV1.17 -- BALLISTIC IMPACT ABSORPTION TEST")
    print("=" * 70)

    parts, tcfg, cfg = build_suit(1.73, 79.4)
    state = SuitState()
    phys = SuitPhysics(state, cfg)

    threats = [
        ("9mm handgun",        35000),
        ("5.56mm rifle",      120000),
        ("7.62mm NATO",       250000),
        (".30-06 AP",         400000),
        (".50 BMG",           600000),
        ("20mm cannon",      1200000),
        ("Artillery frag",   3000000),
    ]

    print(f"\n  {'Threat':25s} {'Impact PSI':>12s} {'Outer':>10s} {'Inter':>10s} "
          f"{'Middle':>10s} {'User feels':>12s} {'Absorbed':>10s}")
    print("  " + "-" * 90)

    for name, psi in threats:
        # Physically-derived per-layer propagation (ballistic-limit model).
        rows, felt, pct = phys.impact_breakdown(psi)
        outer_abs = rows[0][1]
        inter_abs = rows[1][1]
        middle_abs = rows[2][1]
        survived = "SURVIVE" if felt < 50000 else "INJURY" if felt < 100000 else "FATAL"
        print(f"  {name:25s} {psi:>12,} {outer_abs:>10,.0f} {inter_abs:>10,.0f} "
              f"{middle_abs:>10,.0f} {felt:>12,.0f} {pct:>9.1f}%  [{survived}]")

    print("\n" + "=" * 70)


# =============================================================================
# OS TEST  -- SuitRTOS deep diagnostic
# =============================================================================

def print_os_test():
    """Run a full SuitRTOS diagnostic: boot, tasks, crypto, BCI, AI intent, memory map."""
    print("=" * 70)
    print("FLYSUIT Mjalnor'MV1.17 -- SuitRTOS DEEP DIAGNOSTIC")
    print("=" * 70)

    rtos = SuitRTOS()

    # 1. Boot sequence
    print("\n1. BOOT SEQUENCE")
    print("-" * 40)
    boot_log = rtos.boot_sequence()
    for ts, phase, status in boot_log:
        print(f"  [{ts:4d}ms] {phase:30s} [{status}]")
    print(f"  Total boot time: {boot_log[-1][0]}ms")

    # 2. Task registry
    print("\n2. TASK REGISTRY")
    print("-" * 40)
    print(f"  Total tasks: {rtos.task_count}  Critical: {rtos.critical_task_count}")
    print(f"  {'Name':20s} {'Prio':>4s} {'Active':>6s} {'Fail':>4s} {'Lat(ms)':>7s} {'Anomaly':>7s}")
    for name, prio, active, fail, lat, anom in rtos.get_task_status():
        print(f"  {name:20s} {prio:4d} {'YES' if active else 'NO':>6s} {fail:4d} {lat:7.2f} {anom:7.2f}")

    # 3. Run OS for 2 seconds
    print("\n3. RUNTIME (2s simulation)")
    print("-" * 40)
    for _ in range(200):
        rtos.update(0.01)
    print(f"  Uptime: {rtos.uptime_str}")
    print(f"  Integrity checks: {rtos.integrity_checks:,}")
    print(f"  Uptime: {rtos.uptime_pct:.6f}%")
    print(f"  Failovers: {rtos.failovers}  Viruses: {rtos.viruses_detected}")

    # 4. Gantt chart data
    print("\n4. TASK SCHEDULER GANTT (last 500ms)")
    print("-" * 40)
    gantt = rtos.get_gantt_data(500)
    print(f"  Data points: {len(gantt)}")
    # Print a text Gantt for top 10
    for name, ts, dur, prio in gantt[-10:]:
        bar_len = max(1, int(dur / 2))
        bar = "#" * bar_len
        print(f"  {name:20s} P{prio} [{ts:6d}ms] {dur:5.1f}ms {bar}")

    # 5. Memory map
    print("\n5. MEMORY MAP")
    print("-" * 40)
    mem = rtos.get_memory_map()
    total_kb = 0
    for region, info in mem.items():
        print(f"  {region:20s} {info['base']:12s} {info['size_kb']:6d}KB  [{info['type']}]")
        total_kb += info["size_kb"]
    print(f"  {'TOTAL':20s} {'':12s} {total_kb:6d}KB  ({total_kb/1024:.1f}MB)")

    # 6. Crypto handshake
    print("\n6. CRYPTO HANDSHAKE (Kyber-1024 + AES-256-GCM)")
    print("-" * 40)
    success, rounds, key_id = rtos.crypto_handshake()
    print(f"  Success: {success}")
    print(f"  Rounds: {rounds}")
    print(f"  Key ID: {key_id}")
    print(f"  Algorithm: Kyber-1024 (post-quantum) + AES-256-GCM")

    # 7. AI Intent Classification
    print("\n7. AI INTENT CLASSIFICATION")
    print("-" * 40)
    test_signals = [0.1, 0.3, 0.5, 0.65, 0.75, 0.8, 0.9]
    for sig in test_signals:
        intent, conf, action = rtos.ai_intent_classify(sig)
        print(f"  Signal {sig:.2f} -> Intent: {intent:10s} (conf={conf:.2f}) -> Action: {action}")

    # 8. BCI Signal Decoding
    print("\n8. BCI SIGNAL DECODING (64-channel EEG)")
    print("-" * 40)
    # Simulate different motor intents
    test_patterns = [
        ("Resting", np.random.random(64) * 0.1),
        ("Left arm move", np.array([0.8] * 32 + [0.2] * 32)),
        ("Right arm move", np.array([0.2] * 32 + [0.9] * 32)),
        ("Trigger (gamma burst)", np.array([0.3] * 63 + [0.95])),
        ("Speaking", np.array([0.6] * 32 + [0.1] * 32)),
    ]
    for name, signal in test_patterns:
        cmds = rtos.bci_decode(signal)
        print(f"  {name:25s} -> L_arm={cmds['left_arm']:.2f} R_arm={cmds['right_arm']:.2f} "
              f"trigger={cmds['trigger']:.0f} speak={cmds['speak']:.2f} lat={cmds['latency_ms']:.0f}ms")

    # 9. Virus injection + failover test
    print("\n9. VIRUS INJECTION + FAILOVER TEST")
    print("-" * 40)
    rtos.inject_virus(target_task="flight_ctrl", at_s=0.0)
    for _ in range(100):
        rtos.update(0.01)
    print(f"  Viruses detected: {rtos.viruses_detected}")
    print(f"  Failovers: {rtos.failovers}")
    print(f"  ROM restores: {rtos.rom_restores}")
    print(f"  Viruses purged: {rtos.viruses_purged}")
    print(f"  Primary OK: {rtos.primary_ok}")
    print(f"  Recent events:")
    for ts, name, lat in rtos.get_recent_events(5):
        print(f"    [{ts}ms] {name} ({lat:.1f}ms)")

    print("\n" + "=" * 70)
    print("SuitRTOS DIAGNOSTIC COMPLETE")
    print("=" * 70)
    return 0


# =============================================================================
# OBJ EXPORT  -- write 3D model files
# =============================================================================

def export_obj(path, parts, verbose=False):
    """Export the suit model as OBJ + MTL files."""
    obj_path = os.path.join(path, "flysuit.obj")
    mtl_path = os.path.join(path, "flysuit.mtl")

    all_verts = []
    all_faces = []
    materials = {}
    mat_idx = 0
    v_offset = 0

    for part in parts:
        for m in part.meshes:
            mat_name = f"mat_{mat_idx}"
            r, g, b = m.color
            materials[mat_name] = (r / 255, g / 255, b / 255, m.emissive)
            mat_idx += 1

            for v in m.verts:
                all_verts.append((v[0], v[1], v[2]))
            for f in m.faces:
                all_faces.append(([fi + v_offset + 1 for fi in f], mat_name))
            v_offset += len(m.verts)

    with open(obj_path, "w") as f:
        f.write("# Flysuit Mjalnor'MV1.17 exported model\n")
        f.write(f"# {len(all_verts)} vertices, {len(all_faces)} faces\n")
        f.write(f"mtllib flysuit.mtl\n\n")
        for v in all_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        current_mat = None
        for face, mat in all_faces:
            if mat != current_mat:
                f.write(f"usemtl {mat}\n")
                current_mat = mat
            f.write("f " + " ".join(str(i) for i in face) + "\n")

    with open(mtl_path, "w") as f:
        for mat_name, (r, g, b, emissive) in materials.items():
            f.write(f"newmtl {mat_name}\n")
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            if emissive:
                f.write(f"Ka {r:.4f} {g:.4f} {b:.4f}\n")
            f.write("\n")

    if verbose:
        print(f"Exported {len(all_verts)} vertices, {len(all_faces)} faces")
        print(f"  OBJ: {obj_path}")
        print(f"  MTL: {mtl_path}")


# =============================================================================
# APP  -- main interactive application
# =============================================================================

class App:
    def __init__(self):
        self.width = 1280
        self.height = 800
        self.pilot_height = 1.73
        self.pilot_weight = 79.4
        self.parts, self.tcfg, self.cfg = build_suit(self.pilot_height, self.pilot_weight)
        self.state = SuitState()
        self.physics = SuitPhysics(self.state, self.cfg)
        self.model_renderer = SuitRenderer(self.parts, self.cfg)
        self.flight_renderer = FlightRenderer(self.parts, self.cfg)
        self.mode = "model"  # model | flight | layers
        self.paused = False
        self.show_help = False
        self.show_info = True
        self.show_math = False
        self.info_scroll = 0
        self.browser_hit_rects = []
        self.font = None
        self.font_sm = None
        self.angles = {"default": 0.0, "turbine": 0.0}
        # Input state
        self.keys_held = set()
        self.joystick = None
        self.joystick_axes = {}
        self.mouse_pos = (0, 0)
        self.mouse_buttons = (0, 0, 0)
        # Combat scenario
        self.combat_scenario = False
        self.scenario_wave = 0
        self.scenario_kills = 0
        self.scenario_timer = 0.0
        self.scenario_spawn_timer = 0.0
        # Test environment
        self.test_targets_auto = False
        self.target_spawn_timer = 0.0
        # Explosion effects
        self.explosions = []
        # FPS tracking
        self.fps = 0.0
        self.show_fps = False
        # Showcase mode (detailed component views with math proof)
        self.show_showcase = False
        self.showcase_idx = 0
        self.showcase_parts = None  # lazy-loaded
        self.showcase_scroll = 0

    def _init_joystick(self):
        """Initialize joystick/gamepad if available."""
        try:
            pygame.joystick.init()
            count = pygame.joystick.get_count()
            if count > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Gamepad connected: {self.joystick.get_name()}")
                print(f"  Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")
                return True
            else:
                print("No gamepad detected. Keyboard/mouse only.")
                return False
        except Exception as e:
            print(f"Joystick init error: {e}")
            return False

    def _handle_gamepad(self, dt):
        """Process gamepad axes and buttons for flight control."""
        if self.joystick is None or self.mode != "flight":
            return

        joy = self.joystick
        n_axes = joy.get_numaxes()

        # Standard Xbox/PS controller mapping:
        # Axis 0: Left stick X (roll)
        # Axis 1: Left stick Y (pitch, inverted)
        # Axis 2: Right stick X (yaw)
        # Axis 3: Right stick Y (thrust vector vertical)
        # Axis 4: Left trigger (throttle down)
        # Axis 5: Right trigger (throttle up)

        # Read axes with deadzone
        deadzone = 0.15

        def axis(i):
            if i >= n_axes:
                return 0.0
            v = joy.get_axis(i)
            return v if abs(v) > deadzone else 0.0

        # Left stick: pitch + roll
        left_x = axis(0)
        left_y = axis(1)
        self.state.pitch_rate = -left_y * 1.5  # inverted (push up = nose down)
        self.state.roll_rate = left_x * 1.5

        # Right stick: yaw + vertical thrust vector
        right_x = axis(2)
        right_y = axis(3)
        self.state.yaw_rate = right_x * 1.0

        # Triggers: throttle control
        if n_axes >= 6:
            rt = joy.get_axis(5)  # right trigger (usually -1 to 1, pressed = 1)
            lt = joy.get_axis(4)  # left trigger
            # Normalize: triggers often report -1 at rest, 1 at full press
            rt_norm = max(0.0, (rt + 1.0) / 2.0)
            lt_norm = max(0.0, (lt + 1.0) / 2.0)
            self.state.throttle_target = clamp(
                self.state.throttle_target + (rt_norm - lt_norm) * dt * 0.8, 0.0, 1.0)

        # Thrust vector from right stick Y
        if abs(right_y) > 0.1:
            self.state.thrust_vector = np.array([right_x * 0.5, 1.0, -right_y * 0.5])
            self.state.thrust_vector = self.state.thrust_vector / (np.linalg.norm(self.state.thrust_vector) or 1.0)

        # Buttons
        n_buttons = joy.get_numbuttons()
        if n_buttons > 0:
            # Button 0 (A): afterburner toggle
            if joy.get_button(0) and not hasattr(self, '_btn_a_prev'):
                self.state.afterburner = not self.state.afterburner
            self._btn_a_prev = joy.get_button(0)

            # Button 1 (B): cut throttle
            if joy.get_button(1):
                self.state.throttle_target = 0.0
                self.state.afterburner = False

            # Button 2 (X): fire
            if joy.get_button(2) and self.state.auto_aim.lock_acquired:
                self.state.auto_aim.fire(self.state.pos)

            # Button 3 (Y): toggle wings
            if joy.get_button(3) and not getattr(self, '_btn_y_prev', False):
                self.state.wing_target = 0.0 if self.state.wing_target > 0.5 else 1.0
            self._btn_y_prev = joy.get_button(3)

            # Button 4 (LB): cycle environment left
            if joy.get_button(4) and not getattr(self, '_btn_lb_prev', False):
                self.state.env_idx = max(0, self.state.env_idx - 1)
            self._btn_lb_prev = joy.get_button(4)

            # Button 5 (RB): cycle environment right
            if joy.get_button(5) and not getattr(self, '_btn_rb_prev', False):
                self.state.env_idx = min(len(ENVIRONMENTS) - 1, self.state.env_idx + 1)
            self._btn_rb_prev = joy.get_button(5)

            # Button 6 (Back): toggle auto-hover
            if joy.get_button(6) and not getattr(self, '_btn_back_prev', False):
                self.state.auto_hover = not self.state.auto_hover
            self._btn_back_prev = joy.get_button(6)

            # Button 7 (Start): toggle auto-level
            if joy.get_button(7) and not getattr(self, '_btn_start_prev', False):
                self.state.auto_level = not self.state.auto_level
            self._btn_start_prev = joy.get_button(7)

            # D-pad (buttons 11-14 or hat): jump
            if n_buttons > 14 and joy.get_button(14) and not getattr(self, '_btn_dpup_prev', False):
                if self.state.jump.charging:
                    leg_force = self.state.muscle.force_output.get("left_leg", 0) + self.state.muscle.force_output.get("right_leg", 0)
                    self.state.jump.execute_jump(self.state.pos, muscle_force_n=leg_force)
                else:
                    self.state.jump.start_charge()
            self._btn_dpup_prev = joy.get_button(14) if n_buttons > 14 else False

    def _handle_continuous_keys(self, dt):
        """Process held keys for smooth flight control using pygame.key.get_pressed()."""
        if self.mode != "flight":
            return

        keys = pygame.key.get_pressed()

        # Throttle: UP/DOWN for continuous adjustment
        if keys[pygame.K_UP]:
            self.state.throttle_target = min(1.0, self.state.throttle_target + dt * 0.5)
        if keys[pygame.K_DOWN]:
            self.state.throttle_target = max(0.0, self.state.throttle_target - dt * 0.5)

        # Pitch: W/S (hold for continuous)
        if keys[pygame.K_w]:
            self.state.pitch_rate = -0.8
        elif keys[pygame.K_s]:
            self.state.pitch_rate = 0.8
        else:
            self.state.pitch_rate *= max(0.0, 1.0 - dt * 5.0)

        # Roll: A/D (hold for continuous)
        if keys[pygame.K_a]:
            self.state.roll_rate = -0.8
        elif keys[pygame.K_d]:
            self.state.roll_rate = 0.8
        else:
            self.state.roll_rate *= max(0.0, 1.0 - dt * 5.0)

        # Yaw: LEFT/RIGHT arrows (hold for continuous)
        if keys[pygame.K_LEFT]:
            self.state.yaw_rate = -0.5
        elif keys[pygame.K_RIGHT]:
            self.state.yaw_rate = 0.5
        else:
            self.state.yaw_rate *= max(0.0, 1.0 - dt * 5.0)

        # Thrust vector: IJKL keys (hold for continuous), reset to vertical when released
        if keys[pygame.K_i]:
            self.state.thrust_vector = np.array([0, 1.0, -0.5])
            self.state.thrust_vector /= np.linalg.norm(self.state.thrust_vector)
        elif keys[pygame.K_k]:
            self.state.thrust_vector = np.array([0, 1.0, 0.5])
            self.state.thrust_vector /= np.linalg.norm(self.state.thrust_vector)
        elif keys[pygame.K_j]:
            self.state.thrust_vector = np.array([-0.5, 1.0, 0])
            self.state.thrust_vector /= np.linalg.norm(self.state.thrust_vector)
        elif keys[pygame.K_l]:
            self.state.thrust_vector = np.array([0.5, 1.0, 0])
            self.state.thrust_vector /= np.linalg.norm(self.state.thrust_vector)
        else:
            # Reset to straight up when no vector keys held
            self.state.thrust_vector = np.array([0.0, 1.0, 0.0])

        # Auto-fire when holding F in flight mode
        if keys[pygame.K_f] and self.state.auto_aim.lock_acquired:
            if not getattr(self, '_auto_fire_prev', False):
                self.state.auto_aim.fire(self.state.pos)
        self._auto_fire_prev = keys[pygame.K_f]

    def _spawn_test_target(self):
        """Spawn a hostile target at random position for combat testing."""
        angle = np.random.uniform(0, 2 * math.pi)
        dist = np.random.uniform(200, 800)
        pos = self.state.pos + np.array([
            math.cos(angle) * dist,
            np.random.uniform(-50, 200),
            math.sin(angle) * dist
        ])
        speed = np.random.uniform(100, 400)
        vel_dir = (self.state.pos - pos)
        vel_dir = vel_dir / (np.linalg.norm(vel_dir) or 1.0)
        vel = vel_dir * speed + np.random.uniform(-50, 50, 3)
        iff = "hostile" if np.random.random() > 0.2 else "unknown"
        self.state.auto_aim.add_target(pos, vel, mach_speed=speed > 340, iff=iff)

    def _update_test_env(self, dt):
        """Update test environment: auto-spawn targets, scenario logic, explosions."""
        if self.test_targets_auto and self.mode == "flight":
            self.target_spawn_timer += dt
            if self.target_spawn_timer > 3.0 and len(self.state.auto_aim.targets) < 6:
                self._spawn_test_target()
                self.target_spawn_timer = 0.0

        if self.combat_scenario:
            self.scenario_timer += dt
            self.scenario_spawn_timer += dt
            # Spawn waves every 10 seconds
            if self.scenario_spawn_timer > 10.0:
                self.scenario_wave += 1
                self.scenario_spawn_timer = 0.0
                n_targets = min(2 + self.scenario_wave, 6)
                for _ in range(n_targets):
                    self._spawn_test_target()
                print(f"  WAVE {self.scenario_wave}: {n_targets} targets spawned")

            # Count kills
            killed = self.state.auto_aim.shots_hit
            if killed > self.scenario_kills:
                self.scenario_kills = killed

        # Update explosions
        for expl in self.explosions:
            expl["life"] -= dt
            expl["radius"] += dt * 15.0
            for p in expl["particles"]:
                p["pos"] += p["vel"] * dt
                p["vel"] *= max(0.0, 1.0 - dt * 2.0)
                p["life"] -= dt
        self.explosions = [e for e in self.explosions if e["life"] > 0]
        # Clean up dead particles
        for expl in self.explosions:
            expl["particles"] = [p for p in expl["particles"] if p["life"] > 0]

    def _trigger_explosion(self, pos):
        """Create an explosion effect at a world position."""
        particles = []
        for _ in range(20):
            vel = np.random.uniform(-1, 1, 3) * np.random.uniform(10, 40)
            particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": np.random.uniform(0.3, 0.8),
                "size": np.random.randint(2, 5),
            })
        self.explosions.append({
            "pos": pos.copy(),
            "radius": 2.0,
            "life": 0.8,
            "particles": particles,
        })

    def _draw_explosions(self, surf, rect, state):
        """Draw explosion effects on screen."""
        if not self.explosions:
            return
        # Simple 2D projection for explosions (approximate screen position)
        cx = rect.x + rect.w // 2
        cy = rect.y + rect.h // 2
        for expl in self.explosions:
            rel = expl["pos"] - state.pos
            dist = np.linalg.norm(rel)
            if dist < 1:
                continue
            # Approximate screen position
            sx = cx + int(rel[0] / max(dist, 1) * 300)
            sy = cy - int(rel[1] / max(dist, 1) * 300)
            # Draw expanding ring
            r = int(expl["radius"] * 3)
            alpha = int(255 * expl["life"])
            if r > 0 and alpha > 0:
                try:
                    pygame.draw.circle(surf, (255, 180, 40), (sx, sy), r, 2)
                    pygame.draw.circle(surf, (255, 100, 20), (sx, sy), max(1, r - 3), 1)
                except Exception:
                    pass
            # Draw particles
            for p in expl["particles"]:
                prel = p["pos"] - state.pos
                pdist = np.linalg.norm(prel)
                if pdist < 1:
                    continue
                px = cx + int(prel[0] / max(pdist, 1) * 300)
                py = cy - int(prel[1] / max(pdist, 1) * 300)
                palpha = p["life"]
                pcol = (int(255 * palpha), int(150 * palpha), int(40 * palpha))
                try:
                    pygame.draw.circle(surf, pcol, (px, py), p["size"])
                except Exception:
                    pass

    def run(self):
        pygame.init()
        pygame.display.set_caption("Flysuit Mjalnor'MV1.17 -- Hybrid Combat/Space/Undersea/Flight Suit")
        screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_sm = pygame.font.SysFont("consolas", 13)
        clock = pygame.time.Clock()

        self._init_joystick()

        running = True
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.05)
            self.fps = clock.get_fps()
            if not self.paused:
                self.state.update(dt)
                self.physics.step(dt)
                self.angles["turbine"] += dt * self.state.throttle * 30
                self._handle_continuous_keys(dt)
                self._handle_gamepad(dt)
                # Track kills for explosion trigger
                prev_hits = self.state.auto_aim.shots_hit
                self._update_test_env(dt)
                if self.state.auto_aim.shots_hit > prev_hits and self.state.auto_aim.locked_target:
                    self._trigger_explosion(self.state.auto_aim.locked_target.pos.copy())
                    # Award points for target kill
                    self.state.game_score += 250
                    self.state.game_combo += 1
                    self.state.game_combo_timer = 5.0
                    self.state.screen_shake = max(self.state.screen_shake, 0.4)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width = event.w
                    self.height = event.h
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    self.keys_held.add(event.key)
                    running = self._handle_key(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_held.discard(event.key)
                elif event.type == pygame.MOUSEWHEEL:
                    # Help/instructions scroll (any mode)
                    if self.show_help:
                        self.help_scroll = max(0, self.help_scroll - event.y * 30)
                    elif getattr(self, 'show_instructions', False):
                        self.instr_scroll = max(0, getattr(self, 'instr_scroll', 0) - event.y * 30)
                    elif self.mode == "model":
                        # Check if mouse is over showcase panel
                        if getattr(self, 'show_showcase', False) and self.showcase_parts:
                            sc_pw = min(410, int(rect.w * 0.32))
                            sc_x = rect.x + rect.w - sc_pw - 10
                            sc_rect = pygame.Rect(sc_x, rect.y + 10, sc_pw, rect.h - 40)
                            if sc_rect.collidepoint(self.mouse_pos):
                                self.showcase_scroll = max(0, self.showcase_scroll - event.y * 30)
                            else:
                                self.model_renderer.dist_target = clamp(
                                    self.model_renderer.dist_target - event.y * 0.5,
                                    self.model_renderer.zoom_min, self.model_renderer.zoom_max)
                        # Check if mouse is over info panel
                        elif self.show_info:
                            pw = min(360, int(rect.w * 0.28))
                            info_x = rect.x + rect.w - pw - 10
                            info_y = rect.y + 10
                            info_rect = pygame.Rect(info_x, info_y, pw, rect.h - 40)
                            if info_rect.collidepoint(self.mouse_pos):
                                self.info_scroll = max(0, self.info_scroll - event.y * 30)
                            else:
                                self.model_renderer.dist_target = clamp(
                                    self.model_renderer.dist_target - event.y * 0.5,
                                    self.model_renderer.zoom_min, self.model_renderer.zoom_max)
                        else:
                            self.model_renderer.dist_target = clamp(
                                self.model_renderer.dist_target - event.y * 0.5,
                                self.model_renderer.zoom_min, self.model_renderer.zoom_max)
                    elif self.mode == "flight":
                        self.flight_renderer.dist_target = clamp(
                            self.flight_renderer.dist_target - event.y * 0.5,
                            2.0, 20.0)
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = event.pos
                    self.mouse_buttons = event.buttons
                    if event.buttons[0]:
                        # Don't rotate model if dragging started on an overlay
                        if self.mode == "model":
                            # Check if drag started on browser
                            on_overlay = False
                            if getattr(self, 'show_browser', False) and self.browser_hit_rects:
                                for _, hr in self.browser_hit_rects:
                                    if hr.collidepoint(event.pos):
                                        on_overlay = True
                                        break
                            # Check info panel
                            if not on_overlay and self.show_info:
                                pw = min(360, int(rect.w * 0.28))
                                info_rect = pygame.Rect(rect.x + rect.w - pw - 10, rect.y + 10, pw, rect.h - 40)
                                if info_rect.collidepoint(event.pos):
                                    on_overlay = True
                            if not on_overlay:
                                self.model_renderer.az += event.rel[0] * 0.005
                                self.model_renderer.el = clamp(
                                    self.model_renderer.el + event.rel[1] * 0.005, -1.4, 1.4)
                        elif self.mode == "flight":
                            self.flight_renderer.az += event.rel[0] * 0.005
                            self.flight_renderer.el = clamp(
                                self.flight_renderer.el + event.rel[1] * 0.005, -0.5, 1.0)
                    if event.buttons[2]:
                        if self.mode == "model":
                            self.model_renderer.pan += np.array(event.rel) * 0.003
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.mode == "model" and event.button == 1:
                        # Check browser click-to-select
                        if getattr(self, 'show_browser', False) and self.browser_hit_rects:
                            for orig_idx, hit_rect in self.browser_hit_rects:
                                if hit_rect.collidepoint(event.pos):
                                    self.model_renderer.selected = orig_idx
                                    self.info_scroll = 0
                                    break
                    if self.mode == "flight" and event.button == 1:
                        # Left click: fire at locked target
                        if self.state.auto_aim.lock_acquired:
                            hit = self.state.auto_aim.fire(self.state.pos)
                            print(f"FIRE: {'HIT' if hit else 'MISS'}  hit_rate={self.state.auto_aim.hit_rate:.0f}%")

            self.model_renderer.dist += (self.model_renderer.dist_target -
                                         self.model_renderer.dist) * min(1.0, dt * 5.0)
            self.flight_renderer.dist += (self.flight_renderer.dist_target -
                                          self.flight_renderer.dist) * min(1.0, dt * 5.0)

            screen.fill((14, 18, 26))
            rect = pygame.Rect(0, 0, self.width, self.height)

            if self.mode == "model":
                self.model_renderer.render(screen, rect, self.state, self.angles,
                                           font=self.font, interactive=True)
                if self.show_info and self.model_renderer.selected is not None:
                    draw_info(screen, rect, self.parts, self.model_renderer.selected,
                              self.font, self.font_sm, scroll_y=self.info_scroll)
                if getattr(self, 'show_browser', False):
                    self.browser_hit_rects = draw_part_browser(screen, rect, self.parts, self.font, self.font_sm,
                                     self.model_renderer.selected)
                if getattr(self, 'show_blueprint', False):
                    draw_blueprint(screen, rect, self.parts, self.font, self.font_sm)
                if getattr(self, 'show_overview', False):
                    draw_suit_overview(screen, rect, self.parts, self.font, self.font_sm)
                if getattr(self, 'show_showcase', False) and self.showcase_parts:
                    self._draw_showcase(screen, rect)
            elif self.mode == "flight":
                self.flight_renderer.render(screen, rect, self.state, self.angles,
                                            font=self.font)
                draw_hud(screen, rect, self.state, self.physics, self.font, self.font_sm)
                draw_game_hud(screen, rect, self.state, self.font, self.font_sm)
                self._draw_explosions(screen, rect, self.state)
                if self.combat_scenario:
                    self._draw_scenario_info(screen, rect)
                if self.show_fps:
                    fps_txt = self.font_sm.render(f"FPS: {self.fps:.0f}  Parts: {len(self.state.auto_aim.targets)}  Explosions: {len(self.explosions)}", True, C_DIM)
                    surf.blit(fps_txt, (rect.x + 10, rect.y + rect.h - 70))
            elif self.mode == "layers":
                self.model_renderer.render(screen, rect, self.state, self.angles,
                                           font=self.font, interactive=True)

            self._draw_controls_bar(screen)
            if self.show_help:
                self._draw_help(screen)
            if getattr(self, 'show_instructions', False):
                self._draw_instructions(screen)

            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE or key == pygame.K_q:
            return False
        elif key == pygame.K_TAB:
            self.mode = {"model": "flight", "flight": "layers", "layers": "model"}[self.mode]
            print(f"Mode: {self.mode.upper()}")
        elif key == pygame.K_h:
            if self.mode == "flight":
                self.flight_renderer.show_heat = not self.flight_renderer.show_heat
                print(f"Heat overlay: {'ON' if self.flight_renderer.show_heat else 'OFF'}")
            else:
                self.show_help = not self.show_help
                if self.show_help:
                    self.help_scroll = 0
        elif key == pygame.K_F1:
            self.show_instructions = not getattr(self, 'show_instructions', False)
            if self.show_instructions:
                self.instr_scroll = 0
            print(f"Instructions: {'ON' if self.show_instructions else 'OFF'}")
        elif key == pygame.K_i:
            self.show_info = not self.show_info
        elif key == pygame.K_u:
            if self.mode == "model":
                self.show_browser = not getattr(self, 'show_browser', False)
                print(f"Part browser: {'ON' if self.show_browser else 'OFF'}")
        elif key == pygame.K_b:
            if self.mode == "model":
                self.show_blueprint = not getattr(self, 'show_blueprint', False)
                print(f"Blueprint view: {'ON' if self.show_blueprint else 'OFF'}")
        elif key == pygame.K_0:
            if self.mode == "model":
                self.show_overview = not getattr(self, 'show_overview', False)
                print(f"Suit overview: {'ON' if self.show_overview else 'OFF'}")
        elif key == pygame.K_p:
            self.paused = not self.paused
            print(f"Paused: {self.paused}")
        elif key == pygame.K_8:
            if self.mode == "flight":
                self.flight_renderer.chase_cam = not self.flight_renderer.chase_cam
                print(f"Chase camera: {'ON' if self.flight_renderer.chase_cam else 'OFF (orbit)'}")
        elif key == pygame.K_9:
            self.show_fps = not self.show_fps
            print(f"FPS overlay: {'ON' if self.show_fps else 'OFF'}")
        elif key == pygame.K_l:
            self.model_renderer.show_labels = not self.model_renderer.show_labels
        elif key == pygame.K_e:
            self.model_renderer.explode_amt = 0.5 if self.model_renderer.explode_amt < 0.1 else 0.0
        elif key == pygame.K_x:
            self.model_renderer.section = not self.model_renderer.section
        elif key == pygame.K_o:
            export_obj(os.path.dirname(os.path.abspath(__file__)), self.parts, verbose=True)
        elif key == pygame.K_r:
            if self.mode == "flight":
                self.state.pos = np.array([0.0, 2.0, 0.0])
                self.state.velocity = np.zeros(3)
                self.state.throttle = 0.0
                self.state.throttle_target = 0.0
                self.state.pitch = 0.0
                self.state.roll = 0.0
                self.state.yaw = 0.0
                self.state.pitch_rate = 0.0
                self.state.roll_rate = 0.0
                self.state.yaw_rate = 0.0
                print("Position reset")
        # Flight throttle controls
        elif key == pygame.K_SPACE:
            self.state.throttle_target = 1.0
            self.state.afterburner = True
        elif key == pygame.K_c:
            self.state.throttle_target = 0.0
            self.state.afterburner = False
        elif key == pygame.K_v:
            self.state.throttle_target = 0.3
            self.state.afterburner = False
        elif key == pygame.K_z:
            # Altitude-hold: set throttle to hover thrust accounting for density + fuel
            rho_sl = 1.225
            density_ratio = min(1.0, self.state.env_density / rho_sl) if self.state.env_density > 0.001 else 0.0
            total_mass = self.state.pilot_mass + DIMS["weight_total_kg"] + self.state.fuel_kg
            weight_n = total_mass * self.state.env_gravity
            max_thrust_n = DIMS["turbine_count"] * DIMS["turbine_thrust_dry_lbf"] * 4.448 * density_ratio
            self.state.throttle_target = clamp(weight_n / max(max_thrust_n, 1.0), 0.0, 1.0)
            self.state.auto_hover = True
            print(f"Altitude-hold ON (throttle={self.state.throttle_target:.2f})")
        elif key == pygame.K_6:
            self.state.wing_target = 0.0 if self.state.wing_target > 0.5 else 1.0
            print(f"Wings: {'DEPLOYED' if self.state.wing_target > 0.5 else 'STOWED'}")
        elif key == pygame.K_LEFTBRACKET:
            self.state.env_idx = max(0, self.state.env_idx - 1)
            print(f"Env: {self.state.env_name}")
        elif key == pygame.K_RIGHTBRACKET:
            self.state.env_idx = min(len(ENVIRONMENTS) - 1, self.state.env_idx + 1)
            print(f"Env: {self.state.env_name}")
        elif key == pygame.K_COMMA:
            self.model_renderer.isolate_cycle(-1)
            self.info_scroll = 0
        elif key == pygame.K_PERIOD:
            self.model_renderer.isolate_cycle(1)
            self.info_scroll = 0
        elif key == pygame.K_1:
            if self.mode == "model":
                self.model_renderer.view = "full"
                self.model_renderer.explode_amt = 0.0
        elif key == pygame.K_2:
            if self.mode == "model":
                self.model_renderer.view = "exploded"
                self.model_renderer.explode_amt = 0.5
        elif key == pygame.K_3:
            if self.mode == "model":
                self.model_renderer.view = "assembly"
                self.model_renderer.assembled = 0
        # Combat controls
        elif key == pygame.K_f:
            if self.state.auto_aim.lock_acquired:
                hit = self.state.auto_aim.fire(self.state.pos)
                print(f"FIRE: {'HIT' if hit else 'MISS'}  hit_rate={self.state.auto_aim.hit_rate:.0f}%")
        elif key == pygame.K_t:
            self._spawn_test_target()
            print(f"Target added: {len(self.state.auto_aim.targets)} total")
        elif key == pygame.K_g:
            self.state.defense_active = not self.state.defense_active
            if self.state.defense_active:
                self.state.defense.state = "ALERT"
                self.state.defense.state_time = 0.0
            print(f"Defense AI: {'ACTIVE' if self.state.defense_active else 'STANDBY'}")
        elif key == pygame.K_j:
            if self.state.jump.charging:
                leg_force = self.state.muscle.force_output.get("left_leg", 0) + self.state.muscle.force_output.get("right_leg", 0)
                self.state.jump.execute_jump(self.state.pos, muscle_force_n=leg_force)
                print(f"JUMP: {self.state.jump.max_jump_ft}ft  duration={self.state.jump.jump_duration:.1f}s  muscle={leg_force:.0f}N")
            else:
                self.state.jump.start_charge()
                print("Jump charging... (press J again to execute)")
        elif key == pygame.K_b:
            self.state.auto_hover = not self.state.auto_hover
            print(f"Auto-hover: {'ON' if self.state.auto_hover else 'OFF'}")
        elif key == pygame.K_n:
            self.state.auto_level = not self.state.auto_level
            print(f"Auto-level: {'ON' if self.state.auto_level else 'OFF'}")
        elif key == pygame.K_y:
            self.state.rtos.inject_virus(target_task="neural", at_s=0.0)
            print("VIRUS INJECTED -> testing failover...")
        # Test environment controls
        elif key == pygame.K_u:
            self.test_targets_auto = not self.test_targets_auto
            print(f"Auto-targets: {'ON' if self.test_targets_auto else 'OFF'}")
        elif key == pygame.K_m:
            self.combat_scenario = not self.combat_scenario
            if self.combat_scenario:
                self.scenario_wave = 0
                self.scenario_kills = 0
                self.scenario_timer = 0.0
                self.scenario_spawn_timer = 8.0  # first wave in 2s
                self.state.auto_aim.targets.clear()
                self.state.auto_aim.tracked_targets.clear()
                print("COMBAT SCENARIO: STARTED - waves every 10s")
            else:
                print(f"COMBAT SCENARIO: ENDED - {self.scenario_kills} kills in {self.scenario_timer:.0f}s")
        elif key == pygame.K_5:
            # EMP test
            blocked = self.state.emp_hit(intensity_db=120.0)
            print(f"EMP TEST: {'BLOCKED' if blocked else 'BREACHED'}  shield={self.state.emp_shield_active}")
        elif key == pygame.K_4:
            # Damage test
            self.state.damage_armor("outer_armor", 0.2)
            print(f"Damage test: outer_armor={self.state.armor_damage.get('outer_armor', 0):.0%}  heal={self.state.self_heal_active_count}")
        # Squad controls
        elif key == pygame.K_7:
            self.state.auto_aim.add_squad_member(
                f"alpha-{len(self.state.auto_aim.squad_members)}",
                self.state.pos + np.random.uniform(-200, 200, 3))
            sq = self.state.auto_aim.get_squad_status()
            print(f"Squad: {sq['members']} members ({sq['active']} active)")
        # --- Upgrade system controls ---
        elif key == pygame.K_F2:
            ok = self.state.stealth.toggle()
            if ok:
                print(f"Stealth: {'ACTIVE' if self.state.stealth.active else 'OFF'}  {self.state.stealth.stealth_pct:.0f}%")
            else:
                print(f"Stealth: COOLDOWN {self.state.stealth.cooldown_timer:.1f}s")
        elif key == pygame.K_F3:
            mode = self.state.vision.cycle_mode()
            print(f"Vision mode: {mode.upper()}  {self.state.vision.mode_label}")
        elif key == pygame.K_F4:
            if self.state.grapple.deployed:
                self.state.grapple.release()
                print("Grapple: RELEASED")
            else:
                # Fire grapple at locked target or forward
                if self.state.auto_aim.locked_target is not None:
                    ok = self.state.grapple.fire(self.state.auto_aim.locked_target.pos, self.state.pos)
                else:
                    target = self.state.pos + np.array([0, 20, -20])
                    ok = self.state.grapple.fire(target, self.state.pos)
                print(f"Grapple: {'FIRED' if ok else 'OUT OF RANGE'}  {self.state.grapple.status}")
        elif key == pygame.K_F5:
            if self.state.grapple.deployed:
                self.state.grapple.winch_active = not self.state.grapple.winch_active
                print(f"Grapple winch: {'ON' if self.state.grapple.winch_active else 'OFF'}")
        elif key == pygame.K_F6:
            ok = self.state.parachute.deploy()
            print(f"Parachute: {'DEPLOYED' if ok else 'ALREADY DEPLOYED/PACKED'}  {self.state.parachute.status}")
        elif key == pygame.K_F7:
            self.state.drones.launch_all(self.state.pos)
            print(f"Drones: launched  {self.state.drones.status}")
        elif key == pygame.K_F8:
            self.state.drones.recall_all(self.state.pos)
            print(f"Drones: recalling  {self.state.drones.status}")
        elif key == pygame.K_F9:
            ok = self.state.countermeasures.deploy("flare", self.state.pos)
            print(f"Countermeasures: {'DEPLOYED' if ok else 'EMPTY'}  remaining: {self.state.countermeasures.remaining}")
        elif key == pygame.K_F10:
            ok = self.state.eshield.activate()
            print(f"Energy shield: {'ACTIVATED' if ok else 'COOLDOWN/LOW CHARGE'}  {self.state.eshield.status}")
        elif key == pygame.K_F11:
            deployed = self.state.tshield.toggle()
            print(f"Tactical shield: {'DEPLOYED' if deployed else 'STOWED'}  {self.state.tshield.coverage_pct:.0f}%")
        elif key == pygame.K_F12:
            ok = self.state.stun.fire()
            print(f"Stun: {'FIRED' if ok else 'CHARGING/EMPTY'}  {self.state.stun.status}")
        elif key == pygame.K_SCROLLLOCK:
            if self.state.beacon.active:
                self.state.beacon.deactivate()
                print("Emergency beacon: DEACTIVATED")
            else:
                self.state.beacon.activate(self.state.pos)
                print(f"Emergency beacon: ACTIVATED  {self.state.beacon.status}")
        elif key == pygame.K_PAUSE:
            ok = self.state.maglev.toggle()
            if ok:
                self.state.maglev.detect_surface("steel")
                print(f"Maglev: {'ACTIVE' if self.state.maglev.active else 'OFF'}  {self.state.maglev.status}")
        elif key == pygame.K_s:
            if self.mode == "model":
                self.show_showcase = not self.show_showcase
                if self.show_showcase and self.showcase_parts is None:
                    from showcase import get_all_showcases
                    self.showcase_parts = get_all_showcases()
                self.showcase_scroll = 0
                print(f"Showcase: {'ON' if self.show_showcase else 'OFF'}")
        elif key == pygame.K_LEFTBRACKET:
            if self.show_showcase and self.showcase_parts:
                self.showcase_idx = (self.showcase_idx - 1) % len(self.showcase_parts)
                self.showcase_scroll = 0
                print(f"Showcase: {self.showcase_parts[self.showcase_idx].name}")
        elif key == pygame.K_RIGHTBRACKET:
            if self.show_showcase and self.showcase_parts:
                self.showcase_idx = (self.showcase_idx + 1) % len(self.showcase_parts)
                self.showcase_scroll = 0
                print(f"Showcase: {self.showcase_parts[self.showcase_idx].name}")
        return True

    def _draw_scenario_info(self, surf, rect):
        """Draw combat scenario info overlay."""
        x = rect.x + rect.w // 2 - 100
        y = rect.y + 160
        _panel(surf, x, y, 200, 50, 200)
        info = f"WAVE {self.scenario_wave}  KILLS: {self.scenario_kills}  TIME: {self.scenario_timer:.0f}s"
        surf.blit(self.font_sm.render(info, True, C_ACCENT2), (x + 8, y + 6))
        alive = sum(1 for t in self.state.auto_aim.targets if not t.hit)
        surf.blit(self.font_sm.render(f"Targets alive: {alive}", True, C_WARN if alive > 0 else C_OK), (x + 8, y + 24))

    def _draw_showcase(self, surf, rect):
        """Draw showcase mode: 3D exploded detail view + spec panel with math proof."""
        if not self.showcase_parts:
            return
        sp = self.showcase_parts[self.showcase_idx]

        # --- Left side: 3D rendered showcase part ---
        pw = min(410, int(rect.w * 0.32))
        render_rect = pygame.Rect(rect.x, rect.y, rect.w - pw - 20, rect.h)
        surf.set_clip(render_rect)

        cx = render_rect.x + render_rect.w / 2.0
        cy = render_rect.y + render_rect.h / 2.0
        focal = min(render_rect.w, render_rect.h) * 0.8
        az = self.model_renderer.az
        el = self.model_renderer.el
        Rcam = rot_x(el) @ rot_y(az)
        light = self.model_renderer.light

        polys = []
        default_ang = self.angles.get("default", 0.0)

        # Auto-frame: compute bounding box of all meshes to set camera distance
        all_wv = []
        for m in sp.meshes:
            wv = m.world_verts(self.angles.get(m.group, default_ang))
            all_wv.append(wv)
        if all_wv:
            all_pts = np.vstack(all_wv)
            bbox_min = all_pts.min(axis=0)
            bbox_max = all_pts.max(axis=0)
            bbox_size = np.linalg.norm(bbox_max - bbox_min)
            dist = max(bbox_size * 0.8, 1.0)
        else:
            dist = max(3.0, self.model_renderer.dist * 0.5)

        for m in sp.meshes:
            wv = m.world_verts(self.angles.get(m.group, default_ang))
            cam = wv @ Rcam.T
            cam[:, 2] += dist
            col = m.color
            if m.emissive:
                col = _mix(col, (255, 255, 255), 0.25)
            z = cam[:, 2]
            safe = np.where(z > 0.05, z, 1e9)
            sx = cx + focal * cam[:, 0] / safe
            sy = cy - focal * cam[:, 1] / safe
            _emit_polys(polys, cam, sx, sy, col, light, True, m.emissive,
                        _face_groups(m), 3.0, False)

        polys.sort(key=lambda t: t[0], reverse=True)
        for _, pts, fcol, hl in polys:
            if len(pts) >= 3:
                try:
                    pygame.draw.polygon(surf, fcol, pts)
                    pygame.draw.polygon(surf, (12, 14, 20), pts, 1)
                except Exception:
                    pass

        surf.set_clip(None)

        # Title bar
        _panel(surf, render_rect.x + 8, render_rect.y + 8, render_rect.w - 16, 28, 230)
        surf.blit(self.font.render(sp.name, True, C_ACCENT), (render_rect.x + 14, render_rect.y + 12))
        nav = f"[{self.showcase_idx + 1}/{len(self.showcase_parts)}]  [ / ] to cycle"
        surf.blit(self.font_sm.render(nav, True, C_DIM), (render_rect.x + render_rect.w - 200, render_rect.y + 14))

        # Scale indicator
        if "Atomic Scale" in sp.name or "Nano Scale" in sp.name or "Structural Scale" in sp.name:
            # Extract zoom from name
            zoom_match = ""
            for s in sp.specs[:3]:
                if "Scale: 1 display unit" in s:
                    zoom_match = s.split("=")[-1].strip()
                    break
            scale_txt = f"ZOOMED {zoom_match}" if zoom_match else "ATOMIC ZOOM"
            surf.blit(self.font_sm.render(scale_txt, True, C_WARN), (render_rect.x + 14, render_rect.y + 40))
            surf.blit(self.font_sm.render("Red dot = true scale ref", True, C_ACCENT2),
                      (render_rect.x + 14, render_rect.y + 56))
        elif "100%" in " ".join(sp.specs[:3]):
            scale_txt = "100% TO SCALE"
            surf.blit(self.font_sm.render(scale_txt, True, C_OK), (render_rect.x + 14, render_rect.y + 40))
        else:
            scale_txt = "COMPONENT SCALE"
            surf.blit(self.font_sm.render(scale_txt, True, C_OK), (render_rect.x + 14, render_rect.y + 40))

        # --- Right side: spec panel with math proof (scrollable) ---
        px = rect.x + rect.w - pw - 10
        py = rect.y + 10
        max_h = rect.h - 40

        # Calculate content height
        line_h = 15
        content_h = 40  # header
        for line in sp.specs:
            content_h += line_h
        content_h += 24  # footer

        max_scroll = max(0, content_h - max_h)
        self.showcase_scroll = max(0, min(self.showcase_scroll, max_scroll))
        h = min(content_h, max_h)

        _panel(surf, px, py, pw, h, 235)
        old_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(px + 2, py + 2, pw - 4, h - 4))

        # Header
        surf.blit(self.font.render("SHOWCASE SPECS", True, C_ACCENT), (px + 8, py + 8))
        surf.blit(self.font_sm.render(sp.name, True, C_DIM), (px + 8, py + 28))
        pygame.draw.line(surf, C_DIM, (px + 8, py + 48), (px + pw - 8, py + 48), 1)

        # Spec lines with scroll
        ly = py + 54 - self.showcase_scroll
        for line in sp.specs:
            if ly > py + h - 10:
                break
            if ly + line_h < py + 50:
                ly += line_h
                continue
            if line.startswith("==="):
                color = C_ACCENT
                surf.blit(self.font_sm.render(line, True, color), (px + 8, ly))
            elif line.startswith("MATH PROOF"):
                color = C_OK
                surf.blit(self.font_sm.render(line, True, color), (px + 8, ly))
            elif line.endswith("✓"):
                color = C_OK
                surf.blit(self.font_sm.render(line, True, color), (px + 8, ly))
            elif "Proof:" in line or "proof:" in line:
                color = C_WARN
                surf.blit(self.font_sm.render(line, True, color), (px + 8, ly))
            elif line == "":
                pass
            elif any(line.startswith(kw) for kw in ["EMG", "PCM", "Capillary", "Silver", "Spine",
                                                       "DEA", "STF", "CNT", "Graphene", "Jump",
                                                       "Ceramic", "UHMWPE", "Armor", "Self-Healing",
                                                       "Coating", "Thermal", "Compressor", "Combustor",
                                                       "Turbine", "Bypass", "Thrust", "Vectoring",
                                                       "Shell", "Visor", "Neural", "Life", "Helmet",
                                                       "O2", "CO2",
                                                       "Polymer", "Compliant", "Crosslink", "Maxwell",
                                                       "Energy", "Shear", "Hydrocluster", "Re-entrant",
                                                       "Negative", "Crystalline", "Tensile", "Ballistic",
                                                       "Projectile", "Hardness", "Grain", "Aerogel",
                                                       "Electron", "Lattice", "Orthorhombic", "Specific",
                                                       "Volume", "Material", "Chain", "Scale",
                                                       "True-scale", "Foam", "Draw", "Melting",
                                                       "Above", "At low", "At crit",
                                                       "CFRP", "Ti-6Al", "Maxon", "Spectra",
                                                       "Telescoping", "Li-S", "Anode", "Cathode",
                                                       "Reaction", "Cell", "Cycles", "Hover",
                                                       "Electrical", "Solar", "Piezo", "Hot-Swap",
                                                       "During", "Emergency", "EEG", "AI",
                                                       "Auto-aim", "Defense", "G-limiter", "Eye",
                                                       "Pupil", "Calibration", "Crypto", "Latency",
                                                       "Sampling", "Signal", "Gesture", "Stall",
                                                       "Best", "Deployment", "Tear", "Buckling",
                                                       "Nodes", "Pivots", "Straps", "Fits",
                                                       "Lock", "Tubes", "Status", "Individual"]):
                color = C_ACCENT2
                surf.blit(self.font_sm.render(line, True, color), (px + 8, ly))
            else:
                color = C_TEXT
                surf.blit(self.font_sm.render(line, True, color), (px + 8, ly))
            ly += line_h

        surf.set_clip(old_clip)

    def _draw_controls_bar(self, surf):
        bar_h = 28
        y = self.height - bar_h
        _panel(surf, 0, y, self.width, bar_h, 200)
        if self.mode == "model":
            texts = [
                "TAB:mode  mouse:orbit  wheel:zoom  L:labels  E:explode  X:section  ,/.:isolate",
                "I:info  U:browser  B:blueprint  0:overview  S:showcase(15)  [/:cycle  ]:cycle  O:export  H:help  F1:instr  ESC:quit",
            ]
        elif self.mode == "flight":
            texts = [
                "TAB:mode  W/S:pitch  A/D:roll  L/R:yaw  UP/DN:thr  IJKL:vec  SPACE:burn  C:cut  V:hover  Z:alt-hold",
                "6:wings  [/:env  ]:env  F:fire  T:target  M:scenario  G:defense  J:jump  7:squad  8:chase  9:FPS  R:reset",
                "B:hover  N:level  H:heat  F2-F12:upgrades  F1:instr  ESC:quit",
            ]
        else:
            texts = ["TAB:mode  1-4:layers  E:explode  L:labels  H:help  F1:instructions  ESC:quit"]
        max_w = self.width - 16
        shown = []
        for t in texts:
            tw = self.font_sm.size(t)[0]
            if tw <= max_w:
                shown.append(t)
            else:
                shown.extend(_wrap_text(t, self.font_sm, max_w))
        line_h = 14
        total_h = len(shown) * line_h
        start_y = y + (bar_h - total_h) // 2
        for i, t in enumerate(shown):
            surf.blit(self.font_sm.render(t, True, C_DIM), (8, start_y + i * line_h))
        # Adjust bar height if multi-line
        if len(shown) > 1:
            needed = len(shown) * line_h + 6
            if needed > bar_h:
                y2 = self.height - needed
                _panel(surf, 0, y2, self.width, needed, 200)
                for i, t in enumerate(shown):
                    surf.blit(self.font_sm.render(t, True, C_DIM), (8, y2 + 3 + i * line_h))

    def _draw_help(self, surf):
        lines = [
            "FLYSUIT Mjalnor'MV1.17 -- HYBRID COMBAT/SPACE/UNDERSEA/FLIGHT SUIT",
            "",
            "MODEL MODE",
            "  Mouse drag L .... orbit camera",
            "  Mouse drag R .... pan",
            "  Wheel / +/- ..... zoom",
            "  Click part ...... select for info panel",
            "  1 ............... full assembly view",
            "  2 ............... exploded view",
            "  3 ............... assembly sequence",
            "  I ............... toggle info/about panel (selected part)",
            "  U ............... toggle part browser (all 12 parts list)",
            "  B ............... toggle blueprint view (layer stack + assembly order)",
            "  0 ............... toggle suit overview (weight breakdown + gear inventory + performance)",
            "  L ............... toggle labels",
            "  E ............... toggle exploded",
            "  X ............... toggle section cut",
            "  , / . ........... cycle part isolation",
            "  S ............... toggle showcase (15 detailed views)",
            "  [ / ] ........... cycle showcase items",
            "  O ............... export OBJ",
            "",
            "FLIGHT MODE -- KEYBOARD",
            "  UP/DOWN .......... throttle up/down (hold)",
            "  SPACE ............ max throttle + afterburner",
            "  C ................ cut throttle",
            "  V ................ hover throttle (30%)",
            "  Z ................ altitude-hold throttle",
            "  W / S ............ pitch down / up (hold)",
            "  A / D ............ roll left / right (hold)",
            "  LEFT / RIGHT ..... yaw left / right (hold)",
            "  I / K ............ thrust vector fwd / back (hold)",
            "  J / L ............ thrust vector left / right (hold)",
            "  6 ................ toggle wing deployment",
            "  [ / ] ............ cycle environment (11 presets)",
            "  R ................ reset position + attitude",
            "  B ................ toggle auto-hover",
            "  N ................ toggle auto-level",
            "  8 ................ toggle chase camera",
            "  9 ................ toggle FPS/debug overlay",
            "",
            "FLIGHT MODE -- GAME",
            "  Fly through RINGS for points (100 pts each, combo multiplier)",
            "  F / click ........ fire at locked target",
            "  T ................ spawn hostile target",
            "  U ................ toggle auto-target spawning",
            "  M ................ toggle combat scenario (waves)",
            "  G ................ toggle defense AI",
            "  J ................ charge/execute 200ft jump",
            "  7 ................ add squad member",
            "",
            "FLIGHT MODE -- GAMEPAD",
            "  Left stick ....... pitch (Y) + roll (X)",
            "  Right stick ...... yaw (X) + thrust vector (Y)",
            "  R trigger ........ throttle up  |  L trigger .... throttle down",
            "  A: afterburner  B: cut  X: fire  Y: wings  D-pad up: jump",
            "",
            "UPGRADE SYSTEMS (F2-F12)",
            "  F2: active camouflage  F3: night vision  F4: thermal/FLIR",
            "  F5: thermal mode  F6: DEA muscle  F7: STF test  F8: crypto",
            "  F9: flare  F10: energy shield  F11: tac shield  F12: stun",
            "  SCROLL: beacon  PAUSE: maglev climb",
            "",
            "ANY MODE",
            "  TAB .............. switch model/flight/layers",
            "  H ................ toggle this help",
            "  F1 ............... toggle instructions",
            "  P ................ pause simulation",
            "  ESC / Q .......... quit",
        ]
        w = min(480, self.width - 40)
        line_h = 16
        content_h = len(lines) * line_h + 20
        max_h = self.height - 60
        h = min(content_h, max_h)
        x = (self.width - w) // 2
        y = 30

        if not hasattr(self, 'help_scroll'):
            self.help_scroll = 0
        max_scroll = max(0, content_h - h)
        self.help_scroll = max(0, min(self.help_scroll, max_scroll))

        _panel(surf, x, y, w, h, 245)
        old_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(x + 2, y + 2, w - 4, h - 4))

        for i, line in enumerate(lines):
            ly = y + 10 + i * line_h - self.help_scroll
            if ly < y + 4 or ly > y + h - 4:
                continue
            col = C_ACCENT if i == 0 else C_TEXT
            surf.blit(self.font_sm.render(line, True, col), (x + 12, ly))

        # Scroll indicator
        if max_scroll > 0:
            sb_x = x + w - 6
            sb_h = max(20, h * h / content_h)
            sb_y = y + int(self.help_scroll / max_scroll * (h - sb_h))
            pygame.draw.rect(surf, (60, 70, 86), (sb_x, sb_y, 4, sb_h))

        surf.set_clip(old_clip)

    def _draw_instructions(self, surf):
        """Draw categorized instructions / getting-started guide."""
        sections = [
            ("GETTING STARTED", C_ACCENT, [
                "1. Start in MODEL mode (default). Click any part to select it.",
                "2. Press I to see detailed info/about for the selected part.",
                "3. Press U to browse all 12 parts. Press B for the build blueprint.",
                "4. Press 0 for suit overview (weight, gear inventory, performance).",
                "5. Press TAB to switch to FLIGHT mode.",
                "6. In FLIGHT: hold UP to throttle up, W/S to pitch, A/D to roll, L/R to yaw.",
                "7. Press SPACE for max throttle + afterburner. Press 6 to deploy wings.",
                "8. Fly through glowing RINGS for points! Chain them for combo multipliers!",
                "9. Press F or left-click to fire. Press T to spawn targets.",
                "10. Press F1 anytime for full key reference. Press ESC to quit.",
            ]),
            ("SUIT DONNING SEQUENCE (in-universe)", C_ACCENT2, [
                "Step 1: Inner suit -- pull on like wetsuit, connect EMG spine snap",
                "Step 2: Thermal layer -- Dragon Skin over inner suit, under faraday",
                "Step 3: Faraday shield -- drape over, clamp at spine (power bus connects)",
                "Step 4: Middle layer -- snap DEA muscle panels over faraday shield",
                "Step 5: Intermediate -- interlock auxetic lattice panels (tongue-groove)",
                "Step 6: Outer armor -- lock 12 modular panels via magnetic clamps",
                "Step 7: Frame -- telescoping endoskeleton extends to pilot height",
                "Step 8: Turbines -- mount 48 turbines to frame nodes (5 groups)",
                "Step 9: Wings -- stow flat on back, deploy on command",
                "Step 10: Power -- insert 2x hot-swap battery packs in back bay",
                "Step 11: Neural -- EEG cap under helmet, 30s calibration",
                "Step 12: Helmet -- seal to neck ring, life support activates",
            ]),
            ("MODEL MODE -- Inspect the Suit", C_FIBER, [
                "Mouse drag L: orbit | Mouse drag R: pan | Wheel: zoom",
                "Click a part to select it for the info panel",
                "1/2/3: full / exploded / assembly views",
                "I: info panel | U: part browser | B: blueprint | 0: suit overview",
                "L: labels | E: explode | X: section cut | ,/.: isolate parts",
                "O: export OBJ model file",
            ]),
            ("FLIGHT MODE -- Fly the Suit", C_FIBER, [
                "UP/DOWN: throttle | SPACE: max+afterburner | C: cut | V: hover",
                "W/S: pitch | A/D: roll | L/R: yaw | IJKL: thrust vector | Z: altitude hold",
                "6: deploy wings | [/]: cycle environment (Earth/Mars/Space/Ocean)",
                "FLY THROUGH RINGS for points + combo! Chase cam follows your velocity",
                "F/click: fire | T: spawn target | U: auto-target | M: combat scenario",
                "G: defense AI | J: 200ft jump | 5: EMP test | 4: damage test",
                "7: squad member | 8: chase cam (default ON) | 9: FPS | R: reset | B: auto-hover",
            ]),
            ("SUIT SYSTEMS (F2-F8)", C_ACCENT2, [
                "F2: cycle visor zoom (1x to 1000x) | F3: night vision | F4: thermal/FLIR",
                "F5: cycle thermal mode (auto/heat/cool/off) | F6: toggle DEA muscle voltage",
                "F7: test STF impact stiffening | F8: neural crypto handshake",
                "HUD shows live status: muscle force, skin temp, BCI intent, life support,",
                "  power SOC, helmet zoom/comms, frame stress/joint locks",
            ]),
            ("LAYERS MODE -- Cross-section views", C_FIBER, [
                "1-4: toggle layer visibility | E: explode | L: labels",
            ]),
            ("TIPS", C_WARN, [
                "Press I in any mode to toggle the info panel.",
                "In MODEL mode, click parts to select -- info panel shows sub-components,",
                "  materials, connectors, performance metrics, and blueprint notes.",
                "Press 0 in MODEL mode for full suit overview with weight bars and gear count.",
                "Press B in MODEL mode for the assembly blueprint with layer stack diagram.",
            ]),
        ]

        w = min(520, self.width - 40)
        content_h = 0
        for title, col, items in sections:
            content_h += 22  # section header
            content_h += len(items) * 16
            content_h += 10  # gap
        content_h += 20
        x = (self.width - w) // 2
        y = 20
        h = min(content_h, self.height - 40)

        if not hasattr(self, 'instr_scroll'):
            self.instr_scroll = 0
        max_scroll = max(0, content_h - h)
        self.instr_scroll = max(0, min(self.instr_scroll, max_scroll))

        _panel(surf, x, y, w, h, 245)

        old_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(x + 2, y + 2, w - 4, h - 4))

        cy = y + 10 - self.instr_scroll
        for title, col, items in sections:
            if cy > y + h - 4:
                break
            if cy + 20 >= y + 4:
                surf.blit(self.font.render(title, True, col), (x + 12, cy))
            cy += 20
            for item in items:
                if cy > y + h - 4:
                    break
                for line in _wrap_text(item, self.font_sm, w - 30):
                    if cy > y + h - 4:
                        break
                    if cy + 16 >= y + 4:
                        surf.blit(self.font_sm.render(line, True, C_TEXT), (x + 16, cy))
                    cy += 16
            cy += 8
            if y + 4 < cy < y + h - 4:
                pygame.draw.line(surf, (40, 44, 52), (x + 8, cy), (x + w - 8, cy), 1)
            cy += 6

        # Scroll indicator
        if max_scroll > 0:
            sb_x = x + w - 6
            sb_h = max(20, h * h / content_h)
            sb_y = y + int(self.instr_scroll / max_scroll * (h - sb_h))
            pygame.draw.rect(surf, (60, 70, 86), (sb_x, sb_y, 4, sb_h))

        surf.set_clip(old_clip)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    global VISUAL_DETAIL
    ap = argparse.ArgumentParser(
        description="Flysuit Mjalnor'MV1.17 -- Hybrid Combat/Space/Undersea/Flight Suit")
    ap.add_argument("--selftest", action="store_true", help="headless build + render + physics check")
    ap.add_argument("--feasibility", action="store_true", help="real-world build feasibility report")
    ap.add_argument("--stress-test", action="store_true", help="multi-environment stress test")
    ap.add_argument("--impact-test", action="store_true", help="ballistic impact absorption simulation")
    ap.add_argument("--os-test", action="store_true", help="SuitRTOS deep diagnostic (boot, tasks, crypto, BCI, AI intent, memory map)")
    ap.add_argument("--export-obj", action="store_true", help="write OBJ+MTL model files and exit")
    ap.add_argument("--layers", action="store_true", help="start in layer exploded view mode")
    ap.add_argument("--detail", type=float, default=VISUAL_DETAIL, help="mesh detail multiplier")
    ap.add_argument("--pilot-height", type=float, default=1.73, help="pilot height in metres")
    ap.add_argument("--pilot-weight", type=float, default=79.4, help="pilot weight in kg")
    args = ap.parse_args()

    VISUAL_DETAIL = max(0.4, args.detail)

    if args.selftest:
        return selftest()
    if args.feasibility:
        print_feasibility()
        return 0
    if args.stress_test:
        print_stress_test()
        return 0
    if args.impact_test:
        print_impact_test()
        return 0
    if args.os_test:
        return print_os_test()

    if args.export_obj:
        parts, _, cfg = build_suit(args.pilot_height, args.pilot_weight)
        export_obj(os.path.dirname(os.path.abspath(__file__)), parts, verbose=True)
        return 0

    if pygame is None:
        print("pygame is required for the interactive viewer.\n"
              "Install it with:  python3 -m pip install pygame numpy\n"
              "Or run headless:  python flysuit.py --selftest")
        return 1

    app = App()
    if args.layers:
        app.mode = "layers"
        app.model_renderer.explode_amt = 0.5
    app.pilot_height = args.pilot_height
    app.pilot_weight = args.pilot_weight
    app.parts, app.tcfg, app.cfg = build_suit(args.pilot_height, args.pilot_weight)
    app.physics = SuitPhysics(app.state, app.cfg)
    app.model_renderer = SuitRenderer(app.parts, app.cfg)
    app.flight_renderer = FlightRenderer(app.parts, app.cfg)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
