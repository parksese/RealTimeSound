def generate_block(input_data, n_samples):
    """
    Run one audio block of the full acoustic model (offline).
    Returns an (n_samples, 2) stereo pressure array.
    """
    global rpm_filtered, coll_filtered, azimuth

    out_buffer = np.zeros((n_samples, AUDIO_CHANNELS), dtype=np.float32)
    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(n_samples)

    spd = input_data["spd"]
    aos = input_data["aos"]
    aoa = input_data["aoa"]
    tilt = input_data["tilt"]
    rpm_target = input_data["rpm"]
    coll_target = input_data["coll"]

    # Initialize smoothed values
    for rotor_id in range(NUMBER_OF_ROTORS):
        rpm_filtered[rotor_id] = rpm_target[rotor_id]
        coll_filtered[rotor_id] = coll_target[rotor_id]

    # ===================== MAIN LOOP =====================
    for rotor_id in range(NUMBER_OF_ROTORS):

        omega = rotor_direction[rotor_id] * rpm_filtered[rotor_id] * 2*np.pi/60
        if abs(omega) < 1e-9:
            continue

        # lookup coefficients (same as real-time)
        coeffs = lookup.get_coefficients(
            spd=spd,
            aoa=aoa,
            coll=coll_filtered[rotor_id],
            aos=aos
        )
        a0 = coeffs["a0"]
        a1 = coeffs["a1"]
        b1 = coeffs["b1"]
        a2 = coeffs["a2"]
        b2 = coeffs["b2"]

        # rotor tilt transform
        tilt_rad = np.radians(90 - tilt[rotor_id])
        trans_tilt = np.array([
            [np.cos(tilt_rad), 0, -np.sin(tilt_rad)],
            [0,               1,  0              ],
            [np.sin(tilt_rad), 0, np.cos(tilt_rad)]
        ])

        # ---------------- Per-blade loop ----------------
        for blade in range(NUMBER_OF_BLADES):

            source_id = rotor_id * NUMBER_OF_BLADES + blade
            az_start = azimuth[source_id]
            az_block = az_start + omega * dt * n
            azimuth[source_id] = az_block[-1]

            # aerodynamic source term
            L = (a0 +
                 a1 * np.cos(az_block) +
                 b1 * np.sin(az_block) +
                 a2 * np.cos(2 * az_block) +
                 b2 * np.sin(2 * az_block))

            # geometry
            x = rotor_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = rotor_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block)
            z = np.full_like(x, rotor_center[rotor_id][2])
            source_pos = np.stack((x, y, z), axis=1)

            # apply rotor tilt
            source_pos = tilt_center[rotor_id] + (source_pos - tilt_center[rotor_id]) @ trans_tilt.T

            r = observer_position - source_pos
            rmag = np.linalg.norm(r, axis=1)

            # Mach velocity components
            M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
            Mi = np.stack(
                (-M*np.sin(az_block),
                  M*np.cos(az_block),
                  np.zeros_like(az_block)),
                axis=1
            ) @ trans_tilt.T

            Fi = np.stack(
                (np.zeros_like(L), np.zeros_like(L), L),
                axis=1
            ) @ trans_tilt.T

            Mr = np.sum(r*Mi, axis=1) / rmag
            Fr = np.sum(r*Fi, axis=1) / rmag

            p_near = (0.25/pi) * (
                (1/(1-Mr)**2) / (rmag**2) *
                (Fr*(1-M**2)/(1-Mr) - np.sum(Fi*Mi, axis=1))
            )

            out_buffer[:,0] += p_near
            out_buffer[:,1] += p_near

    return out_buffer
    
    def write_tecplot(filename, pressure, sample_rate=AUDIO_SAMPLE_RATE):
    t = np.arange(len(pressure)) / sample_rate

    with open(filename, "w") as f:
        f.write('TITLE = "Rotor Acoustic Pressure"\n')
        f.write('VARIABLES = "t" "p"\n')
        f.write(f'ZONE T="Block" I={len(pressure)}\n')

        for ti, pi in zip(t, pressure):
            f.write(f"{ti:.8f}  {pi:.8e}\n")

    print(f"Tecplot file written: {filename}")
    