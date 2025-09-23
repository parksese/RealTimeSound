def audio_callback(outdata, frames, time_info, status):
    """Vectorized real-time audio callback"""
    if status:
        print(status)

    # Copy current state once (avoid lock inside heavy loop)
    with state_lock:
        rpm = last_state["rpm"].copy()
        collective = last_state["collective"].copy()
        tilt = last_state["tilt"].copy()
        spd = last_state["spd"]
        aoa = last_state["aoa"]
        aos = last_state["aos"]
        azimuth = last_state["azimuth"].copy()
        last_state["azimuth"][:] = azimuth  # will be updated below

    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(frames)  # vector of [0, 1, ..., frames-1]

    # Prepare output buffer
    out_buffer = np.zeros((frames, AUDIO_CHANNELS), dtype=np.float32)

    for rotor_id in range(NUMBER_OF_ROTORS):
        omega = rpm[rotor_id] * 2 * pi / 60
        domega_dt = 0.0  # could be updated from UDP if needed

        for blade in range(NUMBER_OF_BLADES):
            source_id = rotor_id * NUMBER_OF_BLADES + blade

            # Azimuth evolution for this blade over block
            az_start = azimuth[source_id]
            az_block = az_start + omega * dt * n

            # Update azimuth state (for continuity to next block)
            azimuth[source_id] = az_block[-1]

            # Get coefficients (same for all samples in block)
            c = lookup.get_coefficients(spd, aos)
            a0, a1, b1, a2, b2 = c["a0"], c["a1"], c["b1"], c["a2"], c["b2"]

            # Lift (periodic loading)
            L = (a0
                 + a1 * np.cos(az_block) + b1 * np.sin(az_block)
                 + a2 * np.cos(2 * az_block) + b2 * np.sin(2 * az_block))

            # Source position evolution
            x = prop_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = prop_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block)
            z = prop_center[rotor_id][2] + 0.0
            source_pos = np.stack((x, y, np.full_like(x, z)), axis=1)

            # Observer vector
            r = observer_position - source_pos
            rmag = np.linalg.norm(r, axis=1)

            # Mach vector (for simplicity)
            M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
            Mi = np.stack((-M * np.sin(az_block), M * np.cos(az_block), np.zeros_like(az_block)), axis=1)

            # Force vector
            Fi = np.stack((np.zeros_like(az_block),
                           np.zeros_like(az_block),
                           L), axis=1)

            # Dot products
            Mr = np.sum(r * Mi, axis=1) / rmag
            Fr = np.sum(r * Fi, axis=1) / rmag

            # Pressure (vectorized form, near-field only for speed)
            p_near = (0.25 / pi) * (
                1 / (1 - Mr) ** 2 / rmag**2
                * (Fr * (1 - M**2) / (1 - Mr) - np.sum(Fi * Mi, axis=1))
            )

            # Accumulate into output buffer (stereo: duplicate channels)
            out_buffer[:, 0] += p_near
            out_buffer[:, 1] += p_near

    # Save updated azimuth back
    with state_lock:
        last_state["azimuth"][:] = azimuth

    # Apply scaling
    out_buffer *= SCALING_FACTOR

    # Assign to output
    outdata[:] = out_buffer