import numpy as np
import matplotlib.pyplot as plt

# =========================
# Constants
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
T_HIST         = 1.0

# ============================================
# Azimuth history class
# ============================================
class AzimuthHistory:
    def __init__(self):
        self.times = []
        self.omega = []
        self.az    = []
        self.az_current = 0.0

    def update(self, t_sim, omega):
        if self.times:
            dt = t_sim - self.times[-1]
            self.az_current += self.omega[-1] * dt

        self.times.append(float(t_sim))
        self.omega.append(float(omega))
        self.az.append(float(self.az_current))

        cutoff = t_sim - T_HIST
        while len(self.times) > 2 and self.times[0] < cutoff:
            self.times.pop(0)
            self.omega.pop(0)
            self.az.pop(0)

    def azimuth_at(self, t_query):
        # Find location in history
        if len(self.times) < 2:
            return self.az[-1]

        i = np.searchsorted(self.times, t_query) - 1
        if i < 0: 
            i = 0
        if i >= len(self.times)-1:
            i = len(self.times)-2

        t0, t1 = self.times[i], self.times[i+1]
        a0, a1 = self.az[i], self.az[i+1]
        w0, w1 = self.omega[i], self.omega[i+1]

        if t1 == t0:
            return a0

        alpha = (t_query - t0)/(t1 - t0)
        omega_r = w0 + alpha*(w1 - w0)

        return a0 + omega_r*(t_query - t0)


# ============================================
# Blade position
# ============================================
def blade_position(azimuth):
    x = ROTOR_RADIUS * np.cos(azimuth)
    y = ROTOR_RADIUS * np.sin(azimuth)
    return np.array([x, y, 0.0])


# ============================================
# Retarded-time solver
# ============================================
def retarded_time(t_obs, observer_pos, az_hist):
    # initial guess
    az_now = az_hist.azimuth_at(t_obs)
    pos_now = blade_position(az_now)
    R0 = np.linalg.norm(observer_pos - pos_now)
    t_r = t_obs - R0/SPEED_OF_SOUND

    # one refinement
    az_r = az_hist.azimuth_at(t_r)
    pos_r = blade_position(az_r)
    R = np.linalg.norm(observer_pos - pos_r)
    t_r = t_obs - R/SPEED_OF_SOUND

    return t_r


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    observer_pos = np.array([10.0, 0.0, 0.0])

    az_hist = AzimuthHistory()

    # -------------------------------
    # 1. Build source-time history
    # -------------------------------
    t = 0.0
    dt = 0.002
    T_total = 1.0

    while t < T_total:
        rpm = 300 + 60 * t         # accelerating rotor
        omega = rpm * 2*np.pi/60.0
        az_hist.update(t, omega)
        t += dt

    # ----------------------------------------------
    # 2. SOURCE-TIME-BASED: compute observer arrival times
    # ----------------------------------------------
    src_t  = np.array(az_hist.times)
    src_az = np.array(az_hist.az)

    src_obs_times = []
    for ts, azs in zip(src_t, src_az):
        pos = blade_position(azs)
        R = np.linalg.norm(observer_pos - pos)
        tobs = ts + R/SPEED_OF_SOUND
        src_obs_times.append(tobs)

    src_obs_times = np.array(src_obs_times)
    src_az_deg = np.rad2deg(src_az)

    # ----------------------------------------------
    # Observer time window that is physically valid
    # ----------------------------------------------
    t_obs_min = src_obs_times.min()
    t_obs_max = src_obs_times.max()

    # Uniform observer-time grid
    obs_times_uniform = np.linspace(t_obs_min, t_obs_max, 500)

    # -------------------------------------------------
    # 3. INTERPOLATE source-time signal into observer-time
    # -------------------------------------------------
    src_interp_az_deg = np.interp(obs_times_uniform, src_obs_times, src_az_deg)

    # -------------------------------------------------
    # 4. OBSERVER-TIME-BASED RETARDED SOLUTION
    # -------------------------------------------------
    ret_az_deg = []
    for t_obs in obs_times_uniform:
        t_r = retarded_time(t_obs, observer_pos, az_hist)
        ret_az_deg.append(np.rad2deg(az_hist.azimuth_at(t_r)))

    ret_az_deg = np.array(ret_az_deg)

    # -------------------------------------------------
    # 5. PLOT ALL THREE
    # -------------------------------------------------
    plt.figure(figsize=(11, 7))

    # 1) raw source-time → observer-time (non-uniform)
    plt.plot(src_obs_times, src_az_deg, 'o', markersize=4,
             label="(1) Source-time based (non-uniform t_obs)")

    # 2) interpolated to uniform observer-time
    plt.plot(obs_times_uniform, src_interp_az_deg, '-', linewidth=2,
             label="(2) Source-time based interpolated to uniform observer time")

    # 3) observer-time-based retarded-time result
    plt.plot(obs_times_uniform, ret_az_deg, '--', linewidth=3,
             label="(3) Observer-time based retarded-time solution")

    plt.xlabel("Observer time t_obs [s]")
    plt.ylabel("Azimuth [deg]")
    plt.title("Comparison of Source-Time vs Observer-Time Retarded-Azimuth")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    
    
    
    


import numpy as np
import time
import matplotlib.pyplot as plt

# =========================
# GLOBAL CONSTANTS
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0       # m (blade tip radius)
RHO            = 1.225     # air density [kg/m^3] (not heavily used here)

# Simulation / geometry
N_ROTORS = 4
N_BLADES = 5               # per rotor
N_SOURCES = N_ROTORS * N_BLADES

T_TOTAL = 0.5              # total simulation time [s] for building history
DT_SIM  = 0.0005           # source time step [s]


# =========================
# AZIMUTH HISTORY CLASS
# =========================
class AzimuthHistory:
    """
    Stores azimuth and omega for one rotor (hub) vs source time.
    Blades share the same azimuth history but have phase offsets.
    """

    def __init__(self):
        self.times = []
        self.az    = []
        self.omega = []
        self.az_current = 0.0

    def update(self, t_sim, omega_now):
        """Update azimuth history at new simulation time t_sim."""
        if self.times:
            dt = t_sim - self.times[-1]
            # Explicit Euler for azimuth integration using previous omega
            self.az_current += self.omega[-1] * dt

        self.times.append(float(t_sim))
        self.az.append(float(self.az_current))
        self.omega.append(float(omega_now))

    def azimuth_and_omega_at(self, t_query):
        """Return (azimuth, omega) at arbitrary source time by linear interpolation."""
        times = self.times
        if len(times) < 2:
            return self.az[-1], self.omega[-1]

        i = np.searchsorted(times, t_query) - 1
        if i < 0:
            i = 0
        if i >= len(times) - 1:
            i = len(times) - 2

        t0, t1 = times[i], times[i+1]
        az0, az1 = self.az[i], self.az[i+1]
        w0,  w1  = self.omega[i], self.omega[i+1]

        if t1 == t0:
            return az0, w0

        alpha = (t_query - t0) / (t1 - t0)
        az_interp = az0 + alpha * (az1 - az0)
        w_interp  = w0  + alpha * (w1  - w0)
        return az_interp, w_interp


# =========================
# GEOMETRY & KINEMATICS
# =========================
def blade_kinematics(psi, omega, rotor_center, rotor_orientation, radius=ROTOR_RADIUS):
    """
    Given azimuth psi and angular speed omega, return (pos, vel) of a blade tip
    in world coordinates for one source.
    - rotor_center: (3,)
    - rotor_orientation: (3,3) rotation matrix
    """
    # position in rotor-disk coordinates
    x_rel = radius * np.cos(psi)
    y_rel = radius * np.sin(psi)
    z_rel = 0.0
    r_rel = np.array([x_rel, y_rel, z_rel])

    # velocity in rotor-disk coordinates: v = omega k x r
    # k = (0,0,1)
    vx_rel = -omega * radius * np.sin(psi)
    vy_rel =  omega * radius * np.cos(psi)
    vz_rel = 0.0
    v_rel  = np.array([vx_rel, vy_rel, vz_rel])

    pos_world = rotor_center + rotor_orientation @ r_rel
    vel_world = rotor_orientation @ v_rel
    return pos_world, vel_world


# =========================
# LAWSON-STYLE DIPOLE PRESSURE
# =========================
def lawson_pressure(observer_pos, pos, vel, psi, omega, L0=1.0):
    """
    Very simplified Lawson-style rotating dipole:
    p'(t) ~ (1 / (4*pi*c*R)) * d/dt [ n · F / (1 - M_r)^2 ]
    Here:
      - F is assumed vertical, magnitude ~ L0 * cos(psi)
      - n · F = L0 * cos(psi) * n_z
      - d/dt(n·F) = -L0 * sin(psi) * omega * n_z   (since d/dt cos(psi)= -omega sin(psi))

    We keep a basic (1 - M_r)^-2 Doppler-like factor with
      M_r = (n · v) / c

    Returns:
      p  : acoustic pressure [Pa] (relative units)
      R  : distance [m]
    """
    r_vec = observer_pos - pos
    R = np.linalg.norm(r_vec)
    if R < 1e-6:
        return 0.0, 1e-6

    n_hat = r_vec / R

    # dipole direction e_L = z axis => n·e_L = n_z
    n_dot_eL = n_hat[2]

    # convective Mach in radiation direction
    n_dot_v = np.dot(n_hat, vel)
    M_r = n_dot_v / SPEED_OF_SOUND

    # time derivative of n·F
    d_nF_dt = -L0 * np.sin(psi) * omega * n_dot_eL

    # Lawson-like scaling
    denom = (1.0 - M_r)**2 + 1e-6
    p = (1.0 / (4.0 * np.pi * SPEED_OF_SOUND * R)) * d_nF_dt / denom

    return p, R


# =========================
# RETARDED TIME SOLVER
# =========================
def retarded_time_for_source(t_obs, observer_pos, hist, rotor_center, rotor_orientation,
                             blade_phase, n_iter=3):
    """
    Solve t_obs = t_r + R(t_r)/c for one source.
    Uses fixed-point iteration.
    """
    t_r = t_obs  # initial guess
    for _ in range(n_iter):
        az_r, _ = hist.azimuth_and_omega_at(t_r)
        psi_r = az_r + blade_phase
        pos, _ = blade_kinematics(psi_r, 0.0, rotor_center, rotor_orientation)  # omega not needed for distance
        R = np.linalg.norm(observer_pos - pos)
        t_r = t_obs - R / SPEED_OF_SOUND
    return t_r


# =========================
# BENCHMARK: SOURCE-TIME BASED
# =========================
def benchmark_source_time(rotor_histories, source_geoms, observer_pos):
    """
    Source-time-based algorithm:
      1) Build pressure in source time for each source.
      2) Map to non-uniform observer times t_obs = t_s + R(t_s)/c.
      3) Interpolate each source onto a common uniform observer-time grid.
      4) Sum pressures.
    """
    t_s_array = np.arange(0.0, T_TOTAL, DT_SIM)
    n_sources = len(source_geoms)

    arrival_times_list = []
    pressures_list = []

    t0 = time.perf_counter()

    for s_idx in range(n_sources):
        rotor_id, blade_phase, center, orient, radius, L0 = source_geoms[s_idx]
        hist = rotor_histories[rotor_id]

        t_obs_s = []
        p_s = []

        for t_s in t_s_array:
            az, omega = hist.azimuth_and_omega_at(t_s)
            psi = az + blade_phase
            pos, vel = blade_kinematics(psi, omega, center, orient, radius)
            p, R = lawson_pressure(observer_pos, pos, vel, psi, omega, L0)
            t_obs = t_s + R / SPEED_OF_SOUND

            t_obs_s.append(t_obs)
            p_s.append(p)

        t_obs_s = np.array(t_obs_s)
        p_s = np.array(p_s)

        # Ensure sorted in observer time for interpolation
        sort_idx = np.argsort(t_obs_s)
        t_obs_s = t_obs_s[sort_idx]
        p_s = p_s[sort_idx]

        arrival_times_list.append(t_obs_s)
        pressures_list.append(p_s)

    # Determine common observer-time window (intersection across all sources)
    starts = [arr[0] for arr in arrival_times_list]
    ends   = [arr[-1] for arr in arrival_times_list]
    t_obs_min = max(starts)
    t_obs_max = min(ends)

    t_obs_uniform = np.linspace(t_obs_min, t_obs_max, 4096)
    p_total_uniform = np.zeros_like(t_obs_uniform)

    # Interpolate each source onto uniform observer-time grid and sum
    for s_idx in range(n_sources):
        t_obs_s = arrival_times_list[s_idx]
        p_s = pressures_list[s_idx]
        p_interp = np.interp(t_obs_uniform, t_obs_s, p_s)
        p_total_uniform += p_interp

    t1 = time.perf_counter()
    elapsed = t1 - t0
    return t_obs_uniform, p_total_uniform, elapsed


# =========================
# BENCHMARK: OBSERVER-TIME BASED
# =========================
def benchmark_observer_time(rotor_histories, source_geoms, observer_pos, t_obs_uniform):
    """
    Observer-time-based algorithm:
      For each observer time:
        - solve retarded time for each source
        - compute Lawson pressure at that retarded time
        - sum contributions
    """
    n_sources = len(source_geoms)
    p_total = np.zeros_like(t_obs_uniform)

    t0 = time.perf_counter()

    for i, t_obs in enumerate(t_obs_uniform):
        p_sum = 0.0
        for s_idx in range(n_sources):
            rotor_id, blade_phase, center, orient, radius, L0 = source_geoms[s_idx]
            hist = rotor_histories[rotor_id]

            # Solve retarded time
            t_r = retarded_time_for_source(t_obs, observer_pos, hist,
                                           center, orient, blade_phase,
                                           n_iter=3)

            # Evaluate Lawson dipole at t_r
            az_r, omega_r = hist.azimuth_and_omega_at(t_r)
            psi_r = az_r + blade_phase
            pos_r, vel_r = blade_kinematics(psi_r, omega_r, center, orient, radius)
            p_r, _ = lawson_pressure(observer_pos, pos_r, vel_r, psi_r, omega_r, L0)
            p_sum += p_r

        p_total[i] = p_sum

    t1 = time.perf_counter()
    elapsed = t1 - t0
    return p_total, elapsed


# =========================
# MAIN: BUILD DATA & RUN
# =========================
if __name__ == "__main__":

    # ----- 1. Build rotor histories (source-time evolution) -----
    rotor_histories = [AzimuthHistory() for _ in range(N_ROTORS)]

    t = 0.0
    while t < T_TOTAL:
        # Example RPM schedule: same for all rotors (you can change per rotor)
        rpm = 300.0  # constant RPM for now
        omega = rpm * 2.0 * np.pi / 60.0

        for r in range(N_ROTORS):
            rotor_histories[r].update(t, omega)

        t += DT_SIM

    # ----- 2. Build geometry for all sources (rotor x blade) -----
    source_geoms = []
    rotor_centers = [
        np.array([ 5.0,  5.0, 0.0]),
        np.array([-5.0,  5.0, 0.0]),
        np.array([-5.0, -5.0, 0.0]),
        np.array([ 5.0, -5.0, 0.0]),
    ]
    # All rotors flat, no tilt
    rotor_orientation = np.eye(3)

    # One dipole strength for all sources (relative)
    L0 = 1.0

    for r in range(N_ROTORS):
        center = rotor_centers[r]
        for b in range(N_BLADES):
            blade_phase = 2.0 * np.pi * b / N_BLADES
            # (rotor_id, blade_phase, center, orientation, radius, L0)
            source_geoms.append((r, blade_phase, center, rotor_orientation, ROTOR_RADIUS, L0))

    # ----- 3. Define observer position -----
    observer_pos = np.array([0.0, 0.0, 0.0])  # center (e.g., cockpit)

    # ----- 4. Run SOURCE-TIME-BASED benchmark -----
    t_obs_uniform, p_src_uniform, time_src = benchmark_source_time(
        rotor_histories, source_geoms, observer_pos
    )

    # ----- 5. Run OBSERVER-TIME-BASED benchmark (use same t_obs grid) -----
    p_obs_uniform, time_obs = benchmark_observer_time(
        rotor_histories, source_geoms, observer_pos, t_obs_uniform
    )

    # ----- 6. Compare results -----
    # RMS difference between the two pressure signals
    diff = p_src_uniform - p_obs_uniform
    rms_diff = np.sqrt(np.mean(diff**2))
    rel_rms = rms_diff / (np.sqrt(np.mean(p_src_uniform**2)) + 1e-12)

    print("==== BENCHMARK RESULTS ====")
    print(f"Source-time-based runtime   : {time_src:.6f} s")
    print(f"Observer-time-based runtime : {time_obs:.6f} s")
    print(f"RMS difference (Pa, rel)    : {rms_diff:.3e}  (rel = {rel_rms:.3e})")

    # ----- 7. Quick plot for sanity check -----
    plt.figure(figsize=(10,6))
    plt.plot(t_obs_uniform, p_src_uniform, label="Source-time (interp to observer grid)")
    plt.plot(t_obs_uniform, p_obs_uniform, '--', label="Observer-time (retarded)", alpha=0.8)
    plt.xlabel("Observer time t_obs [s]")
    plt.ylabel("Pressure (relative units)")
    plt.title("Lawson-style Dipole: Source-time vs Observer-time")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(t_obs_uniform, diff)
    plt.xlabel("Observer time t_obs [s]")
    plt.ylabel("p_src - p_obs")
    plt.title("Difference between methods")
    plt.grid(True)

    plt.show()

