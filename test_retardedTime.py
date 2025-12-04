import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# Constants & config
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
DT_SIM         = 0.0005
T_TOTAL        = 0.5

N_ROTORS = 4
N_BLADES = 5
N_SOURCES = N_ROTORS * N_BLADES

# =========================
# Azimuth history per rotor
# =========================
class AzimuthHistory:
    def __init__(self):
        self.times = []
        self.az    = []
        self.omega = []
        self.az_current = 0.0

    def update(self, t, omega):
        if self.times:
            dt = t - self.times[-1]
            self.az_current += self.omega[-1] * dt
        self.times.append(float(t))
        self.az.append(float(self.az_current))
        self.omega.append(float(omega))

    def at(self, t_query):
        """Return (az, omega) at arbitrary time by linear interpolation."""
        times = self.times
        if len(times) < 2:
            return self.az[-1], self.omega[-1]

        i = np.searchsorted(times, t_query) - 1
        i = max(0, min(i, len(times)-2))

        t0, t1 = times[i], times[i+1]
        az0, az1 = self.az[i], self.az[i+1]
        w0,  w1  = self.omega[i], self.omega[i+1]

        if t1 == t0:
            return az0, w0

        a = (t_query - t0)/(t1 - t0)
        return az0 + a*(az1-az0), w0 + a*(w1-w0)


# =========================
# Blade kinematics
# =========================
def blade_kinematics(psi, omega, center):
    x = ROTOR_RADIUS * np.cos(psi)
    y = ROTOR_RADIUS * np.sin(psi)
    pos = center + np.array([x, y, 0.0])

    vx = -omega * ROTOR_RADIUS * np.sin(psi)
    vy =  omega * ROTOR_RADIUS * np.cos(psi)
    vel = np.array([vx, vy, 0.0])
    return pos, vel


# =========================
# Lowson rotating dipole (scalar version)
# =========================
def lowson_pressure(obs_pos, src_pos, psi, omega, L0=1.0):
    """
    Single-point Lowson near-field dipole.
    """
    r = obs_pos - src_pos          # (3,)
    rmag = np.linalg.norm(r)
    if rmag < 1e-9:
        return 0.0, 1e-9
    rhat = r / rmag

    # Mach magnitude
    M = omega * ROTOR_RADIUS / SPEED_OF_SOUND

    # Rotating Mach vector M_i
    Mi = np.array([
        -M * np.sin(psi),
         M * np.cos(psi),
         0.0
    ])

    # Dipole force direction Fi (vertical)
    Fi = np.array([0.0, 0.0, L0])

    Mr   = np.dot(rhat, Mi)
    Fr   = np.dot(rhat, Fi)
    FiMi = np.dot(Fi, Mi)

    num   = Fr * (1 - M**2) / (1 - Mr + 1e-12) - FiMi
    denom = (1 - Mr + 1e-12)**2 * (rmag**2 + 1e-12)

    p_near = (0.25/np.pi) * (num / denom)
    return p_near, rmag


# =========================
# Retarded time solver (per source)
# =========================
def solve_retarded_time(t_obs, observer, hist, center, psi_offset, n_iter=3):
    t_r = t_obs
    for _ in range(n_iter):
        az, _ = hist.at(t_r)
        psi = az + psi_offset
        pos, _ = blade_kinematics(psi, 0.0, center)
        R = np.linalg.norm(observer - pos)
        t_r = t_obs - R / SPEED_OF_SOUND
    return t_r


# =========================
# Main: Version A (1-D)
# =========================
if __name__ == "__main__":
    # 1) Build rotor histories
    rotor_hist = [AzimuthHistory() for _ in range(N_ROTORS)]

    t = 0.0
    while t < T_TOTAL:
        rpm = 300.0   # constant RPM; change to ramp if you like
        omega = rpm * 2*np.pi/60.0
        for r in range(N_ROTORS):
            rotor_hist[r].update(t, omega)
        t += DT_SIM

    # 2) Geometry & phase offsets
    rotor_centers = [
        np.array([ 5.0,  5.0, 0.0]),
        np.array([-5.0,  5.0, 0.0]),
        np.array([-5.0, -5.0, 0.0]),
        np.array([ 5.0, -5.0, 0.0]),
    ]
    hub_phase = [0.0, np.pi/2, np.pi, 3*np.pi/2]  # example hub offsets

    # blade list: (rotor_id, blade_phase_total, center)
    blade_info = []
    for r in range(N_ROTORS):
        for b in range(N_BLADES):
            blade_phase = 2*np.pi * b / N_BLADES
            total_phase = hub_phase[r] + blade_phase
            blade_info.append((r, total_phase, rotor_centers[r]))

    observer = np.array([0.0, 0.0, 0.0])

    # =========================
    # SOURCE-TIME PATH
    # =========================
    t_s = np.arange(0.0, T_TOTAL, DT_SIM)

    per_source_tobs = []
    per_source_p    = []

    t0_src_build = time.perf_counter()

    for (r, psi_offset, center) in blade_info:
        hist = rotor_hist[r]
        t_obs_list = []
        p_list     = []

        for ts in t_s:
            az, omega = hist.at(ts)
            psi = az + psi_offset
            pos, _ = blade_kinematics(psi, omega, center)
            p, R = lowson_pressure(observer, pos, psi, omega, L0=1.0)
            t_obs = ts + R / SPEED_OF_SOUND

            t_obs_list.append(t_obs)
            p_list.append(p)

        t_obs_arr = np.array(t_obs_list)
        p_arr     = np.array(p_list)

        sort_idx = np.argsort(t_obs_arr)
        per_source_tobs.append(t_obs_arr[sort_idx])
        per_source_p.append(p_arr[sort_idx])

    t1_src_build = time.perf_counter()

    # find common observer-time window
    starts = [arr[0] for arr in per_source_tobs]
    ends   = [arr[-1] for arr in per_source_tobs]
    tmin = max(starts)
    tmax = min(ends)

    N_OBS = 4096
    t_obs_uniform = np.linspace(tmin, tmax, N_OBS)

    # interpolate & sum
    p_src_uniform = np.zeros_like(t_obs_uniform)

    t0_src_interp = time.perf_counter()

    for s in range(N_SOURCES):
        t_arr = np.asarray(per_source_tobs[s]).ravel()
        p_arr = np.asarray(per_source_p[s]).ravel()
        p_src_uniform += np.interp(t_obs_uniform, t_arr, p_arr)

    t1_src_interp = time.perf_counter()

    T_src_build  = t1_src_build  - t0_src_build
    T_src_interp = t1_src_interp - t0_src_interp
    T_src_total  = T_src_build + T_src_interp

    # =========================
    # OBSERVER-TIME PATH
    # =========================
    p_obs_uniform = np.zeros_like(t_obs_uniform)

    t0_obs_total = time.perf_counter()

    for i, t_obs in enumerate(t_obs_uniform):
        p_sum = 0.0
        for (r, psi_offset, center) in blade_info:
            hist = rotor_hist[r]

            # retarded time
            t_r = solve_retarded_time(t_obs, observer, hist, center, psi_offset, n_iter=3)

            az_r, omega_r = hist.at(t_r)
            psi_r = az_r + psi_offset
            pos_r, _ = blade_kinematics(psi_r, omega_r, center)
            p_r, _ = lowson_pressure(observer, pos_r, psi_r, omega_r, L0=1.0)

            p_sum += p_r
        p_obs_uniform[i] = p_sum

    t1_obs_total = time.perf_counter()
    T_obs_total  = t1_obs_total - t0_obs_total

    # =========================
    # Compare signals
    # =========================
    diff = p_src_uniform - p_obs_uniform
    rms_diff = np.sqrt(np.mean(diff**2))
    rel_rms = rms_diff / (np.sqrt(np.mean(p_src_uniform**2)) + 1e-12)

    print("\n==== TIMING (Version A, 1-D total pressure) ====")
    print(f"Source-time: build   : {T_src_build:.6f} s")
    print(f"Source-time: interp  : {T_src_interp:.6f} s")
    print(f"Source-time: TOTAL   : {T_src_total:.6f} s")
    print(f"Observer-time: TOTAL : {T_obs_total:.6f} s")
    print("\n==== NUMERICAL DIFFERENCE ====")
    print(f"RMS diff (Pa, rel)   : {rms_diff:.3e}  (rel = {rel_rms:.3e})")

    # quick plots
    plt.figure(figsize=(10,6))
    plt.plot(t_obs_uniform, p_src_uniform, label="Source-time (interp)")
    plt.plot(t_obs_uniform, p_obs_uniform, '--', label="Observer-time (retarded)")
    plt.xlabel("Observer time [s]")
    plt.ylabel("Pressure (relative)")
    plt.title("Version A: Total Pressure Comparison")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(t_obs_uniform, diff)
    plt.xlabel("Observer time [s]")
    plt.ylabel("p_src - p_obs")
    plt.title("Pressure Difference")
    plt.grid(True)

    plt.show()














import numpy as np
import matplotlib.pyplot as plt

# =========================
# Constants & config
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
DT_SIM         = 0.0005
T_TOTAL        = 0.5

N_ROTORS = 4
N_BLADES = 5

# =========================
# Azimuth history per rotor
# =========================
class AzimuthHistory:
    def __init__(self):
        self.times = []
        self.az    = []
        self.omega = []
        self.az_current = 0.0

    def update(self, t, omega):
        if self.times:
            dt = t - self.times[-1]
            self.az_current += self.omega[-1] * dt
        self.times.append(float(t))
        self.az.append(float(self.az_current))
        self.omega.append(float(omega))

    def at(self, t_query):
        times = self.times
        if len(times) < 2:
            return self.az[-1], self.omega[-1]

        i = np.searchsorted(times, t_query) - 1
        i = max(0, min(i, len(times)-2))

        t0, t1 = times[i], times[i+1]
        az0, az1 = self.az[i], self.az[i+1]
        w0,  w1  = self.omega[i], self.omega[i+1]

        if t1 == t0:
            return az0, w0

        a = (t_query - t0)/(t1 - t0)
        return az0 + a*(az1-az0), w0 + a*(w1-w0)


# =========================
# Blade kinematics
# =========================
def blade_kinematics(psi, omega, center):
    x = ROTOR_RADIUS * np.cos(psi)
    y = ROTOR_RADIUS * np.sin(psi)
    pos = center + np.array([x, y, 0.0])

    vx = -omega * ROTOR_RADIUS * np.sin(psi)
    vy =  omega * ROTOR_RADIUS * np.cos(psi)
    vel = np.array([vx, vy, 0.0])
    return pos, vel


# =========================
# Lowson rotating dipole (scalar)
# =========================
def lowson_pressure(obs_pos, src_pos, psi, omega, L0=1.0):
    r = obs_pos - src_pos
    rmag = np.linalg.norm(r)
    if rmag < 1e-9:
        return 0.0, 1e-9
    rhat = r / rmag

    M = omega * ROTOR_RADIUS / SPEED_OF_SOUND

    Mi = np.array([
        -M * np.sin(psi),
         M * np.cos(psi),
         0.0
    ])

    Fi = np.array([0.0, 0.0, L0])

    Mr   = np.dot(rhat, Mi)
    Fr   = np.dot(rhat, Fi)
    FiMi = np.dot(Fi, Mi)

    num   = Fr * (1 - M**2) / (1 - Mr + 1e-12) - FiMi
    denom = (1 - Mr + 1e-12)**2 * (rmag**2 + 1e-12)

    p_near = (0.25/np.pi) * (num / denom)
    return p_near, rmag


# =========================
# Retarded time solver
# =========================
def solve_retarded_time(t_obs, observer, hist, center, psi_offset, n_iter=3):
    t_r = t_obs
    for _ in range(n_iter):
        az, _ = hist.at(t_r)
        psi = az + psi_offset
        pos, _ = blade_kinematics(psi, 0.0, center)
        R = np.linalg.norm(observer - pos)
        t_r = t_obs - R / SPEED_OF_SOUND
    return t_r


# =========================
# Main: Version B (3-D)
# =========================
if __name__ == "__main__":
    # 1) Build rotor histories
    rotor_hist = [AzimuthHistory() for _ in range(N_ROTORS)]

    t = 0.0
    while t < T_TOTAL:
        rpm = 300.0
        omega = rpm * 2*np.pi/60.0
        for r in range(N_ROTORS):
            rotor_hist[r].update(t, omega)
        t += DT_SIM

    # 2) Geometry & phase offsets
    rotor_centers = [
        np.array([ 5.0,  5.0, 0.0]),
        np.array([-5.0,  5.0, 0.0]),
        np.array([-5.0, -5.0, 0.0]),
        np.array([ 5.0, -5.0, 0.0]),
    ]
    hub_phase = [0.0, np.pi/2, np.pi, 3*np.pi/2]

    # blade_info: (rotor_id, blade_index, psi_offset, center)
    blade_info = []
    for r in range(N_ROTORS):
        for b in range(N_BLADES):
            blade_phase = 2*np.pi * b / N_BLADES
            psi_offset = hub_phase[r] + blade_phase
            blade_info.append((r, b, psi_offset, rotor_centers[r]))

    observer = np.array([0.0, 0.0, 0.0])

    # 3) Source-time path: build per-blade data
    t_s = np.arange(0.0, T_TOTAL, DT_SIM)

    per_blade_tobs = []   # list of 1D arrays, one per source
    per_blade_p    = []
    per_blade_psi  = []

    for (r, b, psi_offset, center) in blade_info:
        hist = rotor_hist[r]

        t_obs_list = []
        p_list     = []
        psi_list   = []

        for ts in t_s:
            az, omega = hist.at(ts)
            psi = az + psi_offset
            pos, _ = blade_kinematics(psi, omega, center)
            p, R = lowson_pressure(observer, pos, psi, omega, L0=1.0)
            t_obs = ts + R / SPEED_OF_SOUND

            t_obs_list.append(t_obs)
            p_list.append(p)
            psi_list.append(psi)

        t_obs_arr = np.array(t_obs_list)
        p_arr     = np.array(p_list)
        psi_arr   = np.array(psi_list)

        sort_idx = np.argsort(t_obs_arr)
        per_blade_tobs.append(t_obs_arr[sort_idx])
        per_blade_p.append(p_arr[sort_idx])
        per_blade_psi.append(psi_arr[sort_idx])

    # common observer-time window
    starts = [arr[0] for arr in per_blade_tobs]
    ends   = [arr[-1] for arr in per_blade_tobs]
    tmin = max(starts)
    tmax = min(ends)

    N_OBS = 4096
    t_obs_uniform = np.linspace(tmin, tmax, N_OBS)
    Nt = len(t_obs_uniform)

    # 4) Interpolate source-time results into 3D arrays
    p_src  = np.zeros((Nt, N_ROTORS, N_BLADES))
    psi_src = np.zeros((Nt, N_ROTORS, N_BLADES))

    source_idx = 0
    for (r, b, psi_offset, center) in blade_info:
        t_arr   = np.asarray(per_blade_tobs[source_idx]).ravel()
        p_arr   = np.asarray(per_blade_p[source_idx]).ravel()
        psi_arr = np.asarray(per_blade_psi[source_idx]).ravel()

        p_src[:, r, b]  = np.interp(t_obs_uniform, t_arr, p_arr)
        psi_src[:, r, b] = np.interp(t_obs_uniform, t_arr, psi_arr)

        source_idx += 1

    # total pressure by source-time method
    p_total_src = p_src.sum(axis=(1,2))

    # 5) Observer-time method, fill 3D arrays
    p_obs  = np.zeros((Nt, N_ROTORS, N_BLADES))
    psi_obs = np.zeros((Nt, N_ROTORS, N_BLADES))

    for i, t_obs in enumerate(t_obs_uniform):
        for (r, b, psi_offset, center) in blade_info:
            hist = rotor_hist[r]
            # retarded time
            t_r = solve_retarded_time(t_obs, observer, hist, center, psi_offset, n_iter=3)
            az_r, omega_r = hist.at(t_r)
            psi_r = az_r + psi_offset
            pos_r, _ = blade_kinematics(psi_r, omega_r, center)
            p_r, _ = lowson_pressure(observer, pos_r, psi_r, omega_r, L0=1.0)

            p_obs[i, r, b]  = p_r
            psi_obs[i, r, b] = psi_r

    p_total_obs = p_obs.sum(axis=(1,2))

    # 6) Azimuth differences
    psi_src_deg = np.rad2deg(psi_src)
    psi_obs_deg = np.rad2deg(psi_obs)
    psi_diff_deg = psi_src_deg - psi_obs_deg

    # RMS per rotor/blade
    print("\n==== Azimuth RMS differences (deg) per rotor/blade ====")
    for r in range(N_ROTORS):
        for b in range(N_BLADES):
            rms = np.sqrt(np.mean(psi_diff_deg[:, r, b]**2))
            print(f"Rotor {r}, Blade {b}: RMS = {rms:.4f} deg")

    overall_rms = np.sqrt(np.mean(psi_diff_deg**2))
    print(f"\nOverall azimuth RMS = {overall_rms:.4f} deg\n")

    # Pressure difference (total)
    p_diff = p_total_src - p_total_obs
    rms_p = np.sqrt(np.mean(p_diff**2))
    rel_rms_p = rms_p / (np.sqrt(np.mean(p_total_src**2)) + 1e-12)
    print("==== Pressure difference (total) ====")
    print(f"RMS diff (Pa, rel) = {rms_p:.3e}  (rel = {rel_rms_p:.3e})\n")

    # 7) Plots
    # total pressure
    plt.figure(figsize=(10,6))
    plt.plot(t_obs_uniform, p_total_src, label="Source-time (sum)")
    plt.plot(t_obs_uniform, p_total_obs, '--', label="Observer-time (sum)")
    plt.xlabel("Observer time [s]")
    plt.ylabel("Pressure (relative)")
    plt.title("Version B: Total Pressure Comparison")
    plt.legend()
    plt.grid(True)

    # azimuth comparison for first rotor
    plt.figure(figsize=(10,6))
    r0 = 0
    for b in range(N_BLADES):
        plt.plot(t_obs_uniform, psi_src_deg[:, r0, b], label=f"Src R{r0}B{b}")
        plt.plot(t_obs_uniform, psi_obs_deg[:, r0, b], '--', label=f"Obs R{r0}B{b}")
    plt.xlabel("Observer time [s]")
    plt.ylabel("Azimuth [deg]")
    plt.title("Azimuth Comparison (Rotor 0, all blades)")
    plt.legend()
    plt.grid(True)

    # azimuth difference for Rotor 0, Blade 0
    plt.figure(figsize=(10,4))
    plt.plot(t_obs_uniform, psi_diff_deg[:, 0, 0])
    plt.xlabel("Observer time [s]")
    plt.ylabel("Azimuth diff [deg]")
    plt.title("Azimuth Difference (Rotor 0, Blade 0)")
    plt.grid(True)

    plt.show()
    
    
    
















import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# Constants & config
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
DT_SIM         = 0.0005
T_TOTAL        = 0.5

N_ROTORS = 4
N_BLADES = 5
N_SPAN   = 6   # spanwise discretization per blade

# spanwise radii (avoid exact root)
SPAN_RADII = np.linspace(0.3 * ROTOR_RADIUS, ROTOR_RADIUS, N_SPAN)

# total "sources" = rotor × blade × span
N_ELEMENTS = N_ROTORS * N_BLADES * N_SPAN


# =========================
# Azimuth history per rotor
# =========================
class AzimuthHistory:
    def __init__(self):
        self.times = []
        self.az    = []
        self.omega = []
        self.az_current = 0.0

    def update(self, t, omega):
        if self.times:
            dt = t - self.times[-1]
            self.az_current += self.omega[-1] * dt
        self.times.append(float(t))
        self.az.append(float(self.az_current))
        self.omega.append(float(omega))

    def at_fast(self, t_query):
        """
        Faster version assuming uniform time step ~ DT_SIM.
        Returns (az, omega) at t_query via direct index interpolation.
        """
        times = self.times
        N = len(times)
        if N < 2:
            return self.az[-1], self.omega[-1]

        t0 = times[0]
        dt = times[1] - times[0]  # assumed ~ DT_SIM

        # index in [0, N-1]
        idx_float = (t_query - t0) / dt

        if idx_float <= 0:
            return self.az[0], self.omega[0]
        if idx_float >= N - 1:
            return self.az[-1], self.omega[-1]

        i0 = int(idx_float)
        alpha = idx_float - i0

        az0, az1 = self.az[i0], self.az[i0+1]
        w0,  w1  = self.omega[i0], self.omega[i0+1]

        az = az0 + alpha * (az1 - az0)
        w  = w0  + alpha * (w1  - w0)
        return az, w


# =========================
# Blade kinematics
# =========================
def blade_kinematics(psi, omega, center, radius):
    """
    Position and velocity of a spanwise element at given radius.
    """
    x = radius * np.cos(psi)
    y = radius * np.sin(psi)
    pos = center + np.array([x, y, 0.0])

    vx = -omega * radius * np.sin(psi)
    vy =  omega * radius * np.cos(psi)
    vel = np.array([vx, vy, 0.0])
    return pos, vel


# =========================
# Lowson rotating dipole (scalar)
# =========================
def lowson_pressure(obs_pos, src_pos, psi, omega, radius, L0=1.0):
    """
    Single-point Lowson near-field dipole for one spanwise element.
    """
    r = obs_pos - src_pos
    rmag = np.linalg.norm(r)
    if rmag < 1e-9:
        return 0.0, 1e-9
    rhat = r / rmag

    # Mach number at this radius
    M = omega * radius / SPEED_OF_SOUND

    # rotating Mach vector
    Mi = np.array([
        -M * np.sin(psi),
         M * np.cos(psi),
         0.0
    ])

    # dipole force direction (vertical)
    Fi = np.array([0.0, 0.0, L0])

    Mr   = np.dot(rhat, Mi)
    Fr   = np.dot(rhat, Fi)
    FiMi = np.dot(Fi, Mi)

    num   = Fr * (1 - M**2) / (1 - Mr + 1e-12) - FiMi
    denom = (1 - Mr + 1e-12)**2 * (rmag**2 + 1e-12)

    p_near = (0.25/np.pi) * (num / denom)
    return p_near, rmag


# =========================
# Retarded time solver (per element)
# =========================
def solve_retarded_time(t_obs, observer, hist, center, psi_offset, radius, n_iter=3):
    """
    Solve t_obs = t_r + R(t_r)/c for one (rotor, blade, span) element.
    Uses fixed-point iteration with fast azimuth lookup.
    """
    t_r = t_obs
    for _ in range(n_iter):
        az, _ = hist.at_fast(t_r)
        psi = az + psi_offset
        pos, _ = blade_kinematics(psi, 0.0, center, radius)
        R = np.linalg.norm(observer - pos)
        t_r = t_obs - R / SPEED_OF_SOUND
    return t_r


# =========================
# MAIN (Version A with spans)
# =========================
if __name__ == "__main__":
    # 1) Build rotor histories
    rotor_hist = [AzimuthHistory() for _ in range(N_ROTORS)]

    t = 0.0
    while t < T_TOTAL:
        rpm = 300.0
        omega = rpm * 2.0 * np.pi / 60.0
        for r in range(N_ROTORS):
            rotor_hist[r].update(t, omega)
        t += DT_SIM

    # 2) Geometry & phase offsets
    rotor_centers = [
        np.array([ 5.0,  5.0, 0.0]),
        np.array([-5.0,  5.0, 0.0]),
        np.array([-5.0, -5.0, 0.0]),
        np.array([ 5.0, -5.0, 0.0]),
    ]
    hub_phase = [0.0, np.pi/2, np.pi, 3*np.pi/2]

    # Each element: (rotor_id, psi_offset, center, radius)
    element_info = []
    for r in range(N_ROTORS):
        center = rotor_centers[r]
        for b in range(N_BLADES):
            blade_phase = 2.0 * np.pi * b / N_BLADES
            psi0 = hub_phase[r] + blade_phase
            for s in range(N_SPAN):
                radius = SPAN_RADII[s]
                element_info.append((r, psi0, center, radius))

    observer = np.array([0.0, 0.0, 0.0])

    # =========================
    # SOURCE-TIME PATH
    # =========================
    t_s = np.arange(0.0, T_TOTAL, DT_SIM)

    per_elem_tobs = []
    per_elem_p    = []

    t0_src_build = time.perf_counter()

    for (r, psi_offset, center, radius) in element_info:
        hist = rotor_hist[r]

        t_obs_list = []
        p_list     = []

        for ts in t_s:
            az, omega = hist.at_fast(ts)
            psi = az + psi_offset
            pos, _ = blade_kinematics(psi, omega, center, radius)
            p, R = lowson_pressure(observer, pos, psi, omega, radius, L0=1.0)
            t_obs = ts + R / SPEED_OF_SOUND

            t_obs_list.append(t_obs)
            p_list.append(p)

        t_obs_arr = np.array(t_obs_list)
        p_arr     = np.array(p_list)

        # often monotonic, but we keep sort for safety
        if np.all(np.diff(t_obs_arr) >= 0):
            per_elem_tobs.append(t_obs_arr)
            per_elem_p.append(p_arr)
        else:
            idx = np.argsort(t_obs_arr)
            per_elem_tobs.append(t_obs_arr[idx])
            per_elem_p.append(p_arr[idx])

    t1_src_build = time.perf_counter()

    # common observer-time window
    starts = [arr[0] for arr in per_elem_tobs]
    ends   = [arr[-1] for arr in per_elem_tobs]
    tmin = max(starts)
    tmax = min(ends)

    N_OBS = 4096
    t_obs_uniform = np.linspace(tmin, tmax, N_OBS)

    # interpolate & sum
    p_src_uniform = np.zeros_like(t_obs_uniform)

    t0_src_interp = time.perf_counter()

    for e in range(N_ELEMENTS):
        t_arr = np.asarray(per_elem_tobs[e]).ravel()
        p_arr = np.asarray(per_elem_p[e]).ravel()
        p_src_uniform += np.interp(t_obs_uniform, t_arr, p_arr)

    t1_src_interp = time.perf_counter()

    T_src_build  = t1_src_build  - t0_src_build
    T_src_interp = t1_src_interp - t0_src_interp
    T_src_total  = T_src_build + T_src_interp

    # =========================
    # OBSERVER-TIME PATH
    # =========================
    p_obs_uniform = np.zeros_like(t_obs_uniform)

    t0_obs_total = time.perf_counter()

    for i, t_obs in enumerate(t_obs_uniform):
        p_sum = 0.0
        for (r, psi_offset, center, radius) in element_info:
            hist = rotor_hist[r]

            # retarded time for this element
            t_r = solve_retarded_time(t_obs, observer, hist, center, psi_offset, radius, n_iter=3)

            # final evaluation at t_r
            az_r, omega_r = hist.at_fast(t_r)
            psi_r = az_r + psi_offset
            pos_r, _ = blade_kinematics(psi_r, omega_r, center, radius)
            p_r, _ = lowson_pressure(observer, pos_r, psi_r, omega_r, radius, L0=1.0)

            p_sum += p_r

        p_obs_uniform[i] = p_sum

    t1_obs_total = time.perf_counter()
    T_obs_total  = t1_obs_total - t0_obs_total

    # =========================
    # Compare signals
    # =========================
    diff = p_src_uniform - p_obs_uniform
    rms_diff = np.sqrt(np.mean(diff**2))
    rel_rms = rms_diff / (np.sqrt(np.mean(p_src_uniform**2)) + 1e-12)

    print("\n==== TIMING with 6 span stations (Version A) ====")
    print(f"Source-time: build   : {T_src_build:.6f} s")
    print(f"Source-time: interp  : {T_src_interp:.6f} s")
    print(f"Source-time: TOTAL   : {T_src_total:.6f} s")
    print(f"Observer-time: TOTAL : {T_obs_total:.6f} s")
    print("\n==== NUMERICAL DIFFERENCE ====")
    print(f"RMS diff (Pa, rel)   : {rms_diff:.3e}  (rel = {rel_rms:.3e})")

    # quick plots
    plt.figure(figsize=(10,6))
    plt.plot(t_obs_uniform, p_src_uniform, label="Source-time (interp)")
    plt.plot(t_obs_uniform, p_obs_uniform, '--', label="Observer-time (retarded)")
    plt.xlabel("Observer time [s]")
    plt.ylabel("Pressure (relative)")
    plt.title("Total Pressure Comparison (with 6 spanwise elements)")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(t_obs_uniform, diff)
    plt.xlabel("Observer time [s]")
    plt.ylabel("p_src - p_obs")
    plt.title("Pressure Difference")
    plt.grid(True)

    plt.show()
    
    
--------------------    
A-opt1
--------------------

import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# Constants & config
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
DT_SIM         = 0.0005
T_TOTAL        = 0.5

N_ROTORS = 4
N_BLADES = 5
N_SPAN   = 6   # spanwise discretization per blade

# spanwise radii (avoid exact root)
SPAN_RADII = np.linspace(0.3 * ROTOR_RADIUS, ROTOR_RADIUS, N_SPAN)

# total elements = rotor × blade × span
N_ELEMENTS = N_ROTORS * N_BLADES * N_SPAN


# =========================
# Azimuth history per rotor
# =========================
class AzimuthHistory:
    def __init__(self):
        self.times_list = []
        self.az_list    = []
        self.omega_list = []
        self.az_current = 0.0
        # will be filled later
        self.times = None
        self.az    = None
        self.omega = None
        self.dt    = None

    def update(self, t, omega):
        if self.times_list:
            dt = t - self.times_list[-1]
            self.az_current += self.omega_list[-1] * dt
        self.times_list.append(float(t))
        self.az_list.append(float(self.az_current))
        self.omega_list.append(float(omega))

    def finalize(self):
        """Convert lists to arrays and store dt."""
        self.times = np.array(self.times_list, dtype=float)
        self.az    = np.array(self.az_list, dtype=float)
        self.omega = np.array(self.omega_list, dtype=float)
        if len(self.times) >= 2:
            self.dt = self.times[1] - self.times[0]
        else:
            self.dt = DT_SIM

    def at_fast_scalar(self, t_query):
        """Scalar version (not used in main loop now, kept for completeness)."""
        times = self.times
        az    = self.az
        omega = self.omega
        N = len(times)
        if N < 2:
            return az[-1], omega[-1]

        t0 = times[0]
        dt = self.dt
        idx_float = (t_query - t0) / dt

        if idx_float <= 0:
            return az[0], omega[0]
        if idx_float >= N - 1:
            return az[-1], omega[-1]

        i0 = int(idx_float)
        alpha = idx_float - i0

        az0, az1 = az[i0], az[i0+1]
        w0,  w1  = omega[i0], omega[i0+1]

        azq = az0 + alpha * (az1 - az0)
        wq  = w0  + alpha * (w1  - w0)
        return azq, wq


def interp_hist_array(hist: AzimuthHistory, t_query):
    """
    Vectorized interpolation of azimuth & omega for a given rotor history.
    t_query: (K,) array
    Returns:
        az_q   : (K,)
        omega_q: (K,)
    """
    times = hist.times
    az    = hist.az
    omega = hist.omega
    dt    = hist.dt
    N     = len(times)

    if N < 2:
        # trivial fallback
        return np.full_like(t_query, az[-1]), np.full_like(t_query, omega[-1])

    t0 = times[0]

    idx_float = (t_query - t0) / dt
    # clamp to [0, N-1)
    idx_float = np.clip(idx_float, 0.0, N - 1.000001)

    i0 = idx_float.astype(int)
    alpha = idx_float - i0
    i1 = i0 + 1
    i1 = np.clip(i1, 0, N - 1)

    az0 = az[i0]
    az1 = az[i1]
    w0  = omega[i0]
    w1  = omega[i1]

    az_q = az0 + alpha * (az1 - az0)
    w_q  = w0  + alpha * (w1 - w0)
    return az_q, w_q


# =========================
# Lowson rotating dipole (vectorized)
# =========================
def lowson_pressure_array(obs_pos, src_pos, psi, omega, radius, L0=1.0):
    """
    Vectorized Lowson near-field dipole for many elements at once.
    obs_pos : (3,)
    src_pos : (N,3)
    psi     : (N,)
    omega   : (N,)
    radius  : (N,)
    Returns:
        p_near : (N,)
        rmag   : (N,)
    """
    r = obs_pos - src_pos          # (N,3)
    rmag = np.linalg.norm(r, axis=1)
    rmag_safe = rmag + 1e-12
    rhat = r / rmag_safe[:, None]

    # Mach number at each radius
    M = omega * radius / SPEED_OF_SOUND   # (N,)

    # rotating Mach vector Mi
    Mi = np.empty_like(src_pos)
    Mi[:, 0] = -M * np.sin(psi)
    Mi[:, 1] =  M * np.cos(psi)
    Mi[:, 2] =  0.0

    # dipole force (vertical)
    Fi_z = L0
    # Fr = (rhat · Fi) = rhat_z * L0
    Fr = rhat[:, 2] * Fi_z

    # Fi·Mi = L0 * Mi_z = 0 because Mi_z = 0
    FiMi = np.zeros_like(Fr)

    Mr = np.einsum("ij,ij->i", rhat, Mi)

    num   = Fr * (1 - M**2) / (1 - Mr + 1e-12) - FiMi
    denom = (1 - Mr + 1e-12)**2 * (rmag_safe**2)

    p_near = (0.25/np.pi) * (num / denom)
    return p_near, rmag


# =========================
# MAIN (A-opt1)
# =========================
if __name__ == "__main__":
    # 1) Build rotor histories
    rotor_hist = [AzimuthHistory() for _ in range(N_ROTORS)]

    t = 0.0
    while t < T_TOTAL:
        rpm = 300.0
        omega = rpm * 2.0 * np.pi / 60.0
        for r in range(N_ROTORS):
            rotor_hist[r].update(t, omega)
        t += DT_SIM

    # finalize histories: convert to arrays, store dt
    for hist in rotor_hist:
        hist.finalize()

    # 2) Geometry & phase offsets
    rotor_centers = np.array([
        [ 5.0,  5.0, 0.0],
        [-5.0,  5.0, 0.0],
        [-5.0, -5.0, 0.0],
        [ 5.0, -5.0, 0.0],
    ], dtype=float)

    hub_phase = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=float)

    # element arrays: rotor_id, psi0, center, radius
    elem_rotor_id = np.zeros(N_ELEMENTS, dtype=int)
    elem_psi0     = np.zeros(N_ELEMENTS, dtype=float)
    elem_center   = np.zeros((N_ELEMENTS, 3), dtype=float)
    elem_radius   = np.zeros(N_ELEMENTS, dtype=float)

    idx = 0
    for r in range(N_ROTORS):
        center = rotor_centers[r]
        for b in range(N_BLADES):
            blade_phase = 2.0 * np.pi * b / N_BLADES
            psi0 = hub_phase[r] + blade_phase
            for s in range(N_SPAN):
                elem_rotor_id[idx] = r
                elem_psi0[idx]     = psi0
                elem_center[idx]   = center
                elem_radius[idx]   = SPAN_RADII[s]
                idx += 1

    observer = np.array([0.0, 0.0, 0.0])

    # =========================
    # SOURCE-TIME PATH (same structure as before, not heavily optimized yet)
    # =========================
    t_s = np.arange(0.0, T_TOTAL, DT_SIM)

    per_elem_tobs = []
    per_elem_p    = []

    t0_src_build = time.perf_counter()

    for e in range(N_ELEMENTS):
        r_id   = elem_rotor_id[e]
        psi0   = elem_psi0[e]
        center = elem_center[e]
        radius = elem_radius[e]

        hist = rotor_hist[r_id]

        t_obs_list = []
        p_list     = []

        for ts in t_s:
            az, omega = hist.at_fast_scalar(ts)
            psi = az + psi0
            # element position (scalar)
            x = radius * np.cos(psi)
            y = radius * np.sin(psi)
            src_pos = center + np.array([x, y, 0.0])

            p, R = lowson_pressure_array(observer, src_pos[None, :],
                                         np.array([psi]), np.array([omega]),
                                         np.array([radius]))
            t_obs = ts + R[0] / SPEED_OF_SOUND

            t_obs_list.append(t_obs)
            p_list.append(p[0])

        t_obs_arr = np.array(t_obs_list)
        p_arr     = np.array(p_list)

        # often monotonic, but keep sort for safety
        if np.all(np.diff(t_obs_arr) >= 0):
            per_elem_tobs.append(t_obs_arr)
            per_elem_p.append(p_arr)
        else:
            sort_idx = np.argsort(t_obs_arr)
            per_elem_tobs.append(t_obs_arr[sort_idx])
            per_elem_p.append(p_arr[sort_idx])

    t1_src_build = time.perf_counter()

    # common observer-time window
    starts = [arr[0] for arr in per_elem_tobs]
    ends   = [arr[-1] for arr in per_elem_tobs]
    tmin = max(starts)
    tmax = min(ends)

    N_OBS = 4096
    t_obs_uniform = np.linspace(tmin, tmax, N_OBS)

    # interpolate & sum
    p_src_uniform = np.zeros_like(t_obs_uniform)

    t0_src_interp = time.perf_counter()

    for e in range(N_ELEMENTS):
        t_arr = np.asarray(per_elem_tobs[e]).ravel()
        p_arr = np.asarray(per_elem_p[e]).ravel()
        p_src_uniform += np.interp(t_obs_uniform, t_arr, p_arr)

    t1_src_interp = time.perf_counter()

    T_src_build  = t1_src_build  - t0_src_build
    T_src_interp = t1_src_interp - t0_src_interp
    T_src_total  = T_src_build + T_src_interp

    # =========================
    # OBSERVER-TIME PATH (VECTORISED OVER ELEMENTS)
    # =========================
    p_obs_uniform = np.zeros_like(t_obs_uniform)

    t0_obs_total = time.perf_counter()

    # Precompute index masks for each rotor (to avoid repeated np.where)
    rotor_masks = [np.where(elem_rotor_id == r)[0] for r in range(N_ROTORS)]

    # Vectorized retarded-time iterations
    for i, t_obs in enumerate(t_obs_uniform):
        # initial guess for t_r: all equal to t_obs
        t_r = np.full(N_ELEMENTS, t_obs, dtype=float)

        for _ in range(3):  # fixed-point iterations
            az_all   = np.zeros(N_ELEMENTS, dtype=float)
            omega_all = np.zeros(N_ELEMENTS, dtype=float)

            # interpolate az, omega per rotor (vectorized)
            for r in range(N_ROTORS):
                mask = rotor_masks[r]
                if mask.size == 0:
                    continue
                t_r_sub = t_r[mask]
                az_sub, w_sub = interp_hist_array(rotor_hist[r], t_r_sub)
                az_all[mask]   = az_sub
                omega_all[mask] = w_sub

            psi_all = az_all + elem_psi0

            # compute positions for ALL elements at once
            x = elem_radius * np.cos(psi_all)
            y = elem_radius * np.sin(psi_all)
            src_pos = elem_center.copy()
            src_pos[:, 0] += x
            src_pos[:, 1] += y

            # update t_r based on distances
            r_vec = observer - src_pos
            rmag  = np.linalg.norm(r_vec, axis=1)
            t_r   = t_obs - rmag / SPEED_OF_SOUND

        # final Lowson pressure for all elements at converged t_r
        # (we already have psi_all, omega_all, src_pos from last iteration)
        p_all, _ = lowson_pressure_array(observer, src_pos, psi_all, omega_all, elem_radius)
        p_obs_uniform[i] = p_all.sum()

    t1_obs_total = time.perf_counter()
    T_obs_total  = t1_obs_total - t0_obs_total

    # =========================
    # Compare signals
    # =========================
    diff = p_src_uniform - p_obs_uniform
    rms_diff = np.sqrt(np.mean(diff**2))
    rel_rms = rms_diff / (np.sqrt(np.mean(p_src_uniform**2)) + 1e-12)

    print("\n==== TIMING with 6 span stations (Version A-opt1) ====")
    print(f"Source-time: build   : {T_src_build:.6f} s")
    print(f"Source-time: interp  : {T_src_interp:.6f} s")
    print(f"Source-time: TOTAL   : {T_src_total:.6f} s")
    print(f"Observer-time: TOTAL : {T_obs_total:.6f} s")
    print("\n==== NUMERICAL DIFFERENCE ====")
    print(f"RMS diff (Pa, rel)   : {rms_diff:.3e}  (rel = {rel_rms:.3e})")

    # quick plots
    plt.figure(figsize=(10,6))
    plt.plot(t_obs_uniform, p_src_uniform, label="Source-time (interp)")
    plt.plot(t_obs_uniform, p_obs_uniform, '--', label="Observer-time (retarded)")
    plt.xlabel("Observer time [s]")
    plt.ylabel("Pressure (relative)")
    plt.title("Total Pressure Comparison (A-opt1, 6 spanwise elements)")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(t_obs_uniform, diff)
    plt.xlabel("Observer time [s]")
    plt.ylabel("p_src - p_obs")
    plt.title("Pressure Difference")
    plt.grid(True)

    plt.show()
    
    
---------
A-opt2
----------

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit

# =========================
# Constants & config
# =========================
SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
DT_SIM         = 0.0005
T_TOTAL        = 0.5

N_ROTORS = 4
N_BLADES = 5
N_SPAN   = 6   # spanwise discretization per blade

SPAN_RADII = np.linspace(0.3 * ROTOR_RADIUS, ROTOR_RADIUS, N_SPAN)
N_ELEMENTS = N_ROTORS * N_BLADES * N_SPAN


# =========================
# Azimuth history per rotor (Python side)
# =========================
class AzimuthHistory:
    def __init__(self):
        self.times_list = []
        self.az_list    = []
        self.omega_list = []
        self.az_current = 0.0

    def update(self, t, omega):
        if self.times_list:
            dt = t - self.times_list[-1]
            self.az_current += self.omega_list[-1] * dt
        self.times_list.append(float(t))
        self.az_list.append(float(self.az_current))
        self.omega_list.append(float(omega))

    def finalize(self):
        self.times = np.array(self.times_list, dtype=np.float64)
        self.az    = np.array(self.az_list, dtype=np.float64)
        self.omega = np.array(self.omega_list, dtype=np.float64)
        if len(self.times) >= 2:
            self.dt = self.times[1] - self.times[0]
        else:
            self.dt = DT_SIM


# =========================
# Source-time Lowson (vectorized helper, used in source-time path)
# =========================
def lowson_pressure_array(obs_pos, src_pos, psi, omega, radius, L0=1.0):
    """
    Vectorized Lowson near-field dipole for many elements at once.
    obs_pos : (3,)
    src_pos : (N,3)
    psi     : (N,)
    omega   : (N,)
    radius  : (N,)
    Returns:
        p_near : (N,)
        rmag   : (N,)
    """
    r = obs_pos - src_pos
    rmag = np.linalg.norm(r, axis=1)
    rmag_safe = rmag + 1e-12
    rhat = r / rmag_safe[:, None]

    M = omega * radius / SPEED_OF_SOUND

    Mi = np.empty_like(src_pos)
    Mi[:, 0] = -M * np.sin(psi)
    Mi[:, 1] =  M * np.cos(psi)
    Mi[:, 2] =  0.0

    # vertical dipole Fi = (0,0,L0)
    Fr = rhat[:, 2] * L0
    Mr = np.einsum("ij,ij->i", rhat, Mi)

    num   = Fr * (1 - M**2) / (1 - Mr + 1e-12)
    denom = (1 - Mr + 1e-12)**2 * (rmag_safe**2)

    p_near = (0.25/np.pi) * (num / denom)
    return p_near, rmag


# =========================
# Numba-accelerated observer-time kernel
# =========================
@njit
def observer_time_kernel(
    t_obs_uniform,          # (N_obs,)
    times,                  # (N_hist,)
    az_hist,                # (N_rotors, N_hist)
    omega_hist,             # (N_rotors, N_hist)
    dt_hist,                # scalar
    elem_rotor_id,          # (N_elements,)
    elem_psi0,              # (N_elements,)
    elem_center,            # (N_elements, 3)
    elem_radius,            # (N_elements,)
    observer                # (3,)
):
    N_obs = t_obs_uniform.shape[0]
    N_hist = times.shape[0]
    N_elements = elem_rotor_id.shape[0]

    p_obs = np.zeros(N_obs, dtype=np.float64)

    t0 = times[0]
    eps = 1e-12

    for i in range(N_obs):
        t_obs = t_obs_uniform[i]

        # initial guess t_r = t_obs for all elements
        t_r = np.full(N_elements, t_obs, dtype=np.float64)

        # fixed-point iteration for retarded time
        for _ in range(3):
            # we will compute az, omega, psi, pos for all elements
            for e in range(N_elements):
                r_id = elem_rotor_id[e]

                # interpolate az, omega at t_r[e]
                idx_float = (t_r[e] - t0) / dt_hist

                if idx_float <= 0.0:
                    az = az_hist[r_id, 0]
                    w  = omega_hist[r_id, 0]
                elif idx_float >= N_hist - 1:
                    az = az_hist[r_id, N_hist-1]
                    w  = omega_hist[r_id, N_hist-1]
                else:
                    i0 = int(idx_float)
                    alpha = idx_float - i0
                    i1 = i0 + 1
                    az0 = az_hist[r_id, i0]
                    az1 = az_hist[r_id, i1]
                    w0  = omega_hist[r_id, i0]
                    w1  = omega_hist[r_id, i1]
                    az = az0 + alpha*(az1 - az0)
                    w  = w0  + alpha*(w1  - w0)

                psi0 = elem_psi0[e]
                psi  = az + psi0
                radius = elem_radius[e]

                # position at radius, psi
                x = radius * np.cos(psi)
                y = radius * np.sin(psi)

                src_x = elem_center[e, 0] + x
                src_y = elem_center[e, 1] + y
                src_z = elem_center[e, 2]

                dx = observer[0] - src_x
                dy = observer[1] - src_y
                dz = observer[2] - src_z

                R = np.sqrt(dx*dx + dy*dy + dz*dz) + eps

                # update retarded time
                t_r[e] = t_obs - R / SPEED_OF_SOUND

        # final pressure evaluation at converged t_r
        p_sum = 0.0
        for e in range(N_elements):
            r_id = elem_rotor_id[e]

            # interpolate az, omega again at final t_r
            idx_float = (t_r[e] - t0) / dt_hist

            if idx_float <= 0.0:
                az = az_hist[r_id, 0]
                w  = omega_hist[r_id, 0]
            elif idx_float >= N_hist - 1:
                az = az_hist[r_id, N_hist-1]
                w  = omega_hist[r_id, N_hist-1]
            else:
                i0 = int(idx_float)
                alpha = idx_float - i0
                i1 = i0 + 1
                az0 = az_hist[r_id, i0]
                az1 = az_hist[r_id, i1]
                w0  = omega_hist[r_id, i0]
                w1  = omega_hist[r_id, i1]
                az = az0 + alpha*(az1 - az0)
                w  = w0  + alpha*(w1  - w0)

            psi0 = elem_psi0[e]
            psi  = az + psi0
            radius = elem_radius[e]

            # position
            x = radius * np.cos(psi)
            y = radius * np.sin(psi)

            src_x = elem_center[e, 0] + x
            src_y = elem_center[e, 1] + y
            src_z = elem_center[e, 2]

            dx = observer[0] - src_x
            dy = observer[1] - src_y
            dz = observer[2] - src_z

            R = np.sqrt(dx*dx + dy*dy + dz*dz) + eps

            # Lowson dipole (inline, L0=1)
            rhat_x = dx / R
            rhat_y = dy / R
            rhat_z = dz / R

            M = w * radius / SPEED_OF_SOUND

            Mi_x = -M * np.sin(psi)
            Mi_y =  M * np.cos(psi)
            Mi_z =  0.0

            Mr = rhat_x*Mi_x + rhat_y*Mi_y + rhat_z*Mi_z
            Fr = rhat_z * 1.0    # Fi=(0,0,1)

            num   = Fr * (1.0 - M*M) / (1.0 - Mr + eps)
            denom = (1.0 - Mr + eps)**2 * (R*R)

            p = (0.25/np.pi) * (num / denom)
            p_sum += p

        p_obs[i] = p_sum

    return p_obs


# =========================
# MAIN: Source-time vs Numba observer-time
# =========================
if __name__ == "__main__":
    # 1) Build rotor histories
    rotor_hist = [AzimuthHistory() for _ in range(N_ROTORS)]

    t = 0.0
    while t < T_TOTAL:
        rpm = 300.0
        omega = rpm * 2.0 * np.pi / 60.0
        for r in range(N_ROTORS):
            rotor_hist[r].update(t, omega)
        t += DT_SIM

    for hist in rotor_hist:
        hist.finalize()

    # Build shared history arrays for Numba
    times = rotor_hist[0].times.copy()
    N_hist = len(times)
    dt_hist = rotor_hist[0].dt

    az_hist = np.zeros((N_ROTORS, N_hist), dtype=np.float64)
    omega_hist = np.zeros((N_ROTORS, N_hist), dtype=np.float64)
    for r in range(N_ROTORS):
        az_hist[r, :]    = rotor_hist[r].az
        omega_hist[r, :] = rotor_hist[r].omega

    # 2) Geometry & element mapping
    rotor_centers = np.array([
        [ 5.0,  5.0, 0.0],
        [-5.0,  5.0, 0.0],
        [-5.0, -5.0, 0.0],
        [ 5.0, -5.0, 0.0],
    ], dtype=np.float64)

    hub_phase = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float64)

    elem_rotor_id = np.zeros(N_ELEMENTS, dtype=np.int64)
    elem_psi0     = np.zeros(N_ELEMENTS, dtype=np.float64)
    elem_center   = np.zeros((N_ELEMENTS, 3), dtype=np.float64)
    elem_radius   = np.zeros(N_ELEMENTS, dtype=np.float64)

    idx = 0
    for r in range(N_ROTORS):
        center = rotor_centers[r]
        for b in range(N_BLADES):
            blade_phase = 2.0 * np.pi * b / N_BLADES
            psi0 = hub_phase[r] + blade_phase
            for s in range(N_SPAN):
                elem_rotor_id[idx] = r
                elem_psi0[idx]     = psi0
                elem_center[idx]   = center
                elem_radius[idx]   = SPAN_RADII[s]
                idx += 1

    observer = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # =========================
    # SOURCE-TIME PATH (same as before)
    # =========================
    t_s = np.arange(0.0, T_TOTAL, DT_SIM)

    per_elem_tobs = []
    per_elem_p    = []

    t0_src_build = time.perf_counter()

    for e in range(N_ELEMENTS):
        r_id   = elem_rotor_id[e]
        psi0   = elem_psi0[e]
        center = elem_center[e]
        radius = elem_radius[e]

        hist = rotor_hist[r_id]

        t_obs_list = []
        p_list     = []

        for ts in t_s:
            # scalar az, omega from history
            # simple direct index instead of searchsorted
            idx_f = (ts - hist.times[0]) / hist.dt
            if idx_f <= 0.0:
                az = hist.az[0]
                w  = hist.omega[0]
            elif idx_f >= N_hist - 1:
                az = hist.az[-1]
                w  = hist.omega[-1]
            else:
                i0 = int(idx_f)
                a  = idx_f - i0
                i1 = i0 + 1
                az = hist.az[i0] + a*(hist.az[i1] - hist.az[i0])
                w  = hist.omega[i0] + a*(hist.omega[i1] - hist.omega[i0])

            psi = az + psi0

            x = radius * np.cos(psi)
            y = radius * np.sin(psi)
            src_pos = center + np.array([x, y, 0.0])

            p_arr, R_arr = lowson_pressure_array(
                observer, src_pos[None, :],
                np.array([psi]), np.array([w]),
                np.array([radius])
            )
            p = p_arr[0]
            R = R_arr[0]

            t_obs = ts + R / SPEED_OF_SOUND
            t_obs_list.append(t_obs)
            p_list.append(p)

        t_obs_arr = np.array(t_obs_list)
        p_arr     = np.array(p_list)

        if np.all(np.diff(t_obs_arr) >= 0):
            per_elem_tobs.append(t_obs_arr)
            per_elem_p.append(p_arr)
        else:
            sort_idx = np.argsort(t_obs_arr)
            per_elem_tobs.append(t_obs_arr[sort_idx])
            per_elem_p.append(p_arr[sort_idx])

    t1_src_build = time.perf_counter()

    starts = [arr[0] for arr in per_elem_tobs]
    ends   = [arr[-1] for arr in per_elem_tobs]
    tmin = max(starts)
    tmax = min(ends)

    N_OBS = 4096
    t_obs_uniform = np.linspace(tmin, tmax, N_OBS)

    p_src_uniform = np.zeros_like(t_obs_uniform)

    t0_src_interp = time.perf_counter()

    for e in range(N_ELEMENTS):
        t_arr = np.asarray(per_elem_tobs[e]).ravel()
        p_arr = np.asarray(per_elem_p[e]).ravel()
        p_src_uniform += np.interp(t_obs_uniform, t_arr, p_arr)

    t1_src_interp = time.perf_counter()

    T_src_build  = t1_src_build  - t0_src_build
    T_src_interp = t1_src_interp - t0_src_interp
    T_src_total  = T_src_build + T_src_interp

    # =========================
    # OBSERVER-TIME PATH (Numba)
    # =========================
    # First call includes JIT compile time
    t0_obs_total = time.perf_counter()

    p_obs_uniform = observer_time_kernel(
        t_obs_uniform,
        times,
        az_hist,
        omega_hist,
        dt_hist,
        elem_rotor_id,
        elem_psi0,
        elem_center,
        elem_radius,
        observer
    )

    t1_obs_total = time.perf_counter()
    T_obs_total  = t1_obs_total - t0_obs_total

    # =========================
    # Compare signals
    # =========================
    diff = p_src_uniform - p_obs_uniform
    rms_diff = np.sqrt(np.mean(diff**2))
    rel_rms = rms_diff / (np.sqrt(np.mean(p_src_uniform**2)) + 1e-12)

    print("\n==== TIMING with 6 span stations (A-opt2, Numba) ====")
    print(f"Source-time: build   : {T_src_build:.6f} s")
    print(f"Source-time: interp  : {T_src_interp:.6f} s")
    print(f"Source-time: TOTAL   : {T_src_total:.6f} s")
    print(f"Observer-time (Numba): TOTAL : {T_obs_total:.6f} s")
    print("\n==== NUMERICAL DIFFERENCE ====")
    print(f"RMS diff (Pa, rel)   : {rms_diff:.3e}  (rel = {rel_rms:.3e})")

    plt.figure(figsize=(10,6))
    plt.plot(t_obs_uniform, p_src_uniform, label="Source-time (interp)")
    plt.plot(t_obs_uniform, p_obs_uniform, '--', label="Observer-time (Numba)")
    plt.xlabel("Observer time [s]")
    plt.ylabel("Pressure (relative)")
    plt.title("Total Pressure Comparison (A-opt2, 6 spanwise elements)")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(t_obs_uniform, diff)
    plt.xlabel("Observer time [s]")
    plt.ylabel("p_src - p_obs")
    plt.title("Pressure Difference")
    plt.grid(True)

    plt.show()




