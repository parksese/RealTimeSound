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