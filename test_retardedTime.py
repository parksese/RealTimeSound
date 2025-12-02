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