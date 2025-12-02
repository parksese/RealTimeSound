import numpy as np

SPEED_OF_SOUND = 340.0
ROTOR_RADIUS   = 3.0
T_HIST         = 0.5      # 0.5 s of azimuth/omega history

class AzimuthHistory:
    """
    Maintain time history of azimuth and omega for one blade.
    Enough to compute blade azimuth at arbitrary retarded time.
    """

    def __init__(self):
        self.times   = []  # [N_hist]
        self.omega   = []  # [N_hist]
        self.azimuth = []  # [N_hist]  (unwrapped)
        self.t_hist  = T_HIST
        self.az_current = 0.0

    def update(self, t_sim, omega):
        """
        Call this every simulation frame (dt small).
        """
        if self.times:
            dt = t_sim - self.times[-1]
            self.az_current += self.omega[-1] * dt

        self.times.append(t_sim)
        self.omega.append(omega)
        self.azimuth.append(self.az_current)

        # trim old history
        cutoff = t_sim - self.t_hist
        while len(self.times) > 2 and self.times[0] < cutoff:
            self.times.pop(0)
            self.omega.pop(0)
            self.azimuth.pop(0)

    def azimuth_at(self, t_query):
        """
        Return azimuth(t_query) by local linear interpolation.
        """
        times = self.times
        if len(times) < 2:
            return self.azimuth[-1]

        i = np.searchsorted(times, t_query) - 1
        if i < 0:
            i = 0
        if i >= len(times) - 1:
            i = len(times) - 2

        t0, t1 = times[i], times[i+1]
        az0, az1 = self.azimuth[i], self.azimuth[i+1]
        w0, w1   = self.omega[i], self.omega[i+1]

        if t1 == t0:
            return az0

        # interpolated omega
        alpha = (t_query - t0) / (t1 - t0)
        omega_r = w0 + alpha * (w1 - w0)

        # linear integrate from t0 to t_query
        return az0 + omega_r * (t_query - t0)


# ------------------------------
# Blade position
# ------------------------------

def blade_position(azimuth, rotor_center, rotor_orientation):
    """
    Position of blade tip only (simplest version).
    """
    x_rel = ROTOR_RADIUS * np.cos(azimuth)
    y_rel = ROTOR_RADIUS * np.sin(azimuth)
    z_rel = 0.0
    rel = np.array([x_rel, y_rel, z_rel])
    return rotor_center + rotor_orientation @ rel


# ------------------------------
# Retarded time solver
# ------------------------------

def retarded_time(t_obs, observer_pos, rotor_center,
                  rotor_orientation, az_history):
    """
    Solve: t_obs = t_r + R(t_r)/c
    with 1–2 fixed-point iterations.
    """
    # --- initial guess ---
    az_now = az_history.azimuth_at(t_obs)
    pos_now = blade_position(az_now, rotor_center, rotor_orientation)
    R0 = np.linalg.norm(observer_pos - pos_now)
    t_r = t_obs - R0 / SPEED_OF_SOUND

    # --- one refinement (usually enough) ---
    az_r = az_history.azimuth_at(t_r)
    pos_r = blade_position(az_r, rotor_center, rotor_orientation)
    R = np.linalg.norm(observer_pos - pos_r)
    t_r = t_obs - R / SPEED_OF_SOUND

    return t_r
    
if __name__ == "__main__":
    rotor_center = np.array([0,0,0])
    rotor_orientation = np.eye(3)
    observer_pos = np.array([10, 0, 0])   # 10 m front

    hist = AzimuthHistory()

    # Build 1 second of history with slowly changing RPM
    t = 0.0
    dt = 0.002
    while t < 1.0:
        rpm = 300 + 50*t    # slight ramp
        omega = rpm * 2*np.pi/60
        hist.update(t, omega)
        t += dt

    # Compute retarded time at official observer time t_obs = 1.0
    t_obs = 1.0
    t_r = retarded_time(t_obs, observer_pos, rotor_center,
                        rotor_orientation, hist)

    print("Observer time:", t_obs)
    print("Retarded time:", t_r)
    print("Propagation delay:", t_obs - t_r, "seconds")   




import numpy as np
import matplotlib.pyplot as plt

# =========================
# Constants
# =========================
SPEED_OF_SOUND = 340.0     # m/s
ROTOR_RADIUS   = 3.0        # m
T_HIST         = 1.0        # how much history to store

# ============================================================
# Azimuth history class (stores the blade's true motion)
# ============================================================
class AzimuthHistory:
    def __init__(self):
        self.times = []      # t_s history
        self.omega = []      # omega(t_s)
        self.az    = []      # azimuth(t_s) unwrapped
        self.az_current = 0.0

    def update(self, t_sim, omega):
        # integrate azimuth
        if self.times:
            dt = t_sim - self.times[-1]
            self.az_current += self.omega[-1] * dt

        # store
        self.times.append(float(t_sim))
        self.omega.append(float(omega))
        self.az.append(float(self.az_current))

        # trim old history
        cutoff = t_sim - T_HIST
        while len(self.times) > 2 and self.times[0] < cutoff:
            self.times.pop(0)
            self.omega.pop(0)
            self.az.pop(0)

    def azimuth_at(self, t_query):
        # find bracketing indices
        i = np.searchsorted(self.times, t_query) - 1
        if i < 0: i = 0
        if i >= len(self.times)-1: i = len(self.times)-2

        t0, t1 = self.times[i], self.times[i+1]
        a0, a1 = self.az[i], self.az[i+1]
        w0, w1 = self.omega[i], self.omega[i+1]

        if t1 == t0:
            return a0

        alpha = (t_query - t0) / (t1 - t0)
        omega_r = w0 + alpha * (w1 - w0)

        # linear integrate
        return a0 + omega_r * (t_query - t0)


# ============================================================
# Blade position (tip only for simplicity)
# ============================================================
def blade_position(azimuth):
    x = ROTOR_RADIUS * np.cos(azimuth)
    y = ROTOR_RADIUS * np.sin(azimuth)
    return np.array([x, y, 0.0])


# ============================================================
# Retarded time solver
# ============================================================
def retarded_time(t_obs, observer_pos, az_hist):
    # initial guess: use blade at t_obs
    az_now = az_hist.azimuth_at(t_obs)
    pos_now = blade_position(az_now)
    R0 = np.linalg.norm(observer_pos - pos_now)
    t_r = t_obs - R0 / SPEED_OF_SOUND

    # one refinement
    az_r = az_hist.azimuth_at(t_r)
    pos_r = blade_position(az_r)
    R = np.linalg.norm(observer_pos - pos_r)
    t_r = t_obs - R / SPEED_OF_SOUND

    return t_r


# ============================================================
# SIMULATION / LOGGING
# ============================================================
if __name__ == "__main__":
    # observer 10m ahead on x-axis
    observer_pos = np.array([10.0, 0.0, 0.0])

    az_hist = AzimuthHistory()

    # ---- Step 1: Build source-time history ----
    t = 0.0
    dt = 0.002  # simulation step
    T_total = 1.0

    while t < T_total:
        # Slightly accelerating rotor (just a demo)
        rpm = 300 + 60*t       # rpm increases over time
        omega = rpm * 2*np.pi/60

        az_hist.update(t, omega)
        t += dt

    # ---- Step 2: For each observer-time sample, compute t_r and az(t_r) ----
    obs_times = np.linspace(0.5, 1.0, 300)  # choose a time window
    retarded_times = []
    retarded_az = []

    for t_obs in obs_times:
        tr = retarded_time(t_obs, observer_pos, az_hist)
        az_r = az_hist.azimuth_at(tr)

        retarded_times.append(tr)
        retarded_az.append(az_r)

    retarded_times = np.array(retarded_times)
    retarded_az = np.array(retarded_az)

    # ---- Step 3: Extract source-time azimuth history for plotting ----
    source_times = np.array(az_hist.times)
    source_az = np.array(az_hist.az)

    # ============================================================
    # PLOT: Source-time azimuth vs Retarded-time azimuth
    # ============================================================
    plt.figure(figsize=(10, 6))

    plt.plot(source_times, source_az, label="Source-time azimuth(t_s)")
    plt.plot(obs_times, retarded_az, label="Observer-time azimuth(t_r)", linewidth=3)

    plt.xlabel("Time [s]")
    plt.ylabel("Azimuth [rad]")
    plt.title("Source-time vs Retarded-time Blade Azimuth")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    
    
    
    
    

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

        # trim history
        cutoff = t_sim - T_HIST
        while len(self.times) > 2 and self.times[0] < cutoff:
            self.times.pop(0)
            self.omega.pop(0)
            self.az.pop(0)

    def azimuth_at(self, t_query):
        i = np.searchsorted(self.times, t_query) - 1
        if i < 0: i = 0
        if i >= len(self.times)-1: i = len(self.times)-2

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

    # ---- build source-time history ----
    t = 0.0
    dt = 0.002
    T_total = 1.0

    while t < T_total:
        rpm = 300 + 60*t
        omega = rpm * 2*np.pi/60
        az_hist.update(t, omega)
        t += dt

    # ============================================
    # A) SOURCE-TIME-BASED PREDICTION
    # ============================================
    src_t = np.array(az_hist.times)
    src_az = np.array(az_hist.az)

    src_obs_times = []
    for ts, azs in zip(src_t, src_az):
        pos = blade_position(azs)
        R = np.linalg.norm(observer_pos - pos)
        tobs = ts + R/SPEED_OF_SOUND
        src_obs_times.append(tobs)

    src_obs_times = np.array(src_obs_times)

    # ============================================
    # B) OBSERVER-TIME-BASED PREDICTION
    # ============================================
    obs_times = np.linspace(0.5, 1.0, 300)
    obs_ret_az = []

    for t_obs in obs_times:
        t_r = retarded_time(t_obs, observer_pos, az_hist)
        obs_ret_az.append(az_hist.azimuth_at(t_r))

    obs_ret_az = np.array(obs_ret_az)

    # ============================================
    # PLOT FAIR COMPARISON
    # ============================================
    plt.figure(figsize=(10,6))

    # Source-time based: observer arrival time vs azimuth(ts)
    plt.plot(src_obs_times, src_az, label="Source-time-based (t_o = t_s + R/c)")

    # Observer-time based: azimuth at retarded time t_r
    plt.plot(obs_times, obs_ret_az, label="Observer-time-based (azimuth(t_r))", linewidth=3)

    plt.xlabel("Observer time t_o [s]")
    plt.ylabel("Blade azimuth [rad]")
    plt.title("FAIR Comparison: Source-time vs Observer-time Retarded Azimuth")
    plt.grid(True)
    plt.legend()
    plt.show()