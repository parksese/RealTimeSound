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
