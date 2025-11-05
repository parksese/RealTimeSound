import numpy as np
import sounddevice as sd
import threading
import time
import matplotlib
matplotlib.use("TkAgg")  # safer on Windows; avoids WinError 6 from Qt
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt, sosfilt_zi
# import keyboard

# ---------------- Constants ----------------
pi = np.pi
SPEED_OF_SOUND = 334.3
NUMBER_OF_ROTORS = 4
NUMBER_OF_BLADES = 5
NUMBER_OF_SOURCES = NUMBER_OF_ROTORS * NUMBER_OF_BLADES
ROTOR_RADIUS = 3.048

AUDIO_SAMPLE_RATE = 22050 #44100
AUDIO_CHANNELS = 2
AUDIO_BLOCK_SIZE = 4096 #1024
SCALING_FACTOR = 5.0 #0.5 #5 #0.008
VOLUME_RAMP_SPEED = 0.01

volume_gain = 0.0
smoothing_factor = 50 #0.005

# latest_out_buffer = None  ### 

# ---------------- Observer & geometry ----------------
observer_position = np.array([-2.4, 0.0, 0.0])
tilt_center = np.array([[-2.554, -3.962, 0.398], [-2.554, 3.962, 0.398],
                        [4.101, -3.962, 0.667], [4.101, 3.962, 0.667]])
rotor_center = tilt_center + np.array([-0.188, 0, 1.430])
rotor_direction = np.array([-1, 1, -1, 1])

# ---------------- Simulated constant inputs ----------------
# case 1
TEST_INPUT = {
    "rpm": np.array([477.5, 477.5, 477.5, 477.5], dtype=float),
    "coll": np.array([15, 15, 15, 15], dtype=float),
    "tilt": np.array([90, 90, 90, 90], dtype=float),
    "spd": 0.0,
    "aoa": 0.0,
    "aos": 0.0,
    "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES
                         for sid in range(NUMBER_OF_SOURCES)]),
    "last_update_time": time.time(),

}

# case 3
TEST_INPUT = {
    "rpm": np.array([467, 467, 467, 467], dtype=float),
    "coll": np.array([50, 50, 50, 50], dtype=float),
    "tilt": np.array([70, 70, 70, 70], dtype=float),
    "spd": 46.9,
    "aoa": 0.0,
    "aos": 0.0,
    "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES
                         for sid in range(NUMBER_OF_SOURCES)]),
    "last_update_time": time.time(),

}

# # case 4
# TEST_INPUT = {
#     "rpm": np.array([450, 450, 450, 450], dtype=float),
#     "coll": np.array([20, 20, 20, 20], dtype=float),
#     "tilt": np.array([45, 45, 45, 45], dtype=float),
#     "spd": 49.4,
#     "aoa": 0.0,
#     "aos": 0.0,
#     "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES
#                          for sid in range(NUMBER_OF_SOURCES)]),
#     "last_update_time": time.time(),    

# }

# # case 5
# TEST_INPUT = {
#     "rpm": np.array([287, 287, 287, 287], dtype=float),
#     "coll": np.array([30, 30, 30, 30], dtype=float),
#     "tilt": np.array([20, 20, 20, 20], dtype=float),
#     "spd": 54.0,
#     "aoa": 0.0,
#     "aos": 0.0,
#     "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES
#                          for sid in range(NUMBER_OF_SOURCES)]),
#     "last_update_time": time.time(),

# }

# # case 6
# TEST_INPUT = {
#     "rpm": np.array([168, 168, 168, 168], dtype=float),
#     "coll": np.array([50, 50, 50, 50], dtype=float),
#     "tilt": np.array([0, 0, 0, 0], dtype=float),
#     "spd": 67.0,
#     "aoa": 5.0,
#     "aos": 0.0,
#     "azimuth": np.array([2*pi*(sid % NUMBER_OF_BLADES)/NUMBER_OF_BLADES
#                          for sid in range(NUMBER_OF_SOURCES)]),
# #     "last_update_time": time.time(),
# }

# ---------------- Filter setup ----------------
F_LOW = 20
F_HIGH = 1000
FILTER_ORDER = 6 #4
FILTER_ENABLED = False
sos = butter(FILTER_ORDER, [F_LOW, F_HIGH], btype='bandstop', fs=AUDIO_SAMPLE_RATE, output='sos')
zi = np.zeros((sos.shape[0], 2))
filter_lock = threading.Lock()

# ---------------- Plot / window parameters ----------------
PARAM_WINDOW_SEC = 2.0          # seconds for spd, tilt, rpm, aos rolling plot
WAVEFORM_WINDOW_SEC = 2.0       # seconds for waveform rolling plot
PLOT_UPDATE_INTERVAL = 0.2      # seconds

# Buffers for plotting
param_len = int(PARAM_WINDOW_SEC * AUDIO_SAMPLE_RATE / AUDIO_BLOCK_SIZE)
wave_len = int(WAVEFORM_WINDOW_SEC * AUDIO_SAMPLE_RATE)
plot_data = {
    "time": deque(maxlen=param_len),
    "spd": deque(maxlen=param_len),
    "tilt": deque(maxlen=param_len),
    "coll": deque(maxlen=param_len),
    "rpm": deque(maxlen=param_len),
    "aoa": deque(maxlen=param_len),
    "aos": deque(maxlen=param_len),
    "wave": deque(maxlen=wave_len),
    "block_time": deque(maxlen=param_len),
    "block_mean": deque(maxlen=param_len)
}

# # --------------- Mean buffers ---------------
# mean_buffers = {rid: deque(maxlen=44100) for rid in range(NUMBER_OF_ROTORS)} # initial default length
# mean_values = np.zeros(NUMBER_OF_ROTORS)

# ---------------- State variables ----------------
rpm_filtered = np.zeros(NUMBER_OF_ROTORS)
coll_filtered = np.zeros(NUMBER_OF_ROTORS)
azimuth = np.array([2*pi*(sid%NUMBER_OF_BLADES)/NUMBER_OF_BLADES for sid in range(NUMBER_OF_SOURCES)])  # rad

# for debug 
az_prev = np.zeros(NUMBER_OF_SOURCES) # previous azimuths per source
L_end_prev = np.zeros(NUMBER_OF_SOURCES) # last L value from previous block

# ---------------- TableLookup ----------------
# class TableLookup:
#     def __init__(self):
#         table = np.array([
#            # spd  aoa aos coll    a0        a1      b1        a2       b2
#             [0.00, 0, -10, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43], # case1 50.0 rad/s = 477.5 rpm
#             [0.00, 0,  -5, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],
#             [0.00, 0,   0, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],  
#             [0.00, 0,   5, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],
#             [0.00, 0,  10, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43],
#             [5.00, 0, -10, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95], # case2 50.0 rad/s = 477.5 rpm
#             [5.00, 0,  -5, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95],
#             [5.00, 0,   0, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95], 
#             [5.00, 0,   5, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95],
#             [5.00, 0,  10, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95],
#             [46.9, 0, -10, 15, 1896.92, -117.84, 2352.80, -582.05,  -43.35], # case3 48.9 rad/s = 467 rpm
#             [46.9, 0,  -5, 15, 1880.13, -334.72, 2323.99, -567.27, -149.37],
#             [46.9, 0,   0, 15, 1873.06, -549.95, 2279.70, -528.09, -251.81],
#             [46.9, 0,   5, 15, 1880.49, -755.70, 2220.03, -469.80, -344.33],
#             [46.9, 0,  10, 15, 1895.88, -963.69, 2153.17, -400.84, -424.70],
#             [49.4, 0, -10, 20,  776.86,   70.29, 1736.77, -449.34,   79.58], # case4 47.1 rad/s = 450 rpm
#             [49.4, 0,  -5, 20,  729.57, -141.32, 1699.35, -449.35,  -21.83],
#             [49.4, 0,   0, 20,  716.33, -341.13, 1650.76, -425.16, -134.87], 
#             [49.4, 0,   5, 20,  731.30, -550.92, 1617.01, -387.08, -230.59],
#             [49.4, 0,  10, 20,  777.19, -752.89, 1567.50, -316.74, -328.63],
#             [54.0, 0, -10, 35,  -32.84,  240.33,  791.73, -146.91,  147.89], # case5 30.05 rad/s = 287 rpm
#             [54.0, 0,  -5, 35,  -86.97,  -46.00,  758.33, -167.55,   62.44],
#             [54.0, 0,   0, 35, -104.95, -139.91,  722.07, -168.47,  -24.68], 
#             [54.0, 0,   5, 35,  -86.99, -323.71,  687.28, -142.86, -107.53],
#             [54.0, 0,  10, 35,  -29.40, -506.64,  652.07,  -92.83, -185.67],
#             [67.0, 1, -10, 60, 153.86,  457.98,    47.82,   80.31,    4.25], # case6 17.6 rad/s = 168 rpm
#             [67.0, 1,  -5, 60,  80.76,  229.69,   24.315,   19.71,    0.99],
#             # [67.0, 1,   0, 60,  59.69,    0.00,     0.00,    0.00,    0.00], 
#             # [67.0, 1,   0, 60, 108.69,    0.00,     0.00,    0.00,    0.00], 
#             [67.0, 1,   0, 60, 550.69,    0.00,     0.00,    0.00,    0.00], 
#             [67.0, 1,   5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
#             [67.0, 1,  10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25],
#             [85.0, 0, -10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25], # case7 17.6 rad/s = 168 rpm
#             [85.0, 0,  -5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
#             # [85.0, 0,   0, 60,  59.69,    0.00,     0.00,    0.00,    0.00], 
#             # [85.0, 0,   0, 60, 108.69,    0.00,     0.00,    0.00,    0.00], 
#             [85.0, 0,   0, 60, 550.69,    0.00,     0.00,    0.00,    0.00], 
#             [85.0, 0,   5, 60,  80.76, -229.69,  -24.315,   19.71,    0.99],
#             [85.0, 0,  10, 60, 153.86, -457.98,   -47.82,   80.31,    4.25],
#         ])

#         table_vfm = np.array([
#            # spd  aoa aos coll    a0        a1      b1        a2       b2
#             [0.00, 0,   0, 10, 1365.50,    0.81,   3.88,     0.15,    2.63], # case1 50.0 rad/s = 477.5 rpm
#             [0.00, 0,   0, 15, 2111.16,    3.24,  -2.43,     1.31,   -1.43], 
#             [0.00, 0,   0, 20, 2638.02,    0.78,  -0.23,    -0.10,   -0.06],
#             [5.00, 0,   0, 10, 1359.52, -195.00, 241.41,    84.96,   -0.46], # case2 50.0 rad/s = 477.5 rpm
#             [5.00, 0,   0, 15, 2065.45, -203.84, 402.33,    67.18,   -1.95], 
#             [5.00, 0,   0, 20, 2620.15,  -64.57, 429.97,    32.50,   -2.66], 
#         ])
                
#         # table_offset = np.array([
#         #     # spd aoa aos coll offset scaling
#         #     [0.00, 0,   0, 15, -100.0,  1],
#         #     [0.00, 0,   5, 15, -100.0,  1], 
#         #     [5.00, 0,   0, 15,   -4.2,  5], 
#         #     [5.00, 0,   5, 15,   -4.2,  5], 
#         #     [46.9, 0,   0, 15,  -20.0,  1],
#         #     [46.9, 0,   5, 15,  -20.0,  1],
#         #     [49.4, 0,   0, 20,    0.0,  1], 
#         #     [49.4, 0,   5, 20,    0.0,  1], 
#         #     [67.0, 0,   0, 60,   -2.3, 10], 
#         #     [67.0, 0,   5, 60,   -2.3, 10], 
#         #     [85.0, 0,   0, 60,   -2.3, 10], 
#         #     [85.0, 0,   5, 60,   -2.3, 10], 
#         # ])
        
#         names = ["a0", "a1", "b1", "a2", "b2"]
#         # vfm
#         self.spd_vals_vfm = np.unique(table_vfm[:, 0])  # extract unique sorted speed and coll grids
#         self.coll_vals_vfm = np.unique(table_vfm[:, 3])
#         n_spd_vfm = len(self.spd_vals_vfm)
#         n_coll_vfm = len(self.coll_vals_vfm)
#         # build fast interpolators for each coefficient
#         self.coeffs_vfm = {}
#         for j, name in enumerate(names, start=4):       # build fast interpolators for each coefficient
#             grid_vfm = table_vfm[:, j].reshape(n_spd_vfm, n_coll_vfm)
#             self.coeffs_vfm[name] = grid_vfm        
#         # print(self.coeffs_vfm)

#         # tfm, ffm
#         self.spd_vals = np.unique(table[:, 0])          # extract unique sorted speed and aos grids
#         self.aos_vals = np.unique(table[:, 2])
#         n_spd = len(self.spd_vals)
#         n_aos = len(self.aos_vals)
#         # build fast interpolators for each coefficient
#         self.coeffs = {}
#         for j, name in enumerate(names, start=4):       # build fast interpolators for each coefficient
#             grid = table[:, j].reshape(n_spd, n_aos)
#             self.coeffs[name] = grid
#         # print(self.coeffs)
        
#         # # offset
#         # names_offset = ["offset", "scaling"]
#         # self.spd_vals_offset = np.unique(table_offset[:, 0])          # extract unique sorted speed and aos grids
#         # self.aos_vals_offset = np.unique(table_offset[:, 2])
#         # n_spd = len(self.spd_vals_offset)
#         # n_aos = len(self.aos_vals_offset)
#         # # build fast interpolators for each coefficient
#         # self.coeffs_offset = {}
#         # for j, name in enumerate(names_offset, start=4):       # build fast interpolators for each coefficient
#         #     grid_offset = table_offset[:, j].reshape(n_spd, n_aos)
#         #     self.coeffs_offset[name] = grid_offset
        

#     def _bilinear_interp(self, var1_vals, var2_vals, grid, var1, var2):
#         """
#         manual bilinear interpolation of grid at (var1, var2)
#         grid: 2d array indexed as [i_spd, j_aos]
#         """
#         # find indices
#         i = np.searchsorted(var1_vals, var1) - 1
#         j = np.searchsorted(var2_vals, var2) - 1
#         i = np.clip(i, 0, len(var1_vals) - 2)
#         j = np.clip(j, 0, len(var2_vals) - 2)
#         # grid corners
#         x0, x1 = var1_vals[i], var1_vals[i+1]
#         y0, y1 = var2_vals[j], var2_vals[j+1]
#         q11 = grid[i  , j  ]
#         q21 = grid[i+1, j  ]
#         q12 = grid[i  , j+1]
#         q22 = grid[i+1, j+1]
#         # normalize distances
#         tx = (var1-x0) / (x1-x0) if x1 > x0 else 0
#         ty = (var2-y0) / (y1-y0) if y1 > y0 else 0
#         return (q11*(1-tx)*(1-ty) + q21*tx*(1-ty) + q12*(1-tx)*ty + q22*tx*ty)

#     def get_coefficients(self, spd, aos, coll, aoa=None):
#         """
#         get coefficients for (spd, aos)
#         for vfm (spd <=5), only interpolate in spd, aos=0 row        
#         """
#         coeffs_out = {}
#         if spd <= 5: # vfm
#             for name, grid in self.coeffs_vfm.items():
#                 coeffs_out[name] = self._bilinear_interp(self.spd_vals_vfm, self.coll_vals_vfm, grid, spd, coll)
#         else: # tfm, ffm
#             for name, grid in self.coeffs.items():
#                 coeffs_out[name] = self._bilinear_interp(self.spd_vals, self.aos_vals, grid, spd, aos)                
#         return coeffs_out


#     # def get_offset(self, spd, aos, coll, aoa=None):
#     #     """
#     #     get coefficients for (spd, aos)
#     #     for vfm (spd <=5), only interpolate in spd, aos=0 row        
#     #     """
#     #     coeffs_out = {}
#     #     for name, grid in self.coeffs_offset.items():
#     #         coeffs_out[name] = self._bilinear_interp(self.spd_vals_offset, self.aos_vals_offset, grid, spd, aos)                
#     #     return coeffs_out

# ---------------- TableLookup ----------------
class TableLookup:
    def __init__(self):
        table = np.array([
           #  spd tilt aoa aos coll   a0         a1      b1        a2       b2
            [ 0.0, 90, -24,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43], # case1 50.0 rad/s = 477.5 rpm           
            [ 0.0, 90, -16,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  -8,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,   0,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,   8,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  16,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 0.0, 90,  24,   0, 15,  2111.16,    3.24,  -2.43,     1.31,   -1.43],
            [ 5.0, 90, -24,   0, 15,  2019.10, -159.62, 330.10,    45.05,    2.67], # case2 50.0 rad/s = 477.5 rpm
            [ 5.0, 90, -16,   0, 15,  2018.86, -159.24, 373.12,    58.96,  -16.87],
            [ 5.0, 90,  -8,   0, 15,  2043.73, -183.79, 398.20,    70.20,   -7.37],
            [ 5.0, 90,   0,   0, 15,  2065.45, -203.84, 402.33,    67.18,   -1.95],
            [ 5.0, 90,   8,   0, 15,  2076.82, -218.69, 404.37,    73.34,    6.24],
            [ 5.0, 90,  16,   0, 15,  2083.48, -224.29, 403.09,    91.48,   -3.60],
            [ 5.0, 90,  24,   0, 15,  2097.72, -228.09, 374.76,   100.52,   -1.42],         
            [46.9, 70, -24,   0, 15,    89.87, -251.46, 117.25, -302.04, -123.60], # case3 48.9 rad/s = 467 rpm
            [46.9, 70, -16,   0, 15,   649.22, -349.74, 468.23, -404.10, -169.95],          
            [46.9, 70,  -8,   0, 15,  1261.38, -463.84, 909.35, -475.91, -220.62],
            [46.9, 70,   0,   0, 15,  1873.06, -549.95, 279.70, -528.09, -251.81],
            [46.9, 70,   8,   0, 15,  2352.58, -503.41, 464.50, -501.43, -145.07],
            [46.9, 70,  16,   0, 15,  2666.80, -272.75, 490.22, -435.59,   19.22],
            [46.9, 70,  24,   0, 15,  2857.03,  -49.74, 296.69, -280.00,   39.44],                   
            [49.4, 45, -24,   0, 20,  -591.37, -144.36,  594.41,  -98.45,  -41.34], # case4 47.1 rad/s = 450 rpm
            [49.4, 45, -16,   0, 20,  -247.41, -200.25,  893.25, -203.29,  -65.24],
            [49.4, 45,  -8,   0, 20,   199.35, -261.73, 1241.71, -311.33,  -96.86],
            [49.4, 45,   0,   0, 20,   716.33, -341.13, 1650.76, -425.16, -134.87],
            [49.4, 45,   8,   0, 20,  1300.63, -440.39, 2116.61, -530.24, -174.72],
            [49.4, 45,  16,   0, 20,  1874.38, -479.41, 2488.53, -568.15, -162.83],
            [49.4, 45,  24,   0, 20,  2267.01, -357.56, 2575.59, -456.40,   26.17],     
            [54.0, 20, -24,   0, 35,  -455.03,   30.27, -144.29,   -6.21,   -0.45], # case5 30.05 rad/s = 287 rpm
            [54.0, 20, -16,   0, 35,  -455.03,  -30.50,  144.19,   -6.24,   -0.35],
            [54.0, 20,  -8,   0, 35,  -315.31,  -86.77,  413.03,  -71.63,  -10.37],
            [54.0, 20,   0,   0, 35,   -86.97,  -46.00,  758.33, -167.55,   62.44],
            [54.0, 20,   8,   0, 35,   189.60, -177.95, 1052.89, -308.95,  -52.48],        
            [54.0, 20,  16,   0, 35,   533.64, -227.19, 1428.15, -471.48,  -80.31],
            [54.0, 20,  24,   0, 35,   874.65, -227.72, 1687.72, -499.41,  -48.05],
            [67.0,  0, -24,   0, 60,  434.21,   72.22,  -903.17, -223.12,   10.28], # case6 17.6 rad/s = 168 rpm
            [67.0,  0, -16,   0, 60,  296.86,   72.22,  -736.17, -198.20,  -13.09],
            [67.0,  0,  -8,   0, 60,  116.16,   38.72,  -367.46,  -50.77,   -2.52],
            [67.0,  0,   0,   0, 60,   59.69,    0.00,     0.00,    0.00,    0.00],
            # [67.0,  0,   0,   0, 60,   359.69,    0.00,     0.00,    0.00,    0.00],
            [67.0,  0,   8,   0, 60,  116.16,  -38.70,   367.69,  -50.80,   -2.66],
            [67.0,  0,  16,   0, 60,  295.83,  -72.36,   736.25, -198.36,  -12.54],
            [67.0,  0,  24,   0, 60,  434.70,  -74.08,   902.89, -222.21,    8.46],
            [85.0,  0, -24,   0, 60,  434.21,   72.22,  -903.17, -223.12,   10.28], # case7 17.6 rad/s = 168 rpm
            [85.0,  0, -16,   0, 60,  296.86,   72.22,  -736.17, -198.20,  -13.09],
            [85.0,  0,  -8,   0, 60,  116.16,   38.72,  -367.46,  -50.77,   -2.52],
            [85.0,  0,   0,   0, 60,   59.69,    0.00,     0.00,    0.00,    0.00],
            # [85.0,  0,   0,   0, 60,   559.69,    0.00,     0.00,    0.00,    0.00],            
            [85.0,  0,   8,   0, 60,  116.16,  -38.70,   367.69,  -50.80,   -2.66],
            [85.0,  0,  16,   0, 60,  295.83,  -72.36,   736.25, -198.36,  -12.54],
            [85.0,  0,  24,   0, 60,  434.70,  -74.08,   902.89, -222.21,    8.46]
        ])

        table_vfm = np.array([
           #  spd tilt aoa  aos coll    a0        a1      b1        a2       b2
            [ 0.0, 90,   0,   0, 10, 1365.50,    0.81,     3.88,    0.15,    2.63], # case1 50.0 rad/s = 477.5 rpm
            [ 0.0, 90,   0,   0, 15, 2111.16,    3.24,    -2.43,    1.31,   -1.43], 
            [ 0.0, 90,   0,   0, 20, 2638.02,    0.78,    -0.23,   -0.10,   -0.06],
            [ 5.0, 90,   0,   0, 10, 1359.52, -195.00,   241.41,   84.96,   -0.46], # case2 50.0 rad/s = 477.5 rpm
            [ 5.0, 90,   0,   0, 15, 2065.45, -203.84,   402.33,   67.18,   -1.95], 
            [ 5.0, 90,   0,   0, 20, 2620.15,  -64.57,   429.97,   32.50,   -2.66], 
        ])
        
        names = ["a0", "a1", "b1", "a2", "b2"]
        # vfm
        self.spd_vals_vfm = np.unique(table_vfm[:, 0])  # extract unique sorted speed and coll grids
        self.coll_vals_vfm = np.unique(table_vfm[:, 4])
        n_spd_vfm = len(self.spd_vals_vfm)
        n_coll_vfm = len(self.coll_vals_vfm)
        # build fast interpolators for each coefficient
        self.coeffs_vfm = {}
        for j, name in enumerate(names, start=5):       # build fast interpolators for each coefficient
            grid_vfm = table_vfm[:, j].reshape(n_spd_vfm, n_coll_vfm)
            self.coeffs_vfm[name] = grid_vfm
        
        # print(self.coeffs_vfm)

        # tfm, ffm
        self.spd_vals = np.unique(table[:, 0])          # extract unique sorted speed and aos grids
        self.aoa_vals = np.unique(table[:, 2])        
        n_spd = len(self.spd_vals)
        n_aoa = len(self.aoa_vals)
        # build fast interpolators for each coefficient
        self.coeffs = {}
        for j, name in enumerate(names, start=5):       # build fast interpolators for each coefficient
            # grid = table[:, j].reshape(n_spd, n_aos)
            grid = table[:, j].reshape(n_spd, n_aoa)
            self.coeffs[name] = grid
        # print(self.coeffs)

    def _bilinear_interp(self, var1_vals, var2_vals, grid, var1, var2):
        """
        manual bilinear interpolation of grid at (var1, var2)
        grid: 2d array indexed as [i_spd, j_aos]
        """
        # find indices
        i = np.searchsorted(var1_vals, var1) - 1
        j = np.searchsorted(var2_vals, var2) - 1
        i = np.clip(i, 0, len(var1_vals) - 2)
        j = np.clip(j, 0, len(var2_vals) - 2)
        # grid corners
        x0, x1 = var1_vals[i], var1_vals[i+1]
        y0, y1 = var2_vals[j], var2_vals[j+1]
        q11 = grid[i  , j  ]
        q21 = grid[i+1, j  ]
        q12 = grid[i  , j+1]
        q22 = grid[i+1, j+1]
        # normalize distances
        tx = (var1-x0) / (x1-x0) if x1 > x0 else 0
        ty = (var2-y0) / (y1-y0) if y1 > y0 else 0
        return (q11*(1-tx)*(1-ty) + q21*tx*(1-ty) + q12*(1-tx)*ty + q22*tx*ty)

    def get_coefficients(self, spd, aoa, coll, aos=None):
        """
        get coefficients for (spd, aos)
        for vfm (spd <=5), only interpolate in spd, aos=0 row        
        """
        coeffs_out = {}
        if spd <= 5: # vfm
            for name, grid in self.coeffs_vfm.items():
                coeffs_out[name] = self._bilinear_interp(self.spd_vals_vfm, self.coll_vals_vfm, grid, spd, coll)
        else: # tfm, ffm
            for name, grid in self.coeffs.items():
                coeffs_out[name] = self._bilinear_interp(self.spd_vals, self.aoa_vals, grid, spd, aoa)                
        return coeffs_out


lookup = TableLookup()


# ---------------- sound pressure level ----------------
def compute_spl_db(pressure_signal, pref=20e-6):
    """Compute SPL in dB from pressure signal array"""
    rms_pressure = np.sqrt(np.mean(pressure_signal**2))
    if rms_pressure < 1e-12:
        return -np.inf
    spl_db = 20 * np.log10(rms_pressure / pref)
    return spl_db


# ---------------- Audio Callback ----------------
def audio_callback(outdata, frames, time_info, status):
    global rpm_filtered, coll_filtered, azimuth, volume_gain, smoothing_factor
    global latest_out_buffer
    
    if status:
        print(status)
    
    out_buffer = np.zeros((frames, AUDIO_CHANNELS), dtype=np.float32)
    dt = 1.0 / AUDIO_SAMPLE_RATE
    n = np.arange(frames)
    
    spd = TEST_INPUT["spd"]
    aos = TEST_INPUT["aos"]
    aoa = TEST_INPUT["aoa"]
    tilt = TEST_INPUT["tilt"]
    rpm_target = TEST_INPUT["rpm"]
    coll_target = TEST_INPUT["coll"]
    # scaling = TEST_INPUT["scaling"]
    # offset = TEST_INPUT["offset"]
    
    volume_gain += (1.0 - volume_gain) * VOLUME_RAMP_SPEED

    for rotor_id in range(NUMBER_OF_ROTORS):
        rpm_filtered[rotor_id] += 0.02 * (rpm_target[rotor_id] - rpm_filtered[rotor_id])
        coll_filtered[rotor_id] += 0.02 * (coll_target[rotor_id] - coll_filtered[rotor_id])
        omega = rotor_direction[rotor_id] * rpm_filtered[rotor_id] * 2 * np.pi / 60 # rad/s
        if rpm_filtered[rotor_id] < 1e-2:
            continue

        # Get coefficients (same for all samples in block)    
        c = lookup.get_coefficients(spd=spd, aoa=aoa, aos=aos, coll=coll_filtered[rotor_id])
        a0, a1, b1, a2, b2 = c["a0"], c["a1"], c["b1"], c["a2"], c["b2"]
        
        # c2 = lookup.get_offset(spd=spd, aoa=aoa, aos=aos, coll=coll_filtered[rotor_id])
        # offset, scaling = c2["offset"], c2["scaling"]
        # print(f"spd={spd}, coll={coll_filtered}")
        # print(f"c={c}")
        
        # Tilt matrix  (precompute once per rotor)
        tilt_rad = np.radians(90 - tilt[rotor_id]) # tilt: 90 for VFM, 0 for FFM
        aos_rad = np.radians(aos)

        trans_tilt = np.array([
            [np.cos(tilt_rad), 0, -np.sin(tilt_rad)],  # need to double-check the transformation !!!
            [0, 1, 0],
            [np.sin(tilt_rad), 0, np.cos(tilt_rad)]
        ])

        # Simple sinusoidal lift model for demonstration
        for blade in range(NUMBER_OF_BLADES):
            source_id = rotor_id * NUMBER_OF_BLADES + blade
            az_start = azimuth[source_id]
            az_block = az_start + omega * dt * n
            azimuth[source_id] = az_block[-1]

            # Lift (periodic loading)
            if rotor_direction[rotor_id] == 1:  # Counter-clockwise
                L = (a0 + a1 * np.cos(abs(az_block)) + b1 * np.sin(abs(az_block)) + a2 * np.cos(abs(2*az_block)) + b2 * np.sin(abs(2*az_block)))
            else:  # Clockwise
                L = (a0 + a1 * np.cos(abs(az_block)-2.0*aos) + b1 * np.sin(abs(az_block)-2.0*aos) + a2 * np.cos(abs(2*az_block)-2.0*aos) + b2 * np.sin(abs(2*az_block)-2.0*aos))

            # # Lift (periodic loading)  # corrected 2025-10-23
            L = (a0 
            + a1 * np.cos(abs(az_block-aos_rad*rotor_direction[rotor_id])) 
            + b1 * np.sin(abs(az_block-aos_rad*rotor_direction[rotor_id])) 
            + a2 * np.cos(abs(2*(az_block-aos_rad*rotor_direction[rotor_id]))) 
            + b2 * np.sin(abs(2*(az_block-aos_rad*rotor_direction[rotor_id])))
            )

            # # for debug: check azimuth continuity
            # az_diff = (azimuth[source_id] - az_prev[source_id]) % (2*np.pi)
            # expected_diff = (omega * frames * dt) % (2*np.pi)
            # if abs(az_diff - expected_diff) > 1e-3:
            #     print(f"[Azimuth jump] src={source_id} daz={az_diff:.4f}, expected={expected_diff:.4f}")
            # az_prev[source_id] = azimuth[source_id]
            
            # for debug: check loading continuity
            # L_jump = abs(L[0] - L_end_prev[source_id])
            # if L_jump > 1e-1: # adjust threshold if needed
            #     print(f"[L_jump] src={source_id} jump={L_jump:.5f}")
            # L_end_prev[source_id] = L[-1]
            
            # Source position (before tilt)
            x = rotor_center[rotor_id][0] + ROTOR_RADIUS * np.cos(az_block)
            y = rotor_center[rotor_id][1] + ROTOR_RADIUS * np.sin(az_block)
            z = rotor_center[rotor_id][2] * np.ones_like(x)
            source_position = np.stack((x, y, z), axis=1)
            # Apply tilt about tilt_center            
            source_position = tilt_center[rotor_id] + (source_position - tilt_center[rotor_id]) @ trans_tilt.T 
                        
            r = observer_position - source_position
            rmag = np.linalg.norm(r, axis=1)
            
            # Mach vector 
            M = omega * ROTOR_RADIUS / SPEED_OF_SOUND
            Mi = np.stack((-M * np.sin(az_block), M * np.cos(az_block), np.zeros_like(az_block)), axis=1)
            Mi = (Mi @ trans_tilt.T)    # Apply tilt            

            # Force vector
            Fi = np.stack((np.zeros_like(az_block),
                           np.zeros_like(az_block),
                           L), axis=1)
            Fi = (Fi @ trans_tilt.T)    # Apply tilt

            # Dot products
            Mr = np.sum(r * Mi, axis=1) / rmag
            Fr = np.sum(r * Fi, axis=1) / rmag

            # Pressure (vectorized form, near-field only for speed)
            p_near = (0.25 / pi) * (
                1 / (1 - Mr) ** 2 / rmag**2
                * (Fr * (1 - M**2) / (1 - Mr) - np.sum(Fi * Mi, axis=1))
            )
            
            # # rotor-specific running mean subtraction
            # blade_period = 60.0 / (rpm_filtered[rotor_id] * NUMBER_OF_BLADES)
            # mean_window_samples = int(blade_period * AUDIO_SAMPLE_RATE * 3) # Average over 3 blade-passing cycles
            # print(mean_window_samples)
            
            # if mean_window_samples < 10:
            #     mean_window_samples = 10
            # mean_buffers[rotor_id].maxlen = mean_window_samples
            # mean_buffers[rotor_id].extend(p_near)
            # mean_val = np.mean(mean_buffers[rotor_id])
                        
            # plot_data["block_mean"].append(mean_val)
            # plot_data["block_time"].append(time.time())            
            
            out_buffer[:, 0] += p_near #* volume_gain
            out_buffer[:, 1] += p_near #* volume_gain

    # out_buffer = - (out_buffer - offset)
    
    # --- compute dynamic offset using partial mean over one-fifth revolution ---
    # period per revolution (s)
    T_rev = 60.0 / np.mean(rpm_filtered[rpm_filtered > 0]) if np.any(rpm_filtered > 0) else 0.1
    # choose window length = one-fifth revolution
    T_window = T_rev / NUMBER_OF_BLADES
    N_window = int(T_window * AUDIO_SAMPLE_RATE)

    # ensure window smaller than available samples
    if N_window > 0 and N_window < len(out_buffer):
        offset_dynamic = np.mean(out_buffer[:N_window, 0])
    else:
        offset_dynamic = np.mean(out_buffer[:, 0])

    block_mean = offset_dynamic
    plot_data["block_mean"].append(block_mean)
    # plot_data["block_time"].append(len(plot_data["block_time"]) * AUDIO_BLOCK_SIZE / AUDIO_SAMPLE_RATE)
    t_now = time.time()
    plot_data["block_time"].append(t_now)   
    
    # apply offset correction (instead of static TEST_INPUT["offset"])
    out_buffer -= offset_dynamic
    
    # print(" SCALING")
    # print(f"  Before: MIN / MAX = {np.min(out_buffer):.3f} / {np.max(out_buffer):.3f}") # check min/max value for scaling (comment out for normal run)
    out_buffer *= SCALING_FACTOR 
    # out_buffer *= volume_gain
    # print(f"  After : MIN / MAX = {np.min(out_buffer):.3f} / {np.max(out_buffer):.3f}") # check min/max value for scaling (comment out for normal run)   
          
    # --- compute SPL for monitoring ---
    spl_db = compute_spl_db(out_buffer[:,0])
    # print(f"SPL: {spl_db:.2f} dB, Offset applied: {offset_dynamic:.1f} Pa")
    plot_data.setdefault("spl_db", deque(maxlen=param_len))
    plot_data["spl_db"].append(spl_db)
    
    with filter_lock:
        if FILTER_ENABLED:
            out_buffer[:, 0], zi[:, :] = sosfilt(sos, out_buffer[:, 0], zi=zi)
            out_buffer[:, 1], _ = sosfilt(sos, out_buffer[:, 1], zi=zi)
    
    outdata[:] = out_buffer
    
    # store last few seconds for plotting
    # latest_out_buffer = np.copy(out_buffer)
    
    plot_data["wave"].extend(out_buffer[:,0])

    # update param buffers
    plot_data["spd"].append(spd)
    plot_data["tilt"].append(tilt[0])
    plot_data["coll"].append(coll_filtered[0])
    plot_data["rpm"].append(np.mean(rpm_filtered))
    plot_data["aoa"].append(aoa)
    plot_data["aos"].append(aos)
    plot_data["time"].append(time.time() % PARAM_WINDOW_SEC)


# ---------------- Simulate hover -> cruise ---------------- 
def simulate_transition():
    # at phase = 0 -> hover (high RPM, high tilt, low speed)
    # at phase = 1 -> cruise (low RPM, low tilt, high speed)
    start_time = time.time()
    while True:
        t = time.time() - start_time
        transition_duration = 60.0 # seconds
        phase = (t % (2*transition_duration)) / transition_duration # repeating 30-second cycle (15s up, 15s down)
        if phase > 1.0: # go back and forth hover <-> cruise
            phase = 2.0 - phase
        TEST_INPUT["spd"] = 67 * phase
        TEST_INPUT["aoa"] = 10.0 * np.sin(2 * np.pi * 10)
        TEST_INPUT["tilt"] = np.array([90.0 * (1-phase )] * 4)
        TEST_INPUT["rpm"] = np.array([477.5 - 309.5 * phase] * 4)
        TEST_INPUT["coll"] = np.array([15 + 35 * phase] * 4)
        time.sleep(0.05)            


# ---------------- Filter Toggle Thread ----------------

# def filter_toggle_thread():
#     global FILTER_ENABLED
#     print("Press 'f' to toggle 500-2000 Hz band-stop filter")
#     while True:
#         if keyboard.is_pressed('f'):
#             FILTER_ENABLED = not FILTER_ENABLED            
#             print(f"Filter {'ON' if FILTER_ENABLED else 'OFF'}")
#             time.sleep(0.3)
#         elif keyboard.is_pressed('q'):
#             print("Quitting filter thread...")
#             break
#         time.sleep(0.05)

# ---------------- Plotting ----------------
def start_plots():
    fig1, axs1 = plt.subplots(2, 3, figsize=(10,8))
    fig1.subplots_adjust(wspace=0.4, hspace=0.5)
    axs1[0,0].set_title("Speed"); axs1[0,1].set_title("Tilt"); axs1[0,2].set_title("Coll");
    axs1[1,0].set_title("RPM");   axs1[1,1].set_title("AOA");  axs1[1,2].set_title("AOS")
    lines1 = [axs1[0,0].plot([],[])[0],
              axs1[0,1].plot([],[])[0],
              axs1[0,2].plot([],[])[0],
              axs1[1,0].plot([],[])[0],
              axs1[1,1].plot([],[])[0],
              axs1[1,2].plot([],[])[0]]

    fig2, axs2 = plt.subplots(4, 1, figsize=(10,8))
    fig2.subplots_adjust(wspace=0.4, hspace=0.5)

    axs2[0].set_title("Waveform")
    axs2[1].set_title("FFT")
    axs2[2].set_title("Segment Mean Pressure")
    axs2[3].set_title("Block SPL (dB re 20µPa)")

    line_wave = axs2[0].plot([],[])[0]
    line_fft  = axs2[1].plot([],[])[0]
    line_mean = axs2[2].plot([],[])[0]
    line_spl  = axs2[3].plot([],[])[0]


    def update(frame):
        t_vals = list(plot_data["time"])
        # for k, key in enumerate(["spd","tilt","rpm","aos"]):
        #     lines1[k].set_data(t_vals, list(plot_data[key]))
        #     axs1[k//2,k%2].set_xlim(max(0, t_vals[0]), t_vals[-1]+1e-6)
        #     # axs1[k//2,k%2].set_ylim(min(plot_data[key]), max(plot_data[key]))
        
        # corrected 2025-10-24: to avoid list index out-of-range error
        if len(t_vals) < 2:
            return [] # skip until we have data
        for k, key in enumerate(["spd","tilt","coll","rpm","aoa","aos"]):
            y_vals = list(plot_data[key])
            if len(y_vals) < 2:
                continue
            lines1[k].set_data(t_vals, y_vals)
            try:
                axs1[k//3,k%3].set_xlim(max(0, t_vals[0]), t_vals[-1]+1e-6)
            except IndexError:
                # This can happen if axs1 indexing doesn't match grid shape
                continue    
        
        axs1[0,0].set_ylabel('Speed (m/s)')
        axs1[0,0].set_ylim(-10, 100)
        axs1[0,0].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[0,1].set_ylabel('Tilt (deg)')
        axs1[0,1].set_ylim(-10, 100)
        axs1[0,1].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[0,2].set_ylabel('Coll (deg)')
        axs1[0,2].set_ylim(-10, 70)
        axs1[0,2].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[1,0].set_ylabel('Rotating Speed (rpm)')
        axs1[1,0].set_ylim(0, 500)
        axs1[1,0].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[1,1].set_ylabel('AoA (deg)')
        axs1[1,1].set_ylim(-20, 20)
        axs1[1,1].grid(True, axis='y', linestyle='--', alpha=0.7)
        axs1[1,2].set_ylabel('AoS (deg)')
        axs1[1,2].set_ylim(-20, 20)
        axs1[1,2].grid(True, axis='y', linestyle='--', alpha=0.7)

        # waveform
        wave = np.array(plot_data["wave"])
        N = len(wave)
        t_wave = np.linspace(0, WAVEFORM_WINDOW_SEC, N)
        line_wave.set_data(t_wave, wave)
        axs2[0].set_xlim(0, WAVEFORM_WINDOW_SEC)
        # axs2[0].set_ylim(np.min(wave)-0.1*np.abs(np.min(wave)), np.max(wave)+0.1*np.abs(np.min(wave)))
        axs2[0].set_ylim(-12*SCALING_FACTOR, 12*SCALING_FACTOR)
        
        # FFT
        if N > 0:
            freqs = np.fft.rfftfreq(N, 1/AUDIO_SAMPLE_RATE)
            spectrum = np.abs(np.fft.rfft(wave))
            line_fft.set_data(freqs, 20*np.log10(spectrum+1e-12))
            # axs2[1].set_xlim(20, AUDIO_SAMPLE_RATE/2)
            axs2[1].set_xlim(0, 100)
            # axs2[1].set_ylim(np.min(20*np.log10(spectrum+1e-12)), np.max(20*np.log10(spectrum+1e-12))+1)
            axs2[1].set_ylim(40, 120)


        if len(plot_data["block_mean"]) > 0:
            # t_vals = list(plot_data["time"])[-len(plot_data["block_mean"]):]
            t_vals = list(plot_data["block_time"])
            mean_vals = list(plot_data["block_mean"])
            line_mean.set_data(t_vals, mean_vals)
            axs2[2].set_xlim(min(t_vals), max(t_vals) + 1e-6)
            axs2[2].set_ylim(min(mean_vals)-1, max(mean_vals)+1)

        if "spl_db" in plot_data and len(plot_data["spl_db"]) > 0:
            t_spl = list(plot_data["block_time"])[-len(plot_data["spl_db"]):]
            spl_vals = list(plot_data["spl_db"])
            # line_spl.set_data(t_spl, spl_vals)
            line_spl.set_data(t_spl[::2], spl_vals[::2])
            axs2[3].set_xlim(min(t_spl), max(t_spl)+1e-6)
            axs2[3].set_ylim(min(spl_vals)-2, max(spl_vals)+2)
            axs2[3].grid(True, axis='y', linestyle='--', alpha=0.7)
            # axs2[3].set_ylim(40, 120)

        return lines1 + [line_wave, line_fft, line_mean, line_spl]

    # ani1 = FuncAnimation(fig1, update, interval=int(PLOT_UPDATE_INTERVAL*1000), cache_frame_data=False)
    # ani2 = FuncAnimation(fig2, update, interval=int(PLOT_UPDATE_INTERVAL*1000), cache_frame_data=False)
    ani1 = FuncAnimation(fig1, update, interval=int(PLOT_UPDATE_INTERVAL*1000), cache_frame_data=False, save_count=100)
    ani2 = FuncAnimation(fig2, update, interval=int(PLOT_UPDATE_INTERVAL*1000), cache_frame_data=False, save_count=100)
        
    plt.show()

# ---------------- Main ----------------
def main():
    # threading.Thread(target=filter_toggle_thread, daemon=True).start()
    threading.Thread(target=simulate_transition, daemon=True).start()
    print("Starting sound stream...")
    with sd.OutputStream(
        channels=AUDIO_CHANNELS,
        samplerate=AUDIO_SAMPLE_RATE,
        blocksize=AUDIO_BLOCK_SIZE,
        callback=audio_callback
    ):
        start_plots()

if __name__ == "__main__":
    main()