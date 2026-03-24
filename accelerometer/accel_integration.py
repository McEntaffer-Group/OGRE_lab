from typing import Optional, Any

import pandas as pd
import numpy as np
from scipy.integrate import simpson, cumulative_simpson, solve_ivp
from scipy.interpolate import interp1d
import pathlib
import glob
import matplotlib
import matplotlib.pyplot as plt

# Load data
source_folder = pathlib.Path(r"D:\Reverse Telescope Test\accel\Session_2025-10-29_163326")
csv_files = glob.glob(str(source_folder / "*.csv"))
try :
    df_list = [pd.read_csv(f) for f in csv_files[0:10]] # 50 files will eat up about 8 gigs of ram
    df = pd.concat(df_list, ignore_index=True)
except MemoryError as e:
    print("Memory error, you may have too much data.")
    print(e)
    if len(csv_files) > 5 :
        df_list = [pd.read_csv(f) for f in csv_files[0:5]]
    else:
        df_list = [pd.read_csv(csv_files[0])]
    df = pd.read_csv(pathlib.Path.joinpath(source_folder, "*.csv"))
except Exception as e :
    print(e)


t = df['RelativeTime_s'].values
ax = df['Mirror_X_g'].values - df['Mirror_X_g'].median() # accel output from matlab is in g's. g = 386.1 inches/sec^2 For now just get raw V
ay = df['Mirror_Y_g'].values - df['Mirror_Y_g'].median()
az = df['Mirror_Z_g'].values - df['Mirror_Y_g'].median()
aarm = df['Desk_Y_g'].values - df['Desk_Y_g'].median()

def plotaccels(
        t: np.ndarray[Any, np.dtype[np.floating[Any]]],
        a: np.ndarray[Any, np.dtype[np.floating[Any]]],
        axis: str,
        suppressPosition: Optional = False,
        suppressVelocity: Optional = False) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if len(t) != len(a):
        raise ValueError("All input arrays must have the same length.")
    if suppressVelocity :
        suppressPosition = True
    plt.figure(figsize=(10, 6))
    v=None
    p=None
    if suppressPosition and suppressVelocity :
        mean = a.mean()
        std = a.std()
        plt.plot(t, a, label=f'{axis} Acceleration (V) (mean={mean:.2f}, std={std:.2f})')
    else:
        plt.plot(t, a, label=f'{axis} Acceleration (V)')
    if not suppressVelocity :
        # Need to try to detrend the drift in velocity?
        v = cumulative_simpson(y=a, x=t, initial=0)
        plt.plot(t, v, label=f'{axis} Velocity (in/s)')
    if not suppressPosition:
        p = cumulative_simpson(y=v, x=t, initial=0)
        plt.plot(t, p, label=f'{axis} Displacement (in)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title(f'{axis} Acceleration → Velocity → Displacement')
    plt.show()

    return v, p

vx, px = plotaccels(t, ax, "X", suppressPosition=True, suppressVelocity=False)
vy, py = plotaccels(t, ay, "Y", suppressPosition=True, suppressVelocity=False)
vz, pz = plotaccels(t, az, "Z", suppressPosition=True, suppressVelocity=False)
varm, parm = plotaccels(t, aarm, "Desk Y", suppressPosition=True, suppressVelocity=False)
#
# # Integration using Runge-Kutta (solve_ivp)
# # Interpolation for acceleration
# ax_fun = interp1d(t, ax, kind='linear', fill_value='extrapolate')
#
# def odefun_x(tt, y):
#     return [ax_fun(tt), y[0]]  # [dv/dt, dx/dt]
#
# y0 = [0, 0]
# sol_x = solve_ivp(odefun_x, [t[0], t[-1]], y0, t_eval=t, method='RK45')
# velocity_x = sol_x.y[0]
# position_x = sol_x.y[1]
