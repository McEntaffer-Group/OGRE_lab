import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import os
import glob
from pylab import *
from math import e
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import rotate
import matplotlib.animation as animation
from IPython.display import HTML
import pathlib
import pandas as pd
import matplotlib.dates as mdates

# Categories are ["Timestamp", "SHT_Temperature_C",  "MCP_Temperature_C", "HDC_Temperature_C", "SHT_Relative_Humidity", "HDC_Relative_Humidity"]
temps=pd.read_csv('data_frosty.csv')
temps['Timestamp'] = pd.to_datetime(temps['Timestamp'])
df = temps.set_index('Timestamp').resample('1min').mean()

[fig, ax] = plt.subplots(figsize=(12, 6))

# ax.plot(df.index, df['SHT_Temperature_C']*9/5+32, color='blue', linewidth=0.8)
# ax.plot(df.index, df['MCP_Temperature_C']*9/5+32, color='green', linewidth=0.8)
# ax.plot(df.index, df['HDC_Temperature_C']*9/5+32, color='orange', linewidth=0.8)
ax.plot(df.index, df['SHT_Relative_Humidity'], color='teal', linewidth=0.8)
ax.plot(df.index, df['HDC_Relative_Humidity'], color='purple', linewidth=0.8)

# Format x-axis
ax.xaxis.set_major_locator(mdates.DayLocator())        # Big ticks at midnight
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))  # Small ticks at 6-hour intervals
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Grid lines
ax.grid(which='major', color='black', linewidth=1.2)   # Big grid lines
ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.6)  # Small grid lines

plt.title('Frosty Humidity Over Time (resampled to the minute mean)')
plt.xlabel('Date')
plt.ylabel('Relative Humidity (%)')
plt.tight_layout()

plt.savefig('frosty_humidity_overtime.png')
plt.show()