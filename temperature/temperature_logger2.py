"""
This script is a Temperature Logger for the Reverse Telescope system in the OGRE Lab, meant to help diagnose drift and
defocus errors at long time-frames (errors across hours/days) that we believed may be due to the building's HVAC.
Those errors have an unknown relationship (and might be totally independent of) vibrations experienced at short time-frames
(15 hz to 1 hz) that we were addressing through standard vibration mitigation bladders and rubber pads. There is also
a concern that humidity may be effecting the 3D printed parts, and to help assess if those should be replaced with
Aluminum parts.

This temperature logger was meant to run on a Raspberry Pi and multiple independent temperature/humidity sensors.
SHT45 - Sensiron Temperature/Humidity, I2C communication, STEMMA QT connectors
    https://www.adafruit.com/product/6174
     ΔRH = ±1.0 %RH, ΔT = ±0.1 °C
     has a internal heater for if you're trying to operate at low temp (rated for -40 C) or to boil off condensation
MCP9808 - High Precision Temperature, I2C communication, pin connectors only
    https://www.adafruit.com/product/1782#technical-details
    0.25°C typical accuracy over -40°C to 125°C range (0.5°C guaranteed max from -20°C to 100°C)
    0.0625°C resolution
HDC3022 - TI Temperature/Humidity, I2C communication, STEMMA QT connectors
    https://www.adafruit.com/product/5989
    ΔRH = ±.5 %RH, ΔT = ±0.1 °C
This is an upgrade over the BMP388 temperature/pressure sensor, both orginally meant
for use in the Rockets for Inclusive Science Education (RISE) program as an altimeter and video data collector.

Uses adafruit libraries:
    pip install adafruit-circuitpython-sht4x adafruit-circuitpython-mcp9808 adafruit-circuitpython-hdc302x
Commandline stuff to test if i2c is working as intended:
sudo apt-get install i2c-tools
i2cdetect -y 1
"""
import time
import csv
import os
from datetime import datetime
from typing import Tuple
import adafruit_bmp3xx
import board
import busio
import digitalio
import adafruit_sht4x
import adafruit_mcp9808
import adafruit_hdc302x

# Configuration
LOG_INTERVAL_SECONDS = 1  # Change to 1 for per-second logging
HOURLY_STATUS_INTERVAL = 3600  # Seconds in an hour
TEMP_THRESHOLD = 50.0  # Optional alert threshold (disabled by default)
ENABLE_ALERTS = False
ENABLE_AUTOSTART = False # Not yet implemented
SHT45_ENABLED = True
MCP9808_ENABLED = True
HDC3022_ENABLED = True
LOG_FILE = "temperature_log_middecember.csv"

class DisabledSensor:
    @property
    def temperature(self):
        return 0.0
    @property
    def relative_humidity(self):
        return 0.0
    @property
    def measurements(self) -> Tuple[float, float]:
        return (self.temperature, self.relative_humidity)


def initializeSensors(sht45=True, mcp9808=True, hdc3022=True):
    # Initialize sensors
    port = 1
    sht_address = 0x44 # SHT45 I2C address is 0x44 and cannot be changed (a manufacturer limitation)
    mcp_address = 0x18 # Default address printed on circuit card. Can be changed to with soldered jumpers.
    hdc_address = 0x45 # Default is 0x44. Solder Jumper A0 closed to get to 0x45. Can otherwise be 0x46, or 0x47
    i2c = board.I2C()
    if sht45:
        sht = adafruit_sht4x.SHT4x(i2c)
        print("Found SHT45 with serial number", hex(sht.serial_number))
        sht.mode = adafruit_sht4x.Mode.NOHEAT_HIGHPRECISION
        # Can also set the mode to enable heater
        # sht.mode = adafruit_sht4x.Mode.LOWHEAT_100MS
        print("Current mode is: ", adafruit_sht4x.Mode.string[sht.mode])
    else:
        sht = DisabledSensor()
    if mcp9808:
        mcp = adafruit_mcp9808.MCP9808(i2c, address=mcp_address)
        # To initialise using a specified address:
        # Necessary when, for example, connecting A0 to VDD to make address=0x19
        # mcp = adafruit_mcp9808.MCP9808(i2c_bus, address=0x19)
    else:
        mcp = DisabledSensor()
    if hdc3022:
        hdc = adafruit_hdc302x.HDC302x(i2c, address=hdc_address)
    else:
        hdc = DisabledSensor()
    return [sht, mcp, hdc]


def create_log_file(sht45=True, mcp9808=True, hdc3022=True):
    # Create log file with header if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "SHT_Temperature_C",  "MCP_Temperature_C", "HDC_Temperature_C", "SHT_Relative_Humidity", "HDC_Relative_Humidity"])



if __name__ == "__main__":
    # Main loop
    sht, mcp, hdc = initializeSensors()
    create_log_file()
    last_status_time = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_temperatures = [round(sht.temperature, 2), round(mcp.temperature, 2), round(hdc.temperature, 2)]
    start_humidities = [round(sht.relative_humidity,2), round(hdc.relative_humidity,2)]
    print(f"Temperature logging started @ {start_time}: Current Temperatures = {start_temperatures}°C. Current Humidities = {start_humidities}")

    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # data = bme280.sample(bus, address, calibration_params)
            temperatures = [round(sht.temperature, 2), round(mcp.temperature, 2), round(hdc.temperature, 2)]
            humidities = [round(sht.relative_humidity, 2), round(hdc.relative_humidity, 2)]

            # Log to CSV
            with open(LOG_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_time, *temperatures, *humidities])

            # Optional alert
            if ENABLE_ALERTS and temperatures[0] > TEMP_THRESHOLD:
                print(f"ALERT: Temperature exceeded threshold at {current_time}: {temperatures}°C")

            # Hourly status output
            if time.time() - last_status_time >= HOURLY_STATUS_INTERVAL:
                print(f"Status Update @ {current_time}: Current Temperature = {temperatures}°C. Current Humidities = {humidities}")
                last_status_time = time.time()

            time.sleep(LOG_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("Temperature logging stopped by user.")
