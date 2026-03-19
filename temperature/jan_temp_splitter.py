# Microsoft Copilot-generated code to split thankgsiving temperature into three files, matching to the three separate runs
# thanksgiving0001 25-11-25 14-17-25.bmp	2025-11-25 14:17:25
# thanksgiving8557 25-12-01 12-55-43.bmp	2025-12-01 12:55:43
# snowday0001 25-12-01 16-31-28.bmp	        2025-12-01 16:31:28
# snowday2449 25-12-03 09-20-22.bmp	        2025-12-03 09:20:22
# lowhumidity0001 25-12-05 10-32-56.bmp     2025-12-05 10:32:56
# lowhumidity7156 25-12-10 09-52-01.bmp     2025-12-10 09:52:01
import pandas as pd

# --- inputs ---
input_csv = "temperature_log_mid_jan.csv"
output_csv_thanksgiving = "data_mid_jan.csv"
output_csv_snowday = "data_end_jan.csv"
output_csv_lowhumidity = "data_newsecondary.csv"
common_header = "Timestamp,SHT_Temperature_C,MCP_Temperature_C,HDC_Temperature_C,SHT_Relative_Humidity,HDC_Relative_Humidity"
columns = common_header.split(',')
# sample line: 2025-11-25 11:20:52,21.57,21.31,23.0,26.35,31.87

split_time_str1 = "2026-01-22 00:00:00"  # your split point
split_time_str2 = "2026-01-30 00:00:00"  # your split point

# Read CSV and parse the first column as datetime
# Adjust 'parse_dates=[0]' if the datetime column isn't the first, e.g., use ['timestamp'] for a named column.
df = pd.read_csv(
    input_csv,
    parse_dates=[0],            # first column is datetime
    infer_datetime_format=True  # speeds up parsing for common formats
)
df.columns=columns


# Create the split timestamp (timezone-aware to match df if you localized above)
split_time1 = pd.Timestamp(split_time_str1)
split_time2 = pd.Timestamp(split_time_str2)

# Sanity: enforce chronological order if needed
if split_time2 <= split_time1:
    raise ValueError("split_time_str2 must be later than split_time_str1")



# Split into three DataFrames
thanksgiving_df = df[df.iloc[:, 0] < split_time1]
snowday_df      = df[(df.iloc[:, 0] >= split_time1) & (df.iloc[:, 0] < split_time2)]
lowhumidity_df  = df[df.iloc[:, 0] >= split_time2]

# Write out to CSV with the common header and no index
thanksgiving_df.to_csv(output_csv_thanksgiving, index=False, header=True)
snowday_df.to_csv(output_csv_snowday, index=False, header=True)
lowhumidity_df.to_csv(output_csv_lowhumidity, index=False, header=True)

print(f"Wrote {len(thanksgiving_df)} rows to {output_csv_thanksgiving}")
print(f"Wrote {len(snowday_df)} rows to {output_csv_snowday}")
print(f"Wrote {len(lowhumidity_df)} rows to {output_csv_lowhumidity}")
