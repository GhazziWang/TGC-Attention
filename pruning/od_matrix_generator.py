from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, sqrt, pow, unix_timestamp
from datetime import datetime
import torch

# Create a SparkSession
spark = SparkSession.builder \
    .appName("OD Matrix Processing") \
    .getOrCreate()

# List of CSV directory
directory_path = "../data"

# Define start and end dates
start_date = datetime(2022, 1, 1).date()  # Convert to date
end_date = datetime(2024, 2, 28).date()    # Convert to date

# Read each CSV file into a DataFrame
dataframes = spark.read.csv(directory_path, header=True, inferSchema=True)

# Filter for rideable_type = classic_bike
dataframes = dataframes.filter(col("rideable_type") == "classic_bike")

# Preprocess DataFrame to filter out data outside the specified date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 2, 28)

filtered_df = dataframes.filter((col("started_at") >= start_date) & (col("ended_at") <= end_date))

# Filter based on duration (less than 120 minutes) and distance (less than 10 kilometers)
filtered_df = filtered_df.filter((unix_timestamp("ended_at") - unix_timestamp("started_at")) / 60 <= 120) \
                         .filter(sqrt(pow(col("end_lat") - col("start_lat"), 2) + pow(col("end_lng") - col("start_lng"), 2)) * 111.32 <= 10)

print("Number of rows after pruning:", filtered_df.count())

# Step 1: Preprocess DataFrame
processed_df = filtered_df.withColumn("date", to_date("started_at")) \
                         .select("start_station_id", "end_station_id", "date")

# Step 2: Aggregate data to create OD matrix for each day
aggregated_df = processed_df.groupBy("start_station_id", "end_station_id", "date").count()

# Convert aggregated data to PyTorch tensor
unique_start_stations = processed_df.select("start_station_id").distinct().rdd.flatMap(lambda x: x).collect()
unique_end_stations = processed_df.select("end_station_id").distinct().rdd.flatMap(lambda x: x).collect()

# Combine the lists of unique start and end stations
all_unique_stations = set(unique_start_stations + unique_end_stations)

# Calculate the number of unique stations
num_stations = len(all_unique_stations)

# Calculate the number of days between the start and end dates
num_days = (end_date - start_date).days

# Index the all_unique_stations from 0 to num_stations
indexed_stations = {station_id: index for index, station_id in enumerate(all_unique_stations)}

# Initialize the tensor with the correct dimensions
od_tensor = torch.zeros(num_stations, num_stations, num_days)

for row in aggregated_df.collect():
    start_station_id = row['start_station_id']
    end_station_id = row['end_station_id']
    date = row['date']
    day_idx = (date - start_date).days  # Calculate the day index based on start_date
    count = row['count']
    
    # Map station IDs to their corresponding indices
    start_station_idx = indexed_stations.get(start_station_id)
    end_station_idx = indexed_stations.get(end_station_id)
    
    if start_station_idx is not None and end_station_idx is not None:
        # Fill the tensor with the aggregated data
        od_tensor[start_station_idx][end_station_idx][day_idx] = count
    else:
        print(f"Invalid station ID: {start_station_id} or {end_station_id}")

# Save the tensor
torch.save(od_tensor, "od_matrix_filtered.pt")

# Stop the SparkSession
spark.stop()
