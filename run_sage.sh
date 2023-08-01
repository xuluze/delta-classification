#!/bin/bash

# Define the SageMath command you want to execute
SAGE_FILE="/home/xuluze/delta-classification/test_diskcache.sage"

# Number of instances you want to run
NUM_INSTANCES=4

# Function to execute SageMath command
execute_sage() {
    sage "$SAGE_FILE"
}

# Record the start time
START_TIME=$(date +%s)

# Loop to run the instances
for ((i = 1; i <= $NUM_INSTANCES; i++)); do
    execute_sage &
done

# Wait for all instances to complete
wait

# Record the end time
END_TIME=$(date +%s)

# Calculate the elapsed time
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "All instances completed!"
echo "Total execution time: $ELAPSED_TIME seconds."
