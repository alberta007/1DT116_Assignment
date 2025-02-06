#!/bin/bash
# This script runs the sequential demo for each scenario with different tick counts,
# 10 times per combination. It extracts the "Target time" (in milliseconds) from the output,
# calculates the average runtime, and writes the results to a file named results.txt.
#
# Adjust the tick_counts array to vary the number of ticks (simulation steps) as needed.
# This helps you achieve a short test duration while still getting accurate performance readings.

# File to store the results.
output_file="results.txt"

# Overwrite the output file at the start.
echo "Performance Test Results - $(date)" > "$output_file"
echo "----------------------------------------" >> "$output_file"

# Define the tick count values to test.
tick_counts=(500 1000 5000)

# List the scenario files to test.
scenarios=("hugeScenario.xml" "scenario_box.xml" "scenario.xml")

# Loop through each scenario file.
for scenario in "${scenarios[@]}"; do
    echo "==========================================" | tee -a "$output_file"
    echo "Testing scenario: $scenario" | tee -a "$output_file"

    # Loop through each tick count value.
    for tick in "${tick_counts[@]}"; do
        echo "------------------------------------------" | tee -a "$output_file"
        echo "Testing with tick count: $tick" | tee -a "$output_file"
        total_ms=0
        valid_runs=0

        # Run the test 10 times.
        for run in {1..10}; do
            echo "Run $run:" | tee -a "$output_file"
            output=$(./demo/demo "$scenario" --timing-mode --seq --max-steps "$tick")
            echo "$output" | tee -a "$output_file"

            # Extract the target time in milliseconds.
            # Assumes a line like:
            # "Target time: 33 milliseconds, 30303 Frames Per Second."
            target_time=$(echo "$output" | grep "Target time:" | awk '{print $3}')
            
            if [[ -z "$target_time" ]]; then
                echo "Warning: Could not extract target time from the output." | tee -a "$output_file"
            else
                echo "Extracted target time: $target_time milliseconds" | tee -a "$output_file"
                total_ms=$(echo "$total_ms + $target_time" | bc)
                valid_runs=$((valid_runs+1))
            fi
        done

        # Calculate and display the average runtime if at least one run was valid.
        if [[ $valid_runs -gt 0 ]]; then
            avg_ms=$(echo "scale=3; $total_ms / $valid_runs" | bc)
            avg_sec=$(echo "scale=3; $avg_ms / 1000" | bc)
            echo "Average target time for $scenario with tick count $tick: $avg_ms ms ($avg_sec seconds)" | tee -a "$output_file"
        else
            echo "No valid runs for scenario $scenario with tick count $tick" | tee -a "$output_file"
        fi
        echo "" | tee -a "$output_file"
    done
done

echo "Results have been written to $output_file"
