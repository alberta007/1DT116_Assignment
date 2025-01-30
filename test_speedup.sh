#!/usr/bin/env bash

#####################################
# Configurable parameters
#####################################
THREAD_LIST="2 4 8 12 16"         # The thread counts to test
STEP_LIST="1000 3000 5000 10000 100000"  # The step sizes
REPEATS=10                        # How many times to repeat each run
MODE_LIST="o p"                  # 'o' = OpenMP, 'p' = C++ threads

OUTPUT_FILE="average_speedup.txt"

# Clear or initialize the output file with a header
echo "#mode steps threads avg_speedup" > $OUTPUT_FILE

###########################################################
# Helper function to run each combination multiple times
# and average the speedup printed by the program.
###########################################################
run_test() {
  local mode="$1"       # "o" or "p"
  local steps="$2"      # e.g. 1000, 3000, ...
  local threads="$3"    # e.g. 2, 4, 8, ...
  local repeats="$4"

  local sum=0.0
  local count=0

  for ((i=1; i<=repeats; i++)); do
    echo "Run #$i: mode=$mode, steps=$steps, threads=$threads"
    
    # Depending on the mode, set either OMP_NUM_THREADS or CXX_NUM_THREADS
    if [[ "$mode" == "o" ]]; then
      echo "Using OpenMP"
      OMP_NUM_THREADS=$threads make run_timing set_mode=o arg=$steps > tmp_run.log 2>&1
    else
      echo "Using C++ Threads"
      CXX_NUM_THREADS=$threads make run_timing set_mode=p arg=$steps > tmp_run.log 2>&1
    fi

    # Your main program prints something like:
    #
    #   Running reference version...
    #   Reference time: 382 milliseconds, ...
    #   Running target version...
    #   Target time: 419 milliseconds, ...
    #   Speedup: 0.911694
    #
    # We parse the "Speedup: X" line.

    speed_line=$(grep "Speedup:" tmp_run.log)
    # The line might look like: "Speedup: 1.2345"

    speed_val=$(echo "$speed_line" | awk '{print $2}')
    # This extracts the second token (e.g., "1.2345").

    if [[ -z "$speed_val" ]]; then
      speed_val=0.0
      echo "WARNING: Could not parse Speedup. Check your program output."
    fi

    # Accumulate the speedup
    sum=$(awk -v s="$sum" -v t="$speed_val" 'BEGIN { printf "%.6f", s + t }')
    count=$((count+1))
  done

  if [[ $count -gt 0 ]]; then
    avg_speedup=$(awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.6f", s / c }')
  else
    avg_speedup=0.0
  fi

  echo "Average speedup for mode=$mode steps=$steps threads=$threads => $avg_speedup"
  
  # Append to results file: "o 1000 4 1.23"
  echo "$mode $steps $threads $avg_speedup" >> $OUTPUT_FILE
}

#####################################
# Main loops over modes, steps, threads
#####################################
for mode in $MODE_LIST; do
  for steps in $STEP_LIST; do
    for threads in $THREAD_LIST; do
      run_test "$mode" "$steps" "$threads" "$REPEATS"
    done
  done
done

echo "All tests completed. Results saved to $OUTPUT_FILE."
