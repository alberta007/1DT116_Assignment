#!/usr/bin/env bash

#####################################
# Configurable parameters
#####################################
THREAD_LIST="1"                # The thread counts to test.
STEP_LIST="1000 3000 5000 10000 100000"    # The step sizes.
REPEATS=10                               # How many times to repeat each run.
MODE_LIST="simd"                        # 'o' = OpenMP, 'p' = C++ threads, 's' = SIMD

OUTPUT_FILE="average_speedup_simd.txt"

# Clear or initialize the output file with a header.
echo "#mode steps threads avg_speedup" > $OUTPUT_FILE

###########################################################
# Helper function to run each combination multiple times.
# It runs the program and averages the speedup printed by the program.
###########################################################
run_test() {
  local mode="$1"       # "o", "p", or "s"
  local steps="$2"      # e.g., 1000, 3000, ...
  local threads="$3"    # e.g., 2, 4, 8, ...
  local repeats="$4"

  local sum=0.0
  local count=0

  for ((i=1; i<=repeats; i++)); do
    echo "Run #$i: mode=$mode, steps=$steps, threads=$threads"

    # Set the appropriate environment variable or make parameter based on mode.
    # For example, if mode "o" uses OpenMP, "p" uses C++ threads, and "s" uses SIMD.
    if [[ "$mode" == "o" ]]; then
      echo "Using OpenMP"
      OMP_NUM_THREADS=$threads make run_timing set_mode=o arg=$steps > tmp_run.log 2>&1
    elif [[ "$mode" == "p" ]]; then
      echo "Using C++ Threads"
      CXX_NUM_THREADS=$threads make run_timing set_mode=p arg=$steps > tmp_run.log 2>&1
    elif [[ "$mode" == "simd" ]]; then
      echo "Using SIMD"
      # For SIMD mode, you might not need to set a thread count, but we include it for consistency.
      make run_timing set_mode=simd arg=$steps > tmp_run.log 2>&1
    else
      echo "Unknown mode: $mode"
      exit 1
    fi

    # Parse the output (which is assumed to print a line like "Speedup: X").
    speed_line=$(grep "Speedup:" tmp_run.log)
    speed_val=$(echo "$speed_line" | awk '{print $2}')

    if [[ -z "$speed_val" ]]; then
      speed_val=0.0
      echo "WARNING: Could not parse Speedup. Check your program output."
    fi

    sum=$(awk -v s="$sum" -v t="$speed_val" 'BEGIN { printf "%.6f", s + t }')
    count=$((count+1))
  done

  if [[ $count -gt 0 ]]; then
    avg_speedup=$(awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.6f", s / c }')
  else
    avg_speedup=0.0
  fi

  echo "Average speedup for mode=$mode steps=$steps threads=$threads => $avg_speedup"
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
