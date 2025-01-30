arg = 200

.PHONY: clean libpedsim demo

all: libpedsim demo

libpedsim:
	make -C libpedsim

demo:
	make -C demo

clean:
	make -C libpedsim clean
	make -C demo clean
	-rm submission.tar.gz

submission: clean
	mkdir submit
	cp -r demo submit/
	cp -r libpedsim submit/
	cp Makefile submit/
	cp scenario.xml submit/
	cp scenario_box.xml submit/
	cp hugeScenario.xml submit/
	cp lab3-scenario.xml submit/
	tar -czvf submission.tar.gz submit
	rm -rf submit

run:
	demo/demo scenario.xml 

run_export:
	demo/demo scenario.xml --export-trace && \
	python3 visualizer/visualize_export.py export_trace.bin 

run_timing:
	demo/demo hugeScenario.xml --timing-mode --$(set_mode) --max-steps $(arg)

# Example: run multiple tests with different steps and threads
# run_timing: 
#     # 1) Baseline (serial)
#     echo "Running serial for baseline..."
#     demo/demo scenario.xml --timing-mode --set_mode=seq --max-steps 1000 > serial_1000.log
    
#     # 2) OpenMP with different thread counts
#     for t in 1 2 4 8; do \
#       echo "Running OpenMP with $$t threads..."; \
#       OMP_NUM_THREADS=$$t demo/demo scenario.xml --timing-mode --set_mode=o --max-steps 1000 > openmp_1000_$$t.log; \
#     done
    
#     # 3) C++ Threads with different thread counts
#     for t in 1 2 4 8; do \
#       echo "Running C++ Threads with $$t threads..."; \
#       CXX_NUM_THREADS=$$t demo/demo scenario.xml --timing-mode --set_mode=p --max-steps 1000 > threads_1000_$$t.log; \
#     done
    
#     echo "All runs complete."

