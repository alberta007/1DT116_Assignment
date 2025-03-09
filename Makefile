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
	demo/demo scenario.xml --export-trace --o && \
	python3 visualizer/visualize_export.py export_trace.bin 

run_timing:
	demo/demo scenario.xml --timing-mode --$(set_mode) --max-steps $(arg)

valgrind: demo/demo
	@echo "Running Valgrind..."
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes demo/demo hugeScenario.xml --timing-mode --simd --max-steps 100

