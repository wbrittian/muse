.PHONY: build install clean test

RELEASE_TYPE = Release
PY_SRC = src/pysrc
CPP_SRC = src/cppsrc
PYTHON = $(shell poetry run which python)

run: build pyinstall
	poetry run python3 -m pysrc.main

build: cppinstall
	cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=$(RELEASE_TYPE)/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=$(RELEASE_TYPE) -G Ninja -DPython3_EXECUTABLE=$(PYTHON)
	cd build && cmake --build .
	@cp -f build/*.so $(PY_SRC)

test: build
	cd build && ctest --output-on-failure
	poetry run python -m unittest discover -s src/pysrc/test -t src

install: pyinstall cppinstall

pyinstall:
	poetry install

cppinstall:
	conan install . --build=missing

clean:
	rm -f model/museformer.pt model/config.json