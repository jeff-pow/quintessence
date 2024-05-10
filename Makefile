EXE = Titan
LXE = titan

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
else
	NAME := $(LXE)
endif

openbench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo rustc --release -- -C target-cpu=x86-64 --emit link=titan-x64_64-linux-v1
	cargo rustc --release -- -C target-cpu=x86-64-v2 --emit link=titan-x64_64-linux-v2
	cargo rustc --release -- -C target-cpu=x86-64-v3 --emit link=titan-x64_64-linux-v3
	cargo rustc --release -- -C target-cpu=x86-64-v4 --emit link=titan-x64_64-linux-v4
	# cargo rustc --release --target=x86_64-pc-windows-gnu -- -C target-feature=+crt-static -C target-cpu=x86-64 --emit link=Titan-x86_64-windows-v1.exe
	# cargo rustc --release --target=x86_64-pc-windows-gnu -- -C target-feature=+crt-static -C target-cpu=x86-64-v2 --emit link=Titan-x86_64-windows-v2.exe
	# cargo rustc --release --target=x86_64-pc-windows-gnu -- -C target-feature=+crt-static -C target-cpu=x86-64-v3 --emit link=Titan-x86_64-windows-v3.exe
	# cargo rustc --release --target=x86_64-pc-windows-gnu -- -C target-feature=+crt-static -C target-cpu=x86-64-v4 --emit link=Titan-x86_64-windows-v4.exe

avx512:
	cargo rustc --release --features avx512 -- -C target-cpu=native --emit link=$(NAME)

bench:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)
	./$(NAME) bench

ancient:
	cargo rustc --release -- -C target-cpu=x86-64 --emit link=$(NAME)
