CUDA_HOME   = /usr/local/cuda
CUDA_ARCH = -gencode=arch=compute_86,code=sm_86 \
	                -gencode=arch=compute_89,code=compute_89


NVCC        = $(CUDA_HOME)/bin/nvcc
# Base compiler flags (can be overridden). Keep -lineinfo for better mapping.
NVCC_FLAGS ?= -lineinfo -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(CUDA_ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
# NVCC_FLAGS  = -lineinfo -Xcompiler -g -Xcompiler -O0 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(CUDA_ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
#PROG_FLAGS  = -DNumElem=512
PROG_FLAGS  = -DNumElem=1024

# Enable debug easily with: make DEBUG=1 kernel09.exe
ifeq ($(DEBUG),1)
	# Device debug (-G) and host symbols, disable optimizations.
	NVCC_FLAGS += -G -g -O0 -Xcompiler -g -Xcompiler -O0
	LD_FLAGS   += -g
endif

EXE01	        = kernel01.exe
EXE02	        = kernel02.exe
EXE03	        = kernel03.exe
EXE04	        = kernel04.exe
EXE05	        = kernel05.exe
EXE06	        = kernel06.exe
EXE07	        = kernel07.exe
EXE08	        = kernel08.exe
EXE09	        = kernel09.exe
EXE10	        = kernel10.exe
OBJ01	        = main01.o
OBJ02	        = main02.o
OBJ03	        = main03.o
OBJ04	        = main04.o
OBJ05	        = main05.o
OBJ06	        = main06.o
OBJ07	        = main07.o
OBJ08	        = main08.o
OBJ09	        = main09.o
OBJ10	        = main10.o

TST             = test.exe
TSTO            = test.o

EXE = $(EXE01) $(EXE02) $(EXE03) $(EXE04) $(EXE05) $(EXE06) $(EXE07) $(EXE08) $(EXE09) $(EXE10)

default: $(EXE)


test.o: test.cu
	$(NVCC) -c -o $@ test.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main01.o: main01.cu
	$(NVCC) -c -o $@ main01.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main02.o: main02.cu
	$(NVCC) -c -o $@ main02.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main03.o: main03.cu
	$(NVCC) -c -o $@ main03.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main04.o: main04.cu
	$(NVCC) -c -o $@ main04.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main05.o: main05.cu
	$(NVCC) -c -o $@ main05.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main06.o: main06.cu
	$(NVCC) -c -o $@ main06.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main07.o: main07.cu
	$(NVCC) -c -o $@ main07.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main08.o: main08.cu
	$(NVCC) -c -o $@ main08.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main09.o: main09.cu
	$(NVCC) -c -o $@ main09.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main10.o: main10.cu
	$(NVCC) -c -o $@ main10.cu $(NVCC_FLAGS) $(PROG_FLAGS)

$(TST): $(TSTO)
	$(NVCC) $(TSTO) -o $(TST) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) vector.cu -o $(EXE01) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

$(EXE02): $(OBJ02)
	$(NVCC) $(OBJ02) vector.cu -o $(EXE02) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

$(EXE03): $(OBJ03)
	$(NVCC) $(OBJ03) vector.cu -o $(EXE03) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

$(EXE04): $(OBJ04)
	$(NVCC) $(OBJ04) vector.cu -o $(EXE04) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

$(EXE05): $(OBJ05)
	$(NVCC) $(OBJ05) vector.cu -o $(EXE05) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS) 

$(EXE06): $(OBJ06)
	$(NVCC) $(OBJ06) vector.cu -o $(EXE06) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS) 

$(EXE07): $(OBJ07)
	$(NVCC) $(OBJ07) vector.cu -o $(EXE07) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS) 

$(EXE08): $(OBJ08)
	$(NVCC) $(OBJ08) vector.cu -o $(EXE08) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS) 

$(EXE09): $(OBJ09)
	$(NVCC) $(OBJ09) vector.cu -o $(EXE09) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

$(EXE10): $(OBJ10)
	$(NVCC) $(OBJ10) vector.cu -o $(EXE10) $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

all:	$(EXE01) $(EXE02) $(EXE03) $(EXE04) $(EXE05) $(EXE06) $(EXE07) $(EXE08) $(EXE09) $(EXE10)

# Generic rule: link a single object into an executable using nvcc.
# This prevents make from falling back to the default `cc` linker when
# the user invokes e.g. `make main09` (which would try to link with cc).
%: %.o
	$(NVCC) $< vector.cu -o $@ $(NVCC_FLAGS) $(PROG_FLAGS) $(LD_FLAGS)

.PHONY: debug
debug:
	# Ensure a full rebuild with debug flags (host -g, device -G)
	$(MAKE) clean
	$(MAKE) -B DEBUG=1 kernel09.exe

clean:
	rm -rf *.o $(EXE)

ultraclean:
	rm -rf *.o $(EXE) job-*e* job*.o*

