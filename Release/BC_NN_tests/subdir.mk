################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_NN_tests/MNIST_test.cpp \
../BC_NN_tests/deprecated\ Tests.cpp 

OBJS += \
./BC_NN_tests/MNIST_test.o \
./BC_NN_tests/deprecated\ Tests.o 

CPP_DEPS += \
./BC_NN_tests/MNIST_test.d \
./BC_NN_tests/deprecated\ Tests.d 


# Each subdirectory must supply rules for building sources it contributes
BC_NN_tests/%.o: ../BC_NN_tests/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -I/usr/include/atlas -O3 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_NN_tests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -I/usr/include/atlas -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

BC_NN_tests/deprecated\ Tests.o: ../BC_NN_tests/deprecated\ Tests.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -I/usr/include/atlas -O3 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_NN_tests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -I/usr/include/atlas -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


