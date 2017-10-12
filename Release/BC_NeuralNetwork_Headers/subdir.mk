################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_NeuralNetwork_Headers/nonLinearityFunction.cu 

OBJS += \
./BC_NeuralNetwork_Headers/nonLinearityFunction.o 

CU_DEPS += \
./BC_NeuralNetwork_Headers/nonLinearityFunction.d 


# Each subdirectory must supply rules for building sources it contributes
BC_NeuralNetwork_Headers/%.o: ../BC_NeuralNetwork_Headers/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -I/usr/include/atlas -O3 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_NeuralNetwork_Headers" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -I/usr/include/atlas -O3 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


