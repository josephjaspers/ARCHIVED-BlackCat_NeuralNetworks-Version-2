################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_NN_tests/MNIST_test.cpp \
../BC_NN_tests/Simple_QlearningTest.cpp \
../BC_NN_tests/SnakeGame.cpp \
../BC_NN_tests/SnakeTrainer.cpp \
../BC_NN_tests/deprecated\ Tests.cpp 

OBJS += \
./BC_NN_tests/MNIST_test.o \
./BC_NN_tests/Simple_QlearningTest.o \
./BC_NN_tests/SnakeGame.o \
./BC_NN_tests/SnakeTrainer.o \
./BC_NN_tests/deprecated\ Tests.o 

CPP_DEPS += \
./BC_NN_tests/MNIST_test.d \
./BC_NN_tests/Simple_QlearningTest.d \
./BC_NN_tests/SnakeGame.d \
./BC_NN_tests/SnakeTrainer.d \
./BC_NN_tests/deprecated\ Tests.d 


# Each subdirectory must supply rules for building sources it contributes
BC_NN_tests/%.o: ../BC_NN_tests/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -std=c++17 -fopenmp -I/usr/include/atlas -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

BC_NN_tests/deprecated\ Tests.o: ../BC_NN_tests/deprecated\ Tests.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C++ Compiler'
	g++ -std=c++17 -fopenmp -I/usr/include/atlas -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/cuda-workspace/BLACKCAT_NeuralNetworks/BC_NeuralNetwork_Headers -O2 -g -Wall -c -fmessage-length=0 -MMD -MP -MF"BC_NN_tests/deprecated Tests.d" -MT"BC_NN_tests/deprecated\ Tests.d" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


