################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: RecommendSystemCoreTest

dependents:
	-cd /Users/Firfi/work/c++/workspace/RecommendSystemCore/Debug && $(MAKE) all

# Tool invocations
RecommendSystemCoreTest: $(OBJS) $(USER_OBJS) /Users/Firfi/work/c++/workspace/RecommendSystemCore/Debug/RecommendSystemCore
	@echo 'Building target: $@'
	@echo 'Invoking: Cross G++ Linker'
	g++ -L/Users/Firfi/work/c++/lib/gtest-1.6.0/xcode/build/Release/ -o "RecommendSystemCoreTest" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(C++_DEPS)$(OBJS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) RecommendSystemCoreTest
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:
/Users/Firfi/work/c++/workspace/RecommendSystemCore/Debug/RecommendSystemCore:

-include ../makefile.targets
