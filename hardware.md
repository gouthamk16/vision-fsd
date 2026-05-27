## Lightning fast inference for self-driving vehicles

**The problem**: How to get actuator predictions from machine learning models (multiple) in the order of milliseconds for autonomous driving systems. For eg: Tesla's NPU chip enables its model to do this in 28ms. For context a human blink takes 100-400ms. i.e., Tesla makes a decision 10x faster than you can blink.

**The Solution**: A combination of really optimized, powerful hardware and intelligent software. Most of this article will be on how Tesla approached this problem.

### Tesla's Full Self-Driving Computer

Basic goals for a FSD chip:
1. Can handle atleast 50 TFLOPS
2. High utilization (~80%). Ooptimized for worst-case scenario (batch-size of 1)
3. Sub 40W/Chip
4. Combination of GPU's and CPU's for post processing and general purpose needs

...