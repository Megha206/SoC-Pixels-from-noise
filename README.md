# SoC-Pixels-from-noise
This repository contains all the things I learnt over the course of SoC 2025. 

**Midterm Report**

1. Week1:

During this week I just spent some time learning various libraries in python like Numpy and matplotlib, which were necessary for the upcoming weeks. 

I was also introduced to basics of signal processing. I learned how periodic signals, even discontinuous ones, can be represented using sums of sinusoids through Fourier series. We began with understanding how time-domain signals often fail to reveal important characteristics, especially in audio, and how frequency-domain representations (using harmonics) can bring out structure and consonance.

I explored how to compute Fourier coefficients by integrating the signal with cosine and sine basis functions. I also applied this to triangle and square waves, showing how the series approximates the original signal better as more harmonics are added. The famous *Gibbs phenomenon* emerged as I saw how Fourier approximations overshoot near discontinuities.

I also did some additional reading on aliasing and sampling of signals. To mitigate aliasing, I discovered the concept of anti-aliasing, where high-frequency components are removed before sampling. The lecture also introduced the Nyquist frequency as the maximum frequency that can be accurately captured—half the sampling rate. I also briefly read on quantization, where the number of bits per sample affects signal fidelity. Lower bit depths lead to audible distortion, reinforcing the importance of resolution in digital representation. This helped me understand how images are represented/converted from continuous to discrete and how this affects the clarity of the image. 

2. Week2&3:

These two weeks focused on understanding theoretical concepts necesssary for Computer vision.
 I explored the Discrete-Time Fourier Series, a powerful way to represent periodic discrete-time signals using a finite number of frequency components.

I saw how DTFS compares to its continuous-time counterpart—while the continuous version needs infinitely many terms, the discrete version only needs N terms if the signal has period N. This makes DTFS more compact and computationally efficient. At the heart of many image processing tasks is convolution—a mathematical operation that blends two signals (like an image and a filter) to produce effects such as blurring, sharpening, or edge detection. The properties of convolution (linearity, shift-invariance, commutativity) make it ideal for image filters and feature extraction.

To analyze signals in the frequency domain, we use the Fourier Transform. The 2D Fourier Transform decomposes images into their spatial frequency components, revealing texture, edges, and patterns not obvious in the pixel domain. This is essential in applications like compression, denoising, and pattern recognition.

The Discrete-Time Fourier Transform (DTFT) provides a complete frequency description of discrete signals but is continuous in frequency and not practical for computation. Instead, we use the Discrete Fourier Transform (DFT), which samples the DTFT at fixed intervals. The Fast Fourier Transform (FFT) is an efficient algorithm to compute the DFT, reducing complexity from O(N²) to O(N log N), and is widely used in real-time systems.

A key issue in frequency analysis is aliasing, where high-frequency content is misrepresented as low-frequency due to insufficient sampling. This can distort images or signals unless mitigated with proper filtering (anti-aliasing).

All these concepts operate under the framework of Linear Shift-Invariant (LSI) systems, which model many real-world image processing pipelines. LSI systems' properties ensure predictable responses to image inputs, making them fundamental to convolutional neural networks (CNNs) and classical filtering.

3. Week4:

In this week, I started writing code for the things that I studied about in the previous three weeks. The objective of this week was to write the code for a few commonly used image filters like, inversion filter, correlation, blurring, sharpening and edge detection. 

I have included a file called **Filters.py** in the repo that consists of the code for the same. 

In addition to the materials provided by my mentor, I explored OpenCV further, and did a few small projects. These include exploring the usage of Haar Cascade Filters for face and eye detection, canny edge detection and a simple RBG color detector. This helped me explore the applications of OpenCV, using the theory that I learnt during these 4 weeks. I have uploaded the codes for the above too. 

















