---
title: Spiking Neural Networks
description: An in depth explanation of Spiking Neural Networks 
author: Suyash Kumar
date: 2025-02-09 09:30:00 +0530
categories: [Machine Learning, Spiking Neural Networks]
tags: [Neural Networks]
pin: true
math: true
mermaid: true
image:
  path: /assets/img/SNNs.png
  # lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Responsive rendering of Chirpy theme on multiple devices.
---

# Spiking Neural Networks
![neural_net-03](/images/SNN/snn-1.png)


Spiking Neural Networks(SNNs) are the third generation of neural network models. They possess a unique ability to mimic a biological neuron closely. Biological neurons communicate via spikes—brief, discrete events in time that are Highs and lows in electrical signals. SNNs simulate this process by using spikes as the fundamental unit of information and computation. This fundamental difference from traditional artificial neural networks (ANNs), which process continuous data, makes SNNs closer to the human brain.

## 1) What are SNNs? How is it different from ANN?
<div style="text-align: center;">
  <img src="/images/SNN/snn-2.png" alt="Spikes">
</div>



Most modern AI algorithms are based on the workings of the human brain. While traditional models like artificial neural networks (ANNs) replicate brain function at a higher level, spiking neural networks(SNNs) do it at a low level by closely mimicking the actual working of biological neurons.

The primary difference between ANNs and SNN neurons is the nature of data processing. While ANNs process continuous data without incorporating time information, SNNs handle discrete data with time dynamics. Unlike ANNs, where neurons maintain their constant state regardless of time, the state of SNN neurons changes over time, either growing or decaying in response to stimuli, reflecting the stimulus-driven behavior of biological neurons.

The state of an SNN neuron is referred to as its "membrane potential". When this potential reaches a threshold, the neuron "fires", transferring data to the next neuron. This discrete firing mechanism highlights the unique nature of SNNs, enabling them to reduce computation and making them more efficient than traditional ANNs.
## 2) The Why?

You must be wondering, with the widespread use and impressive performance of ANN-based models like GPT-4, BERT, LLaMA, and others, why we would still consider older, more specialized architectures like Spiking Neural Networks (SNNs). This is because SNNs are more energy efficient, possess temporal processing along with close resemblance to biological neurons.
### 2.1) More Energy Efficient
One of the most important aspects of SNN is its energy efficiency while traditional ML models like ANN have their neurons' state updated(calculated) continuously which leads to high computation and thus higher energy cost, whereas SNN utilizes an event-driven mechanism where neurons are active only when spikes are generated which drastically reduces the computation as only a small number of neuron are active at a moment.

The efficiency of SNN leads to only necessary computation being done to update a neuron's state based on incoming spikes. This makes SNNs very energy efficient, making them very apt to be used on low-power devices without compromising their high performance.

### 2.2) Enables Temporal Processing
Let's first start by discussing what temporal data is- it refers to data having time information along with it, i.e., where the timing and sequence of events carry significant information, such as speech recognition, video analysis, sensory data processing, and real-time control systems.

SNN communicates through "spikes"- discrete events happening at specific moments in time- to encode and transmit information, which makes SNNs a strong tool for processing temporal data. Unlike ANNs that process inputs in a single time step, SNNs handle data as a series of events distributed over time. Unlike ANNs, such as LLMs, which often require additional mechanisms in the form of RNNs or LSTM units for memory management or temporal information, spiking neural networks inherently integrate time as an intrinsic element and thus make it more efficient in handling temporal patterns.

### 2.3) Closely mimics biological neuron

SNNs very closely mimic the actual working of biological neurons, while ANNs have their neurons to process continuous data & have almost every neuron connected to every other neuron, SNNs have these properties like actual neurons. A biological neuron works on discrete spikes of electric signals depicted by high & low voltages, also a single neuron is connected to only nearby neurons. SNN also follows a similar rule by working on spikes of discrete data and having a direct connection to only nearby neurons.

With regard to learning, SNN utilizes the STDP(Spike Timing Dependent Plasticity) mechanism, a biologically inspired rule that adjusts the strength of connections between neurons based on the timing of their spikes. SNNs make use of STDP to learn from temporal details in data, just like the human brain learns from experiences over time. We will talk about STDP in much detail in section [3.4](#34-Learning--STDP) below.

## 3) Key Concepts in SNN

Having discussed the WHAT & WHY question about Spiking Neural Networks, let's now delve into much detail about SNNs. We see some key concepts like-
* Spikes
* Neuron- The LIF model
* Temporal Encoding
* Learning- STDP

### 3.1) Spikes
<div style="text-align: center;">
  <img src="/images/SNN/snn-3.png" alt="Neuron">
</div>
=

Spikes are brief, discrete events in time, characterized by their binary nature, they are primary information carriers in SNNs. Spikes in SNNs refer to events when a particular neuron fires, i.e., allows data to be transmitted to its next neuron, it is these discrete spikes that give SNNs their distinct and unique nature.
### 3.2) Neuron- The LIF model
The working of SNNs' neurons is quite different from ANNs' neurons, SNNs employ specialized models to replicate the dynamics of biological neurons. The Leaky Integrate-and-Fire (LIF) Model is the most widely accepted model for an SNN neuron. The LIF neuron model simulates how a neuron integrates input signals over time and generates spikes when the accumulated input reaches a certain threshold. It is a depiction of how biological neurons behave, propagating signals based on electric potential.

**Some key components are-**
1. **Membrane Potential (*V(t)*):**
    * Membrane Potential represents the electric charge of a neuron at a specific time *t*.
    * The Membrane Potential changes with input spikes and leakage over time.
2. **Threshold Potential (*V~th~*):**
    * Threshold Potential is the voltage at the neuron fires, i.e., transmits information to the next neuron in connection.
    * After firing the neuron resets to zero or default reseting value as set.
3. **Leakage:**
    * Over time the potential of neurons drops if there isn't an incoming spike, this is referred to as Leakage.
    * Mathematically it is represented by exponential decay of *V(t)*.



This whole LIF model can be mathematically represented by a differential model, as follows
$$
{dV(t) \over dt}= {-V(t) \over {\tau}} + I(t)
$$
Where,
* *V(t)* = Membrane Potential
* *$\tau$* = time constant with which V(t) decays
* *I(t)* = Incoming current/spike
### 3.3) Temporal Encoding
As SNNs process data in the temporal domain, the data needs to be encoded to represent this temporal information. There are various encoding methods, such as.
* **Rate Coding:** We can use the rate of spikes to encode temporal dynamics, this refers to Rate Coding.
* **Temporal Coding:** If the exact timing of spikes is used it is called Temporal Coding.
* **Population Coding:** When information is Distributed across multiple neurons, allowing collective capture of the spiking pattern.

More about Temporal encoding in Section [4](#4-Encoding-Data-in-SNNs).
### 3.4) Learning through STDP
SNNs leverage a special learning mechanism named Spike-Timing Dependent Plasticity (STDP), which is a biologically inspired system of updating weights based on the timing of the spike. Let's first look at some related concepts-
* **Synapse:**  Synapse refers to the connection between two neurons.
* **Pre-Synaptic Neuron:** The neuron that fires the spike is called Pre-Synaptic Neuron.
* **Post-Synaptic Neuron:**  The neuron that receives the spike is called Post-Synaptic Neuron.

Let's now see how weights are updated in STDP, there are two possible circumstances-
* The Post-Synaptic neuron fires after the Pre-Synaptic spike reaches, in this case, the weight between the Post and pre-synaptic neuron strengthens, as it shows Pre & Post-Synaptic neurons are interlinked as one affects the other. **(Long-Term Potentiation, LTP)**
* If otherwise, the connection weakens, as the Pre-Synaptic neuron doesn't affect the Post-Synaptic one. **(Long-Term Depression, LTD)**

This temporal sensitivity allows SNNs to learn and adapt to patterns in the input data. Mathematically it can be represented as follows,
$$
\Delta w =
\begin{cases} 
A_+ e^{-\frac{\Delta t}{\tau_+}}, & \text{if } \Delta t > 0 \\ 
-A_- e^{\frac{\Delta t}{\tau_-}}, & \text{if } \Delta t < 0
\end{cases}
$$
#### Where:
- **$\Delta$ w:** Represents changes in synaptic weight.
- **$\Delta$ t:** Time difference between the post-synaptic and pre-synaptic spikes. It has below two scenarios-
  - **$\Delta$ t > 0:** Pre-synaptic spike occurs first (**LTP**).
  - **$\Delta$ t < 0:** Post-synaptic spike occurs first (**LTD**).
- **$A_+$ and $A_-$:** Maximum changes in synaptic weight for LTP and LTD, respectively.
- **$\tau_+$ and $\tau_-$:** Time constants for the exponential decay of weight changes.

<!-- ## 4) Architecture of SNN -->

## 4) Encoding Data in SNNs
<div style="text-align: center;">
  <img src="/images/SNN/snn-4.jpg" alt="Temporal Encoding">
</div>

Temporal Encoding refers to how incoming data, in the form of images, audio, or sensor readings, needs to be converted into spike trains for processing in Spiking Neural Networks (SNNs). Unlike traditional artificial neural networks, where inputs are represented as continuous values, SNNs require data to be represented in the form of discrete spikes over time.

Following up is the different types of Temporal Encoding,
1. **Rate Encoding :**
It is one of the easiest yet intuitive methods to encode incoming spikes based on the rate or frequency of spikes over a given time window.
    * **Working-**
        * The number of spikes in a fixed time period depends on the magnitude of the input value.
        * Higher input values produce more frequent spikes, while lower values result in fewer spikes.
    * **Advantages**:
        * It is easier to implement and understand.
        * It is not affected by noise in timing.
    * **Limitations**:
        * Temporal information is not utilized efficiently.
        * It is not effective if data is less, a large window size is required to grasp the temporal pattern of incoming spikes.
2. **Temporal Encoding :**
In this unlike 'Rate Encoding' exact timing of the spike is used instead of frequency, exploiting temporal dynamics in more detail.
    * **Working-**
        * The precise time of neuron firing is encoded relative to a reference point.
        * Earlier spikes indicate higher input intensities and later spikes indicate lower intensities.
    * **Advantages**:
        * Temporal patterns of incoming data can be captured in fewer spikes.
        * It very closely captures the temporal information by relying on firing time instead of firing frequency.
    * **Limitations**:
        * It is susceptible to noise in the timing of spike.
        * The time being captured needs to be perfectly synchronized at every spike.
2. **Population Encoding :**
In Population Encoding a group of neurons is used to represent a piece of information. Each neuron in the population is mapped to a specific range or aspect of input.
    * **Working-**
        * A specific group of neurons is tuned for specific information or features of incoming data.
        * The collective activity of the population represents the input.
    * **Advantages**:
        * Complex and high-dimensional data can be easily represented by assigning different features or dimensions to a specific group of neurons.
        * Robust to noise as multiple neurons contribute to the representation.
    * **Limitations**:
        * As this encoding utilizes a large number of neurons at once, the computational cost is larger than other encoding methods.
2. **Latency Encoding :**
Here the latency, that is, the delay of spikes relative to a stimulus onset is emphasized for encoding temporal information in input.
    * **Working-**
        * The strength of the input is inversely related to the latency of the spike.
        * Stronger inputs produce earlier spikes.
    * **Advantages**:
        * Captures temporal dynamics on more natural 'delay' or 'latency' based parameters.
        * It is very suitable for dynamic sensory data, such as vision or sound.
    * **Limitations**:
        * Requires precise temporal control, everything should be perfectly synchronized.



<!-- ## 6) Training methods -->

## 5) Applications of SNNs
<div style="text-align: center;">
  <img src="/images/SNN/snn-5.png" alt="Appication of SNN">
</div>

In recent times SNNs have become one of the most important research areas in the Artificial Intelligence(AI) domain, the SNNs are more robust than ANNs with their more biological design. This unique approach has paved the way for a variety of innovative applications across diverse fields.
1. **Robotics and Autonomous Systems-**
SNNs offer a robust, well-suited approach for robotics, where real-time processing and adaptive behavior are critical. SNNs' ability to process temporal data efficiently enables robots to navigate efficiently through dynamic environments, detect objects, or even interact with their surrounding.  For instance, SNNs can control prosthetic limbs or guide autonomous drones by interpreting surrounding input and making decisions in real time.
2. **Edge Computing and IoT Devices:**
SNNs are very energy-efficient and offer asynchronous processing, which is perfect for edge computing and Internet of Things (IoT) applications. Low-power devices can leverage SNNs to perform intelligent tasks locally, reducing latency and the need for cloud-based processing. This is particularly useful in applications like smart sensors, wearable devices, and smart home systems.
3. **Energy-Efficient AI:**
The low power consumption of SNNs makes them the best solution for low-energy environments. SNNs are increasingly becoming popular and are being used in scenarios where traditional AI models are too resource-intensive, such as in mobile devices, wearables, and remote sensing applications. The low energy consumption of SNN can tackle the power constraints of existing AI models.
4. **Signal Processing and Sensory Data:**
The spike-based nature of SNNs makes them suitable for processing continuous sensory data, like those of auditory and visual nature. In auditory processing, SNNs are used in speech recognition and sound localization tasks, as these data have temporal information very closely associated with them. Similarly, in the vision domain, event-based cameras, which capture changes in the scene rather than static frames, utilize SNNs to increase speed, thus enabling more efficient processing for tasks like motion detection and object tracking.

## 6) Challenges & Limitations
Although SNNs promise a more energy-efficient and robust algorithm with the advantage of temporal processing, it is still in the development phase and have some major challenges & limitations that need to be tackled before they can be mass deployed to replace ANNs.

One of the major drawbacks of SNNs is their inability to leverage traditional computation accelerators like GPUs and TPUs. SNNs require a special neuromorphic chip for its deployment. The current lack of widespread adoption of neuromorphic hardware makes it difficult for us to unlock the full potential of SNNs. As the technology matures and becomes more accessible, it is expected that the compatibility between SNNs and neuromorphic chips will drive a new wave of AI applications, characterized by their efficiency, responsiveness, and closer resemblance to natural intelligence.

Another drawback of SNNs is the difficulty to train them. Traditional training algorithms like backpropagation, widely used in ANNs, are less effective for the discrete, non-differentiable nature of spikes. Researchers are actively exploring alternative training methods, such as surrogate gradient approaches, to overcome this limitation.

