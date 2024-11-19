# **1. Channel Models**

Wireless communication channels are inherently unpredictable and vary across time, frequency, and space due to the dynamic nature of the propagation environment. Accurate channel modeling is essential for designing and evaluating modern communication systems, particularly in 5G and beyond. This chapter provides an overview of wireless channel characteristics, common models, and their simulation techniques.

---

## **1.1 Wireless Channel Characteristics**
Wireless channels exhibit unique characteristics due to the interaction between transmitted signals and the surrounding environment. The primary factors affecting wireless channels are:

!!! note "small-scale fading"
    
    ### **1.1.1 Multipath Propagation**
    - **Definition**: Signals from a transmitter reach the receiver via multiple paths caused by reflection, diffraction, and scattering.
    - **Impacts**:
    
        - **Constructive or destructive interference**: Leads to signal fading.
        - **Delay spread**: Causes inter-symbol interference (ISI).
        - **Frequency selectivity**: Variations in channel gain across different frequencies.
    
    ### **1.1.2 Doppler Effect**
    - **Definition**: Frequency shift caused by the relative motion between transmitter and receiver.
    - **Impacts**:
        
        - **Doppler spread**: Results in time-selective fading.
        - **Coherence time**: Defines how long the channel remains static.

!!! note "large-scale fading"
    
    ### **1.1.3 Shadowing**
    - **Definition**: Signal attenuation caused by large obstacles (e.g., buildings, mountains).
    - **Modeling**:
    
          - Typically modeled as a log-normal distribution over distance.
    
    ### **1.1.4 Path Loss**
    - **Definition**: Reduction in signal power as it propagates through space.
    
    - **Impacts**:
      
          - Determines the communication range.
          - Dominates over large distances.

---

## **1.2 Wireless Channel Classification**
Wireless channels are commonly classified based on their characteristics:

### **1.2.1 Classification by Fading**
1. **Flat Fading**:

       - All frequency components of the signal experience the same fading.
       - Occurs when the channel bandwidth is much larger than the signal bandwidth.

2. **Frequency-Selective Fading**:

       - Different frequency components experience different fading.
       - Caused by multipath propagation with delay spread larger than the signal's symbol duration.

### **1.2.2 Classification by Time Variability**
1. **Fast Fading**:

       - Channel changes rapidly within the symbol duration.
       - Caused by high mobility or large Doppler spread.

2. **Slow Fading**:

       - Channel remains constant over several symbol durations.
       - Results from slow environmental changes.

### **1.2.3 Classification by LOS/NLOS**
1. **Line-of-Sight (LOS)**:

       - Direct path exists between transmitter and receiver.
       - Typically observed in free-space or rural environments.

2. **Non-Line-of-Sight (NLOS)**:

       - No direct path due to obstructions.
       - Common in urban and indoor environments.

---

## **1.3 Common Channel Models**
### **1.3.1 Free-Space Path Loss Model**
- **Description**:

      - Idealized model for LOS communication.
      - Path loss is proportional to the square of the distance.

- **Path Loss Formula**:

  $$
  PL(d) = 20 \log_{10}(d) + 20 \log_{10}(f) + 20 \log_{10}(\frac{4\pi}{c}) \quad (\text{dB})
  $$

  where \(d\) denotes distance (in meters), \(f\) denotes frequency (in Hz), and \(c\) denotes the speed of light (in m/s).

### **1.3.2 Rayleigh Fading Model**
- **Description**:

      - Assumes no dominant LOS path.
      - Signal amplitude follows a Rayleigh distribution.

- **Applications**:

      - Urban environments with dense scatterers.

- **Probability Density Function (PDF)**:

  $$
  f_R(r) = \frac{r}{\sigma^2} e^{-r^2 / (2\sigma^2)}, \quad r \geq 0
  $$

### **1.3.3 Rician Fading Model**
- **Description**:

      - Incorporates both LOS and scattered components.
      - Signal amplitude follows a Rician distribution.

- **Applications**:

      - Environments with a strong LOS component (e.g., highways, rural areas).

- **PDF**:
  $$
  f_R(r) = \frac{r}{\sigma^2} e^{-(r^2 + A^2) / (2\sigma^2)} I_0\left(\frac{Ar}{\sigma^2}\right), \quad r \geq 0
  $$

  where \(A\) denotes amplitude of the LOS component, and \(I_0\) denotes modified Bessel function of the first kind.

### **1.3.4 Log-Normal Shadowing Model**
- **Description**:

      - Models large-scale signal variations due to shadowing.
      - Shadowing effects are modeled as a Gaussian random variable in dB scale.

- **Path Loss with Shadowing**:
  $$
  PL(d) = PL_0 + 10\beta \log_{10}(d/d_0) + X_\sigma
  $$
  
  where \(PL_0\) denotes path loss at reference distance \(d_0\), and \(X_\sigma\) denotes zero-mean Gaussian random variable with standard deviation \(\sigma\).

---

## **1.4 Python Simulation**
The following Python code demonstrates the simulation of common channel models.

### **Large-scale Path Loss Simulation**
``` py
import numpy as np
import matplotlib.pyplot as plt


def free_space_path_loss(fc, dist, Gt=1, Gr=1):
    """
    Computes the free-space path loss (FSPL).

    Args:
        fc (float): Carrier frequency in Hz.
        dist (float or ndarray): Distance between base station and mobile station in meters.
        Gt (float, optional): Transmitter gain. Defaults to 1.
        Gr (float, optional): Receiver gain. Defaults to 1.

    Returns:
        float or ndarray: Path loss in dB.
    """
    lamda = 3e8 / fc  # Wavelength in meters
    tmp = lamda / (4 * np.pi * dist)  # Free-space propagation factor
    if Gt > 0:
        tmp *= np.sqrt(Gt)
    if Gr > 0:
        tmp *= np.sqrt(Gr)
    PL = -20 * np.log10(tmp)  # Convert to dB
    return PL


def log_distance_or_shadowing_path_loss(fc, d, d0, n, sigma=0):
    """
    Computes the log-distance or log-normal shadowing path loss.

    Args:
        fc (float): Carrier frequency in Hz.
        d (float or ndarray): Distance between base station and mobile station in meters.
        d0 (float): Reference distance in meters.
        n (float): Path loss exponent.
        sigma (float, optional): Standard deviation for shadowing in dB. Defaults to 0.

    Returns:
        float or ndarray: Path loss in dB.
    """
    lamda = 3e8 / fc  # Wavelength in meters
    PL = -20 * np.log10(lamda / (4 * np.pi * d0)) + 10 * n * np.log10(d / d0)  # Log-distance model
    if sigma > 0:
        PL += sigma * np.random.randn(*np.shape(d))  # Add shadowing
    return PL


def simulate_path_loss():
    """
    Simulates and plots path loss for free-space, log-distance, and log-normal shadowing models.
    """
    fc = 1.5e9  # Carrier frequency in Hz
    d0 = 100  # Reference distance in meters
    sigma = 3  # Shadowing standard deviation in dB
    distance = np.array([i**2 for i in range(1, 32, 2)])  # Quadratic distance scale
    Gt = [1, 1, 0.5]  # Transmitter gain values
    Gr = [1, 0.5, 0.5]  # Receiver gain values
    Exp = [2, 3, 6]  # Path loss exponents

    # Compute path loss for each model
    y_Free = np.array([free_space_path_loss(fc, distance, Gt[i], Gr[i]) for i in range(3)])
    y_logdist = np.array([log_distance_or_shadowing_path_loss(fc, distance, d0, Exp[i]) for i in range(3)])
    y_lognorm = np.array([log_distance_or_shadowing_path_loss(fc, distance, d0, Exp[0], sigma) for _ in range(3)])

    # Plot Free Space Path Loss
    plt.subplot(131)
    for i in range(3):
        plt.semilogx(distance, y_Free[i], label=f"Gt={Gt[i]}, Gr={Gr[i]}")
    plt.grid(True)
    plt.title(f"Free Space Path Loss, fc={fc/1e6:.1f} MHz")
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.legend()

    # Plot Log-Distance Path Loss
    plt.subplot(132)
    for i in range(3):
        plt.semilogx(distance, y_logdist[i], label=f"n={Exp[i]}")
    plt.grid(True)
    plt.title(f"Log-Distance Path Loss, fc={fc/1e6:.1f} MHz")
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.legend()

    # Plot Log-Normal Shadowing Path Loss
    plt.subplot(133)
    for i in range(3):
        plt.semilogx(distance, y_lognorm[i], label=f"path {i+1}")
    plt.grid(True)
    plt.title(f"Log-Normal Path Loss, fc={fc/1e6:.1f} MHz, σ={sigma} dB")
    plt.xlabel("Distance [m]")
    plt.ylabel("Path Loss [dB]")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_path_loss()
```

### **Small-scale Fading Simulation**
``` py
import numpy as np
import matplotlib.pyplot as plt


def rayleigh_channel(L):
    """
    Generates Rayleigh fading channel realizations.

    Args:
        L (int): Number of channel realizations.

    Returns:
        ndarray: Rayleigh fading channel vector of size (L,).
    """
    real_part = np.random.normal(0, 1, L)  # Real part of the channel
    imag_part = np.random.normal(0, 1, L)  # Imaginary part of the channel
    H = (real_part + 1j * imag_part) / np.sqrt(2)  # Normalize
    return H


def rician_channel(K_dB, L):
    """
    Generates Rician fading channel realizations.

    Args:
        K_dB (float): Rician K-factor in dB.
        L (int): Number of channel realizations.

    Returns:
        ndarray: Rician fading channel vector of size (L,).
    """
    K = 10 ** (K_dB / 10)  # Convert K-factor from dB to linear scale
    rayleigh = rayleigh_channel(L)  # Rayleigh component
    los_component = np.sqrt(K / (K + 1))  # Line-of-sight (LOS) component
    scattered_component = np.sqrt(1 / (K + 1)) * rayleigh  # Scattered component
    H = los_component + scattered_component
    return H


def plot_fading_channels():
    """
    Simulates and plots the amplitude distribution of Rayleigh and Rician fading channels.
    """
    N = 200000  # Number of channel realizations
    level = 30  # Number of histogram bins
    K_dB = [-40, 15]  # Rician K-factors in dB

    # Generate Rayleigh fading channel
    rayleigh_ch = rayleigh_channel(N)
    temp, x = np.histogram(np.abs(rayleigh_ch), bins=level, density=True)
    plt.plot(x[:-1], temp, 'k-s', label='Rayleigh')

    # Generate Rician fading channels
    for i, k in enumerate(K_dB):
        rician_ch = rician_channel(k, N)
        temp, x = np.histogram(np.abs(rician_ch), bins=level, density=True)
        plt.plot(x[:-1], temp, 'k-o' if i == 0 else 'k-^', label=f'Rician, K={k} dB')

    # Plot customization
    plt.xlabel('Amplitude')
    plt.ylabel('Occurrence')
    plt.legend()
    plt.grid(True)
    plt.title('Amplitude Distribution of Fading Channels')
    plt.show()


if __name__ == "__main__":
    plot_fading_channels()
```

---

## **1.5 Conclusion**
Wireless channel modeling is a critical step in understanding and designing robust communication systems. From basic free-space models to advanced fading and shadowing models, these tools enable accurate performance analysis and system optimization. Through simulation, engineers can gain insights into real-world channel behaviors and test algorithms under realistic conditions.

---

## **References**
- Cho, Yong Soo, Jaekwon Kim, Won Y. Yang, and Chung G. Kang. MIMO-OFDM wireless communications with MATLAB. John Wiley & Sons, 2010.
