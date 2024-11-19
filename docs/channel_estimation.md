# **2. Channel Estimation**

Accurate channel estimation is essential in wireless communication systems for efficient signal processing, especially in scenarios involving MIMO, OFDM, and beamforming techniques. It allows the transmitter or receiver to acquire knowledge of the channel state information (CSI), which is critical for equalization, interference mitigation, and adaptive modulation.

---

## **2.1 What is Channel Estimation?**

Channel estimation refers to the process of determining the channel characteristics between the transmitter and receiver. It involves estimating parameters such as:

  - Path gains and delays.
  - Phase and amplitude distortions.
  - Frequency-selective or time-varying fading.

### Importance of Channel Estimation
- **Equalization**: Removes channel-induced distortions.
- **Beamforming**: Requires accurate CSI to align transmitted signals.
- **Adaptive Modulation**: Enables the system to dynamically adjust modulation schemes based on channel conditions.

---

## **2.2 Channel Estimation Techniques**

Channel estimation methods can be broadly categorized based on the availability of pilot symbols and system assumptions:

### **2.2.1 Pilot-based Estimation**
- **Pilot Symbols**: Predefined symbols known to both transmitter and receiver are periodically inserted into the transmitted signal.
- **Advantages**:

    - Provides accurate CSI for deterministic scenarios.
    - Well-suited for time-invariant or slow-fading channels.

- **Methods**:

    1. **Least Squares (LS)**:

         - Solves for channel coefficients by minimizing the squared error between received and expected signals.
         - Formula:

            $$
              \boldsymbol{Y} = \boldsymbol{X}\boldsymbol{H} + \boldsymbol{Z}
            $$
      
            where \(Y\) is the received signal, \(X\) is the pilot symbol (diagonal matrix),
            
            $$
            \begin{aligned}
            J(\hat{\boldsymbol{H}}) & =\|\boldsymbol{Y}-\boldsymbol{X} \hat{\boldsymbol{H}}\|^2 \\
            & =(\boldsymbol{Y}-\boldsymbol{X} \hat{\boldsymbol{H}})^{\mathrm{H}}(\boldsymbol{Y}-\boldsymbol{X} \hat{\boldsymbol{H}}) \\
            & =\boldsymbol{Y}^{\mathrm{H}} \boldsymbol{Y}-\boldsymbol{Y}^{\mathrm{H}} \boldsymbol{X} \hat{\boldsymbol{H}}-\hat{\boldsymbol{H}}^{\mathrm{H}} \boldsymbol{X}^{\mathrm{H}} \boldsymbol{Y}+\hat{\boldsymbol{H}}^{\mathrm{H}} \boldsymbol{X}^{\mathrm{H}} \boldsymbol{X} \hat{\boldsymbol{H}}.
            \end{aligned}
            $$
      
            Let the partial derivative of $J(\hat{\boldsymbol{H}})$ with respect to $\hat{\boldsymbol{H}}$ be equal to 0,
           
            $$ \frac{\partial J(\hat{\boldsymbol{H}})}{\partial \hat{\boldsymbol{H}}}=-2\left(\boldsymbol{X}^{\mathrm{H}} \boldsymbol{Y}\right)^*+2\left(\boldsymbol{X}^{\mathrm{H}} \boldsymbol{X} \hat{\boldsymbol{H}}\right)^*=0.
            $$
      
            Then, we can get the solution of LS,
           
             $$\hat{\boldsymbol{H}}_{\mathrm{LS}}=\left(\boldsymbol{X}^{\mathrm{H}} \boldsymbol{X}\right)^{-1} \boldsymbol{X}^{\mathrm{H}} \boldsymbol{Y}=\boldsymbol{X}^{-1} \boldsymbol{Y},
             $$

             where $\hat{\boldsymbol{H}}$ denots the estimation of $\boldsymbol{H}$. Furthermore, we can calculate the mean square error (MSE) of LS method as follows:

              $$
                \begin{aligned}
                \mathrm{MSE}_{\mathrm{LS}} & =E\left\{\left(\boldsymbol{H}-\hat{\boldsymbol{H}}_{\mathrm{LS}}\right)^{\mathrm{H}}\left(\boldsymbol{H}-\boldsymbol{H}_{\mathrm{LS}}\right)\right\} \\
                & =E\left\{\left(\boldsymbol{H}-\boldsymbol{X}^{-1} \boldsymbol{Y}\right)^{\mathrm{H}}\left(\boldsymbol{H}-\boldsymbol{X}^{-1} \boldsymbol{Y}\right)\right\} \\
                & =E\left\{\left(\boldsymbol{X}^{-1} \boldsymbol{Z}\right)^{\mathrm{H}}\left(\boldsymbol{X}^{-1} \boldsymbol{Z}\right)\right\} \\
                & =E\left\{\boldsymbol{Z}^{\mathrm{H}}\left(\boldsymbol{X} \boldsymbol{X}^{\mathrm{H}}\right)^{-1} \boldsymbol{Z}\right\} \\
                & =\frac{\sigma_z^2}{\sigma_x^2},
                \end{aligned}
               $$

               where MSE is inversely proportional to the signal-to-noise (SNR) ratio, $\frac{\sigma_x^2}{\sigma_z^2}$. This implies that the lower SNR, the less effective the LS is, because it amplifies the noise.

    2. **Minimum Mean Square Error (MMSE)**:
       
         - Incorporates noise statistics and prior knowledge of the channel.
         - Formula:

           $$
           \hat{H}_{MMSE} = R_H (R_H + \sigma^2 I)^{-1} \hat{H}_{LS},
           $$

           where \(R_H\) is the channel covariance matrix and \(\sigma^2\) is noise variance.

- **LS vs. MMSE Comparison**

      1. **Noise Sensitivity**:
      
          - LS estimation is highly susceptible to noise, especially in low-SNR environments, as it does not account for noise statistics.
          - MMSE estimation leverages noise characteristics to suppress the impact of noise, providing superior accuracy in noisy conditions.
      
      2. **Use of Prior Information**:
      
          - LS estimation assumes no prior knowledge of the channel and performs a straightforward estimation.
          - MMSE estimation utilizes statistical properties of the channel (e.g., correlation, delay spread), resulting in a more robust estimation.
      
      3. **Complexity**:
      
          - LS estimation is computationally simple, requiring only a division operation, making it suitable for systems with stringent latency or processing constraints.
          - MMSE estimation involves matrix inversion and requires the covariance matrix, resulting in higher computational complexity.
      
      4. **Mean Square Error (MSE)**:
      
          - LS estimation minimizes instantaneous errors but does not guarantee the smallest MSE.
          - MMSE estimation explicitly minimizes the MSE, ensuring optimal performance in terms of estimation accuracy when channel statistics are available.

### **2.2.2 Blind Estimation**
- **Description**: Estimates the channel without relying on pilot symbols, using the statistical properties of received signals.
- **Advantages**:
    
    - Higher spectral efficiency due to no overhead for pilot symbols.

- **Challenges**:
    
    - Requires a large number of observations for accurate results.
    - Computationally complex.

### **2.2.3 Semi-blind Estimation**
- Combines pilot-based and blind estimation techniques.
- Uses pilot symbols for initial estimates and refines them using blind methods.

---

## **2.3 Channel Estimation in OFDM Systems**

In OFDM systems, channel estimation faces unique challenges due to:
1. **Frequency Selectivity**:
   - Multipath propagation causes varying gains across subcarriers.
2. **Pilot Subcarrier Allocation**:
   - Pilots are inserted at specific subcarriers to estimate the channel at those frequencies, followed by interpolation for other subcarriers.
3. **Interpolation Techniques**:
   - **Linear Interpolation**:
     - Straightforward but may result in poor accuracy for rapidly changing channels.
   - **Spline Interpolation**:
     - Provides smoother transitions for high-variability channels.
   - **DFT-based Interpolation**:
     - Exploits the sparsity of the channel in the delay domain.

---

## **2.4 Python Simulation**

The following Python code demonstrates LS and MMSE-based channel estimation in an OFDM system.

### **Code Implementation**
``` py
import numpy as np
from scipy.interpolate import interp1d

def interpolate(H_est, pilot_loc, Nfft, method):
    """
    Interpolates the channel estimate over all subcarriers.

    Args:
        H_est (ndarray): Channel estimate using pilot sequence.
        pilot_loc (ndarray): Location of pilot sequence.
        Nfft (int): FFT size.
        method (str): Interpolation method ('linear' or 'spline').

    Returns:
        ndarray: Interpolated channel estimate over all subcarriers.
    """
    # Extend at the beginning if necessary
    if pilot_loc[0] > 0:
        slope_start = (H_est[1] - H_est[0]) / (pilot_loc[1] - pilot_loc[0])
        extrapolated_start = H_est[0] - slope_start * (pilot_loc[0] - 0)
        H_est = np.insert(H_est, 0, extrapolated_start)
        pilot_loc = np.insert(pilot_loc, 0, 0)

    # Extend at the end if necessary
    if pilot_loc[-1] < Nfft - 1:
        slope_end = (H_est[-1] - H_est[-2]) / (pilot_loc[-1] - pilot_loc[-2])
        extrapolated_end = H_est[-1] + slope_end * (Nfft - 1 - pilot_loc[-1])
        H_est = np.append(H_est, extrapolated_end)
        pilot_loc = np.append(pilot_loc, Nfft - 1)

    # Interpolate
    if method.lower() == 'linear':
        interp_fn = interp1d(pilot_loc, H_est, kind='linear', fill_value="extrapolate")
    else:
        interp_fn = interp1d(pilot_loc, H_est, kind='cubic', fill_value="extrapolate")

    return interp_fn(np.arange(Nfft))


def ls_channel_estimation(Y, Xp, pilot_loc, Nfft, Nps, int_opt):
    """
    LS channel estimation function.

    Args:
        Y (ndarray): Frequency-domain received signal.
        Xp (ndarray): Pilot signal.
        pilot_loc (ndarray): Pilot locations.
        Nfft (int): FFT size.
        Nps (int): Pilot spacing.
        int_opt (str): Interpolation method ('linear' or 'spline').

    Returns:
        ndarray: LS channel estimate.
    """
    Np = Nfft // Nps  # Number of pilots
    LS_est = Y[pilot_loc] / Xp  # LS channel estimation
    H_LS = interpolate(LS_est, pilot_loc, Nfft, int_opt)  # Interpolation

    return H_LS


def mmse_channel_estimation(Y, Xp, pilot_loc, Nfft, Nps, h, SNR_dB):
    """
    MMSE channel estimation function.

    Args:
        Y (ndarray): Frequency-domain received signal.
        Xp (ndarray): Pilot signal.
        pilot_loc (ndarray): Pilot locations.
        Nfft (int): FFT size.
        Nps (int): Pilot spacing.
        h (ndarray): Channel impulse response.
        SNR_dB (float): Signal-to-Noise Ratio in dB.

    Returns:
        ndarray: MMSE channel estimate.
    """
    snr = 10 ** (SNR_dB / 10)  # Convert SNR to linear scale
    Np = Nfft // Nps  # Number of pilots
    H_tilde = Y[pilot_loc] / Xp  # LS estimate

    # Compute RMS delay spread from channel impulse response
    k = np.arange(len(h))
    hh = np.sum(h * np.conj(h))
    tmp = h * np.conj(h) * k
    r = np.sum(tmp) / hh
    r2 = np.sum(tmp * k) / hh
    tau_rms = np.sqrt(r2 - r**2)

    # Frequency-domain correlation
    df = 1 / Nfft  # Subcarrier spacing
    j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df

    # Correlation matrices
    K1 = np.tile(np.arange(Nfft).reshape(-1, 1), (1, Np))
    K2 = np.tile(np.arange(Np), (Nfft, 1))
    rf = 1 / (1 + j2pi_tau_df * (K1 - K2 * Nps))

    K3 = np.tile(np.arange(Np).reshape(-1, 1), (1, Np))
    K4 = np.tile(np.arange(Np), (Np, 1))
    rf2 = 1 / (1 + j2pi_tau_df * Nps * (K3 - K4))

    Rhp = rf
    Rpp = rf2 + np.eye(Np) / snr
    H_MMSE = np.matmul(Rhp, np.linalg.inv(Rpp)).dot(H_tilde)

    return H_MMSE
```

---

## **2.5 Concludsion**
Channel estimation is a cornerstone of modern wireless communication systems, enabling equalization, beamforming, and adaptive modulation. Techniques like LS and MMSE are widely used due to their simplicity and accuracy, while interpolation methods ensure reliable channel estimates across all subcarriers in OFDM systems.

---

## **References**
- Cho, Yong Soo, Jaekwon Kim, Won Y. Yang, and Chung G. Kang. MIMO-OFDM wireless communications with MATLAB. John Wiley & Sons, 2010.
