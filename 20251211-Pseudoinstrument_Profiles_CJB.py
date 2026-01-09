#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.interpolate import interp1d

# ============================================================================
# SUB-FUNCTIONS
# ============================================================================

def apply_noise(data, noise_param, noise_range=(0.0, 5.0)):
    """
    Apply Y-direction Gaussian noise based on signal RMS.
    
    Parameters:
        data (np.ndarray): Spectrum data, shape (N,) or (M, N)
        noise_param (float): Parameter in [0, 1] range (0 = no noise)
        noise_range (tuple): (min_percent, max_percent) for noise std as % of RMS
    
    Returns:
        np.ndarray: Noisy spectrum, same shape as input
    """
    if not (0 <= noise_param <= 1):
        raise ValueError("noise_param must be in [0, 1] range")
    
    # Linear map [0, 1] to noise range
    noise_percent = noise_range[0] + noise_param * (noise_range[1] - noise_range[0])
    
    if noise_percent == 0:
        return data
    
    if data.ndim == 1:
        # Single spectrum
        rms = np.sqrt(np.mean(data ** 2))
        noise_std = (noise_percent / 100.0) * rms
        noise = np.random.normal(0, noise_std, size=data.shape)
        return data + noise
    else:
        # Multiple spectra
        noisy_signal = np.zeros_like(data)
        for i in range(data.shape[0]):
            rms = np.sqrt(np.mean(data[i] ** 2))
            noise_std = (noise_percent / 100.0) * rms
            noise = np.random.normal(0, noise_std, size=data[i].shape)
            noisy_signal[i] = data[i] + noise
        return noisy_signal


def apply_x_perturbations(data, x_coords, x_params, 
                          offset_range=(-30, 30), 
                          dilation_range=(-1.5, 1.5)):
    """
    Apply X-axis offset and dilation, then interpolate back to original grid.
    
    Parameters:
        data (np.ndarray): Spectrum data, shape (N,) or (M, N)
        x_coords (np.ndarray): Original x-axis coordinates (wavenumbers), shape (N,)
        x_params (array-like): [x_offset_param, x_center_param, dilation_param] 
                               each in [-1, 1] (0 = inactive)
        offset_range (tuple): (min, max) offset in cm⁻¹, default (-30, 30)
        dilation_range (tuple): (min, max) dilation in percent, default (-1.5, 1.5)
    
    Returns:
        np.ndarray: X-perturbed spectrum, same shape as input
    """
    # Early exit if no perturbation
    if x_params[0] == 0 and x_params[2] == 0:
        return data

    original_shape = data.shape
    signal = data.reshape(-1, data.shape[-1])
    
    # Map offset: 0 → 0 offset
    x_offset = np.interp(x_params[0], [-1, 1], offset_range)
    
    # Map center: 0 → middle of x_coords
    x_center_norm = (x_params[1] + 1) / 2
    x_center = x_coords[0] + x_center_norm * (x_coords[-1] - x_coords[0])
    
    # Map dilation: 0 → 0% dilation
    dilation_pct = np.interp(x_params[2], [-1, 1], dilation_range)
    
    # Apply offset and dilation
    x_perturbed = x_offset + x_center + (x_coords - x_center) * (1 + dilation_pct / 100)
    
    # Interpolate back to original grid
    perturbed_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        f_interp = interp1d(
            x_perturbed, signal[i],
            kind='linear',
            bounds_error=False,
            fill_value=(signal[i, 0], signal[i, -1])
        )
        perturbed_signal[i] = f_interp(x_coords)
    
    return perturbed_signal.reshape(original_shape)


def add_cosine_baseline(data, x_coords, mag_param, period_param, phase_param,
                        mag_range=(0.0, 30),
                        period_range=(300, 1500),
                        phase_range=(0, 2*np.pi),
                        return_baseline=False):
    """
    Add cosine baseline component to spectrum.
    
    Parameters:
        data (np.ndarray): Spectrum data, shape (N,) or (M, N)
        x_coords (np.ndarray): X-axis coordinates, shape (N,)
        mag_param (float): Magnitude parameter in [0, 1] (0 = inactive)
        period_param (float): Period parameter in [-1, 1]
        phase_param (float): Phase parameter in [-1, 1]
        mag_range (tuple): (min, max) magnitude as % of mean signal
        period_range (tuple): (min, max) period in cm⁻¹
        phase_range (tuple): (min, max) phase in radians
        return_baseline (bool): If True, return (data + baseline, baseline)
    
    Returns:
        np.ndarray or tuple: Spectrum with baseline added, or (spectrum, baseline) if return_baseline=True
    """
    if not (0 <= mag_param <= 1):
        raise ValueError("mag_param must be in [0, 1] range")
    
    original_shape = data.shape
    signal = data.reshape(-1, data.shape[-1])
    
    # Map magnitude: [0, 1] → [0, max] so 0 = inactive
    mag_pct = mag_range[0] + mag_param * (mag_range[1] - mag_range[0])
    
    if mag_pct == 0:  # Early exit if no baseline
        if return_baseline:
            return data, np.zeros_like(data)
        return data
    
    # Map period and phase
    period = np.interp(period_param, [-1, 1], period_range)
    phase = np.interp(phase_param, [-1, 1], phase_range)
    
    # Calculate baseline for each spectrum
    result = np.zeros_like(signal)
    baseline_total = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        mean_signal = np.mean(np.abs(signal[i]))
        magnitude = (mag_pct / 100) * mean_signal
        cosine = magnitude * np.cos(2 * np.pi * x_coords / period + phase)
        baseline_total[i] = cosine
        result[i] = signal[i] + cosine
    
    if return_baseline:
        return result.reshape(original_shape), baseline_total.reshape(original_shape)
    return result.reshape(original_shape)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def apply_instrument_profile(spectra, x_coords, instrument_params,
                             noise_range=(0.0, 5.0),
                             offset_range=(-30, 30),
                             dilation_range=(-1.5, 1.5),
                             baseline_mag_range=(0.0, 30),
                             baseline_period_range=(300, 1500),
                             baseline_phase_range=(0, 2*np.pi),
                             return_baseline=False):
    """
    Apply complete instrument profile to spectrum/spectra.
    
    Parameters:
        spectra (np.ndarray): Input spectrum, shape (N,) or (M, N) for M spectra
        x_coords (np.ndarray): X-axis coordinates (wavenumbers), shape (N,)
        instrument_params (array-like): 13 parameters, ALL with 0 = inactive:
            [0]: noise [0, 1] - 0 = no noise
            [1]: x_offset [-1, 1] - 0 = no offset
            [2]: x_center [-1, 1] - 0 = middle of range
            [3]: dilation [-1, 1] - 0 = no dilation
            [4-6]: cos1 [0,1], [-1,1], [-1,1] - mag=0 = inactive
            [7-9]: cos2 [0,1], [-1,1], [-1,1] - mag=0 = inactive
            [10-12]: cos3 [0,1], [-1,1], [-1,1] - mag=0 = inactive
        
        Tuning ranges (optional):
            noise_range (tuple): (min, max) noise as % of RMS, default (0, 5)
            offset_range (tuple): (min, max) X-offset in cm⁻¹, default (-30, 30)
            dilation_range (tuple): (min, max) X-dilation in %, default (-1.5, 1.5)
            baseline_mag_range (tuple): (min, max) baseline magnitude as % of mean
            baseline_period_range (tuple): (min, max) baseline period in cm⁻¹
            baseline_phase_range (tuple): (min, max) baseline phase in radians
            return_baseline (bool): If True, return (perturbed, baseline_profile)
    
    Returns:
        np.ndarray or tuple: Perturbed spectrum/spectra, or (perturbed, baseline) if return_baseline=True
    
    Example - No perturbation:
        >>> params = np.zeros(13)  # All zeros = no effect
        >>> result = apply_instrument_profile(spectrum, x, params)
        >>> np.allclose(result, spectrum)  # Should be True
    
    Example - Only offset:
        >>> params = np.zeros(13)
        >>> params[1] = 0.5  # +15 cm⁻¹ offset
        >>> result = apply_instrument_profile(spectrum, x, params)
    """
    instrument_params = np.asarray(instrument_params)
    if len(instrument_params) != 13:
        raise ValueError("instrument_params must have exactly 13 elements")
    
    # Validate parameter ranges
    if not (0 <= instrument_params[0] <= 1):
        raise ValueError("noise_param (index 0) must be in [0, 1]")
    if not all(0 <= instrument_params[i] <= 1 for i in [4, 7, 10]):
        raise ValueError("magnitude params (indices 4, 7, 10) must be in [0, 1]")
    if not all(-1 <= instrument_params[i] <= 1 for i in [1, 2, 3, 5, 6, 8, 9, 11, 12]):
        raise ValueError("offset/dilation/period/phase params must be in [-1, 1]")
    
    # Ensure 2D for processing
    original_shape = spectra.shape
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)
    
    # Apply transformations in order
    result = spectra.copy()
    
    # 1. Noise
    result = apply_noise(result, instrument_params[0], noise_range)
    
    # 2. X-axis perturbations
    result = apply_x_perturbations(
        result, x_coords, instrument_params[1:4],
        offset_range, dilation_range
    )
    
    # 3. Three cosine baseline components
    if return_baseline:
        baseline_total = np.zeros_like(result)
        
        result, baseline1 = add_cosine_baseline(
            result, x_coords, instrument_params[4], instrument_params[5], instrument_params[6],
            baseline_mag_range, baseline_period_range, baseline_phase_range,
            return_baseline=True
        )
        baseline_total += baseline1
        
        result, baseline2 = add_cosine_baseline(
            result, x_coords, instrument_params[7], instrument_params[8], instrument_params[9],
            baseline_mag_range, baseline_period_range, baseline_phase_range,
            return_baseline=True
        )
        baseline_total += baseline2
        
        result, baseline3 = add_cosine_baseline(
            result, x_coords, instrument_params[10], instrument_params[11], instrument_params[12],
            baseline_mag_range, baseline_period_range, baseline_phase_range,
            return_baseline=True
        )
        baseline_total += baseline3
        
        return result.reshape(original_shape), baseline_total.reshape(original_shape)
    else:
        result = add_cosine_baseline(
            result, x_coords, instrument_params[4], instrument_params[5], instrument_params[6],
            baseline_mag_range, baseline_period_range, baseline_phase_range
        )
        result = add_cosine_baseline(
            result, x_coords, instrument_params[7], instrument_params[8], instrument_params[9],
            baseline_mag_range, baseline_period_range, baseline_phase_range
        )
        result = add_cosine_baseline(
            result, x_coords, instrument_params[10], instrument_params[11], instrument_params[12],
            baseline_mag_range, baseline_period_range, baseline_phase_range
        )
        
        return result.reshape(original_shape)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic spectrum
    N = 884
    x_raman = np.linspace(400, 1800, N)
    
    true_spectrum = (
        1000 * np.exp(-((x_raman - 600) / 20)**2) +
        800 * np.exp(-((x_raman - 900) / 25)**2) +
        600 * np.exp(-((x_raman - 1200) / 30)**2) +
        400 * np.exp(-((x_raman - 1500) / 35)**2) +
        100
    )
    
    # Test 1: Zero vector = no change
    zero_params = np.zeros(13)
    no_change = apply_instrument_profile(true_spectrum, x_raman, zero_params)
    print(f"Zero params identical: {np.allclose(no_change, true_spectrum)}")
    
    # Test 2: Only noise
    noise_only = np.zeros(13)
    noise_only[0] = 0.5  # 2.5% noise
    noisy = apply_instrument_profile(true_spectrum, x_raman, noise_only)
    
    # Test 3: Full perturbation with baseline profile
    full_params = np.array([
        0.4,    # noise: 2% RMS
        1,      # x_offset: +30 cm⁻¹
        0.0,    # x_center: middle
        1,      # dilation: +1.5%
        1, 1, 1,   # cos1: 4% mag
        0.3, -0.5, 0.8,  # cos2: 2.4% mag
        0.2, 0.7, -0.4   # cos3: 1.6% mag
    ])
    perturbed, baseline = apply_instrument_profile(true_spectrum, x_raman, full_params, return_baseline=True)
    
    # Visualize
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(x_raman, true_spectrum, 'k-', linewidth=2, label='Original')
    axes[0,0].plot(x_raman, no_change, 'r--', alpha=0.7, label='Zero params')
    axes[0,0].set_title('Test: Zero params = No change')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(x_raman, true_spectrum, 'k-', linewidth=2, label='Original')
    axes[0,1].plot(x_raman, noisy, alpha=0.7, label='Noise only')
    axes[0,1].set_title('Test: Noise only (2.5% RMS)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(x_raman, true_spectrum, 'k-', linewidth=2, label='Original')
    axes[1,0].plot(x_raman, perturbed, alpha=0.7, label='Full perturbation')
    axes[1,0].plot(x_raman, baseline, 'g--', alpha=0.7, label='Baseline profile')
    axes[1,0].set_title('Test: Full instrument profile')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(x_raman, perturbed - true_spectrum, 'g-', label='Difference')
    axes[1,1].plot(x_raman, baseline, 'r--', alpha=0.7, label='Baseline only')
    axes[1,1].set_title('Difference vs Baseline')
    axes[1,1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Intensity')
    
    plt.tight_layout()
    plt.show()


# In[4]:


# ============================================================================
# CELL 0: PLOTTING FUNCTION
# ============================================================================

def plot_sweep(x_raman, true_spectrum, params_list, param_name, param_values, 
               figsize_width=9.8, figsize_height_per_plot=1.75):
    """
    Plot a parameter sweep with vertically stacked short wide plots.
    
    Parameters:
        x_raman: X-axis coordinates
        true_spectrum: Original spectrum
        params_list: List of 13-element parameter arrays
        param_name: Name for plot titles (e.g., "Noise", "X-Offset")
        param_values: List of parameter values for labeling (e.g., ["0%", "1%", "2%"])
        figsize_width: Width of entire figure (default 9.8, ~30% smaller than 14)
        figsize_height_per_plot: Height of each subplot (default 1.75, ~30% smaller than 2.5)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n = len(params_list)
    
    fig, axes = plt.subplots(n, 1, figsize=(figsize_width, figsize_height_per_plot * n))
    if n == 1:
        axes = [axes]
    
    for idx, (params, value) in enumerate(zip(params_list, param_values)):
        perturbed, baseline = apply_instrument_profile(true_spectrum, x_raman, params, return_baseline=True)
        
        # Plot perturbed and baseline first
        axes[idx].plot(x_raman, perturbed, 'r-', linewidth=1.5, label='Perturbed')
        if np.any(baseline != 0):
            axes[idx].plot(x_raman, baseline, 'g--', linewidth=1, alpha=0.7, label='Baseline')
        
        # Plot original on top (dotted)
        axes[idx].plot(x_raman, true_spectrum, 'k:', linewidth=2, alpha=0.6, label='Original')
        
        axes[idx].set_title(f'{param_name}: {value}', loc='left', fontweight='bold')
        axes[idx].set_xlabel('Raman Shift (cm⁻¹)')
        axes[idx].set_ylabel('Intensity')
        axes[idx].legend(loc='upper right', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# In[5]:


# ============================================================================
# SWEEP 1: NOISE
# ============================================================================

noise_levels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
params_list = []

for noise in noise_levels:
    params = np.zeros(13)
    params[0] = noise
    params_list.append(params)

param_values = [f'{n*5:.1f}% RMS' for n in noise_levels]
plot_sweep(x_raman, true_spectrum, params_list, 'Noise', param_values)


# In[6]:


# ============================================================================
# SWEEP 2: X-OFFSET
# ============================================================================

offsets = [-1, -0.6, -0.2, 0.2, 0.6, 1.0]
params_list = []

for offset in offsets:
    params = np.zeros(13)
    params[1] = offset
    params_list.append(params)

param_values = [f'{o*30:+.0f} cm⁻¹' for o in offsets]
plot_sweep(x_raman, true_spectrum, params_list, 'X-Offset', param_values)


# In[7]:


# ============================================================================
# SWEEP 3: X-DILATION (left-centered)
# ============================================================================

dilations = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
params_list = []

for dilation in dilations:
    params = np.zeros(13)
    params[2] = -1  # Left edge as center
    params[3] = dilation
    params_list.append(params)

param_values = [f'{d*1.5:+.2f}%' for d in dilations]
plot_sweep(x_raman, true_spectrum, params_list, 'Dilation (left-centered)', param_values)


# In[8]:


# ============================================================================
# SWEEP 4: X-DILATION CENTER (max dilation, sweep center)
# ============================================================================

centers = [-1, -0.6, -0.2, 0.2, 0.6, 1.0]
params_list = []

for center in centers:
    params = np.zeros(13)
    params[2] = center  # Dilation center
    params[3] = 1.0     # Max dilation (1.5%)
    params_list.append(params)

# Calculate actual center positions
center_values = []
for c in centers:
    x_center_norm = (c + 1) / 2
    x_center_actual = 400 + x_center_norm * (1800 - 400)
    center_values.append(f'{x_center_actual:.0f} cm⁻¹')

plot_sweep(x_raman, true_spectrum, params_list, 'Dilation Center (1.5%)', center_values)


# In[9]:


# ============================================================================
# SWEEP 5: BASELINE MAGNITUDE (single cosine)
# ============================================================================

magnitudes = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
params_list = []

for mag in magnitudes:
    params = np.zeros(13)
    params[4] = mag     # Magnitude
    params[5] = 0.0     # Period (middle of range: 900 cm⁻¹)
    params[6] = 0.0     # Phase (middle of range: π),
    params_list.append(params)

param_values = [f'{m*30:.1f}%' for m in magnitudes]
plot_sweep(x_raman, true_spectrum, params_list, 'Baseline Magnitude', param_values)


# In[10]:


# ============================================================================
# SWEEP 6: BASELINE PERIOD (single cosine, fixed magnitude)
# ============================================================================

periods = [-1, -0.6, -0.2, 0.2, 0.6, 1.0]
params_list = []

for period in periods:
    params = np.zeros(13)
    params[4] = 0.5     # Fixed magnitude (15%)
    params[5] = period  # Period
    params[6] = 0.0     # Phase
    params_list.append(params)

param_values = [f'{300 + (p+1)/2 * 1200:.0f} cm⁻¹' for p in periods]
plot_sweep(x_raman, true_spectrum, params_list, 'Baseline Period (15% mag)', param_values)


# In[11]:


# ============================================================================
# SWEEP 7: BASELINE PHASE (single cosine, fixed magnitude & period)
# ============================================================================

phases = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
params_list = []

for phase in phases:
    params = np.zeros(13)
    params[4] = 0.5     # Fixed magnitude (15%)
    params[5] = 0.0     # Fixed period (900 cm⁻¹)
    params[6] = phase*2 - 1  # Map [0,1] to [-1,1]
    params_list.append(params)

param_values = [f'{(p*2-1+1)/2 * 2 * np.pi:.2f} rad' for p in phases]
plot_sweep(x_raman, true_spectrum, params_list, 'Baseline Phase (15% mag)', param_values)


# In[17]:


# ============================================================================
# SWEEP 8: RANDOM BASELINES (all three cosines active)
# ============================================================================

np.random.seed(42)
n_samples = 6
params_list = []

for i in range(n_samples):
    params = np.zeros(13)
    # Random baseline parameters for all 3 cosines
    params[4:13] = np.random.uniform(-1, 1, 9)
    params[4] = np.random.uniform(0, 1)   # Magnitude 1 [0,1]
    params[7] = np.random.uniform(0, 1)   # Magnitude 2 [0,1]
    params[10] = np.random.uniform(0, 1)  # Magnitude 3 [0,1]
    params_list.append(params)

param_values = [f'Random #{i+1}' for i in range(n_samples)]
plot_sweep(x_raman, true_spectrum, params_list, 'Random Baseline', param_values)


# In[12]:


# ============================================================================
# TEST: BATCH PROCESSING - SAME PROFILE ON MULTIPLE SPECTRA
# ============================================================================

# Generate 4 different synthetic Raman spectra
N = 884
x_raman = np.linspace(400, 1800, N)

spectra_batch = np.zeros((4, N))

# Spectrum 1: Original from example
spectra_batch[0] = (
    1000 * np.exp(-((x_raman - 600) / 20)**2) +
    800 * np.exp(-((x_raman - 900) / 25)**2) +
    600 * np.exp(-((x_raman - 1200) / 30)**2) +
    400 * np.exp(-((x_raman - 1500) / 35)**2) +
    100
)

# Spectrum 2: Different peak positions
spectra_batch[1] = (
    900 * np.exp(-((x_raman - 550) / 25)**2) +
    700 * np.exp(-((x_raman - 850) / 30)**2) +
    500 * np.exp(-((x_raman - 1300) / 35)**2) +
    300 * np.exp(-((x_raman - 1600) / 40)**2) +
    120
)

# Spectrum 3: Different intensities
spectra_batch[2] = (
    1200 * np.exp(-((x_raman - 650) / 22)**2) +
    600 * np.exp(-((x_raman - 950) / 28)**2) +
    800 * np.exp(-((x_raman - 1250) / 32)**2) +
    350 * np.exp(-((x_raman - 1450) / 38)**2) +
    90
)

# Spectrum 4: Broader peaks
spectra_batch[3] = (
    800 * np.exp(-((x_raman - 700) / 40)**2) +
    1000 * np.exp(-((x_raman - 1000) / 45)**2) +
    700 * np.exp(-((x_raman - 1400) / 50)**2) +
    110
)

# Define a moderately complex instrument profile
instrument_profile = np.array([
    0.3,    # 1.5% noise
    0.5,    # +15 cm⁻¹ offset
    0.0,    # center at middle
    0.4,    # +0.6% dilation
    0.4, 0.2, 0.3,   # cos1: 12% mag, ~780 cm⁻¹ period
    0.3, -0.4, 0.7,  # cos2: 9% mag, ~660 cm⁻¹ period
    0.2, 0.6, -0.5   # cos3: 6% mag, ~1140 cm⁻¹ period
])

# Apply same profile to all spectra
perturbed_batch, baseline_batch = apply_instrument_profile(
    spectra_batch, x_raman, instrument_profile, return_baseline=True
)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

for i in range(4):
    axes[i].plot(x_raman, spectra_batch[i], 'k:', linewidth=2, alpha=0.6, label='Original')
    axes[i].plot(x_raman, perturbed_batch[i], 'r-', linewidth=1.5, label='Perturbed')
    axes[i].plot(x_raman, baseline_batch[i], 'g--', linewidth=1, alpha=0.7, label='Baseline')
    
    axes[i].set_title(f'Spectrum {i+1}: Same Instrument Profile Applied', loc='left', fontweight='bold')
    axes[i].set_xlabel('Raman Shift (cm⁻¹)')
    axes[i].set_ylabel('Intensity')
    axes[i].legend(loc='upper right', fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




