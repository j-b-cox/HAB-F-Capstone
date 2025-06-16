import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolate_rrs(rrs_vals, rrs_wavelengths, target_wavelengths, method='linear'):
    """
    Interpolates Rrs values to target wavelengths.
    """
    mask = np.isfinite(rrs_vals)
    if np.count_nonzero(mask) < 2:
        return np.full(len(target_wavelengths), np.nan, dtype=np.float32)

    interp_func = interp1d(
        rrs_wavelengths[mask], rrs_vals[mask],
        kind=method, bounds_error=False, fill_value=np.nan
    )
    return interp_func(target_wavelengths)

def check_wavelength_coverage(rrs_wavelengths, target_wavelengths, max_gap=40):
    """
    Check if input wavelengths reasonably cover the target wavelengths.
    """
    rrs_min, rrs_max = min(rrs_wavelengths), max(rrs_wavelengths)
    target_min, target_max = min(target_wavelengths), max(target_wavelengths)
    return (rrs_min <= target_min and rrs_max >= target_max and
            np.max(np.diff(sorted(rrs_wavelengths))) <= max_gap)

def plot_interpolation(rrs_vals, rrs_wavelengths, target_wavelengths, interp_vals, title="Interpolation Check"):
    """
    Plot original and interpolated spectral Rrs.
    """
    plt.figure()
    plt.plot(rrs_wavelengths, rrs_vals, 'o-', label='Original')
    plt.plot(target_wavelengths, interp_vals, 'x--', label='Interpolated')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Rrs")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()
