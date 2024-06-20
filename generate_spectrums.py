import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from catppuccin import PALETTE
from catppuccin.models import Flavor

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def create_spectrum(
    audio_file: Path,
    window: str = "hann",
    nfft: int = 1024,
    noverlap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a frequency spectrum from an audio file.

    Parameters:
    audio_file (Path): Path to the audio file.
    window (str): Type of window to use for the FFT.
    nfft (int): Number of points in the FFT.
    noverlap (Optional[int]): Number of points to overlap between segments.

    Returns:
    pd.DataFrame: DataFrame containing frequency and power level in dB.
    """
    sample_rate, data = wav.read(audio_file)
    if len(data.shape) > 1:
        data = data[:, 0]  # Use the first channel if stereo
    data = data / np.max(np.abs(data))  # Normalize data
    f, power = signal.welch(
        data,
        fs=sample_rate,
        window=window,
        nperseg=nfft,
        noverlap=noverlap,
        scaling="spectrum",
    )
    power_dB = 10 * np.log10(power + 1e-10)  # Convert power to dB
    return pd.DataFrame({"Frequency (Hz)": f, "Level (dB)": power_dB})


def get_style(theme: Flavor) -> Dict[str, Any]:
    """
    Get the style dictionary for plotting based on the given theme.

    Parameters:
    theme (Flavor): Theme object containing color information.

    Returns:
    Dict[str, Any]: Dictionary of style parameters for plotting.
    """
    return {
        "axes.facecolor": theme.colors.base.hex,
        "grid.color": theme.colors.overlay1.hex,
        "axes.grid": True,
        "axes.edgecolor": theme.colors.mantle.hex,
        "text.color": theme.colors.text.hex,
        "xtick.color": theme.colors.subtext0.hex,
        "ytick.color": theme.colors.subtext0.hex,
        "axes.labelcolor": theme.colors.subtext1.hex,
        "figure.facecolor": theme.colors.mantle.hex,
    }


def convert_ticks_to_labels(ticks: List[int]) -> List[str]:
    """
    Convert a list of tick values to human-readable labels.

    Parameters:
    ticks (List[int]): List of tick values.

    Returns:
    List[str]: List of formatted tick labels.
    """
    labels = []
    for tick in ticks:
        if tick >= 1000:
            labels.append(f"{tick // 1000} kHz")
        else:
            labels.append(f"{tick} Hz")
    return labels


def generate_spectrums(
    file_pairs: List[Tuple[Path, ...]],
    output_dir: Path = Path("output_files"),
    theme: Flavor = PALETTE.latte,
    window: str = "hann",
    nfft: int = 1024,
    fig_size: Tuple[float, float] = (11.69, 8.27),
    major_ticks: Optional[List[int]] = None,
    legend_labels: Optional[List[str]] = None,
) -> None:
    """
    Generate and save frequency spectrums for given audio file pairs.

    Parameters:
    file_pairs (List[Tuple[Path, ...]]): List of tuples containing paths to audio files.
    output_dir (Path): Directory to save the output images.
    theme (Flavor): Theme for the plots.
    window (str): Type of window to use for the FFT.
    nfft (int): Number of points in the FFT.
    fig_size (Tuple[float, float]): Size of the figure.
    major_ticks (Optional[List[int]]): Major ticks for the x-axis.
    legend_labels (Optional[List[str]]): Labels for the legend.
    """
    ctp_styles = get_style(theme)
    sns.set_theme(context="notebook", style="darkgrid", rc=ctp_styles)
    colors = [
        theme.colors.blue.hex,
        theme.colors.maroon.hex,
        theme.colors.green.hex,
        theme.colors.lavender.hex,
        theme.colors.flamingo.hex,
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_tuple in file_pairs:
        plt.figure(figsize=fig_size)

        for i, audio_file in enumerate(file_tuple):
            df = create_spectrum(audio_file, window=window, nfft=nfft)
            label = (
                legend_labels[i]
                if legend_labels and i < len(legend_labels)
                else f"File {i+1}"
            )
            linestyle = "-" if i == 0 else "--"
            linewidth = 1.2 if i == 0 else 0.9
            color = colors[i % len(colors)]

            sns.lineplot(
                x="Frequency (Hz)",
                y="Level (dB)",
                data=df,
                color=color,
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
            )

        plt.ylim(-120, -10)
        plt.xscale("log")
        plt.xlim(47, 24000)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dB]")
        plt.title(f"{Path(file_tuple[0]).stem} Frequency Spectrum")
        plt.legend()

        plt.grid(True, which="both", axis="both", color=theme.colors.overlay1.hex)
        plt.minorticks_on()
        plt.grid(
            which="minor",
            linestyle=":",
            linewidth="0.5",
            color=theme.colors.overlay1.hex,
        )
        plt.grid(
            which="major",
            linestyle="-",
            linewidth="0.5",
            color=theme.colors.overlay1.hex,
        )

        if major_ticks:
            major_labels = convert_ticks_to_labels(major_ticks)
            plt.xticks(major_ticks, major_labels)

        output_path = output_dir / f"{Path(file_tuple[0]).stem}_combined.png"
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=600)
        plt.close()


# Example usage
if __name__ == "__main__":
    file_pairs = [
        (
            Path("data/original.wav"),
            Path("data/change_1.wav"),
            Path("data/change_2.wav"),
        )
    ]
    legend_labels = ["Original", "Change 1", "Change 2"]
    major_ticks = [100, 1000, 10000]
    generate_spectrums(
        file_pairs,
        output_dir=Path("data/output_images/"),
        legend_labels=legend_labels,
        major_ticks=major_ticks,
        window="hann",
        nfft=2**11,
        fig_size=(11.69 * 1.5, 8.27),
    )
