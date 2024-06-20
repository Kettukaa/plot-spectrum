# Visualize Spectrum
I wanted a tool similar to Audacity's `Analyze>Plot Spectrum` option that I could automate better and create nicer looking graphs.


## Example Use
```py
file_pairs = [
    (
        Path("data/original.wav"),
        Path("data/change_1.wav"),
        Path("data/change_2.wav"),
    )
]
output_dir = Path("data/output_images/")
legend_labels = ["Original", "Change 1", "Change 2"]
major_ticks = [100, 1000, 10000]
nfft = 2**11
fig_size = (11.69 * 1.5, 8.27)
generate_spectrums(
    file_pairs,
    output_dir=output_dir,
    legend_labels=legend_labels,
    major_ticks=major_ticks,
    window="hann",
    nfft=nfft,
    fig_size=fig_size,
)
```