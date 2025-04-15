# M3N-VC: Multi-Modality Multi-Node Vehicle Classification Dataset

The M3N-VC dataset (Multi-Modality Multi-Node Vehicle Classification) is a
research dataset designed for, but not limited to, distributed sensing,
multi-modal signal processing, IoT foundation model pre-training and
fine-tuning, and test-time adaptation using data from spatially distributed
sensor networks.

It contains synchronized microphone and geophone recordings of four different
vehicles, captured in multiple real-world scenes using spatially distributed
sensor nodes.

## Use M3N-VC

Download M3N-VC here: <https://doi.org/10.5281/zenodo.15215210>

If you use M3N-VC dataset in your work, please cite

``` Bibtex
@inproceedings{li2025restoreml,
  author    = {Jinyang Li and Yizhuo Chen and Ruijie Wang and Tomoyoshi Kimura and Tianshi Wang and You Lyu and Hongjue Zhao and Binqi Sun and Shangchen Wu and Yigong Hu and Denizhan Kara and Beitong Tian and Klara Nahrstedt and Suhas Diggavi and Jae H. Kim and Greg Kimberly and Guijun Wang and Maggie Wigness and Tarek Abdelzaher},
  title     = {{RestoreML}: Practical Unsupervised Tuning of Deployed Intelligent IoT Systems},
  booktitle = {2025 The 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)},
  year      = {2025},
  publisher = {IEEE}
}
```

## Dataset Overview

- Modalities
  - Microphone (1.6 kHz)
  - Geophone (200 Hz)
  - GPS (1 Hz)
- Scenes: 6 unique environments
- Targets
  - Mazda CX-30 (C)
  - Mercedes-Benz GLE 350 (G)
  - Ford Mustang (M)
  - Mazda MX-5 (X)
- Nodes per Scene: 6–8 nodes equipped with both sensors
- Synchronization: All nodes are GPS time-synchronized
- Labels: Ground truth vehicle type and timestamp annotations

| ID  | Terrain          | Weather | Targets    | # Nodes | Length | Scenario                        |
|-----|------------------|---------|------------|---------|--------|---------------------------------|
| h08 | Asphalt & gravel | Sunny   | C, G, M, X | 6       | 2.77 h | single target; multi-view       |
| h24 | Asphalt & gravel | Rainy   | C, G, M, X | 6       | 3.43 h | single target; multi-view       |
| s31 | Dirt & gravel    | Sunny   | C, G, M, X | 6       | 2.78 h | single-target; multi-vantage    |
| a06 | Asphalt          | Sunny   | C, G, X    | 6       | 2.14 h | single-target; multi-vantage    |
| i29 | Concrete         | Windy   | C, G, M, X | 8       | 4.14 h | single-target; multi-vantage    |
| i22 | Concrete         | Sunny   | C, M, X    | 8       | 3.00 h | multi-target (2); multi-vantage |

Structure

``` text
M3N-VC/
├── h08.zip
├── h24.zip
├── s31.zip
├── a06.zip
├── i29.zip
└── i22.zip
```

Each zip file contains time-synchronized sensor data and metadata in `.parquet` format:

- `run_ids.parquet` -- Metadata for each vehicle run (timestamps, labels, etc.)
- `sensor_location.parquet` -- GPS coordinates of all sensor nodes
- `run<n>_rs<m>_mic.parquet` - Microphone time series data from node `m` during run `n`
- `run<n>_rs<m>_geo.parquet` -- Geophone time series data from node `m` during run `n`
- `run<n>_rs<m>_dis.parquet` -- Distance readings from node `m` during run `n`
- `run<n>_gps.parquet` -- GPS trajectory of the target vehicle during run `n`

## Data Sample

Use any library that can read `parquet` file. Here we use `polars` as an example.

``` python-console
>>> import polars as pl
```

Run metadata example:

``` python-console
>>> pl.read_parquet('s31/run_ids.parquet')
shape: (8, 6)
┌────────┬─────────┬───────┬─────────────────────────┬─────────────────────────┬──────────────┐
│ run_id ┆ label   ┆ set   ┆ start_time              ┆ end_time                ┆ length       │
│ ---    ┆ ---     ┆ ---   ┆ ---                     ┆ ---                     ┆ ---          │
│ i64    ┆ str     ┆ str   ┆ datetime[μs, Etc/UTC]   ┆ datetime[μs, Etc/UTC]   ┆ duration[μs] │
╞════════╪═════════╪═══════╪═════════════════════════╪═════════════════════════╪══════════════╡
│ 0      ┆ gle350  ┆ train ┆ 2023-12-31 03:33:20 UTC ┆ 2023-12-31 04:04:55 UTC ┆ 31m 35s      │
│ 1      ┆ gle350  ┆ test  ┆ 2023-12-31 04:05:19 UTC ┆ 2023-12-31 04:17:33 UTC ┆ 12m 14s      │
│ 2      ┆ mustang ┆ train ┆ 2023-12-31 04:26:17 UTC ┆ 2023-12-31 04:56:31 UTC ┆ 30m 14s      │
│ 3      ┆ mustang ┆ test  ┆ 2023-12-31 04:56:51 UTC ┆ 2023-12-31 05:07:25 UTC ┆ 10m 34s      │
│ 4      ┆ miata   ┆ train ┆ 2023-12-31 05:18:06 UTC ┆ 2023-12-31 05:48:44 UTC ┆ 30m 38s      │
│ 5      ┆ miata   ┆ test  ┆ 2023-12-31 05:49:54 UTC ┆ 2023-12-31 06:00:23 UTC ┆ 10m 29s      │
│ 6      ┆ cx30    ┆ train ┆ 2023-12-31 06:04:46 UTC ┆ 2023-12-31 06:35:32 UTC ┆ 30m 46s      │
│ 7      ┆ cx30    ┆ test  ┆ 2023-12-31 06:36:02 UTC ┆ 2023-12-31 06:46:20 UTC ┆ 10m 18s      │
└────────┴─────────┴───────┴─────────────────────────┴─────────────────────────┴──────────────┘
```

Sensor location example:

``` python-console
>>> pl.read_parquet('s31/sensor_location.parquet')
shape: (6, 4)
┌───────────┬───────────┬────────────┬──────────────┐
│ sensor_id ┆ latitude  ┆ longitude  ┆ terrain      │
│ ---       ┆ ---       ┆ ---        ┆ ---          │
│ str       ┆ f64       ┆ f64        ┆ str          │
╞═══════════╪═══════════╪════════════╪══════════════╡
│ rs1       ┆ 40.096379 ┆ -88.224714 ┆ asphalt_road │
│ rs2       ┆ 40.096373 ┆ -88.225168 ┆ road_curb    │
│ rs5       ┆ 40.096089 ┆ -88.224689 ┆ asphalt_road │
│ rs6       ┆ 40.096109 ┆ -88.22536  ┆ asphalt_road │
│ rs7       ┆ 40.096668 ┆ -88.224669 ┆ asphalt_road │
│ rs8       ┆ 40.096638 ┆ -88.225354 ┆ asphalt_road │
└───────────┴───────────┴────────────┴──────────────┘
```

Geophone data example:

``` python-console
>>> pl.read_parquet('s31/run0_rs8_geo.parquet')
shape: (379_000, 3)
┌───────────┬─────────┬─────────┐
│ timestamp ┆ channel ┆ samples │
│ ---       ┆ ---     ┆ ---     │
│ f64       ┆ str     ┆ i64     │
╞═══════════╪═════════╪═════════╡
│ 1.7040e9  ┆ SH3     ┆ 16667   │
│ 1.7040e9  ┆ SH3     ┆ 16694   │
│ 1.7040e9  ┆ SH3     ┆ 16695   │
│ 1.7040e9  ┆ SH3     ┆ 16705   │
│ 1.7040e9  ┆ SH3     ┆ 16723   │
│ …        ┆ …      ┆ …      │
│ 1.7040e9  ┆ SH3     ┆ 14452   │
│ 1.7040e9  ┆ SH3     ┆ 15202   │
│ 1.7040e9  ┆ SH3     ┆ 16092   │
│ 1.7040e9  ┆ SH3     ┆ 16998   │
│ 1.7040e9  ┆ SH3     ┆ 17672   │
└───────────┴─────────┴─────────┘
```

Microphone data example:

``` python-console
>>> pl.read_parquet('s31/run0_rs8_mic.parquet')
shape: (3_032_000, 2)
┌───────────┬─────────┐
│ timestamp ┆ samples │
│ ---       ┆ ---     │
│ f64       ┆ i64     │
╞═══════════╪═════════╡
│ 1.7040e9  ┆ -6      │
│ 1.7040e9  ┆ -39     │
│ 1.7040e9  ┆ -34     │
│ 1.7040e9  ┆ -16     │
│ 1.7040e9  ┆ 26      │
│ …        ┆ …      │
│ 1.7040e9  ┆ 51      │
│ 1.7040e9  ┆ -24     │
│ 1.7040e9  ┆ 9       │
│ 1.7040e9  ┆ 11      │
│ 1.7040e9  ┆ -75     │
└───────────┴─────────┘
```

Distance between target and node example:

``` python-console
>>> pl.read_parquet('s31/run0_rs8_dis.parquet')
shape: (1_869, 2)
┌─────────────────────┬────────────┐
│ time                ┆ distance   │
│ ---                 ┆ ---        │
│ datetime[μs]        ┆ f64        │
╞═════════════════════╪════════════╡
│ 2023-12-31 03:33:20 ┆ 124.024754 │
│ 2023-12-31 03:33:21 ┆ 123.023777 │
│ 2023-12-31 03:33:22 ┆ 121.911582 │
│ 2023-12-31 03:33:23 ┆ 120.798464 │
│ 2023-12-31 03:33:24 ┆ 119.575043 │
│ …                  ┆ …         │
│ 2023-12-31 04:04:51 ┆ 28.04795   │
│ 2023-12-31 04:04:52 ┆ 37.486873  │
│ 2023-12-31 04:04:53 ┆ 45.047892  │
│ 2023-12-31 04:04:54 ┆ 51.400186  │
│ 2023-12-31 04:04:55 ┆ 57.004715  │
└─────────────────────┴────────────┘
```

Target GPS example

``` python-console
>>> pl.read_parquet('s31/run0_gps.parquet')
shape: (1_869, 4)
┌─────────────────────┬───────────┬────────────┬───────────┐
│ time                ┆ latitude  ┆ longitude  ┆ elevation │
│ ---                 ┆ ---       ┆ ---        ┆ ---       │
│ datetime[μs]        ┆ f64       ┆ f64        ┆ f64       │
╞═════════════════════╪═══════════╪════════════╪═══════════╡
│ 2023-12-31 03:33:20 ┆ 40.078171 ┆ -88.21878  ┆ 216.1     │
│ 2023-12-31 03:33:21 ┆ 40.078162 ┆ -88.21878  ┆ 216.1     │
│ 2023-12-31 03:33:22 ┆ 40.078152 ┆ -88.21878  ┆ 216.1     │
│ 2023-12-31 03:33:23 ┆ 40.078142 ┆ -88.218779 ┆ 216.1     │
│ 2023-12-31 03:33:24 ┆ 40.078131 ┆ -88.218779 ┆ 216.1     │
│ …                  ┆ …        ┆ …         ┆ …        │
│ 2023-12-31 04:04:51 ┆ 40.076804 ┆ -88.218752 ┆ 215.6     │
│ 2023-12-31 04:04:52 ┆ 40.076719 ┆ -88.218758 ┆ 215.6     │
│ 2023-12-31 04:04:53 ┆ 40.076651 ┆ -88.218768 ┆ 215.5     │
│ 2023-12-31 04:04:54 ┆ 40.076594 ┆ -88.218778 ┆ 215.5     │
│ 2023-12-31 04:04:55 ┆ 40.076544 ┆ -88.218794 ┆ 215.5     │
└─────────────────────┴───────────┴────────────┴───────────┘
```

## License

This dataset is released under the Creative Commons Attribution 4.0
International Public License (CC BY 4.0).

See LICENSE.txt for details.
