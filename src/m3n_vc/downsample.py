# %%
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io.wavfile import write as write_wav
from scipy.signal import decimate


def downsample_df(df, original_rate, target_rate):
    decim_factor = original_rate // target_rate
    assert original_rate % target_rate == 0

    samples = df['samples'].to_numpy()
    timestamps = df['timestamp'].to_numpy()

    down_samples = decimate(samples, decim_factor, ftype='fir', zero_phase=True)

    start = df['timestamp'][0]
    sample_period = 1.0 / target_rate
    down_timestamps = np.arange(len(down_samples)) * sample_period + start

    down_df = pl.DataFrame({'timestamp': down_timestamps, 'samples': down_samples})

    return down_df


# %%
data_dir = Path('data')
raw_dir = data_dir / 'raw'
zenodo_dir = data_dir / 'zenodo'

# %%
for data_id in ['h08', 'h24', 's31', 'a06', 'i29', 'i22']:
    # for data_id in ['i22']:
    print('processing', data_id)
    (zenodo_dir / data_id).mkdir(exist_ok=True)
    (zenodo_dir / (data_id + '_wav')).mkdir(exist_ok=True)

    files = sorted((raw_dir / data_id).glob('*'))
    files_geo = sorted((raw_dir / data_id).glob('*geo*'))
    files_mic = sorted((raw_dir / data_id).glob('*mic*'))
    files_dis = sorted((raw_dir / data_id).glob('*dis*'))
    files_gps = sorted((raw_dir / data_id).glob('*gps*'))
    run_ids = raw_dir / data_id / 'run_ids.parquet'
    sensor_location = raw_dir / data_id / 'sensor_location.parquet'
    assert (
        len(files)
        == len(files_geo) + len(files_mic) + len(files_dis) + len(files_gps) + 2
    )

    _ = shutil.copy(run_ids, zenodo_dir / data_id / run_ids.name)
    _ = shutil.copy(sensor_location, zenodo_dir / data_id / sensor_location.name)
    continue

    for f in files_mic:
        df_mic = pl.read_parquet(f).sort('timestamp')
        # 16kHz -> 1.6kHz
        # df_mic = downsample_df(df_mic, 16_000, 1_600)
        df_mic = df_mic.select(['timestamp', 'samples'])[::10]
        df_mic.write_parquet(zenodo_dir / data_id / f.name)
        write_wav(
            zenodo_dir / (data_id + '_wav') / f.with_suffix('.wav').name,
            1600,
            df_mic['samples'].to_numpy().astype(np.int16),
        )

    for f in files_geo:
        _ = shutil.copy(f, zenodo_dir / data_id / f.name)
        # break

    for f in files_dis:
        _ = shutil.copy(f, zenodo_dir / data_id / f.name)
        # break

    for f in files_gps:
        _ = shutil.copy(f, zenodo_dir / data_id / f.name)
        # break
