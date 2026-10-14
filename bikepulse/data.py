from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'


def load_data():
    path = ROOT / 'data' / 'hour.csv'
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.endswith('hour.csv')]
            if len(names) != 1:
                raise ValueError('Expected one hourly dataset')
            csv = archive.read(names[0])
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(csv)
        (path.parent / 'provenance.json').write_text(json.dumps({
            'source': URL, 'archive_sha256': hashlib.sha256(payload).hexdigest(),
            'csv_sha256': hashlib.sha256(csv).hexdigest()
        }, indent=2))
    frame = pd.read_csv(path)
    frame['timestamp'] = pd.to_datetime(frame['dteday']) + pd.to_timedelta(frame['hr'], unit='h')
    frame = frame.sort_values('timestamp').set_index('timestamp')
    if frame.index.has_duplicates or (frame['cnt'] < 0).any():
        raise ValueError('Invalid hourly observations')
    if not (frame['cnt'] == frame['casual'] + frame['registered']).all():
        raise ValueError('Rental totals do not reconcile')
    return frame
