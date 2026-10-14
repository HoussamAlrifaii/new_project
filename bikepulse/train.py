import json
import pickle
from bikepulse.data import ROOT, load_data
from bikepulse.model import train


def main():
    model, metrics, predictions = train(load_data())
    destination = ROOT / 'artifacts'
    destination.mkdir(exist_ok=True)
    (destination / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    predictions.to_csv(destination / 'predictions.csv')
    with (destination / 'model.pkl').open('wb') as handle:
        pickle.dump(model, handle)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
