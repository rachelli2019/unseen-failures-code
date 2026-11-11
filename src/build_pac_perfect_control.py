"""Build a perfect-prediction control while preserving Luna uncertainty scores.

This changes only the input predictions; the runner still applies the imperfect
labeler PAC branch, including calibration and the epsilon/alpha budget splits.
"""
import json
from pathlib import Path
from run_pac_labeling import load_population


def main():
    config=json.loads(Path('configs/run_pac_labeling_luna_fullcorpus.json').read_text())
    ids,truth,_,uncertainty,_=load_population(config)
    control=json.loads(Path('configs/run_pac_labeling_perfect_control.json').read_text())
    output=Path(control['predictions_jsonl']);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(dict(session_id=s,predicted_unsafe=int(y),uncertainty_score=float(u)))+'\n' for s,y,u in zip(ids,truth,uncertainty)))
    print(f'Wrote {len(ids)} perfect predictions to {output}')

if __name__=='__main__':main()
