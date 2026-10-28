"""Recover saved single-shot Well evaluations without repeating test inference.

Reads HDF5 metadata for expected window identities, never field arrays or a model.
Published-reference comparisons are deliberately omitted if protocol parity is not
established. CSV rounding limits mean precision to roughly5e-7 per row.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import well2d


def sha(path):
    hasher=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):hasher.update(block)
    return hasher.hexdigest()


def read_scores(path,expected):
    scores={}
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['traj','t0','vrmse']:raise ValueError('Unexpected CSV schema')
        for row in reader:
            key=(int(row['traj']),int(row['t0']));value=float(row['vrmse'])
            if key in scores:raise ValueError(f'Duplicate window {key}')
            if key not in expected:raise ValueError(f'Unexpected window {key}')
            if not math.isfinite(value) or value<0:raise ValueError(f'Invalid metric {key}')
            scores[key]=value
    if scores.keys()!=expected:raise ValueError(f'Missing {len(expected-scores.keys())} windows')
    return scores


def audit(dataset,folder):
    split=well2d.Split(dataset,'test')
    try:
        n,T=split.n,split.T
        expected={(k,t) for k in range(n) for t in range(well2d.N_HIST-1,T-1)}
        data_files=[dict(path=str(f),size_bytes=Path(f).stat().st_size,mtime_ns=Path(f).stat().st_mtime_ns) for f in split.files]
    finally:split.close()
    names=['official']+[f'{arm}_s{seed}' for arm in ('vanilla','mlp30','mr30ryd') for seed in (1337,1338,1339)]
    records=[];missing=[];invalid=[];by_name={}
    for name in names:
        path=folder/(name+'_test.csv')
        if not path.exists():missing.append(str(path));continue
        try:scores=read_scores(path,expected)
        except (ValueError,KeyError) as error:
            invalid.append(dict(path=str(path),error=str(error)));continue
        values=list(scores.values())
        checkpoint=folder/(name+'.pt') if name!='official' else None
        row=dict(name=name,windows=len(values),mean_vrmse=statistics.mean(values),median_vrmse=statistics.median(values),
                 csv_sha256=sha(path),checkpoint_sha256=sha(checkpoint) if checkpoint and checkpoint.exists() else None)
        records.append(row);by_name[name]=row
    aggregates={}
    for arm in ('vanilla','mlp30','mr30ryd'):
        keys=[f'{arm}_s{s}' for s in (1337,1338,1339)]
        if all(k in by_name for k in keys):
            means=[by_name[k]['mean_vrmse'] for k in keys]
            aggregates[arm]=dict(mean=statistics.mean(means),sample_sd=statistics.stdev(means),seeds=3)
    return dict(dataset=dataset,status='complete_csv_grid' if not missing and not invalid else 'partial_csv_grid',
                records=records,missing=missing,invalid=invalid,aggregate_complete_arms=aggregates,
                expected_trajectories=n,frames=T,expected_windows=len(expected),data_metadata=data_files,
                new_test_inference=False,source_sha256=sha(Path(__file__)),
                limitation='CSV coverage and descriptive metrics only; no source-checkpoint equivalence or published-protocol proof')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset');parser.add_argument('folder',type=Path)
    args=parser.parse_args();report=audit(args.dataset,args.folder)
    dest=args.folder/'csv_audit.json';dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
