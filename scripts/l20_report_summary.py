import glob
import json
import os
base = '/data/neg_opinion/outputs/structure_v1'
ev = base + '/evaluation_v1'
state = json.load(open(base + '/state.json'))
print('STATE', json.dumps(state, ensure_ascii=False))
for name in ['joint', 'future']:
    path = ev + '/' + name + '/report.json'
    if os.path.exists(path):
        print(name.upper(), json.dumps(json.load(open(path))['report'], ensure_ascii=False))
    else:
        print(name.upper(), 'no report yet')
for name in ['joint', 'extraction', 'future']:
    rec = ev + '/' + name + '/records'
    count = len(glob.glob(rec + '/*.json')) if os.path.isdir(rec) else 0
    print('RECORDS', name, count)
print('SUMMARY_EXISTS', os.path.exists(ev + '/SUMMARY.json'))
