import json

file = 'errorframe.boris'

data = json.load(open(file, 'r'))

outdictboris = dict()
for obser in data['observations'].values():
    v = obser['file']['1'][0]
    tsec = [evt[0] for evt in obser['events']]
    fps = list(obser['media_info']['fps'].values())[0]
    tframe = [int(t*fps) for t in tsec]
    outdictboris[v] = tframe