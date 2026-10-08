import io,zipfile,csv,collections,statistics,requests,mido,hashlib,json,time
URL="https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
print("download beginning",flush=True);b=requests.get(URL,timeout=120).content
print("bytes",len(b),"sha256",hashlib.sha256(b).hexdigest(),flush=True)
z=zipfile.ZipFile(io.BytesIO(b))
info=[n for n in z.namelist() if n.endswith('info.csv')]
print("info paths",info)
data=list(csv.DictReader(io.StringIO(z.read(info[0]).decode('utf-8-sig'))))
print("columns",list(data[0]),"n",len(data),"examples",data[:2],flush=True)
stats=[]
for d in data:
 try:
  member=next(n for n in z.namelist() if n.endswith(d['midi_filename']))
  mf=mido.MidiFile(file=io.BytesIO(z.read(member)))
  ticks=0;events=[]
  for track in mf.tracks:
   ticks=0
   for m in track:
    ticks+=m.time
    if m.type=='note_on' and m.velocity>0:events.append((ticks/mf.ticks_per_beat,m.note))
  events.sort()
  maxt=events[-1][0] if events else 0
  nbar=int(maxt//4)
  if nbar>=8 and len(events)>=40:
   counts=[sum(4*j<=t<4*(j+1) for t,p in events) for j in range(min(nbar,64))]
   mn=statistics.mean(counts);sd=statistics.pstdev(counts)
   stats.append({'path':d['midi_filename'],'drummer':d['drummer'],'split':d['split'],'beat_type':d['beat_type'],'nbar':nbar,'nnotes':len(events),'cv_count':sd/mn if mn else 0,'time_signature':d['time_signature'],'bpm':d['bpm']})
 except Exception as e:
  print("PARSE_ERROR",d['midi_filename'],str(e)[:120])
print("n eligible",len(stats),"metercounts",dict(collections.Counter(x['time_signature'] for x in stats)),"beattypes",dict(collections.Counter(x['beat_type'] for x in stats)),flush=True)
for nbar in [12,16,24,32,48,64]:
 arr=[x for x in stats if x['nbar']>=nbar and x['time_signature']=='4-4']
 print("nbar >=",nbar,"count",len(arr),"mean cv",round(statistics.mean(x['cv_count'] for x in arr),3) if arr else '-', "p10p50p90",sorted(x['cv_count'] for x in arr)[int(len(arr)*.1)] if arr else None,sorted(x['cv_count'] for x in arr)[int(len(arr)*.5)] if arr else None,sorted(x['cv_count'] for x in arr)[int(len(arr)*.9)] if arr else None,flush=True)
print("top variety",sorted(stats,key=lambda x:x['cv_count'],reverse=True)[:5])
open('groove_scout.json','w').write(json.dumps(stats,indent=2))
