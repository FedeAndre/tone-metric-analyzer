import requests
u='https://gitlab.com/api/v4/projects/47366628/repository/tree'
r=requests.get(u,params={'path':'Code','recursive':'true','per_page':100},timeout=30)
print('STATUS',r.status_code)
try:
 items=r.json()
 for x in items:print('FILE',x.get('path'),x.get('type'),flush=True)
 for x in items:
  nm=x.get('path','').lower()
  if any(z in nm for z in ['peak','bpm','ippm','breath','find']):
   uri='https://gitlab.com/api/v4/projects/47366628/repository/files/'+requests.utils.quote(x['path'],safe='')+'/raw?ref=main'
   a=requests.get(uri,timeout=30)
   print('CODE',x['path'],a.status_code,len(a.content),a.text[:6500],flush=True)
except Exception as e:print('ERROR',repr(e),r.text[:1000])
