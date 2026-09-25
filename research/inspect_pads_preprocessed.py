import requests
b="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/scripts/"
for f in ["plot_example_hc_signal_processed.py","run_preprocessing.py","utils/io.py","utils/file_io.py"]:
 r=requests.get(b+f,timeout=60);print("\n###",f,r.status_code,len(r.text));print(r.text[:12000])
# inspect one preprocessed file signature
u="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/preprocessed/movement/001_ml.bin"
r=requests.get(u,timeout=60);print("\nBIN",r.status_code,len(r.content),r.content[:64])
