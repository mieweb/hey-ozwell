# Train the hey-ozwell verifier on browser-captured embeddings (same approach as ozwell-done).
import json, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
d=json.load(open("../captures/verifier-capture-hey.json"))["hey-ozwell"]
pos=np.array(d["pos"],dtype="float32"); neg=np.array(d["neg"],dtype="float32")
X=np.concatenate([pos,neg]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
print(f"hey-ozwell: pos {len(pos)} | neg {len(neg)}\n")
flat=lambda a:a.reshape(len(a),-1)
# CV
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
pk={t:[] for t in (0.3,0.5,0.65)}; nr={t:[] for t in (0.3,0.5,0.65)}
for tr,te in skf.split(X,y):
    m=MLPClassifier(hidden_layer_sizes=(32,),alpha=1.0,max_iter=4000,random_state=0).fit(X[tr],y[tr])
    P=m.predict_proba(X[te])[:,1]; yt=y[te]
    for t in pk: pk[t].append((P[yt==1]>=t).mean()*100); nr[t].append((P[yt==0]<t).mean()*100)
for t in (0.3,0.5,0.65):
    print(f"  t={t}: real-wake kept {np.nanmean(pk[t]):3.0f}% | false-fire rejected {np.nanmean(nr[t]):3.0f}%")
# final + export (MLP -> wasm-compatible)
final=MLPClassifier(hidden_layer_sizes=(32,),alpha=1.0,max_iter=4000,random_state=0).fit(X,y)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onx=convert_sklearn(final,initial_types=[("input",FloatTensorType([None,1536]))],options={id(final):{"zipmap":False}})
import os; fp="../../prod/js/models/hey-ozwell-verifier.onnx"
open(fp,"wb").write(onx.SerializeToString())
print(f"\nexported {fp} ({os.path.getsize(fp)//1024} KB)")
print("TRAIN_HEY_DONE")
