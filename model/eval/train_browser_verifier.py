# Train the verifier on BROWSER-CAPTURED embeddings (same representation as runtime -> no mismatch).
# Cross-validate to confirm it separates real wakes from false-fires, then export ONNX for prod.
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
rng=np.random.default_rng(0)
d=json.load(open("../captures/verifier-capture.json"))
pos=np.array(d["pos"],dtype="float32"); neg=np.array(d["neg"],dtype="float32")
X=np.concatenate([pos,neg]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
print(f"pos {len(pos)} | neg {len(neg)}\n")

def mk(name):
    if name=="logistic": return LogisticRegression(max_iter=3000,C=0.1,class_weight="balanced")
    if name=="mlp32":     return MLPClassifier(hidden_layer_sizes=(32,),alpha=1.0,max_iter=3000,random_state=0)
    if name=="mlp256_64": return MLPClassifier(hidden_layer_sizes=(256,64),alpha=0.1,max_iter=3000,early_stopping=True,random_state=0)

# 5-fold CV: report held-out pos-kept and neg-rejected at thresholds
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
for name in ["logistic","mlp32","mlp256_64"]:
    pk={t:[] for t in (0.1,0.3,0.5)}; nr={t:[] for t in (0.1,0.3,0.5)}
    for tr,te in skf.split(X,y):
        m=mk(name).fit(X[tr],y[tr])
        P=m.predict_proba(X[te])[:,1]; yt=y[te]
        for t in pk:
            pk[t].append((P[yt==1]>=t).mean()*100); nr[t].append((P[yt==0]<t).mean()*100)
    print(f"[{name}]")
    for t in (0.1,0.3,0.5):
        print(f"   t={t}: real-wake kept {np.mean(pk[t]):3.0f}%  | false-fire rejected {np.mean(nr[t]):3.0f}%")
    print()

# Train FINAL on all data with the most robust model, export ONNX
final=mk("logistic").fit(X,y)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
onx=convert_sklearn(final,initial_types=[("input",FloatTensorType([None,1536]))],options={id(final):{"zipmap":False}})
import os; fp="../../prod/js/models/ozwell-i'm-done-verifier.onnx"
open(fp,"wb").write(onx.SerializeToString())
print(f"exported {fp} ({os.path.getsize(fp)//1024} KB) trained on {len(pos)} pos / {len(neg)} neg (browser-native)")
print("TRAIN_DONE")
