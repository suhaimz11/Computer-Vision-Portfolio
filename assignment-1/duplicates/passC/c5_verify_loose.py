import os
from _paths import IMAGE_DIR, load_reference_labels
import cv2, numpy as np, pickle, csv, json, collections
names,P,D,DF=pickle.load(open("sift.pkl","rb"))
cand=pickle.load(open("cand3.pkl","rb"))
B=45
grp,uns=load_reference_labels()
bf=cv2.BFMatcher(cv2.NORM_L2)
IM=str(IMAGE_DIR)+os.sep
sift=cv2.SIFT_create(nfeatures=200,contrastThreshold=0.02)
_fc={}
def flipped(a):
    if a not in _fc:
        kp,d=sift.detectAndCompute(cv2.flip(cv2.imread(IM+names[a],0),1),None)
        if d is None: _fc[a]=(np.zeros((0,2),np.float32),np.zeros((0,128),np.float32))
        else: _fc[a]=(np.array([k.pt for k in kp],np.float32),np.sqrt(d/(d.sum(1,keepdims=True)+1e-7)).astype(np.float32))
    return _fc[a]
def ver(a,b,m):
    pa,Da=flipped(a) if m else (P[a],D[a])
    if len(Da)<4 or len(D[b])<4: return 0,0.
    ms=bf.knnMatch(Da,D[b],k=2)
    g=[x for x,y in (q for q in ms if len(q)==2) if x.distance<0.8*y.distance]
    if len(g)<4: return len(g),0
    # flipped query pts: mirror x
    g=[x for x in g if B<pa[x.queryIdx][0]<416-B and B<pa[x.queryIdx][1]<416-B and B<P[b][x.trainIdx][0]<416-B and B<P[b][x.trainIdx][1]<416-B]
    if len(g)<4: return len(g),0
    s=np.float32([pa[x.queryIdx] for x in g]); d=np.float32([P[b][x.trainIdx] for x in g])
    M,inl=cv2.estimateAffinePartial2D(s,d,ransacReprojThreshold=4)
    if M is None: return len(g),0
    sc=float(np.hypot(M[0,0],M[1,0]))
    ii=inl.ravel()>0
    if ii.sum()<4: return 0,sc
    ext=float(np.ptp(d[ii,0])*np.ptp(d[ii,1])/416**2)
    if ext<0.12: return 0,sc
    return int(ii.sum()),sc
out=[]
for (a,b),v in cand.items():
    best=max((ver(a,b,m)+(m,) for m in (0,1)),key=lambda t:t[0])
    out.append((a,b,v,*best))
rows=[]
for a,b,v,inl,sc,m in out:
    ga,gb=grp.get(names[a]),grp.get(names[b])
    known = "SAME_GROUP" if ga and ga==gb else ("UNSURE" if frozenset((names[a],names[b])) in uns else "NEW")
    rows.append((names[a],names[b],v,inl,round(sc,2),m,known))
rows.sort(key=lambda r:-r[3])
with open("verified3.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["a","b","votes","inliers","scale","mirror","status"]);w.writerows(rows)
c=collections.Counter(r[6] for r in rows if r[3]>=15)
print("inliers>=15 by status:",c)
print("known dup pairs recovered (in-group pairs with >=15 inl):",c["SAME_GROUP"])
for t in (15,20,25,40):print(t,collections.Counter(r[6] for r in rows if r[3]>=t))
