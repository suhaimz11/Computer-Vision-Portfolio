from _paths import IMAGE_DIR, load_reference_labels
import cv2, numpy as np, pickle, collections, json, time
cv2.setNumThreads(4)
names,P,D,DF=pickle.load(open("sift.pkl","rb"))
N=len(names)
allD=np.concatenate([d for d in D if len(d)]); owner=np.concatenate([np.full(len(d),i,np.int32) for i,d in enumerate(D) if len(d)])
print("index",allD.shape,flush=True)
fl=cv2.flann_Index(allD,dict(algorithm=1,trees=4))
votes=collections.defaultdict(int)  # (a,b,mirror)->votes
t=time.time()
for i in range(N):
    for mir,Q in ((0,D[i]),(1,DF[i])):
        if len(Q)<3: continue
        idx,dist=fl.knnSearch(Q,3,params=dict(checks=48))
        for r in range(len(Q)):
            seen=set()
            # skip self-matches, ratio test on first two non-self
            c=[(dist[r,k],owner[idx[r,k]]) for k in range(3) if owner[idx[r,k]]!=i]
            if len(c)>=2 and c[0][0]<0.8**2*c[1][0] and c[0][1] not in seen:
                votes[(i,int(c[0][1]),mir)]+=1
            elif len(c)==1: votes[(i,int(c[0][1]),mir)]+=1
    if i%500==0: print(i,len(votes),round(time.time()-t),flush=True)
cand=collections.Counter()
for (a,b,m),v in votes.items():
    if v>=3: cand[(min(a,b),max(a,b))]=max(cand[(min(a,b),max(a,b))],v)
print("candidate pairs",len(cand),flush=True)
pickle.dump(dict(cand),open("cand3.pkl","wb"))
