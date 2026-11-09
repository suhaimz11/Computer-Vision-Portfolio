from _paths import IMAGE_DIR, load_reference_labels
import cv2, numpy as np, os
from multiprocessing import Pool
cv2.setNumThreads(1)
IM=str(IMAGE_DIR)
names=sorted(name for name in os.listdir(IM) if os.path.splitext(name)[1].lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
if not names: raise ValueError(f'No supported images in {IM}')
def work(i):
    img=cv2.imread(os.path.join(IM,names[i]),0)
    if img is None: raise ValueError(f'Cannot decode image: {names[i]}')
    if img.shape != (416,416): raise ValueError(f'Expected 416x416 image: {names[i]} has shape {img.shape}')
    s=cv2.SIFT_create(nfeatures=200,contrastThreshold=0.02)
    kp,d=s.detectAndCompute(img,None)
    if d is None: return i,np.zeros((0,2),np.float32),np.zeros((0,128),np.float32),np.zeros((0,128),np.float32)
    p=np.array([k.pt for k in kp],np.float32)
    d=d/ (d.sum(1,keepdims=True)+1e-7); d=np.sqrt(d).astype(np.float32)
    f=cv2.flip(img,1); kf,df=s.detectAndCompute(f,None)
    if df is None: df=np.zeros((0,128),np.float32)
    else: df=np.sqrt(df/(df.sum(1,keepdims=True)+1e-7)).astype(np.float32)
    return i,p,d,df
if __name__=="__main__":
    with Pool(4) as pool:
        res=[]
        for n,r in enumerate(pool.imap(work,range(len(names)),chunksize=20)):
            res.append(r)
            if n%1000==0: print("extracted",n,flush=True)
    res.sort(key=lambda r:r[0])
    import pickle
    pickle.dump((names,[r[1] for r in res],[r[2] for r in res],[r[3] for r in res]),open("sift.pkl","wb"),protocol=4)
    print("done",sum(len(r[2]) for r in res),"descriptors",flush=True)
