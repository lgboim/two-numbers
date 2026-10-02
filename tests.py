# Tests for the tool, built from edge cases found in review. Each test fails if the tool reverts to the old behaviour.
import sys, numpy as np
sys.path.insert(0,__file__.rsplit("/",1)[0]); import two_numbers as T
FAIL=[]
def check(name,cond):
    print(("PASS " if cond else "FAIL ")+name); (None if cond else FAIL.append(name))
# 1. Means from training rows only; an unseen group gets the global mean; changing test labels does not change them
y=np.array([0.,2,10,12,5,7]); g=np.array(["a","a","b","b","c","c"]); tr=np.array([True,True,True,True,False,False])
C,u=T.group_means(y,g,tr); y2=y.copy(); y2[4:]+=100; C2,_=T.group_means(y2,g,tr)
check("train-only means, unseen fallback",np.allclose(C.ravel(),[1,1,11,11,6,6]) and u==2 and np.allclose(C,C2))
# 2. A large gain that comes entirely from correcting cell means: must not be reported as 'near zero' or 'interval includes zero'
rng=np.random.default_rng(0); G=24; gg=np.repeat(np.arange(G),4); trn=np.tile([True,True,False,False],G)
yy=rng.normal(0,3,G)[gg]+np.where(trn,0,rng.normal(0,1,G)[gg])+np.tile([-.01,.01,-.01,.01],G)   # on test rows the cell mean shifts relative to training
cell=np.array([yy[~trn&(gg==k)].mean() for k in range(G)]); pred=np.where(trn,0,cell[gg])      # prediction: the test cell mean, with no ordering inside it
R=T.report(yy,gg,trn,pred=pred,n_boot=200,n_perm=50); txt=T.format_report(R)
check("cell-mean-only gain is not called near zero",R["CPG"]>0.5 and abs(R["within_cell_part"])<1e-9 and "includes zero" not in txt and "near zero" not in txt and "correcting cell means" in txt)
# 3. Perfect baseline: no crash, and CPG is marked undefined; also when only one dimension is perfect
yz=np.array([1.,1,2,2,1,1,2,2]); gz=np.array([0,0,1,1,0,0,1,1]); tz=np.array([True]*4+[False]*4)
R=T.report(yz,gz,tz,pred=yz+0.1,n_boot=20,n_perm=5); check("perfect baseline -> undefined, no crash","CPG_undefined" in R and "undefined" in T.format_report(R))
y2d=np.column_stack([yz,np.arange(8.)]); R=T.report(y2d,gz,tz,pred=y2d+0.1,n_boot=20,n_perm=5); check("one perfect axis -> undefined","CPG_undefined" in R)
# 4. Missing group label: a clear input error
try: T.report(yz,np.array([0,0,1,None,0,0,1,1],dtype=object),tz,pred=yz); check("missing group -> clear ValueError",False)
except ValueError as e: check("missing group -> clear ValueError","missing group" in str(e))
try: T.report(yz,gz,tz,pred=np.r_[yz[:4],np.nan,yz[5:]]); check("non-finite test prediction -> ValueError",False)
except ValueError: check("non-finite test prediction -> ValueError",True)
# 5. Inner penalty selection: training targets in each fold do not depend on the label of a row held out in that fold
n=200; gx=np.repeat(np.arange(100),2); X=rng.normal(0,1,(n,5)); yx=rng.normal(0,1,n); trx=np.ones(n,bool); trx[-20:]=False
cap=[]; orig=T._ridge_path
def spy(Xt,Yt,Xv,al): cap.append(Yt.copy()); return orig(Xt,Yt,Xv,al)
T._ridge_path=spy; T._fit_residual(X,yx,gx,trx,joint=False); A=list(cap); cap.clear()
folds=np.array_split(np.random.default_rng(0).permutation(np.where(trx)[0]),5); r0=folds[0][0]
yb=yx.copy(); yb[r0]+=100; T._fit_residual(X,yb,gx,trx,joint=False); Bc=list(cap); T._ridge_path=orig
check("inner-fold targets ignore the held-out label",np.allclose(A[0],Bc[0]))
# 6. Rows of groups unseen in training are shown in the text
gu=np.array([0,0,1,1,0,0,2,2]); R=T.report(np.arange(8.),gu,tz,pred=np.arange(8.)+0.3,n_boot=20,n_perm=5)
check("unseen-group test rows are shown",R["unseen_test_rows"]==2 and "unseen" in T.format_report(R))
# 7. C494: an undefined interval is not called "includes zero"; the share of undefined draws is shown; too few training rows for the feature route is an error
R=T.report(np.arange(8.),gu,tz,pred=np.arange(8.)+0.3,n_boot=20,n_perm=5); R["CPG_ci_rows"]=[float("nan"),float("nan")]; R["ci_rows_undefined_share"]=1.0
txt=T.format_report(R); check("undefined interval is reported as undefined","undefined" in txt and "includes zero" not in txt and "100.0%" in txt)
try: T.report(np.arange(12.),np.array([0]*6+[1]*6),np.array([True]*5+[False]*7),X=np.random.default_rng(0).normal(size=(12,3))); check("feature route with 5 training rows -> ValueError",False)
except ValueError as e: check("feature route with 5 training rows -> ValueError","at least 10 training rows" in str(e))
print("ALL TESTS PASS" if not FAIL else "FAILED: "+"; ".join(FAIL)); sys.exit(1 if FAIL else 0)
