# two_numbers: a small tool for checking a claim of the form 'the model encodes the variable' when the variable is grouped (country, century, protein site, topic).
# Number 1, B: a predictor that uses only the group mean, estimated from training rows only, and scored with the same metric as the headline H.
#   If B reaches H, the headline shows nothing beyond what the group alone already gives.
# Number 2, CPG: the out-of-sample gain beyond the group mean, 1 minus the ratio of the model's sum of squared errors to the group mean's.
#   Reported with confidence intervals, a cell decomposition and a within-group permutation test. An interval that includes zero is not evidence of absent information.
# Two ways to use it: (a) pass features X and the tool fits a standard probe and a residual probe itself; (b) pass ready out-of-sample predictions.
# Dependencies: numpy, pandas, scipy, scikit-learn. No downloads, no network.
# Internal revision notes before release 1.0.0: penalty selection for the residual probe uses fold-local group means; separate wording for size, uncertainty, decomposition and permutation;
# CPG is undefined when the baseline is perfect; input validation; rows of groups unseen in training are reported; the profile is labelled as fixed predictions.
__version__ = "1.0.0"
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

ALPHAS=np.logspace(-2,6,25)

def _2d(a):
    a=np.asarray(a,float); return a.reshape(len(a),-1)

def _validate(y,g,train,X=None,pred=None):
    # Input validation: equal lengths, finite target, no missing group labels, at least one training and one test row, and finite predictions on test rows.
    n=len(y); g=pd.Series(np.asarray(g,dtype=object)); train=np.asarray(train)
    if len(g)!=n or len(train)!=n: raise ValueError("y, groups and train must have the same length")
    if train.dtype!=bool: raise ValueError("train must be a boolean array (True = training row)")
    if not np.isfinite(_2d(y)).all(): raise ValueError("y has missing or non-finite values")
    if g.isna().any(): raise ValueError(f"{int(g.isna().sum())} rows have a missing group label; drop them or give them an explicit label")
    if train.all() or not train.any(): raise ValueError("need at least one training row and one test row")
    if X is not None:
        X=np.asarray(X,float)
        if len(X)!=n or not np.isfinite(X).all(): raise ValueError("X must have one finite row per target row")
        if train.sum()<10: raise ValueError("the feature route needs at least 10 training rows (penalty selection uses 5 inner folds)")
    if pred is not None:
        P=_2d(pred)
        if len(P)!=n or P.shape[1]!=_2d(y).shape[1]: raise ValueError("pred must have the same shape as y")
        if not np.isfinite(P[~train]).all(): raise ValueError("pred has missing or non-finite values on test rows")

def group_means(y,g,train,min_size=1):
    # Group mean from training rows only. A group absent from training, or with fewer than min_size training rows, gets the global training mean, and is counted.
    y=_2d(y); g=np.asarray(g,dtype=object); G=pd.DataFrame(y[train]).groupby(g[train]); s=G.mean()[G.size()>=min_size]
    C=s.reindex(g).to_numpy(copy=True); miss=np.isnan(C).any(1); C[miss]=y[train].mean(0)
    return C,int((miss&~train).sum())

def score(y,p,metric):
    # The paper's metric. With more than one target dimension: the mean over dimensions.
    y=_2d(y); p=_2d(p); out=[]
    for k in range(y.shape[1]):
        a,b=y[:,k],p[:,k]
        if metric=="r2": out.append(1-((a-b)**2).sum()/((a-a.mean())**2).sum())
        elif metric=="pearson": out.append(pearsonr(a,b)[0])
        elif metric=="spearman": out.append(spearmanr(a,b)[0])
        elif metric=="auroc": out.append(roc_auc_score(a,b))
        else: raise ValueError("metric must be r2, pearson, spearman or auroc")
    return float(np.mean(out))

def cpg(y,p,C):
    # Gain beyond the group mean: per dimension and then averaged; also the pooled version over all dimensions.
    # If the baseline has zero error on any dimension, CPG is undefined (nan is returned): there is nothing to improve on.
    y=_2d(y); p=_2d(p); C=_2d(C); sf=((y-p)**2).sum(0); sr=((y-C)**2).sum(0)
    if (sr<=0).any(): return float("nan"),float("nan")
    return float(np.mean(1-sf/sr)),float(1-sf.sum()/sr.sum())

def decompose(y,p,C,g):
    # Exact decomposition: how much of the gain comes from correcting cell means and how much from predicting group-centred values within cells. The two parts sum to CPG.
    t=_2d(y)-_2d(C); q=_2d(p)-_2d(C); out=[]
    for k in range(t.shape[1]):
        d=pd.DataFrame({"g":np.asarray(g,dtype=object),"t":t[:,k],"q":q[:,k]}); m=d.groupby("g")[["t","q"]].transform("mean")
        sR=float((d.t**2).sum())
        if sR<=0: return float("nan"),float("nan")
        bF=float(((m.t-m.q)**2).sum()); wF=float((((d.t-m.t)-(d.q-m.q))**2).sum()); bR=float((m.t**2).sum())
        out.append(((bR-bF)/sR,((sR-bR)-wF)/sR))
    b,w=np.mean(out,0); return float(b),float(w)

def _ridge_path(Xt,Yt,Xv,alphas):
    # Ridge with intercept, for all penalty values at once: eigendecomposition of the feature Gram matrix if rows >= features, otherwise an SVD.
    mx=Xt.mean(0); my=Yt.mean(0); Xc=Xt-mx; Yc=Yt-my
    if Xc.shape[0]>=Xc.shape[1]:
        ev,V=np.linalg.eigh(Xc.T@Xc); ev=np.clip(ev,0,None); B=V.T@(Xc.T@Yc); Q=(Xv-mx)@V
        for a in alphas: yield Q@((1.0/(ev+a))[:,None]*B)+my
    else:
        U,s,Vt=np.linalg.svd(Xc,full_matrices=False); UtY=U.T@Yc; Q=(Xv-mx)@Vt.T
        for a in alphas: yield Q@((s/(s**2+a))[:,None]*UtY)+my

def _fit_residual(X,y,g,train,joint,k=5,seed=0):
    # Residual probe beyond the group mean. Penalty chosen by inner folds: in each fold, group means (and, for the joint fit, the feature centring)
    # are computed from that fold's training rows only, so a held-out row's label never enters other rows' targets.
    y=_2d(y); tr=np.where(train)[0]; folds=np.array_split(np.random.default_rng(seed).permutation(tr),k); err=np.zeros(len(ALPHAS))
    for f in folds:
        ft=np.zeros(len(y),bool); ft[np.setdiff1d(tr,f)]=True
        Cf,_=group_means(y,g,ft); Xf=X-group_means(X,g,ft)[0] if joint else X
        for j,P in enumerate(_ridge_path(Xf[ft],(y-Cf)[ft],Xf[f],ALPHAS)): err[j]+=float(((y[f]-Cf[f]-P)**2).sum())
    a=float(ALPHAS[int(np.argmin(err))]); C,_=group_means(y,g,train); Xa=X-group_means(X,g,train)[0] if joint else X
    P=next(_ridge_path(Xa[train],(y-C)[train],Xa,[a]))
    return C+P,a

def fit_from_features(X,y,g,train):
    # Route (a): a standard probe predicts y (no group information); a residual probe predicts y minus the group mean;
    # and a joint fit, in which the features are also centred on group means.
    X=np.asarray(X,float); y=_2d(y); g=np.asarray(g,dtype=object)
    p_std=RidgeCV(alphas=ALPHAS).fit(X[train],y[train]).predict(X).reshape(len(y),-1)
    p_add,a_add=_fit_residual(X,y,g,train,joint=False); p_joint,a_joint=_fit_residual(X,y,g,train,joint=True)
    return p_std,p_add,p_joint,dict(alpha_additive=a_add,alpha_joint=a_joint,alpha_at_edge=bool(a_add in (ALPHAS[0],ALPHAS[-1]) or a_joint in (ALPHAS[0],ALPHAS[-1])))

def _ci(draws):
    d=np.asarray(draws,float); ok=np.isfinite(d)
    return ([float(v) for v in np.percentile(d[ok],[2.5,97.5])] if ok.any() else [float("nan")]*2),float(1-ok.mean())

def report(y,g,train,X=None,pred=None,metric="r2",headline=None,levels=None,n_boot=2000,n_perm=200,seed=0):
    # The full report. train: boolean mask of training rows; all other rows are test rows.
    # pred: ready predictions of y for all rows (only test rows are scored). levels: dict of extra grouping levels, coarse to fine.
    if X is None and pred is None: raise ValueError("give X or pred")
    _validate(y,g,train,X=X,pred=pred)
    y=_2d(y); g=np.asarray(g,dtype=object); train=np.asarray(train,bool); te=~train; rng=np.random.default_rng(seed)
    C,unseen=group_means(y,g,train)
    R=dict(metric=metric,n_test=int(te.sum()),n_groups_test=int(len(pd.unique(g[te]))),unseen_test_rows=unseen,
           fallback="test rows of groups unseen in training are predicted by the global training mean")
    if X is not None: p_std,p_full,p_joint,info=fit_from_features(X,y,g,train); R.update(info)
    else: p_std=p_full=np.nan_to_num(_2d(pred)); p_joint=None
    R["B"]=score(y[te],C[te],metric)
    R["H_computed"]=score(y[te],p_std[te],metric); R["H"]=headline if headline is not None else R["H_computed"]
    R["verdict"]="B reaches H" if R["B"]>=R["H"] else "H exceeds B"
    yt,pt,Ct,gt=y[te],p_full[te],C[te],g[te]
    R["CPG"],R["CPG_pooled"]=cpg(yt,pt,Ct)
    if not np.isfinite(R["CPG"]):   # perfect baseline on some dimension: no CPG, and no intervals are computed
        R["CPG_undefined"]="the group baseline has zero error on the test rows for at least one target axis"; return R
    if p_joint is not None: R["CPG_joint"]=cpg(yt,p_joint[te],Ct)[0]
    R["between_cell_part"],R["within_cell_part"]=decompose(yt,pt,Ct,gt)
    n=len(yt); R["CPG_ci_rows"],R["ci_rows_undefined_share"]=_ci([cpg(yt[i],pt[i],Ct[i])[0] for i in (rng.integers(0,n,n) for _ in range(n_boot))])
    u=np.array(sorted(pd.unique(gt),key=str),dtype=object)   # fixed group order (as np.unique), so the resampling matches the audited version
    if len(u)>=20:   # whole-group interval, only with at least 20 groups; conditional on the fixed predictions, no refitting
        rows={k:np.where(gt==k)[0] for k in u}; bc=[]
        for _ in range(n_boot):
            i=np.concatenate([rows[k] for k in rng.choice(u,len(u))]); bc.append(cpg(yt[i],pt[i],Ct[i])[0])
        R["CPG_ci_groups"],R["ci_groups_undefined_share"]=_ci(bc)
    q=pt-Ct; idx=pd.Series(np.arange(n)).groupby(gt).indices; null=[]   # permutation test: shuffles predictions within each group; keeps the cell-mean part and tests only the within-cell component
    for _ in range(n_perm):
        perm=np.arange(n)
        for ix in idx.values(): perm[ix]=rng.permutation(ix)
        null.append(cpg(yt,Ct+q[perm],Ct)[0])
    R["null_95"]=float(np.nanpercentile(null,95)); R["null_p"]=float((1+np.sum(np.array(null)>=R["CPG"]))/(1+n_perm))
    if levels:   # profile: the same fixed predictions against the mean of each grouping level (no refitting per level)
        R["profile"]={}
        for name,lab in levels.items():
            Cl,_=group_means(y,np.asarray(lab,dtype=object),train); R["profile"][name]=dict(B=score(yt,Cl[te],metric),CPG_fixed_predictions=cpg(yt,pt,Cl[te])[0])
    return R

TXT={"he":dict(head="שני המספרים",groups="קבוצות",B="הבסיס של הקבוצה",H="המספר המפורסם",cpg="המספר השני, מעבר לקבוצה",rows="מרווח לפי שורות",grp="מרווח לפי קבוצות",
               joint="בהתאמה משותפת",reach="הבסיס מגיע למספר המפורסם: המספר המפורסם לבדו לא מראה ידע מעבר לקבוצה.",exceed="המספר המפורסם גבוה מהבסיס.",
               undefined="המספר השני לא מוגדר: לבסיס של הקבוצה אין שום שגיאה בשורות המבחן, לפחות בציר אחד.",
               dec="פירוק: תיקון ממוצעי התאים %.3f, החלק שבתוך התאים %.3f",
               perm="מבחן ערבוב בתוך הקבוצה (בודק רק את החלק שבתוך התאים): p = %.3f",
               unseen="%d שורות מבחן שייכות לקבוצות שלא הופיעו באימון, והבסיס שלהן הוא הממוצע הכללי; חלק מהמספר השני יכול לבוא מהן.",
               pos="קריאה: המספר השני חיובי, והמרווח לפי שורות לא כולל אפס.",ci_undef="קריאה: המרווח לא מוגדר, כי אף דגימה חוזרת לא נתנה ערך מוגדר.",drop="  %.1f%% מהדגימות החוזרות לפי שורות לא היו מוגדרות והושמטו.",gdrop="  %.1f%% מהדגימות החוזרות לפי קבוצות לא היו מוגדרות והושמטו.",neg="קריאה: המספר השני שלילי: המודל גרוע מממוצע הקבוצה.",
               inc="קריאה: המרווח כולל אפס. אין שיפור ברור מעבר לקבוצה בגודל המדגם הזה, וזו לא ראיה להיעדר ידע.",
               within="רוב השיפור בא מחיזוי הערכים הממורכזים בתוך התאים.",between="רוב השיפור בא מתיקון ממוצעי התאים, ולא מחיזוי בתוכם.",
               perm_yes="החלק שבתוך התאים עובר את מבחן הערבוב.",perm_no="החלק שבתוך התאים לא נבדל ממבחן הערבוב.",
               prof="פרופיל (אותן תחזיות קבועות מול ממוצעי רמות אחרות, בלי התאמה מחדש)"),
     "en":dict(head="Two numbers",groups="groups",B="group baseline B",H="headline H",cpg="conditional predictive gain (CPG)",rows="row CI",grp="group CI",
               joint="joint fit",reach="B reaches H: the headline alone shows nothing beyond the group.",exceed="H exceeds B.",
               undefined="CPG is undefined: the group baseline has zero error on the test rows for at least one target axis.",
               dec="decomposition: cell-mean part %.3f, within-cell part %.3f",
               perm="within-group permutation (tests only the within-cell component): p = %.3f",
               unseen="%d test rows belong to groups unseen in training; their baseline is the global training mean, and part of CPG may come from them.",
               pos="reading: CPG is positive and its row interval excludes zero.",ci_undef="reading: the interval is undefined; no bootstrap draw gave a defined value.",drop="  %.1f%% of row bootstrap draws were undefined and were excluded.",gdrop="  %.1f%% of group bootstrap draws were undefined and were excluded.",neg="reading: CPG is negative: the model predicts worse than the group mean.",
               inc="reading: the interval includes zero. No clear gain beyond the group at this sample size; this is not evidence that information is absent.",
               within="Most of the gain comes from predicting group-centred values within cells.",between="Most of the gain comes from correcting cell means, not from predicting within them.",
               perm_yes="The within-cell component exceeds the permutation null.",perm_no="The within-cell component is not distinguishable from the permutation null.",
               prof="profile (the same fixed predictions against means of other groupings; not refit)")}

def format_report(R,lang="en"):
    # Each line states one thing: size, uncertainty, source of the gain, permutation test. No line infers 'no information' from a test that did not pass.
    T=TXT[lang]; L=[f"== {T['head']} ({R['metric']}, n={R['n_test']}, {R['n_groups_test']} {T['groups']})"]
    L.append(f"{T['B']} {R['B']:.3f} | {T['H']} {R['H']:.3f}  ->  {T['reach'] if R['B']>=R['H'] else T['exceed']}")
    if R.get("unseen_test_rows",0)>0: L.append("  "+T["unseen"]%R["unseen_test_rows"])
    if "CPG_undefined" in R: L.append(T["undefined"]); return "\n".join(L)
    s=f"{T['cpg']} {R['CPG']:.3f}  {T['rows']} [{R['CPG_ci_rows'][0]:.3f}, {R['CPG_ci_rows'][1]:.3f}]"
    if "CPG_ci_groups" in R: s+=f"  {T['grp']} [{R['CPG_ci_groups'][0]:.3f}, {R['CPG_ci_groups'][1]:.3f}]"
    if "CPG_joint" in R: s+=f"  | {T['joint']} {R['CPG_joint']:.3f}"
    L.append(s); L.append("  "+T["dec"]%(R["between_cell_part"],R["within_cell_part"])); L.append("  "+T["perm"]%R["null_p"])
    lo,hi=R["CPG_ci_rows"]
    if R.get("ci_rows_undefined_share",0)>0: L.append(T["drop"]%(100*R["ci_rows_undefined_share"]))
    if R.get("ci_groups_undefined_share",0)>0: L.append(T["gdrop"]%(100*R["ci_groups_undefined_share"]))
    L.append("  "+(T["ci_undef"] if not (np.isfinite(lo) and np.isfinite(hi)) else T["pos"] if lo>0 else T["neg"] if hi<0 else T["inc"]))
    if R["CPG"]>0: L.append("  "+(T["within"] if R["within_cell_part"]>=R["between_cell_part"] else T["between"]))
    L.append("  "+(T["perm_yes"] if R["null_p"]<0.05 else T["perm_no"]))
    if "profile" in R:
        L.append(T["prof"]+": "+" | ".join(f"{k}: B {v['B']:.3f}, CPG {v['CPG_fixed_predictions']:.3f}" for k,v in R["profile"].items()))
    return "\n".join(L)

def main():
    # Command-line use on a table: target columns, group column, test column (1 on test rows) and prediction columns.
    # Example: python3 two_numbers.py data.csv --y lat,lon --group country --test is_test --pred p_lat,p_lon --metric r2
    import argparse, json
    ap=argparse.ArgumentParser(description="Two numbers: group baseline B and conditional predictive gain (CPG)")
    ap.add_argument("csv"); ap.add_argument("--y",required=True); ap.add_argument("--group",required=True); ap.add_argument("--test",required=True)
    ap.add_argument("--pred",required=True); ap.add_argument("--metric",default="r2"); ap.add_argument("--headline",type=float)
    ap.add_argument("--levels",default="",help="extra grouping columns, coarse to fine"); ap.add_argument("--lang",default="en"); ap.add_argument("--json")
    a=ap.parse_args(); df=pd.read_csv(a.csv); ycols=a.y.split(","); pcols=a.pred.split(",")
    if len(ycols)!=len(pcols): raise SystemExit("--y and --pred need the same number of columns")
    nd=len(df); df=df.dropna(subset=ycols+[a.group,a.test])
    if len(df)<nd: print(f"note: {nd-len(df)} rows with a missing target, group or test flag were dropped")
    train=~df[a.test].astype(bool).values
    lev={c:df[c].values for c in a.levels.split(",") if c}
    try: R=report(df[ycols].values,df[a.group].values,train,pred=df[pcols].to_numpy(dtype=float,copy=True),metric=a.metric,headline=a.headline,levels=lev or None)
    except ValueError as e: raise SystemExit(f"input error: {e}")
    print(format_report(R,a.lang))
    if a.json: json.dump(R,open(a.json,"w"),indent=1)

if __name__=="__main__": main()
