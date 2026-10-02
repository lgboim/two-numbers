# Three synthetic examples with known answers; they double as a check: the script fails if the tool gets them wrong.
# (1) pass: the features know the group mean and the deviation within it. Expected: H above B, and a high CPG.
# (2) fail: the features know only which group a unit is in. Expected: B reaches H, and CPG near zero.
# (3) finer category: the features know only a subgroup. Expected: positive CPG beyond the group, but zero beyond the subgroup.
import sys, numpy as np
sys.path.insert(0,__file__.rsplit("/",1)[0]); import two_numbers as tn
LANG=sys.argv[1] if len(sys.argv)>1 else "en"
rng=np.random.default_rng(0); n,G,S,d=6000,60,5,64
g=rng.integers(0,G,n); sub=g*S+rng.integers(0,S,n)                # 60 groups, each with five subgroups
mu_g=rng.normal(0,1.0,G); mu_s=rng.normal(0,0.5,G*S); w=rng.normal(0,0.5,n)
y=mu_g[g]+mu_s[sub]+w                                             # target: group, subgroup and individual deviation
train=rng.random(n)<0.7
Eg=rng.normal(0,1,(G,d)); a=rng.normal(0,1,d); b=rng.normal(0,1,d); noise=lambda: 0.3*rng.normal(0,1,(n,d))
CASES={"pass":Eg[g]+np.outer(mu_g[g],a)+np.outer(mu_s[sub]+w+rng.normal(0,0.5,n),b)+noise(),   # partial knowledge of the deviation, with noise
       "fail":Eg[g]+np.outer(mu_g[g],a)+noise(),
       "finer category":Eg[g]+np.outer(mu_g[g],a)+np.outer(mu_s[sub],b)+noise()}   # knows the subgroup mean, not the individual deviation
ok=True
for name,X in CASES.items():
    R=tn.report(y,g,train,X=X,levels={"group":g,"subgroup":sub},n_boot=500,n_perm=100)
    print(f"\n### {name}"); print(tn.format_report(R,LANG))
    if name=="pass": ok&=R["H"]>R["B"]+0.05 and R["CPG"]>0.3 and R["CPG"]>R["null_95"]
    if name=="fail": ok&=R["B"]>=R["H"]-0.01 and abs(R["CPG"])<0.03
    if name=="finer category": ok&=R["CPG"]>0.2 and R["profile"]["subgroup"]["CPG_fixed_predictions"]<0.1
print("\nALL EXAMPLES AS EXPECTED" if ok else "\nEXAMPLE CHECK FAILED"); sys.exit(0 if ok else 1)
