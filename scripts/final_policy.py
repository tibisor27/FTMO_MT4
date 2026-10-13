"""Calibrarea finala pentru FTMO 2-Step: rezolva HJB, exporta politica, o VALIDEAZA
printr-un Monte Carlo inainte (doua metode independente trebuie sa fie de acord).

    python scripts/final_policy.py [orizont_zile] [fisier_iesire]
    python scripts/final_policy.py 126 configs/live/sizing_policy_126d.csv

Tabelul scris e EXACT politica pe care o citeste src/live/sizing.py in live.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

D="results/backtest/noise_area_USATECHIDXUSD_1m"
d=pd.read_csv(f"{D}/excursion_opposite_band.csv")
R=d["ret"].to_numpy(float); L=(d["min_intra"]-d["ret"]).to_numpy(float)
SIG=R.std(ddof=1)
T=int(sys.argv[1]) if len(sys.argv)>1 else 252
OUT=sys.argv[2] if len(sys.argv)>2 else f"configs/live/sizing_policy_{T}d.csv"
DAILY,MAXL,T1,T2=0.05,0.10,0.10,0.05
E=np.linspace(-MAXL,T1,61); i0=int(np.searchsorted(E,0.0))
VOLS=np.array([0.30,0.50,0.75,1.00,1.50,2.00,2.50])
VAL,C,SH=0.137*0.9,0.0055,0.95
BASE=(R-R.mean())+SH*SIG/np.sqrt(252)

W1=np.zeros((T+1,len(E))); W2=np.zeros((T+1,len(E)))
P1=np.zeros((T+1,len(E))); P2=np.zeros((T+1,len(E)))
F1=np.zeros((T+1,len(E))); F2=np.zeros((T+1,len(E)))
A1=np.zeros((T+1,len(E))); A2=np.zeros((T+1,len(E)))
for t in range(1,T+1):
    rW,rP,rF=W1[t-1][i0]-C,P1[t-1][i0],F1[t-1][i0]+1.0
    for ph in (2,1):
        tgt=T1 if ph==1 else T2
        best=np.full(len(E),-1e9); arg=np.zeros(len(E)); bp=np.zeros(len(E)); bf=np.zeros(len(E))
        for v in VOLS:
            k=(v/100.0)/SIG; rk=BASE*k; lok=(BASE+L)*k
            brk=(lok<=-DAILY)[None,:]|((E[:,None]+lok[None,:])<=-MAXL)
            en=E[:,None]+rk[None,:]; fl=en.ravel(); w=en>=tgt
            if ph==1:
                nw=np.where(w,W2[t-1][i0],np.interp(fl,E,W1[t-1]).reshape(en.shape))
                np_=np.where(w,P2[t-1][i0],np.interp(fl,E,P1[t-1]).reshape(en.shape))
                nf=np.where(w,F2[t-1][i0],np.interp(fl,E,F1[t-1]).reshape(en.shape))
            else:
                nw=np.where(w,VAL,np.interp(fl,E,W2[t-1]).reshape(en.shape))
                np_=np.where(w,1.0,np.interp(fl,E,P2[t-1]).reshape(en.shape))
                nf=np.where(w,0.0,np.interp(fl,E,F2[t-1]).reshape(en.shape))
            val=np.where(brk,rW,nw).mean(axis=1); m=val>best
            best[m]=val[m]; arg[m]=v
            bp[m]=np.where(brk,rP,np_).mean(axis=1)[m]; bf[m]=np.where(brk,rF,nf).mean(axis=1)[m]
        if ph==1: W1[t],A1[t],P1[t],F1[t]=best,arg,bp,bf
        else:     W2[t],A2[t],P2[t],F2[t]=best,arg,bp,bf

# ---- VALIDARE: Monte Carlo inainte sub exact aceeasi politica ----------
NS,BLOCK=20000,20
rng=np.random.default_rng(31)
nb=int(np.ceil(T/BLOCK))
st=rng.integers(0,max(1,len(R)-BLOCK+1),size=(NS,nb))
IX=np.clip((st[:,:,None]+np.arange(BLOCK)[None,None,:]).reshape(NS,-1)[:,:T],0,len(R)-1)
B0,L0=BASE[IX],(BASE+L)[IX]
def forward(sh_real=SH):
    b=(R-R.mean())+sh_real*SIG/np.sqrt(252)
    bb,ll=b[IX],(b+L)[IX]
    eq=np.zeros(NS); ph=np.ones(NS,int); fd=np.full(NS,T+1); fees=np.ones(NS,int)
    for t in range(T):
        live=fd>T
        if not live.any(): break
        rem=T-t
        j=np.clip(np.searchsorted(E,eq),0,len(E)-1)
        v=np.where(ph==1,A1[rem][j],A2[rem][j])
        k=(v/100.0)/SIG; rt=bb[:,t]*k; lt=ll[:,t]*k
        br=live&((eq+lt<=-MAXL)|(lt<=-DAILY))
        eq[br]=0.0; ph[br]=1; fees[br]+=1
        ad=live&~br; eq[ad]+=rt[ad]
        tg=np.where(ph==1,T1,T2); hit=ad&(eq>=tg)
        p2=hit&(ph==2); p1=hit&(ph==1); fd[p2]=t; eq[p1]=0.0; ph[p1]=2
    return fd,fees

fd,fe=forward()
print("=== VALIDARE: programare dinamica vs Monte Carlo inainte, aceeasi politica ===")
print(f"{'':>12} {'HJB':>10} {'Monte Carlo':>13} {'diferenta':>11}")
print(f"{'P(orizont)':>12} {P1[T][i0]:10.3f} {(fd<=T).mean():13.3f} {abs(P1[T][i0]-(fd<=T).mean()):11.3f}")
print(f"{'E[taxe]':>12} {F1[T][i0]:10.2f} {fe.mean():13.2f} {abs(F1[T][i0]-fe.mean()):11.2f}")

print("\n=== REZULTATE (politica optima, plafon 2.5%/zi) ===")
print(f"{'orizont':>10} {'P(finantat)':>12} {'E[taxe]':>9} {'valoare neta':>13} {'$ pe 100k':>11}")
for h in sorted({max(1,T//4), max(1,T//2), max(1,3*T//4), T}):
    print(f"{str(h)+' zile':>10} {P1[h][i0]:12.3f} {F1[h][i0]:9.2f} {W1[h][i0]:13.2%} {W1[h][i0]*100000:10,.0f}")
_fin=fd[fd<=T]
print(f"\nmediana zilelor pana la finantare: {np.median(_fin):.0f}" if len(_fin) else "\nnicio finantare in orizont")

print("\n=== ROBUSTETE: politica e fixa, Sharpe-ul real variaza ===")
print(f"{'Sharpe real':>12} {'P(12 luni)':>11} {'P(6 luni)':>10} {'E[taxe]':>9}")
for s in (1.35,1.19,0.95,0.70,0.50,0.30,0.00):
    f2,e2=forward(s)
    print(f"{s:12.2f} {(f2<=T).mean():11.3f} {(f2<=T//2).mean():10.3f} {e2.mean():9.2f}")

rows=[]
for ph,A in ((1,A1),(2,A2)):
    for h in range(1,T+1):
        for j,e in enumerate(E): rows.append((ph,h,round(e,4),A[h][j]))
pd.DataFrame(rows,columns=["faza","zile_ramase","echitate","vol_target_pct"]).to_csv(OUT,index=False)
print(f"\nscris: {OUT} ({len(rows)} randuri: faza x zile_ramase x echitate -> marime)")
print(f"geometrie: tinta faza1 {T1:.0%}, tinta faza2 {T2:.0%}, "
      f"pierdere zilnica {DAILY:.0%}, pierdere totala {MAXL:.0%}")
print(f"grila de volatilitati oferita solverului: {list(VOLS)}")
