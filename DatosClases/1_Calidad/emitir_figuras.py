#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emite figuras-datos.tex: macros pgfplots cuyas coordenadas son EXACTAMENTE
los datos de Conservas del Itata. Asi ninguna carta de las laminas esta
dibujada a mano ni difiere de lo que produce el codigo que ven los alumnos."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

D = Path(__file__).resolve().parent
peso = pd.read_csv(D / "itata_peso.csv")
atr = pd.read_csv(D / "itata_atributos.csv")
msa = pd.read_csv(D / "itata_msa.csv")

A2, D3, D4, d2 = 0.729, 0.000, 2.282, 2.059
LIE, LSE, NOM = 465.0, 495.0, 480.0

f1 = peso[peso.fase == "I"].groupby("subgrupo")["peso_g"]
xbar1, R1 = f1.mean(), f1.max() - f1.min()
XBB, RB = xbar1.mean(), R1.mean()
LCSX, LCIX = XBB + A2 * RB, XBB - A2 * RB
LCSR, LCIR = D4 * RB, D3 * RB
SIGMA = RB / d2

todos = peso.groupby("subgrupo")["peso_g"]
xbar, R = todos.mean(), todos.max() - todos.min()


def num(v, dec=2):
    """Numero con coma decimal, para ETIQUETAS. Nunca usar en coordenadas."""
    return f"{v:.{dec}f}".replace(".", "{,}")


def coords(serie, fmt="{:.3f}"):
    return " ".join(f"({i},{fmt.format(v)})" for i, v in serie.items())


L = []
A = L.append

A("% " + "=" * 70)
A("% figuras-datos.tex  --  GENERADO por emitir_figuras.py, no editar a mano.")
A("% Coordenadas exactas de Conservas del Itata S.A.")
A("% " + "=" * 70)
A("")

# ---------------------------------------------------------------- carta Xbar
dentro = xbar[xbar.index <= 25]
fuera = xbar[(xbar.index > 25) & (xbar > LCSX)]
dentro2 = xbar[(xbar.index > 25) & (xbar <= LCSX)]
A(r"\newcommand{\cartaxbarrafig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=11.4cm,height=4.1cm,xlabel={{subgrupo}},"
  rf"ylabel={{$\bar x$ (g)}},grid=major,xmin=0,xmax=31,ymin=477,ymax=497,"
  rf"xtick={{5,10,15,20,25,30}},legend style={{font=\tiny,at={{(0.02,0.97)}},"
  rf"anchor=north west,legend columns=3}},tick label style={{font=\tiny}},"
  rf"label style={{font=\scriptsize}}]")
A(rf"\addplot[iiBlue,thick,mark=*,mark size=1.2pt] coordinates {{{coords(dentro)}}};"
  r"\addlegendentry{Fase I: estable}")
A(rf"\addplot[iiOrange,thick,mark=*,mark size=1.2pt,forget plot] coordinates "
  rf"{{{coords(xbar[xbar.index >= 25])}}};")
A(rf"\addplot[only marks,mark=*,mark size=2.6pt,iiRed] coordinates "
  rf"{{{coords(fuera)}}};" r"\addlegendentry{fuera de control}")
A(rf"\draw[iiDark,thick] (axis cs:0,{XBB:.3f}) -- (axis cs:31,{XBB:.3f})"
  rf" node[right,font=\tiny,iiDark]{{}};")
A(rf"\draw[iiRed,densely dashed,thick] (axis cs:0,{LCSX:.3f}) -- (axis cs:31,{LCSX:.3f});")
A(rf"\draw[iiRed,densely dashed,thick] (axis cs:0,{LCIX:.3f}) -- (axis cs:31,{LCIX:.3f});")
A(rf"\node[font=\tiny,iiRed,anchor=west,fill=white,inner sep=1pt] at "
  rf"(axis cs:25.3,{LCSX+0.9:.2f}) {{LCS = {num(LCSX)}}};")
A(rf"\node[font=\tiny,iiDark,anchor=west,fill=white,inner sep=1pt] at "
  rf"(axis cs:0.4,{XBB+0.9:.2f}) {{LC = {num(XBB)}}};")
A(rf"\node[font=\tiny,iiRed,anchor=west,fill=white,inner sep=1pt] at "
  rf"(axis cs:0.4,{LCIX+0.9:.2f}) {{LCI = {num(LCIX)}}};")
A(rf"\draw[iiGreen,densely dotted,very thick] (axis cs:20.5,477) -- (axis cs:20.5,497);")
A(rf"\node[font=\tiny,iiGreen,anchor=south,rotate=90] at (axis cs:20.2,483) "
  rf"{{Fase I $\to$ Fase II}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- carta R
A(r"\newcommand{\cartarangofig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=11.4cm,height=2.9cm,xlabel={{subgrupo}},ylabel={{$R$ (g)}},"
  rf"grid=major,xmin=0,xmax=31,ymin=0,ymax=19,xtick={{5,10,15,20,25,30}},"
  rf"tick label style={{font=\tiny}},label style={{font=\scriptsize}}]")
A(rf"\addplot[iiPurple,thick,mark=*,mark size=1.2pt] coordinates {{{coords(R)}}};")
A(rf"\draw[iiDark,thick] (axis cs:0,{RB:.3f}) -- (axis cs:31,{RB:.3f});")
A(rf"\draw[iiRed,densely dashed,thick] (axis cs:0,{LCSR:.3f}) -- (axis cs:31,{LCSR:.3f});")
A(rf"\node[font=\tiny,iiRed,anchor=west,fill=white,inner sep=1pt] at "
  rf"(axis cs:0.4,{LCSR+0.9:.2f}) {{LCS = {num(LCSR)}}};")
A(rf"\node[font=\tiny,iiDark,anchor=west,fill=white,inner sep=1pt] at "
  rf"(axis cs:0.4,{RB+0.9:.2f}) {{$\bar R$ = {num(RB)}}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ------------------------------------------- limites de control vs especificacion
A(r"\newcommand{\limitesvsespecfig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=11.0cm,height=3.9cm,xlabel={{peso escurrido (g)}},"
  rf"ylabel={{densidad}},grid=major,xmin=462,xmax=500,ymin=0,domain=462:500,"
  rf"samples=200,legend style={{font=\tiny,at={{(0.02,0.97)}},anchor=north west}},"
  rf"tick label style={{font=\tiny}},label style={{font=\scriptsize}},"
  rf"ytick=\empty]")
A(rf"\addplot[iiBlue,thick]{{exp(-((x-{XBB:.2f})^2)/(2*{SIGMA:.3f}^2))"
  rf"/({SIGMA:.3f}*sqrt(2*pi))}};" r"\addlegendentry{individuos ($\sigma$)}")
sx = SIGMA / 2.0
A(rf"\addplot[iiPurple,thick,densely dashed]{{exp(-((x-{XBB:.2f})^2)"
  rf"/(2*{sx:.4f}^2))/({sx:.4f}*sqrt(2*pi))}};"
  r"\addlegendentry{medias $\bar x$ ($\sigma/\sqrt{n}$)}")
A(rf"\draw[iiRed,very thick] (axis cs:{LIE},0) -- (axis cs:{LIE},0.115);")
A(rf"\draw[iiRed,very thick] (axis cs:{LSE},0) -- (axis cs:{LSE},0.115);")
A(rf"\node[font=\tiny,iiRed,anchor=south] at (axis cs:{LIE},0.115) {{LIE 465}};")
A(rf"\node[font=\tiny,iiRed,anchor=south] at (axis cs:{LSE},0.115) {{LSE 495}};")
A(rf"\draw[iiGreen,very thick,densely dotted] (axis cs:{LCIX:.2f},0) -- "
  rf"(axis cs:{LCIX:.2f},0.09);")
A(rf"\draw[iiGreen,very thick,densely dotted] (axis cs:{LCSX:.2f},0) -- "
  rf"(axis cs:{LCSX:.2f},0.09);")
A(rf"\node[font=\tiny,iiGreen,anchor=south] at (axis cs:{LCIX:.2f},0.09) {{LCI}};")
A(rf"\node[font=\tiny,iiGreen,anchor=south] at (axis cs:{LCSX:.2f},0.09) {{LCS}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- Cp / Cpk
Cp = (LSE - LIE) / (6 * SIGMA)
Cpk = min((LSE - XBB), (XBB - LIE)) / (3 * SIGMA)
A(r"\newcommand{\cpkgeomfig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=11.0cm,height=4.4cm,xlabel={{peso escurrido (g)}},"
  rf"grid=major,xmin=462,xmax=500,ymin=0,ymax=0.135,domain=462:500,samples=200,"
  rf"tick label style={{font=\tiny}},label style={{font=\scriptsize}},ytick=\empty,"
  rf"legend style={{font=\tiny,at={{(0.02,0.97)}},anchor=north west}}]")
A(rf"\addplot[iiBlue,thick,name path=act]{{exp(-((x-{XBB:.2f})^2)/(2*{SIGMA:.3f}^2))"
  rf"/({SIGMA:.3f}*sqrt(2*pi))}};"
  rf"\addlegendentry{{actual: $\mu={num(XBB,1)}$, $C_{{pk}}={num(Cpk)}$}}")
A(rf"\addplot[iiGreen,thick,densely dashed]{{exp(-((x-480)^2)/(2*{SIGMA:.3f}^2))"
  rf"/({SIGMA:.3f}*sqrt(2*pi))}};"
  rf"\addlegendentry{{recentrado: $\mu=480$, $C_{{pk}}={num(Cp)}$}}")
A(rf"\path[name path=cero] (axis cs:462,0) -- (axis cs:500,0);")
A(rf"\addplot[iiRed!45] fill between[of=act and cero,soft clip={{domain=495:500}}];")
A(rf"\draw[iiRed,very thick] (axis cs:{LIE},0) -- (axis cs:{LIE},0.125)"
  rf" node[above,font=\tiny]{{LIE}};")
A(rf"\draw[iiRed,very thick] (axis cs:{LSE},0) -- (axis cs:{LSE},0.125)"
  rf" node[above,font=\tiny]{{LSE}};")
A(rf"\draw[<->,iiDark,thick] (axis cs:{LIE},0.118) -- (axis cs:{LSE},0.118)"
  rf" node[midway,fill=white,inner sep=1pt,font=\tiny]"
  rf"{{tolerancia = 30 g \; ; \; $6\sigma$ = {num(6*SIGMA,1)} g $\Rightarrow C_p={num(Cp)}$}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- carta p
pb = atr.n_defectuosas.sum() / atr.n_inspeccionadas.sum()
atr = atr.assign(p=atr.n_defectuosas / atr.n_inspeccionadas)
atr = atr.assign(
    lcs=pb + 3 * np.sqrt(pb * (1 - pb) / atr.n_inspeccionadas),
    lci=(pb - 3 * np.sqrt(pb * (1 - pb) / atr.n_inspeccionadas)).clip(lower=0))
pin = atr[atr.p <= atr.lcs]
pout = atr[atr.p > atr.lcs]
A(r"\newcommand{\cartapfig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=11.4cm,height=4.7cm,xlabel={{lote}},"
  rf"ylabel={{fracci\'on defectuosa $p$}},grid=major,xmin=0,xmax=25,ymin=0,ymax=0.105,"
  rf"xtick={{4,8,12,16,20,24}},tick label style={{font=\tiny}},"
  rf"label style={{font=\scriptsize}},legend style={{font=\tiny,"
  rf"at={{(0.02,0.97)}},anchor=north west,legend columns=2}}]")
A(rf"\addplot[iiBlue,thick,mark=*,mark size=1.2pt] coordinates "
  rf"{{{' '.join(f'({int(r.lote)},{r.p:.4f})' for _, r in atr.iterrows())}}};"
  r"\addlegendentry{$p_i$}")
A(rf"\addplot[iiRed,densely dashed,thick,const plot mark left] coordinates "
  rf"{{{' '.join(f'({int(r.lote)},{r.lcs:.4f})' for _, r in atr.iterrows())}}};"
  r"\addlegendentry{LCS variable ($n_i$ distinto)}")
A(rf"\addplot[only marks,mark=*,mark size=2.6pt,iiRed,forget plot] coordinates "
  rf"{{{' '.join(f'({int(r.lote)},{r.p:.4f})' for _, r in pout.iterrows())}}};")
A(rf"\draw[iiDark,thick] (axis cs:0,{pb:.4f}) -- (axis cs:25,{pb:.4f});")
A(rf"\node[font=\tiny,iiDark,anchor=west,fill=white,inner sep=1pt] at "
  rf"(axis cs:0.3,{pb+0.006:.4f}) {{$\bar p$ = {num(pb,4)}}};")
for _, r in pout.iterrows():
    A(rf"\node[font=\tiny,iiRed,anchor=south] at (axis cs:{int(r.lote)},{r.p+0.004:.4f})"
      rf" {{lote {int(r.lote)}}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- Pareto
par = atr.groupby("causa_principal")["n_defectuosas"].sum().sort_values(ascending=False)
par = par[par > 0]
tot = par.sum()
acum = (par.cumsum() / tot * 100)
etq = {"Sello defectuoso": "Sello", "Peso fuera de rango": "Peso",
       "Rotulado": "Rotulado", "Abolladura": "Abolladura",
       "Cuerpo extranio": "Cuerpo ext."}
xt = ",".join(str(i) for i in range(1, len(par) + 1))
xl = ",".join(etq.get(c, c) for c in par.index)
A(r"\newcommand{\paretofig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=8.4cm,height=4.5cm,ybar,bar width=15pt,"
  rf"ylabel={{unidades defectuosas}},grid=major,ymin=0,ymax={par.max()*1.25:.0f},"
  rf"xtick={{{xt}}},xticklabels={{{xl}}},x tick label style={{font=\tiny,rotate=25,"
  rf"anchor=east}},tick label style={{font=\tiny}},label style={{font=\scriptsize}},"
  rf"axis y line*=left,xmin=0.4,xmax={len(par)+0.6}]")
A(rf"\addplot[fill=iiBlue,draw=iiBlue] coordinates "
  rf"{{{' '.join(f'({i},{v})' for i, v in enumerate(par.values, start=1))}}};")
A(r"\end{axis}")
A(rf"\begin{{axis}}[width=8.4cm,height=4.5cm,ymin=0,ymax=105,axis y line*=right,"
  rf"axis x line=none,ylabel={{\% acumulado}},xmin=0.4,xmax={len(par)+0.6},"
  rf"tick label style={{font=\tiny}},label style={{font=\scriptsize}},"
  rf"ytick={{0,20,40,60,80,100}}]")
A(rf"\addplot[iiOrange,very thick,mark=*,mark size=1.6pt] coordinates "
  rf"{{{' '.join(f'({i},{v:.1f})' for i, v in enumerate(acum.values, start=1))}}};")
A(rf"\draw[iiRed,densely dashed] (axis cs:0.4,80) -- (axis cs:{len(par)+0.6},80);")
A(rf"\node[font=\tiny,iiRed,anchor=west] at (axis cs:0.5,85) {{80\,\%}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- curva OC
def oc_pts(n, c):
    ps = np.arange(0, 0.1401, 0.002)
    return " ".join(f"({p:.4f},{stats.binom.cdf(c, n, p):.4f})" for p in ps)


A(r"\newcommand{\ocfig}{%")
A(r"\begin{tikzpicture}")
A(r"\begin{axis}[width=9.2cm,height=5.0cm,xlabel={fracci\'on defectuosa del lote $p$ (\%)},"
  r"ylabel={$P_a$},grid=major,xmin=0,xmax=0.14,ymin=0,ymax=1.02,"
  r"tick label style={font=\tiny},label style={font=\scriptsize},"
  r"xticklabel={\pgfmathparse{\tick*100}\pgfmathprintnumber[precision=0]{\pgfmathresult}}]")
A(rf"\addplot[iiBlue,very thick] coordinates {{{oc_pts(80, 2)}}};")
A(r"\draw[iiGreen,densely dashed,thick] (axis cs:0.01,0) -- (axis cs:0.01,0.9534);")
A(r"\draw[iiGreen,densely dashed,thick] (axis cs:0,0.9534) -- (axis cs:0.01,0.9534);")
A(r"\node[font=\tiny,iiGreen,anchor=west] at (axis cs:0.012,0.955) "
  r"{AQL $=1$\,\%: $P_a=0{,}953$ \ ($\alpha=0{,}047$)};")
A(r"\draw[iiRed,densely dashed,thick] (axis cs:0.0652,0) -- (axis cs:0.0652,0.10);")
A(r"\draw[iiRed,densely dashed,thick] (axis cs:0,0.10) -- (axis cs:0.0652,0.10);")
A(r"\node[font=\tiny,iiRed,anchor=west] at (axis cs:0.068,0.16) "
  r"{LTPD $=6{,}5$\,\%: $\beta=0{,}10$};")
A(r"\node[font=\tiny,iiDark,anchor=north east] at (axis cs:0.138,0.98) "
  r"{plan simple $n=80$, $c=2$};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

A(r"\newcommand{\ocncfig}{%")
A(r"\begin{tikzpicture}")
A(r"\begin{axis}[width=9.2cm,height=4.8cm,xlabel={$p$ (\%)},ylabel={$P_a$},grid=major,"
  r"xmin=0,xmax=0.14,ymin=0,ymax=1.02,tick label style={font=\tiny},"
  r"label style={font=\scriptsize},legend style={font=\tiny,at={(0.98,0.97)},"
  r"anchor=north east},xticklabel={\pgfmathparse{\tick*100}"
  r"\pgfmathprintnumber[precision=0]{\pgfmathresult}}]")
for (n, c, col, sty) in [(80, 2, "iiBlue", "very thick"),
                         (160, 4, "iiOrange", "very thick,densely dashed"),
                         (40, 1, "iiGreen", "very thick,densely dotted")]:
    A(rf"\addplot[{col},{sty}] coordinates {{{oc_pts(n, c)}}};"
      rf"\addlegendentry{{$n={n}$, $c={c}$}}")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- AOQ
ps = np.arange(0.0005, 0.1401, 0.0001)
aoq = ps * stats.binom.cdf(2, 80, ps) * (2000 - 80) / 2000
A(r"\newcommand{\aoqfig}{%")
A(r"\begin{tikzpicture}")
A(r"\begin{axis}[width=8.6cm,height=4.2cm,xlabel={$p$ entrante (\%)},"
  r"ylabel={AOQ saliente (\%)},grid=major,xmin=0,xmax=0.14,ymin=0,"
  r"tick label style={font=\tiny},label style={font=\scriptsize},"
  r"xticklabel={\pgfmathparse{\tick*100}\pgfmathprintnumber[precision=0]{\pgfmathresult}},"
  r"yticklabel={\pgfmathparse{\tick*100}\pgfmathprintnumber[precision=1]{\pgfmathresult}}]")
A(rf"\addplot[iiPurple,very thick] coordinates "
  rf"{{{' '.join(f'({p:.4f},{a:.5f})' for p, a in zip(ps, aoq))}}};")
imax = int(np.argmax(aoq))
A(rf"\draw[iiRed,densely dashed] (axis cs:0,{aoq[imax]:.5f}) -- "
  rf"(axis cs:{ps[imax]:.4f},{aoq[imax]:.5f});")
A(rf"\node[font=\tiny,iiRed,anchor=south west] at "
  rf"(axis cs:{ps[imax]:.4f},{aoq[imax]:.5f}) "
  rf"{{AOQL $={num(100*aoq[imax],2)}$\,\% en $p={num(100*ps[imax],1)}$\,\%}};")
A(r"\end{axis}\end{tikzpicture}}")
A("")

# ---------------------------------------------------------------- R&R
med_op = msa.groupby("operador")["medicion_g"].mean()
piv = msa.pivot_table(index=["pieza", "operador"], columns="repeticion", values="medicion_g")
Rbb = (piv[1] - piv[2]).abs().mean()
EV = Rbb * 0.8862
Xdiff = med_op.max() - med_op.min()
AV = np.sqrt(max((Xdiff * 0.5231) ** 2 - EV ** 2 / 20, 0))
GRR = np.sqrt(EV ** 2 + AV ** 2)
PV = (msa.groupby("pieza")["medicion_g"].mean().max()
      - msa.groupby("pieza")["medicion_g"].mean().min()) * 0.3146
TV = np.sqrt(GRR ** 2 + PV ** 2)
A(r"\newcommand{\varianzafig}{%")
A(r"\begin{tikzpicture}")
A(rf"\begin{{axis}}[width=8.8cm,height=3.5cm,xbar stacked,bar width=15pt,"
  rf"xmin=0,xmax={TV**2*1.08:.2f},ytick={{0,1}},yticklabels={{"
  rf"{{lo que \emph{{quiero}} ver}},{{lo que la carta \emph{{muestra}}}}}},"
  rf"y tick label style={{font=\tiny,align=right}},tick label style={{font=\tiny}},"
  rf"xlabel={{varianza (g$^2$)}},label style={{font=\scriptsize}},"
  rf"legend style={{font=\tiny,at={{(0.5,-0.42)}},anchor=north,legend columns=2}}]")
A(rf"\addplot[fill=iiBlue!35,draw=iiBlue] coordinates "
  rf"{{({PV**2:.3f},0) ({PV**2:.3f},1)}};" r"\addlegendentry{proceso (PV$^2$)}")
A(rf"\addplot[fill=iiRed!35,draw=iiRed] coordinates "
  rf"{{(0,0) ({GRR**2:.3f},1)}};" r"\addlegendentry{medici\'on (GRR$^2$)}")
A(r"\end{axis}\end{tikzpicture}}")
A("")

Path(D.parent / "figuras-datos.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"figuras-datos.tex escrito: {len(L)} lineas, 10 macros.")
print(f"  Xbb={XBB:.3f} Rb={RB:.3f} LCSX={LCSX:.3f} LCIX={LCIX:.3f} sigma={SIGMA:.3f}")
print(f"  Cp={Cp:.3f} Cpk={Cpk:.3f}  pbar={pb:.4f}")
print(f"  EV={EV:.3f} AV={AV:.3f} GRR={GRR:.3f} PV={PV:.3f} TV={TV:.3f}")
