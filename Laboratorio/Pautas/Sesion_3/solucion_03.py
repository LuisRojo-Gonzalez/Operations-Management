"""Solución docente de Gestión de Operaciones. Profesor: Manuel López."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
datos = base / "datos"
salida = base / "resultados"
salida.mkdir(exist_ok=True)
parametros = json.loads((datos / "parametros.json").read_text(encoding="utf-8"))

def tabla(df, nombre):
    df.to_csv(salida / (nombre + ".csv"), index=False)
    visible = df if len(df) <= 28 else df.head(12)
    print("\n" + nombre + "\n" + visible.to_string(index=False))
    if len(df) > 28:
        print(f"Vista inicial: {len(df)} filas completas en {nombre}.csv")

def figura(fig, nombre):
    fig.tight_layout()
    fig.savefig(salida / (nombre + ".png"), dpi=150, bbox_inches="tight")
    if "__file__" not in globals():
        plt.show()
    plt.close(fig)

def resumen(obj):
    def convertir(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    (salida / "resumen.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=convertir),
        encoding="utf-8")

ajuste=pd.read_csv(datos / "ajuste.csv")
validacion=pd.read_csv(datos / "validacion.csv")
m=parametros["estacionalidad"]
w=parametros["ventana_media_movil"]

def pronosticos(y):
    y=np.asarray(y,dtype=float)
    ingenuo=np.r_[np.nan,y[:-1]]
    mm=np.full(len(y),np.nan); ie=mm.copy()
    for t in range(w,len(y)):
        mm[t]=np.mean(y[t-w:t])
    for t in range(m,len(y)):
        ie[t]=y[t-m]
    f={"ingenuo":ingenuo,"media_movil":mm,"ingenuo_estacional":ie}
    for alpha in parametros["alphas"]:
        ses=np.empty(len(y)); ses[0]=y[0]
        for t in range(1,len(y)):
            ses[t]=alpha*y[t-1]+(1-alpha)*ses[t-1]
        f[f"ses_{alpha:g}"]=ses
    return f

def metricas(y,f,indices):
    e=y[indices]-f[indices]
    return float(np.mean(np.abs(e))),float(np.mean(e))

# Selección cerrada antes de leer la prueba final.
pasado=pd.concat([ajuste,validacion],ignore_index=True)
y=pasado.demanda_unidades.to_numpy(float)
iv=np.arange(len(ajuste),len(pasado))
f=pronosticos(y)
alpha=min(parametros["alphas"],key=lambda a:(metricas(y,f[f"ses_{a:g}"],iv)[0],a))
candidatos=["ingenuo","media_movil","ingenuo_estacional",f"ses_{alpha:g}"]
seleccionado=min(candidatos,key=lambda nombre:
                 (metricas(y,f[nombre],iv)[0],candidatos.index(nombre)))
seleccion={"alpha_ses":alpha,"metodo":seleccionado,
           "ultimo_periodo_seleccion":int(pasado.periodo.max())}
(salida / "seleccion_validacion.json").write_text(json.dumps(seleccion,indent=2))

# Ahora se habilita la prueba. Los bucles solo utilizan y anteriores a t.
prueba=pd.read_csv(datos / "prueba_final.csv")
df=pd.concat([pasado,prueba],ignore_index=True)
assert df.periodo.tolist()==list(range(1,29))
y=df.demanda_unidades.to_numpy(float); f=pronosticos(y)
it=np.arange(len(pasado),len(df))
filas=[]
for nombre,pred in f.items():
    a,b=metricas(y,pred,iv); c,d=metricas(y,pred,it)
    filas.append({"metodo":nombre,"MAE_validacion":a,"sesgo_validacion":b,
                  "MAE_prueba":c,"sesgo_prueba":d})
tabla(pd.DataFrame(filas),"metricas")
tabla(pd.DataFrame({"periodo":df.periodo,"demanda":y,**f}),"pronosticos")
fig,ax=plt.subplots(figsize=(9,4.5))
ax.plot(df.periodo,y,"o-",label="Demanda",color="black")
for nombre in candidatos:
    ax.plot(df.periodo,f[nombre],"--",label=nombre)
ax.axvline(20.5,color="gray"); ax.axvline(24.5,color="gray")
ax.set(xlabel="Período",ylabel="Prendas",xlim=(17.5,28.5))
ax.legend(fontsize=8,ncol=2)
figura(fig,"pronosticos")
assert seleccionado=="ingenuo_estacional"
assert metricas(y,f[seleccionado],it)==(4.0,4.0)
# Perturbar solo el futuro no cambia pronósticos previos.
y_alt=y.copy(); y_alt[25:]+=500
f_alt=pronosticos(y_alt)
for nombre in f:
    assert np.allclose(f[nombre][:26],f_alt[nombre][:26],equal_nan=True)
resumen({**seleccion,"metricas":filas})
