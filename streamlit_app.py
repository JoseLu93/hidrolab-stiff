import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Stiff & Davis (Mariposa)", layout="wide")
st.title("Stiff & Davis – Gráfica ‘Mariposa’")

# ---------- equivalentes (meq por mg) ----------
EQW = {
    "Na": 22.98976928, "Ca": 20.039, "Mg": 12.1525, "Fe": 27.9225,
    "Cl": 35.45, "HCO3": 61.016, "SO4": 48.03, "CO3": 30.004
}

def normalize(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    df = df.copy()
    df["Ion"] = df["Ion"].astype(str).str.strip()
    df["Group"] = df["Group"].astype(str).str.lower().str.strip()
    df["Conc"] = pd.to_numeric(df["Conc"], errors="coerce").fillna(0)
    rows = []
    for _, r in df.iterrows():
        ion, grp, c = r["Ion"], r["Group"], r["Conc"]
        if unit == "mg/L":
            mg, meq = c, (c / EQW[ion]) if ion in EQW else np.nan
        else:
            meq, mg = c, (c * EQW[ion]) if ion in EQW else np.nan
        rows.append({"Ion": ion, "Group": grp, "mgL": mg, "meqL": meq})
    return pd.DataFrame(rows)

# ---------- mariposa ----------
CAT_ORDER = ["Na","Ca","Mg","Fe"]
ANI_ORDER = ["Cl","HCO3","SO4","CO3"]

def stiff_plot(df: pd.DataFrame, title: str) -> go.Figure:
    # separar y ordenar
    cat = df[(df["Group"]=="cation") & (df["Ion"].isin(CAT_ORDER))].copy()
    ani = df[(df["Group"]=="anion")  & (df["Ion"].isin(ANI_ORDER))].copy()
    cat["Ion"] = pd.Categorical(cat["Ion"], categories=CAT_ORDER, ordered=True)
    ani["Ion"] = pd.Categorical(ani["Ion"], categories=ANI_ORDER, ordered=True)
    cat, ani = cat.sort_values("Ion"), ani.sort_values("Ion")

    # posiciones Y fijas (alineadas arriba→abajo)
    y_left  = {"Na":8, "Ca":7, "Mg":6, "Fe":5}
    y_right = {"Cl":8, "HCO3":7, "SO4":6, "CO3":5}
    y_cat = np.array([y_left[i]  for i in cat["Ion"]], dtype=float)
    y_ani = np.array([y_right[i] for i in ani["Ion"]], dtype=float)
    y_center = 6.5  # entre Fe (5) y Cl (8)

    # X = ±|log10(meq/L)|; si meqL==0 → x=0 (centro)
    meq_cat = cat["meqL"].to_numpy()
    meq_ani = ani["meqL"].to_numpy()
    dist_cat = np.where(meq_cat > 0, np.abs(np.log10(meq_cat)), 0.0)
    dist_ani = np.where(meq_ani > 0, np.abs(np.log10(meq_ani)), 0.0)
    x_cat = -dist_cat
    x_ani =  dist_ani

    # polilínea unida en el centro + hover correcto
    x_line = list(x_cat) + [0.0] + list(x_ani)
    y_line = list(y_cat) + [y_center] + list(y_ani)
    custom = list(zip(cat["Ion"].tolist(), meq_cat.tolist())) + [("", None)] + \
             list(zip(ani["Ion"].tolist(), meq_ani.tolist()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines+markers",
        line=dict(width=3), marker=dict(size=7),
        customdata=custom,
        hovertemplate="%{customdata[0]}: %{customdata[1]:.4g} meq/L<extra></extra>"
    ))

    # eje X: etiquetas en meq/L reales
    tickvals = list(range(-4,5))  # -4..4
    ticktext = []
    for v in tickvals:
        if v == 0:
            ticktext.append("0")  # centro
        else:
            ticktext.append(f"{10**abs(v):g}" if v > 0 else f"{10**abs(v):g}")

    # línea central
    fig.add_vline(x=0, line_width=2, line_color="#222")

    # y: solo cationes visbles a la izq; aniones como anotaciones a la derecha
    y_ticks = [8,7,6,5]
    y_left_labels = ["Na","Ca","Mg","Fe"]
    right_annos = [
        dict(xref="paper", yref="y", x=0.985, y=float(y_right[ion]),
             text=ion, showarrow=False, xanchor="right", font=dict(size=12))
        for ion in ANI_ORDER
    ]
    side_titles = [
        dict(xref="paper", yref="y", x=0.02,  y=6.5, text="<b>Cationes</b>",
             showarrow=False, xanchor="left",  font=dict(size=14)),
        dict(xref="paper", yref="y", x=0.985, y=6.5, text="<b>Aniones</b>",
             showarrow=False, xanchor="right", font=dict(size=14)),
    ]

    fig.update_layout(
        title=title, height=520, margin=dict(l=90, r=120, t=60, b=50),
        xaxis=dict(
            title="Concentración (meq/L) – décadas (distancia = |log10|)",
            tickmode="array", tickvals=tickvals,
            # mostramos 0.0001 0.001 0.01 0.1 | 0 | 10 100 1000 10000
            ticktext=[("0" if v==0 else f"{10**abs(v):g}") for v in tickvals],
            range=[-4,4], zeroline=True, zerolinewidth=2, gridcolor="#eee"
        ),
        yaxis=dict(
            title="", tickmode="array",
            tickvals=y_ticks, ticktext=y_left_labels,
            gridcolor="#eee"
        ),
        showlegend=False,
        annotations=right_annos + side_titles
    )
    return fig

# ---------- UI mínima ----------
st.subheader("1) Captura/edita datos")
unit = st.radio("Unidad de la columna Conc", ["mg/L","meq/L"], horizontal=True, index=0)

df_default = pd.DataFrame({
    "Ion":   ["Na","Ca","Mg","Fe","Cl","HCO3","SO4","CO3"],
    "Group": ["cation","cation","cation","cation","anion","anion","anion","anion"],
    "Conc":  [27713, 2600, 510, 2, 48521, 854, 60, 0]   # ejemplo: tu brine en mg/L
})
df_edit = st.data_editor(df_default, use_container_width=True, num_rows="dynamic")

st.subheader("2) Graficar")
if st.button("Graficar Mariposa"):
    dfN = normalize(df_edit, unit)
    st.dataframe(dfN[["Ion","Group","meqL"]], use_container_width=True)
    st.plotly_chart(stiff_plot(dfN, "Mariposa — muestra"), use_container_width=True)
else:
    st.info("Elige la unidad correcta, edita la tabla y pulsa **Graficar Mariposa**.")

