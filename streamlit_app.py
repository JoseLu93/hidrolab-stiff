import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hidrolab – Stiff & Davis", layout="wide")
st.title("Hidrolab – Stiff & Davis (Mariposa)")

# ------------------ Conversión mg/L <-> meq/L ------------------
EQW = {
    "Na": 22.98976928, "K": 39.0983, "Ca": 20.039, "Mg": 12.1525, "Fe": 27.9225,
    "Sr": 43.81, "Ba": 68.6635, "Cl": 35.45, "HCO3": 61.016, "CO3": 30.004,
    "SO4": 48.03, "NO3": 62.0049, "F": 18.998403163
}

def normalize(df: pd.DataFrame, unit: str = "mg/L") -> pd.DataFrame:
    df = df.copy()
    df["Ion"] = df["Ion"].astype(str).str.strip()
    df["Group"] = df["Group"].astype(str).str.lower().str.strip()
    df["Conc"] = pd.to_numeric(df["Conc"], errors="coerce").fillna(0).clip(lower=0)
    rows = []
    for _, r in df.iterrows():
        ion, grp, c = r["Ion"], r["Group"], r["Conc"]
        if unit == "mg/L":
            mg, meq = c, c / EQW.get(ion, np.nan)
        else:
            meq, mg = c, c * EQW.get(ion, np.nan)
        rows.append({"Ion": ion, "Group": grp, "mgL": mg, "meqL": meq})
    return pd.DataFrame(rows)

# ------------------ Gráfica Mariposa ------------------
CAT_ORDER = ["Na", "Ca", "Mg", "Fe"]
ANI_ORDER = ["Cl", "HCO3", "SO4", "CO3"]

def stiff_plot(df: pd.DataFrame, title: str) -> go.Figure:
    # separar y ordenar
    cat = df[(df["Group"].str.strip().str.lower() == "cation") & (df["Ion"].isin(CAT_ORDER))].copy()
    ani = df[(df["Group"].str.strip().str.lower() == "anion")  & (df["Ion"].isin(ANI_ORDER))].copy()
    cat["Ion"] = pd.Categorical(cat["Ion"], categories=CAT_ORDER, ordered=True)
    ani["Ion"] = pd.Categorical(ani["Ion"], categories=ANI_ORDER, ordered=True)
    cat, ani = cat.sort_values("Ion"), ani.sort_values("Ion")

    # posiciones Y alineadas (arriba→abajo)
    y_left_map  = {"Na": 8, "Ca": 7, "Mg": 6, "Fe": 5}
    y_right_map = {"Cl": 8, "HCO3": 7, "SO4": 6, "CO3": 5}
    y_cat = np.array([y_left_map[i]  for i in cat["Ion"]], dtype=float)
    y_ani = np.array([y_right_map[i] for i in ani["Ion"]], dtype=float)
    y_center = 6.5  # entre Fe (5) y Cl (8)

    # X = ±|log10(meq/L)|
    d_cat = np.abs(np.log10(cat["meqL"].clip(lower=1e-6))).to_numpy()
    d_ani = np.abs(np.log10(ani["meqL"].clip(lower=1e-6))).to_numpy()
    x_cat, x_ani = -d_cat, d_ani

    # polilínea unida en el centro
    x_line = list(x_cat) + [0.0] + list(x_ani)
    y_line = list(y_cat) + [y_center] + list(y_ani)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines+markers",
        line=dict(width=3), marker=dict(size=7), name="Perfil"
    ))

    # línea central
    fig.add_vline(x=0, line_width=2, line_color="#222")

    # eje Y: solo cationes visibles a la izquierda; aniones como anotaciones a la derecha
    y_ticks = [8, 7, 6, 5]
    y_left_text = ["Na", "Ca", "Mg", "Fe"]

    right_annos = [
        dict(xref="paper", yref="y", x=0.985, y=float(y_right_map[ion]),
             text=ion, showarrow=False, xanchor="right", font=dict(size=12))
        for ion in ["Cl", "HCO3", "SO4", "CO3"]
    ]

    side_titles = [
        dict(xref="paper", yref="y", x=0.02,  y=6.5,
             text="<b>Cationes</b>", showarrow=False, xanchor="left",  font=dict(size=14)),
        dict(xref="paper", yref="y", x=0.985, y=6.5,
             text="<b>Aniones</b>",  showarrow=False, xanchor="right", font=dict(size=14)),
    ]

    fig.update_layout(
        title=title, height=520, margin=dict(l=90, r=120, t=60, b=50),
        xaxis=dict(
            title="Concentración (meq/L) – décadas (distancia = |log10|)",
            tickmode="array",
            tickvals=[-4, -3, -2, -1, 0, 1, 2, 3, 4],
            ticktext=["10000", "1000", "100", "10", "1", "10", "100", "1000", "10000"],
            range=[-4, 4], zeroline=True, zerolinewidth=2, gridcolor="#eee"
        ),
        yaxis=dict(
            title="", tickmode="array",
            tickvals=y_ticks, ticktext=y_left_text,
            gridcolor="#eee"
        ),
        showlegend=False,
        annotations=right_annos + side_titles
    )
    return fig

# ------------------ UI ------------------
st.subheader("1) Captura/edita tus datos (Ion, Group, Conc)")
unit = st.radio("Unidad de la columna Conc", ["mg/L", "meq/L"], horizontal=True, index=0)

default_df = pd.DataFrame({
    "Ion":   ["Na","Ca","Mg","Fe","Cl","HCO3","SO4","CO3"],
    "Group": ["cation","cation","cation","cation","anion","anion","anion","anion"],
    "Conc":  [3500, 800, 120, 2, 3700, 200, 50, 0.1]  # ejemplo en mg/L
})
df_edit = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

st.subheader("2) Graficar")
if st.button("Graficar Mariposa"):
    dfN = normalize(df_edit, unit)

    # Debug opcional
    dbg = dfN.copy()
    dbg["|log10(meq/L)|"] = np.abs(np.log10(dbg["meqL"].clip(lower=1e-6)))
    dbg["Lado"] = np.where(dbg["Group"]=="cation", "Izquierda (Cat)", "Derecha (Ani)")
    st.markdown("**Vista previa de posiciones (meq/L → |log10| y lado):**")
    st.dataframe(dbg[["Ion","Group","meqL","|log10(meq/L)|","Lado"]], use_container_width=True)

    st.plotly_chart(stiff_plot(dfN, "Mariposa – muestra"), use_container_width=True)
    st.success("Listo. Mariposa alineada: Na–Fe a la izquierda y Cl–CO3 a la derecha.")
else:
    st.info("Tip: Edita la tabla y pulsa **Graficar Mariposa**.")

