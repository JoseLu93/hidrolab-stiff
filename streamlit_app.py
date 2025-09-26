# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------
# Conversión mg/L -> meq/L  (meq/L = mg/L * valencia / peso molecular)
# Ajusta valencias/pesos si en tu laboratorio usan otros supuestos.
MW = {
    "Na":   22.989769,   # valencia 1
    "Ca":   40.078,      # valencia 2
    "Mg":   24.305,      # valencia 2
    "Fe":   55.845,      # valencia 2
    "Cl":   35.453,      # valencia 1
    "HCO3": 61.016,      # valencia 1
    "SO4":  96.06,       # valencia 2
    "CO3":  60.008,      # valencia 2
}
VAL = {
    "Na": 1, "Ca": 2, "Mg": 2, "Fe": 2,
    "Cl": 1, "HCO3": 1, "SO4": 2, "CO3": 2,
}
FACTOR = {ion: VAL[ion]/MW[ion] for ion in MW}   # mg/L -> meq/L

# --------------------------------------------------------------------
def normalize(df_edit: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: Ion, Group, meqL (normalizada)."""
    df = df_edit.copy()
    df.columns = [c.strip().title() if c.lower()=="group" else c for c in df.columns]
    # Normalizo nombres de columnas esperadas
    # Ion, Group, Conc (si viene mg/L) o meqL (si ya viene en meq)
    lower = {c.lower(): c for c in df.columns}
    if "ion" not in lower:
        raise ValueError("Falta columna 'Ion'")
    if "group" not in lower:
        raise ValueError("Falta columna 'Group'")

    if unit == "mg/L":
        if "conc" not in lower:
            raise ValueError("Con unit='mg/L' se espera columna 'Conc'")
        df["meqL"] = df["Conc"]
        # convierte por ion (si no está en la tabla, deja NaN)
        df["meqL"] = df.apply(
            lambda r: r["Conc"]*FACTOR.get(r["Ion"], np.nan), axis=1
        )
    else:  # unit == "meq/L"
        if "meqL" not in lower:
            raise ValueError("Con unit='meq/L' se espera columna 'meqL'")
        # nada que hacer, solo aseguro tipo numérico
        df["meqL"] = pd.to_numeric(df["meqL"], errors="coerce")

    # Filtra solo iones soportados y limpia
    df = df[df["Ion"].isin(MW.keys())].copy()
    df["Group"] = df["Group"].str.strip().str.lower()
    return df[["Ion", "Group", "meqL"]]

# --------------------------------------------------------------------
CAT_ORDER = ["Na","Ca","Mg","Fe"]
ANI_ORDER = ["Cl","HCO3","SO4","CO3"]

def stiff_plot(df: pd.DataFrame, title: str) -> go.Figure:
    # --- separar y ordenar ---
    cat = df[(df["Group"]=="cation") & (df["Ion"].isin(CAT_ORDER))].copy()
    ani = df[(df["Group"]=="anion")  & (df["Ion"].isin(ANI_ORDER))].copy()
    cat["Ion"] = pd.Categorical(cat["Ion"], categories=CAT_ORDER, ordered=True)
    ani["Ion"] = pd.Categorical(ani["Ion"], categories=ANI_ORDER, ordered=True)
    cat, ani = cat.sort_values("Ion"), ani.sort_values("Ion")

    # --- posiciones Y fijas ---
    y_left  = {"Na":8, "Ca":7, "Mg":6, "Fe":5}
    y_right = {"Cl":8, "HCO3":7, "SO4":6, "CO3":5}
    y_cat = np.array([y_left[i]  for i in cat["Ion"]], dtype=float)
    y_ani = np.array([y_right[i] for i in ani["Ion"]], dtype=float)

    # ---- X = ±|log10(meq) + 1| (centro 0, 0.1 meq/L pegado al centro) ----
    meq_cat = cat["meqL"].to_numpy()
    meq_ani = ani["meqL"].to_numpy()
    dist_cat = np.where(meq_cat > 0, np.abs(np.log10(meq_cat) + 1.0), 0.0)
    dist_ani = np.where(meq_ani > 0, np.abs(np.log10(meq_ani) + 1.0), 0.0)
    x_cat = -dist_cat
    x_ani =  dist_ani

    # ---- rango dinámico en décadas coherente con esta escala ----
    meq_all = pd.concat([cat["meqL"], ani["meqL"]], ignore_index=True)
    meq_pos = meq_all[meq_all > 0]
    max_dec = int(np.ceil(np.nanmax(np.abs(np.log10(meq_pos) + 1.0)))) if not meq_pos.empty else 1

    tickvals = list(range(-max_dec, max_dec + 1))      # ... -2 -1 0 1 2 ...
    # v = 0 -> "0"; v != 0 -> 10**(|v|-1)   (1, 10, 100...; 0.1 queda pegado al centro)
    ticktext = [("0" if v == 0 else f"{10**(abs(v)-1):g}") for v in tickvals]

    fig = go.Figure()

    # --- cationes ---
    fig.add_trace(go.Scatter(
        x=x_cat, y=y_cat, mode="lines+markers",
        name="Cationes", line=dict(width=3), marker=dict(size=9),
        customdata=list(zip(cat["Ion"], meq_cat)),
        hovertemplate="%{customdata[0]}: %{customdata[1]:.4g} meq/L<extra></extra>"
    ))
    # --- aniones ---
    fig.add_trace(go.Scatter(
        x=x_ani, y=y_ani, mode="lines+markers",
        name="Aniones", line=dict(width=3), marker=dict(size=9),
        customdata=list(zip(ani["Ion"], meq_ani)),
        hovertemplate="%{customdata[0]}: %{customdata[1]:.4g} meq/L<extra></extra>"
    ))

    # línea central
    fig.add_vline(x=0, line_width=2, line_color="#222")

    # etiquetas laterales
    y_ticks = [8,7,6,5]
    left_labels = ["Na","Ca","Mg","Fe"]
    right_annos = [
        dict(xref="paper", yref="y", x=0.985, y=float(y_right[i]), text=i,
             showarrow=False, xanchor="right", font=dict(size=12)) for i in ANI_ORDER
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
            title="Concentración (meq/L) – décadas (distancia = |log10(meq)+1|)",
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            range=[-max_dec, max_dec], zeroline=True, zerolinewidth=2, gridcolor="#eee"
        ),
        yaxis=dict(
            title="", tickmode="array", tickvals=y_ticks, ticktext=left_labels, gridcolor="#eee"
        ),
        hovermode="closest",
        showlegend=False,
        annotations=right_annos + side_titles
    )
    return fig
    # ---- CONSTANTES ----
    MW  = {"Na":22.989769, "Ca":40.078, "Mg":24.305, "Fe":55.845,
       "Cl":35.453,   "HCO3":61.016, "SO4":96.06, "CO3":60.008}
    VAL = {"Na":1, "Ca":2, "Mg":2, "Fe":2, "Cl":1, "HCO3":1, "SO4":2, "CO3":2}

    FACTOR = {ion: VAL[ion]/MW[ion] for ion in MW}   # mg/L -> meq/L


# =====================  UI mínima  =====================
st.subheader("1) Captura/edita datos")
unit = st.radio("Unidad de la columna Conc", ["mg/L","meq/L"], horizontal=True, index=0)

df_default = pd.DataFrame({
    "Ion":   ["Na","Ca","Mg","Fe","Cl","HCO3","SO4","CO3"],
    "Group": ["cation","cation","cation","cation","anion","anion","anion","anion"],
    "Conc":  [27713.0, 2600.0, 510.0, 2.0, 48521.0, 854.0, 60.0, 0.0]   # ejemplo en mg/L
})
df_edit = st.data_editor(df_default, use_container_width=True, num_rows="dynamic",
                         column_config={"Conc": st.column_config.NumberColumn("Conc", help="Concentración (mg/L o meq/L, segun selección)",
                                                                             min_value=0.0, step=0.01, format="%4f"),
                                                                             "Group": st.column_config.SelectboxColumn(
                                                                                 "Group", options=["cation","anion"]),
                         },
                                                                             )

st.subheader("2) Graficar")
if st.button("Graficar Mariposa"):
    dfN = normalize(df_edit, unit)                           # <-- ahora sí existe
    st.dataframe(dfN[["Ion","Group","meqL"]], use_container_width=True)
    st.plotly_chart(stiff_plot(dfN, "Mariposa — muestra"), use_container_width=True)
else:
    st.info("Elige la unidad correcta, edita la tabla y pulsa **Graficar Mariposa**.")

