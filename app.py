import pandas as pd
import numpy as np 
import Funciones_MCF as MCF
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

UNIVERSO_ACTIVOS = {
    "Sectores":["XLK", "XLY", "XLC", "XLV", "XLP", "XLF", "XLU", "XLB", "XLRE", "XLI", "XLE"],
    "Regiones":["SPLG", "EWC", "IEUR", "EEM", "EWJ", "EWW", "EWZ"]
}

TEMP_UNIVERSO_INVERTIBLE = []

with st.sidebar:
    st.header("Configuración Activos", divider="red")

    checkbox_sectores = st.checkbox("Usar Sectores")
    if checkbox_sectores:
        TEMP_UNIVERSO_INVERTIBLE.extend(UNIVERSO_ACTIVOS["Sectores"])

    checkbox_regiones = st.checkbox("Usar Regiones")
    if checkbox_regiones:
        TEMP_UNIVERSO_INVERTIBLE.extend(UNIVERSO_ACTIVOS["Regiones"])

    UNIVERSO_INVERTIBLE = st.multiselect(
        "Universo Invertible:",
        [etf for sublist in  [UNIVERSO_ACTIVOS[key] for key in UNIVERSO_ACTIVOS] for etf in sublist],
        default=TEMP_UNIVERSO_INVERTIBLE,
    )

    col1, col2 = st.columns(2)

    fecha_inicial = col1.date_input("Fecha Inicial", datetime.today() - timedelta(days=365))
    fecha_final = col2.date_input("Fecha Final", "today")

    TIPO_ANALISIS = st.selectbox(
        "Tipo de Análisis:",
        ("Manual", "Mean-Variance", "Black-Litterman"),
    )

    col1, col2 = st.columns(2)

    if col1.button("Aplicar", width="stretch"):
        st.session_state["aplicar_btn"] = True

    if col2.button("Reset", type="primary", width="stretch"):
        for key in st.session_state.keys():
            del st.session_state[key]

st.header('Proyecto Seminario de Finanzas', divider="red")

fecha_inicial = datetime(year=fecha_inicial.year, month=fecha_inicial.month, day=fecha_inicial.day)
fecha_final = datetime(year=fecha_final.year, month=fecha_final.month, day=fecha_final.day)

if "aplicar_btn" in st.session_state:
    if st.session_state["aplicar_btn"]:
        precios_df = MCF.obtener_datos(UNIVERSO_INVERTIBLE, start=fecha_inicial, end=fecha_final)
        rendimientos_df = MCF.calcular_rendimientos(precios_df)

        precios_sp500_df = MCF.obtener_datos(["^GSPC"], start=fecha_inicial, end=fecha_final)
        precios_sp500_df.rename(columns={"^GSPC":"SP500"}, inplace=True)
        rendimientos_sp500_df = MCF.calcular_rendimientos(precios_sp500_df)
        
        if TIPO_ANALISIS == "Manual":
            st.subheader('Definición del Portafolio')
            pesos_df = pd.DataFrame({"Activo":UNIVERSO_INVERTIBLE, "Peso (%)":[1/len(UNIVERSO_INVERTIBLE)]*len(UNIVERSO_INVERTIBLE)})
            pesos_df = st.data_editor(
                pesos_df, hide_index=True, disabled=["Activo"],
                column_config={
                    "Peso (%)": st.column_config.NumberColumn(format="percent", min_value=0, max_value=1)
                }
            )
            pesos_df.set_index("Activo", inplace=True)
        
        elif TIPO_ANALISIS == "Mean-Variance":
            st.subheader('Definición del Portafolio')

            rendimientos_promedio = rendimientos_df.mean()
            cov_mat = rendimientos_df.cov()

            portfolio_return = lambda x: x @ rendimientos_promedio.values
            portfolio_variance = lambda x: (x.reshape(-1,1).T @ cov_mat.values @ x.reshape(-1,1)).item()
            portfolio_volatility = lambda x: portfolio_variance(x)**(1/2)
            portfolio_sharpe = lambda x: portfolio_return(x)/portfolio_volatility(x)

            tipo_optimizacion = st.selectbox(
                "Seleccione el tipo de optimización que desea realizar:",
                ("Min Variance", "Max Sharpe", "Markowitz"),
            )

            x = np.array([1/len(rendimientos_promedio.values)]*len(rendimientos_promedio.values))
            bounds = [(0, 1) for _ in range(len(rendimientos_promedio.values))]
            if tipo_optimizacion == "Markowitz":
                target_return = st.number_input("Rendimiento Objetivo", value=0.001, format="%0.4f", step=0.0001)
                constraints = [{"type":"eq", "fun":lambda x: portfolio_return(x) - target_return},
                               {"type":"eq", "fun":lambda x: sum(x) - 1}]
                fun = portfolio_variance
            elif tipo_optimizacion == "Max Sharpe":
                constraints = [{"type":"eq", "fun":lambda x: sum(x) - 1}]
                fun = lambda x: -portfolio_sharpe(x)
            elif tipo_optimizacion == "Min Variance":
                constraints = [{"type":"eq", "fun":lambda x: sum(x) - 1}]
                fun = portfolio_variance

            res = minimize(
                fun=fun, x0=x, constraints=constraints, method="SLSQP", bounds=bounds, 
                options={"ftol":1e-12, "maxiter":2000, "eps":1e-12, "disp":False}
            )
            pesos_df = pd.DataFrame({"Activo":rendimientos_promedio.index.copy(), "Peso (%)":res.x})

            st.dataframe(
                pesos_df, hide_index=True,
                column_config={
                    "Peso (%)": st.column_config.NumberColumn(format="percent", min_value=0, max_value=1)
                }
            )
            pesos_df.set_index("Activo", inplace=True)
        
        elif TIPO_ANALISIS == "Black-Litterman":
            market_weights = {
                "XLC":0.0999,
                "XLY":0.1025,
                "XLP":0.0482,
                "XLE":0.0295,
                "XLF":0.1307,
                "XLV":0.0958,
                "XLI":0.0809,
                "XLB":0.0166,
                "XLRE":0.0187,
                "XLK":0.3535,
                "XLU":0.0237
            }

            pesos_bmk_df = pd.DataFrame({"Activo":UNIVERSO_INVERTIBLE})
            pesos_bmk_df["Peso (%)"] = pesos_bmk_df["Activo"].map(market_weights)

            st.subheader('Pesos del Benchmark')
            st.dataframe(
                pesos_bmk_df, hide_index=True,
                column_config={
                    "Peso (%)": st.column_config.NumberColumn(format="percent", min_value=0, max_value=1)
                }
            )

            if "views" not in st.session_state:
                st.session_state["views"] = {}
                st.session_state["views"]["views_idx"] = []
            if "views_cntr" not in st.session_state:
                st.session_state["views_cntr"] = 0

            col1, col2, col3, col4 = st.columns([0.1, 0.15, 0.12, 0.63])
            col1.subheader("Views")
            
            if col2.button("Agregar View Absoluto"):
                st.session_state["views"][st.session_state["views_cntr"]] = {
                    "Tipo":"Absoluto"
                }
                st.session_state["views"]["views_idx"].append(st.session_state["views_cntr"])
                st.session_state["views_cntr"] += 1

            if col3.button("Agregar View Relativo"):
                st.session_state["views"][st.session_state["views_cntr"]] = {
                    "Tipo":"Relativo"
                }
                st.session_state["views"]["views_idx"].append(st.session_state["views_cntr"])
                st.session_state["views_cntr"] += 1

            # st.write(st.session_state["views"])

            for i in st.session_state["views"]["views_idx"]:
                view_data = st.session_state["views"][i]
                if view_data["Tipo"] == "Absoluto":
                    col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
                    with col1:
                        st.markdown("#####")
                        view_data["Activo"] = st.selectbox(
                            "Activo",
                            UNIVERSO_INVERTIBLE,
                            index=(UNIVERSO_INVERTIBLE.index(view_data["Activo"]) if view_data["Activo"] != None else None) \
                                  if "Activo" in view_data else None,
                            key=f"View_Activo_{i}"
                        )

                    with col2:
                        st.markdown("#####")
                        view_data["Rendimiento"] = st.number_input(
                            "Rendimiento", 
                            value=0.05, 
                            format="%0.2f", 
                            step=0.05,
                            key=f"View_Rendimiento_{i}"
                        )

                    with col3:
                        st.markdown("##")
                        if st.button("Borrar", width="stretch", key=f"View_btn_{i}"):
                            st.session_state["views"]["views_idx"].remove(i)
                            st.rerun()

                else:
                    col1, col2, col3 ,col4 = st.columns([0.40, 0.40, 0.1, 0.1])
                    with col1:
                        st.markdown("#####")
                        view_data["Outperformers"] = st.multiselect(
                            "Outperformers:",
                            UNIVERSO_INVERTIBLE,
                            key=f"View_Overperformers_{i}",
                            default=None,
                        )

                    with col2:
                        st.markdown("#####")
                        view_data["Underperformers"] = st.multiselect(
                            "Underperformers:",
                            UNIVERSO_INVERTIBLE,
                            key=f"View_Underperformers_{i}",
                            default=None,
                        )
                    
                    with col3:
                        st.markdown("#####")
                        view_data["Rendimiento"] = st.number_input(
                            "Rendimiento", 
                            value=0.05, 
                            format="%0.2f", 
                            step=0.05,
                            key=f"View_Rendimiento_{i}"
                        )
                    
                    with col4:
                        st.markdown("##")
                        if st.button("Borrar", width="stretch", key=f"View_btn_{i}"):
                            st.session_state["views"]["views_idx"].remove(i)
                            st.rerun()


            # st.write(st.session_state["views"])
            picking_mtx = []
            for i in st.session_state["views"]["views_idx"]:
                view_data = st.session_state["views"][i]
                picking_mtx_row = [0]*len(UNIVERSO_INVERTIBLE)

                if view_data["Tipo"] == "Absoluto":
                    if view_data["Activo"] != None:
                        temp_idx = pesos_bmk_df[pesos_bmk_df["Activo"] == view_data["Activo"]].index[0]
                        picking_mtx_row[temp_idx] = 1

                elif view_data["Tipo"] == "Relativo":
                    if len(view_data["Underperformers"]) > 0 and len(view_data["Outperformers"]) > 0:
                        temp_idx = pesos_bmk_df[pesos_bmk_df["Activo"].isin(view_data["Outperformers"])].index
                        pesos_relativos = pesos_bmk_df.loc[temp_idx, "Peso (%)"]
                        pesos_relativos = pesos_relativos/pesos_relativos.sum()
                        for idx in temp_idx:
                            picking_mtx_row[idx] = pesos_relativos.loc[idx].item()

                        temp_idx = pesos_bmk_df[pesos_bmk_df["Activo"].isin(view_data["Underperformers"])].index
                        pesos_relativos = pesos_bmk_df.loc[temp_idx, "Peso (%)"]
                        pesos_relativos = -pesos_relativos/pesos_relativos.sum()
                        for idx in temp_idx:
                            picking_mtx_row[idx] = pesos_relativos.loc[idx].item()


                picking_mtx.append(picking_mtx_row)

            st.subheader("Picking Matrix")
            st.write(np.array(picking_mtx))

        # rendimientos_port_df = rendimientos_df[UNIVERSO_INVERTIBLE].values @ pesos_df.loc[UNIVERSO_INVERTIBLE, "Peso (%)"].values.reshape(-1,1)
        # rendimientos_port_df = rendimientos_port_df.flatten()
        # rendimientos_port_df = pd.DataFrame({"Rendimiento Portafolio":rendimientos_port_df})
        # rendimientos_port_df.index = rendimientos_df.index.copy()

        # valor_port_df = rendimientos_port_df.copy()
        # valor_port_df.loc[precios_df.iloc[0].name, "Rendimiento Portafolio"] = 0
        # valor_port_df.sort_index(inplace=True)
        # valor_port_df += 1
        # valor_port_df = valor_port_df.cumprod()
        # valor_port_df.rename(columns={"Rendimiento Portafolio":"Valor Portafolio"}, inplace=True)

        # drawdowns_df = (valor_port_df["Valor Portafolio"].cummax() - valor_port_df["Valor Portafolio"])/valor_port_df["Valor Portafolio"].cummax()
        # drawdowns_df = drawdowns_df.to_frame()
        # drawdowns_df.rename(columns={"Valor Portafolio":"Drawdown"}, inplace=True)
        
        # st.subheader('Análisis del Portafolio', divider="red")
        # temp_fig = px.line(valor_port_df, x=valor_port_df.index, y="Valor Portafolio")
        # temp_fig.update_traces(line_color="red")

        # backtesting_fig = make_subplots(
        #     rows=2, cols=1,
        #     shared_xaxes=True,
        #     row_heights=[0.8, 0.2],
        #     vertical_spacing=0.00
        # )

        # for trace in temp_fig.data:
        #     backtesting_fig.add_trace(trace, row=1, col=1)

        # backtesting_fig.add_trace(
        #     go.Scatter(
        #         x=drawdowns_df.index,
        #         y=drawdowns_df["Drawdown"],
        #         line=dict(color="white"),
        #         name="Drawdown",
        #         fill="tozeroy",
        #         fillcolor="rgba(255,255,255,0.2)",
        #         showlegend=True
        #     ),
        #     row=2, col=1
        # )

        # backtesting_fig.update_xaxes(showticklabels=False, row=1, col=1)
        # backtesting_fig.update_xaxes(
        #     tickfont=dict(size=26), showline=True, linecolor="white", linewidth=2, gridcolor='rgba(255,255,255,0.2)', mirror=True
        # )
        # backtesting_fig.update_yaxes(
        #     tickfont=dict(size=26), showline=True, linecolor="white", linewidth=2, gridcolor='rgba(255,255,255,0.2)', mirror=True
        # )
        # backtesting_fig.update_yaxes(tickformat=".0%", row=2, col=1)
        # backtesting_fig.update_layout(
        #     margin=dict(t=50, b=35, l=70, r=2), plot_bgcolor="rgba(0,0,0,0)", height=800, title="Valor del Portafolio y Drawdown",
        #     legend=dict(x=0.99, y=0.20, xanchor="right", yanchor="top", bgcolor="rgba(0,0,0,0)", font=dict(size=20, color="white"))
        # )
        # st.plotly_chart(backtesting_fig, theme=None)

        # cols = st.columns([0.4, 0.06, 0.18, 0.18, 0.18])

        # rendimientos_fig = px.histogram(rendimientos_port_df, x="Rendimiento Portafolio")
        # rendimientos_fig.update_traces(marker_line_width=2, marker_line_color="red", marker_color="rgba(255,0,0,0.2)")
        # rendimientos_fig.update_yaxes(
        #     title=None, showticklabels=False, linecolor="white", linewidth=2, gridcolor='rgba(255,255,255,0.2)', mirror=True
        # )
        # rendimientos_fig.update_xaxes(
        #     title=None, tickfont=dict(size=26), showline=True, linecolor="white", linewidth=2, gridcolor='rgba(255,255,255,0.2)', mirror=True
        # )
        # rendimientos_fig.update_layout(
        #     margin=dict(t=50, b=35, l=70, r=2), plot_bgcolor="rgba(0,0,0,0)", title="Rendimientos Diarios Portafolio", height=525
        # )

        # cols[0].markdown("######")
        # cols[0].plotly_chart(rendimientos_fig, theme=None)

        # metricas_portafolio_dict = {}

        # rendimientos_vs_sp500_df = pd.merge(rendimientos_port_df, rendimientos_sp500_df, left_index=True, right_index=True, how="left")
        # X = rendimientos_vs_sp500_df[["SP500"]].values
        # y = rendimientos_vs_sp500_df["Rendimiento Portafolio"]
        # reg = LinearRegression().fit(X, y)

        # metricas_portafolio_dict["BETA"] = reg.coef_
        # metricas_portafolio_dict["Media"] = rendimientos_port_df["Rendimiento Portafolio"].mean()
        # metricas_portafolio_dict["Volatilidad"] = rendimientos_port_df["Rendimiento Portafolio"].std()
        # metricas_portafolio_dict["Max Drawdown"] = drawdowns_df["Drawdown"].max()
        # metricas_portafolio_dict["Kurtosis"] = rendimientos_port_df["Rendimiento Portafolio"].kurtosis()
        # metricas_portafolio_dict["Sesgo"] = rendimientos_port_df["Rendimiento Portafolio"].skew()
        # metricas_portafolio_dict["VaR_95"] = rendimientos_port_df["Rendimiento Portafolio"].quantile(0.05)
        # metricas_portafolio_dict["cVaR_95"] = rendimientos_port_df["Rendimiento Portafolio"][rendimientos_port_df["Rendimiento Portafolio"] <= \
        #                                       metricas_portafolio_dict["VaR_95"]].mean()
        # metricas_portafolio_dict["Sharpe"] = metricas_portafolio_dict["Media"]/metricas_portafolio_dict["Volatilidad"]
        # metricas_portafolio_dict["Sortino"] = metricas_portafolio_dict["Media"] / \
        #                                       rendimientos_port_df["Rendimiento Portafolio"][rendimientos_port_df["Rendimiento Portafolio"] <= 0].std()

        # for i in range(2, 5):
        #     if i != 3:
        #         cols[i].markdown('#')
        #         cols[i].markdown('##')
        #         cols[i].markdown("#")
        #     else:
        #         cols[i].markdown("#")
        #         cols[i].metric("BETA", np.round(metricas_portafolio_dict["BETA"], decimals=2), border=True, width="content")
        #         cols[i].markdown("######")

        # for i, key in enumerate(metricas_portafolio_dict):
        #     if key != "BETA":
        #         cols[(i - 1) % 3 + 2].metric(key, np.round(metricas_portafolio_dict[key], decimals=4))











