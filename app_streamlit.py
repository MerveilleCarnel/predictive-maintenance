"""
app_streamlit.py
----------------
Interface web Streamlit pour la prediction de la condition de la valve.
Appelle l'API FastAPI sur http://localhost:8000
"""

import requests
import streamlit as st

# ── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Maintenance Prédictive",
    page_icon="🏭",
    layout="centered",
)

API_URL = "http://localhost:8000"

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; }
    h1 { color: #f8fafc !important; }
    .result-optimal {
        background-color: #052e1c;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-non-optimal {
        background-color: #2d0a0a;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏭 Maintenance Prédictive")
st.markdown("**Prédiction de la condition de la valve hydraulique**")
st.markdown("---")

# ── Vérification santé API ────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"✅ API connectée — Modèle : **{health['model']}** | "
               f"Cycles disponibles : **{health['total_cycles']}**")
except Exception:
    st.error("❌ API non disponible — Assurez-vous que l'API tourne sur http://localhost:8000")
    st.stop()

st.markdown("---")

# ── Formulaire de prédiction ──────────────────────────────────────────────────
st.subheader("🔍 Lancer une prédiction")

col1, col2 = st.columns([2, 1])

with col1:
    cycle_index = st.number_input(
        label="Numéro de cycle",
        min_value=0,
        max_value=health.get("total_cycles", 2204) - 1,
        value=42,
        step=1,
        help="Index du cycle de production à analyser (0 à 2204)"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Prédire", use_container_width=True, type="primary")

# ── Résultat ──────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"cycle_index": int(cycle_index)},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                is_optimal = data["valve_optimal"] == 1

                st.markdown("---")
                st.subheader("📊 Résultat")

                # ── Carte résultat ──
                if is_optimal:
                    st.markdown(f"""
                    <div class="result-optimal">
                        <h2 style="color:#10b981">✅ {data['label']}</h2>
                        <p style="color:#a7f3d0">La valve fonctionne de manière optimale (100%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-non-optimal">
                        <h2 style="color:#ef4444">⚠️ {data['label']}</h2>
                        <p style="color:#fca5a5">La valve présente une dégradation — maintenance recommandée</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Métriques ──
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Cycle analysé", f"#{data['cycle_index']}")
                with col_b:
                    st.metric("Probabilité", f"{data['probability']*100:.1f}%")
                with col_c:
                    st.metric("Confiance", data['confidence'])

                # ── Barre de probabilité ──
                st.markdown("**Probabilité que la valve soit optimale :**")
                st.progress(float(data["probability"]))

                # ── Interprétation ──
                st.markdown("---")
                st.subheader("💡 Interprétation")

                if is_optimal:
                    st.info(
                        f"Le modèle prédit avec une probabilité de **{data['probability']*100:.1f}%** "
                        f"que la valve du cycle **#{data['cycle_index']}** est dans un état optimal. "
                        f"Aucune intervention n'est nécessaire."
                    )
                else:
                    st.warning(
                        f"Le modèle prédit avec une probabilité de **{(1-data['probability'])*100:.1f}%** "
                        f"que la valve du cycle **#{data['cycle_index']}** est dégradée. "
                        f"Une inspection ou maintenance est recommandée."
                    )

            else:
                error = response.json()
                st.error(f"❌ Erreur API : {error.get('detail', 'Erreur inconnue')}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de contacter l'API. Vérifiez qu'elle tourne sur http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {str(e)}")

# ── Exploration par plage ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 Explorer une plage de cycles"):
    st.markdown("Analysez plusieurs cycles d'un coup pour détecter les anomalies.")

    col_start, col_end = st.columns(2)
    with col_start:
        start = st.number_input("Cycle de départ", min_value=0, max_value=2200, value=0, step=1)
    with col_end:
        end = st.number_input("Cycle de fin", min_value=1, max_value=2204, value=20, step=1)

    if st.button("📈 Analyser la plage", use_container_width=True):
        if start >= end:
            st.error("Le cycle de départ doit être inférieur au cycle de fin.")
        else:
            results = []
            progress = st.progress(0)
            total = end - start

            for i, idx in enumerate(range(start, end + 1)):
                try:
                    r = requests.post(
                        f"{API_URL}/predict",
                        json={"cycle_index": idx},
                        timeout=5
                    ).json()
                    results.append({
                        "Cycle"       : idx,
                        "Prédiction"  : r["label"],
                        "Probabilité" : f"{r['probability']*100:.1f}%",
                        "Confiance"   : r["confidence"],
                        "Optimal"     : "✅" if r["valve_optimal"] == 1 else "⚠️"
                    })
                except Exception:
                    pass
                progress.progress((i + 1) / (total + 1))

            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                n_optimal = sum(1 for r in results if r["Optimal"] == "✅")
                n_total   = len(results)

                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("Cycles analysés", n_total)
                with col_y:
                    st.metric("Cycles optimaux", f"{n_optimal}/{n_total} ({n_optimal/n_total*100:.0f}%)")

                st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "📊 Dataset : [UCI Hydraulic Systems](https://archive.ics.uci.edu/dataset/447) &nbsp;|&nbsp; "
    "🔗 [API Docs](http://localhost:8000/docs) &nbsp;|&nbsp; "
    "💻 [GitHub](https://github.com/MerveilleCarnel/predictive-maintenance)",
    unsafe_allow_html=True
)
