import io
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Hashtag Miner", layout="wide")

# --- GLOBAL DARK THEME & NAVBAR ---
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top left, #0f172a 0%, #1e293b 90%) !important;
    color: #e2e8f0;
}

header, header * {
    background-color: #1e293b !important; /* dark background */
    color: #e2e8f0 !important;           /* light text/icons */
}

header [data-testid="stSidebarNav"] svg {
    fill: #e2e8f0 !important;
}

header .css-18ni7ap.e8zbici2 {
    color: #60a5fa !important;
}


.navbar {
    background-color: #1e293b;
    padding: 12px 28px;
    border-bottom: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.navbar-title {
    color: #60a5fa;
    font-size: 1.4rem;
    font-weight: 600;
}
.navbar-subtitle {
    color: #9ca3af;
    font-size: 0.9rem;
}


.card {
    background: #1e293b;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}

h1, h2, h3, h4, h5, label {
    color: #60a5fa !important;
}
p, span, div {
    color: #e2e8f0 !important;
}

input, textarea, select {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
}

[data-testid="stFileUploader"] section {
    background-color: #1e293b !important;
    border: 1px dashed #475569 !important;
    color: #e2e8f0 !important;
}
[data-testid="stFileUploader"] label div {
    color: #e2e8f0 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #334155 !important;
    color: #e2e8f0 !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #475569 !important;
}

.stSlider > div[data-baseweb="slider"] {
    color: #60a5fa;
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1;
    background-color: transparent;
}
.stTabs [aria-selected="true"] {
    color: #60a5fa;
    border-bottom: 2px solid #60a5fa;
}

[data-testid="stSidebar"] {
    background-color: #1e293b !important;
}

[data-testid="stDataFrame"] {
    background-color: #1e293b !important;
    border-radius: 12px;
    color: #e2e8f0 !important;
}
thead tr th {
    background-color: #0f172a !important;
    color: #60a5fa !important;
}
tbody tr:nth-child(odd) {
    background-color: #1e293b !important;
}
tbody tr:nth-child(even) {
    background-color: #0f172a !important;
}
tbody tr:hover {
    background-color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)

# NAVBAR 
st.markdown("""
<div class="navbar">
    <div>
        <span class="navbar-title">📊 Hashtag Miner</span><br>
        <span class="navbar-subtitle">Apriori-based Hashtag Recommendation</span>
    </div>
</div>

<!-- Add spacing below navbar -->
<div style="margin-bottom: 25px;"></div>
""", unsafe_allow_html=True)


# Input Card 
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2 = st.columns([1.3, 2])
with c1:
    uploaded_file = st.file_uploader("📂 Upload CSV or JSON file", type=["csv", "json"])
with c2:
    hashtag_input = st.text_input("Enter hashtags (comma-separated, e.g. #ai, #foodie)", value="")

c3, c4, c5, c6 = st.columns([1.3, 1.3, 1.3, 2])
with c3:
    min_support = st.slider("Min Support", 0.01, 0.5, 0.1, 0.01)
with c4:
    min_confidence = st.slider("Min Confidence", 0.1, 0.95, 0.5, 0.05)
with c5:
    top_k = st.number_input("Top-K Results", 1, 50, 10)
with c6:
    include_lift = st.checkbox("Include Lift ≥ 1", True)
st.markdown("</div>", unsafe_allow_html=True)


# Main Logic
if uploaded_file is not None and hashtag_input.strip():
    hashtags_entered = [h.strip().lower() for h in hashtag_input.split(",") if h.strip()]

    @st.cache_data
    def load_transactions(data_bytes, filename):
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data_bytes))
            if "hashtags" not in df.columns:
                st.error("CSV must contain a 'hashtags' column.")
                return []
            txs = []
            for tags_str in df["hashtags"].astype(str):
                tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
                txs.append(sorted(set(tags)))
            return txs
        elif filename.endswith(".json"):
            data = json.load(io.BytesIO(data_bytes))
            txs = []
            for entry in data:
                tags = entry.get("hashtags", [])
                cleaned = [t.strip().lower() for t in tags if isinstance(t, str)]
                txs.append(sorted(set(cleaned)))
            return txs
        return []

    transactions = load_transactions(uploaded_file.read(), uploaded_file.name)
    if not transactions:
        st.warning("No valid transactions found.")
    else:
        # Tabs 
        tab1, tab2, tab3, tab4 = st.tabs(["🧾 Data Preview", "🔍 Frequent Itemsets", "🔗 Association Rules", "📈 Visualizations"])

        with tab1:
            st.markdown("### Sample Transactions")
            for tx in transactions[:5]:
                st.write(", ".join(tx))
            all_items = sorted(set(item for tx in transactions for item in tx))
            encoded_vals = [{item: (item in tx) for item in all_items} for tx in transactions]
            df = pd.DataFrame(encoded_vals)
            st.dataframe(df.head())

        with tab2:
            frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
            filtered_itemsets = frequent_itemsets[
                frequent_itemsets['itemsets'].apply(lambda x: all(h in [xx.lower() for xx in x] for h in hashtags_entered))
            ]
            st.markdown("### Frequent Combinations with Selected Hashtag(s)")
            if filtered_itemsets.empty:
                st.info("No frequent itemsets with your hashtag(s). Try adjusting support.")
            else:
                st.dataframe(filtered_itemsets.sort_values(by='support', ascending=False).head(top_k))

        with tab3:
            try:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            except TypeError:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            if include_lift:
                rules = rules[rules['lift'] >= 1]
            rules['antecedents'] = rules['antecedents'].apply(lambda x: [xx.lower() for xx in x])
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            selected_rules = rules[
                rules['antecedents'].apply(lambda x: all(h in x for h in hashtags_entered))
            ]
            st.markdown("### Association Rules with Hashtag(s) as Antecedent")
            if selected_rules.empty:
                st.info("No rules found for your hashtag(s) at these settings.")
            else:
                st.dataframe(
                    selected_rules[['antecedents', 'consequents', 'confidence', 'support', 'lift']]
                    .sort_values(by=['confidence', 'lift'], ascending=False)
                    .head(top_k).reset_index(drop=True)
                )
                
            # Association Rules Calculation
            try:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            except TypeError:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            if include_lift:
                rules = rules[rules['lift'] >= 1]
            rules['antecedents'] = rules['antecedents'].apply(lambda x: [xx.lower() for xx in x])
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            selected_rules = rules[
                rules['antecedents'].apply(lambda x: all(h in x for h in hashtags_entered))
            ]

            if selected_rules.empty:
                st.info("No rules found for your hashtag(s) at these settings.")
            else:
                for idx, row in selected_rules.head(top_k).iterrows():
                    antecedents = ", ".join(row['antecedents'])
                    consequents = row['consequents']
                    support_val = row['support'] * 100
                    confidence_val = row['confidence'] * 100
                    lift_val = row['lift']

                    st.markdown(
                        f"<b style='color:#a3e635;'>Rule:</b> <span style='color:#38bdf8;'>If</span> <b>[{antecedents}]</b> <span style='color:#38bdf8;'>then</span> <b>[{consequents}]</b>", 
                        unsafe_allow_html=True
                    )
                    with st.expander("Show explanation / insight"):
                        st.write(
                            f"If a post uses **{antecedents}**, then it is likely to also use **{consequents}**.\n"
                            f"\n- **Support:** {support_val:.2f}% of all posts"
                            f"\n- **Confidence:** In {confidence_val:.2f}% of those cases, [{consequents}] also appeared"
                            f"\n- **Lift:** {lift_val:.2f} (how much more likely than random chance)"
                        )

            # Custom Hashtag Recommendations 
            input_set = set(hashtags_entered)
            recommendations = []

            # Go through each rule and recommend new tags from the consequents
            for _, row in selected_rules.iterrows():
                for rec_tag in row['consequents'].split(','):
                    rec_tag = rec_tag.strip()
                    if rec_tag and rec_tag not in input_set:
                        recommendations.append({
                            "Recommended Hashtag": rec_tag,
                            "Confidence (%)": row['confidence'] * 100,
                            "Lift": row['lift'],
                            "Support (%)": row['support'] * 100
                        })

            # Remove duplicates and sort by confidence and lift
            recommendations_df = pd.DataFrame(recommendations)
            if not recommendations_df.empty:
                recommendations_df = (
                    recommendations_df
                    .sort_values(by=['Confidence (%)', 'Lift'], ascending=False)
                    .drop_duplicates('Recommended Hashtag')
                    .head(top_k)
                )
                st.markdown("### Top Hashtag Add-on Recommendations")
                st.dataframe(recommendations_df)
                for _, row in recommendations_df.iterrows():
                    st.write(
                        f"**Add `{row['Recommended Hashtag']}` :** "
                        f"{row['Confidence (%)']:.2f}% confidence, "
                        f"Lift {row['Lift']:.2f}, "
                        f"Support {row['Support (%)']:.2f}%"
                    )
            else:
                st.markdown("### Top Hashtag Add-on Recommendations")
                st.info("No recommended hashtags found for your input.")


        with tab4:
            st.markdown("### Visual Insights")

            if frequent_itemsets.empty and selected_rules.empty:
                st.info("No visualizations available. Run Apriori first.")
            else:
                (subtab1,) = st.tabs(["🔗 Association Network"])


                # Subtab 1: Network Graph
                with subtab1:
                    if not selected_rules.empty:
                        G = nx.DiGraph()
                        for _, row in selected_rules.iterrows():
                            for a in row['antecedents']:
                                G.add_edge(a, row['consequents'], weight=row['confidence'])

                        plt.figure(figsize=(8, 6)) 
                        pos = nx.spring_layout(G, k=0.6, iterations=20)
                        nx.draw(
                            G,
                            pos,
                            with_labels=True,
                            node_color="#60a5fa",
                            node_size=850,  # smaller node bubbles
                            edge_color="#94a3b8",
                            font_size=8.5,
                            font_weight='bold'
                        )
                        plt.tight_layout()
                        st.pyplot(plt)
                    else:
                        st.warning("No association rules available for network visualization.")



else:
    st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown("⬆️ <span style='color:#60a5fa'>Upload your dataset and enter hashtags above to get started.</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


