# test_pathway.py
from pathway_analyzer import PathwayAnalyzer, PathwayVisualizer

# Küçük bir örnek gen listesi
genes = ["TP53", "EGFR", "BRCA1", "PIK3CA", "KRAS"]

# Enrichr kütüphanesini doğrudan seçiyoruz (Streamlit app'te de bu yapılıyor)
pa = PathwayAnalyzer(
    libraries=["KEGG_2021_Human"],  # isterseniz: "Reactome_2016", "GO_Biological_Process_2021" vb.
    use_gseapy=True,
    organism="Human"
)

results = pa.analyze_pathways(
    features=genes,
    max_pathways=15,
    max_p_value=0.1
)

print(f"Toplam pathway sayısı: {len(results)}")
for r in results[:5]:
    print(f"- {r['pathway_name']} | p={r['p_value']:.2e} | q={r.get('p_adj', 1.0):.2e} | genes={r['feature_count']}")

# (Opsiyonel) Görselleştirmeye uygun tablo
viz = PathwayVisualizer()
df = viz.prepare_results_for_plotting(results)
try:
    import pandas as pd
    if isinstance(df, list):
        df = pd.DataFrame(df)
    print(df.head(10))
except Exception:
    pass
