import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from Bio import Entrez, SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------- CONFIG ---------------- #
Entrez.email = "your.email@example.com"  # Replace with your email
parker_hydro_scale = {
    'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
    'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
    'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
    'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
}

ORGANISMS = [
    "Homo sapiens", "Plasmodium falciparum", "Toxocara canis", "Ascaris suum",
    "Schistosoma mansoni", "Fasciola hepatica", "Other..."
]

# ---------------- Styles ---------------- #
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #fafafa; color: #222; }
    .main > div { max-width: 900px; margin: auto; padding: 20px 15px; }
    .stButton>button {
        background-color: #1976d2; color: white; border-radius: 5px; padding: 8px 16px;
        font-weight: 600; border: none; transition: background-color 0.3s ease;
    }
    .stButton>button:hover {background-color: #115293;}
    .card {
        background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-radius: 10px; padding: 20px; margin-bottom: 20px;
    }
    .peptide-details {
        display: flex; flex-wrap: nowrap; gap: 20px; font-weight: 600;
        margin-bottom: 16px; flex-wrap: wrap;
    }
    .peptide-details > div { white-space: nowrap; }
    .explanation-text {
      font-style: italic; color: #555; margin-bottom: 8px;
    }
    .color-square {
      width: 20px;
      height: 20px;
      border-radius: 3px;
      display: inline-block;
      margin-right: 6px;
      vertical-align: middle;
    }
    @media (prefers-color-scheme: dark) {
        body { background-color: #121212; color: #eee; }
        .card { background: #1e1e1e; box-shadow: 0 1px 5px rgba(255,255,255,0.1); }
        .stButton>button { background-color: #0d47a1; }
        .stButton>button:hover { background-color: #1565c0; }
        .explanation-text { color: #ccc; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Functions ---------------- #
def fetch_protein_sequence(organism, protein_term):
    try:
        query = f"{protein_term} AND {organism}[Organism]"
        ids = Entrez.read(Entrez.esearch(db="protein", term=query, retmax=1))["IdList"]
        if not ids:
            return {"error": f"No protein found for '{protein_term}' in '{organism}'"}
        uid = ids[0]
        record = SeqIO.read(Entrez.efetch(db="protein", id=uid, rettype="gb", retmode="text"), "genbank")
        fasta_str = io.StringIO()
        SeqIO.write(record, fasta_str, "fasta")
        ncbi_url = f"https://www.ncbi.nlm.nih.gov/protein/{uid}"
        return {"record": record, "FASTA": fasta_str.getvalue(), "ncbi_url": ncbi_url}
    except Exception as e:
        return {"error": str(e)}

def trypsin_digest(seq):
    seq = str(seq)
    peptides = []
    start = 0
    for i in range(len(seq)-1):
        if seq[i] in ["K","R"] and seq[i+1] != "P":
            peptides.append(seq[start:i+1])
            start = i+1
    peptides.append(seq[start:])
    return peptides

def calculate_protein_hydrophobicity(seq):
    vals = [parker_hydro_scale.get(aa, 0) for aa in seq.upper()]
    if vals:
        return sum(vals)/len(vals)
    return 0

def calculate_aliphatic_index(seq):
    pa = ProteinAnalysis(str(seq))
    aa_percent = pa.amino_acids_percent
    a = aa_percent.get('A',0)*100
    v = aa_percent.get('V',0)*100
    i = aa_percent.get('I',0)*100
    l = aa_percent.get('L',0)*100
    return a + 2.9*v + 3.9*(i+l)

def bepi_pred_like(seq, threshold=1.0, window=7):
    seq = seq.upper()
    scores = []
    for i in range(len(seq)):
        win = seq[max(0,i-window//2):min(len(seq),i+window//2+1)]
        avg_score = np.mean([parker_hydro_scale.get(aa,0) for aa in win])
        scores.append(avg_score)
    epitopes = []
    cur = ""
    for i,aa in enumerate(seq):
        if scores[i]>=threshold:
            cur += aa
        else:
            if len(cur)>=4:
                epitopes.append(cur)
            cur = ""
    if len(cur)>=4:
        epitopes.append(cur)
    return epitopes, scores

def analyse_peptides_with_epitopes(peptides, threshold, window):
    rows = []
    for pep in peptides:
        pa = ProteinAnalysis(pep)
        epis,_ = bepi_pred_like(pep, threshold, window)
        rows.append({
            "Select": False,
            "Predicted Epitopes": ", ".join(epis) if epis else "-",
            "Peptide": pep,
            "Length": len(pep),
            "MW": round(pa.molecular_weight(),2),
            "Instability": round(pa.instability_index(),2),
            "GRAVY": round(pa.gravy(),3),
            "Hydrophobicity": round(calculate_protein_hydrophobicity(pep),3),
            "Aliphatic Index": round(calculate_aliphatic_index(pep),2)
        })
    return pd.DataFrame(rows)

def plot_epitope_match_heatmap(peptide, epitope):
    max_len = max(len(peptide), len(epitope))
    mat = np.zeros((1, max_len))
    for i in range(min(len(peptide), len(epitope))):
        mat[0, i] = 1 if peptide[i] == epitope[i] else 0
    fig = px.imshow(mat,
                    color_continuous_scale='YlGnBu',
                    labels=dict(x="Residue Position in Peptide", y="", color="Match (1=Yes)"),
                    x=list(peptide) + ['']*(max_len - len(peptide)),
                    y=["Match"])
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Residue-by-residue Match Heatmap between Selected Peptide and Theoretical Epitope",
        coloraxis_colorbar=dict(
            title="Match",
            tickvals=[0, 0.5, 1],
            ticktext=["Bad Match (0)", "Medium Match (~0.5)", "Good Match (1)"]
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_score_profile(seq, scores, threshold):
    fig, ax = plt.subplots(figsize=(min(12,len(seq)*0.3),2.5))
    ax.plot(range(1,len(seq)+1), scores, color="#1976d2", linewidth=2, label='Epitope Score')
    ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold Cutoff')
    ax.set_xticks(range(1,len(seq)+1))
    ax.set_xticklabels(list(seq), rotation=90)
    ax.set_ylabel("Score")
    ax.set_xlabel("Residue Position")
    ax.set_title("Amino Acid Residue Epitope Score Profile (BepiPred-like Parker Scale)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_color_gradient_legend():
    gradient = np.linspace(0,1,300).reshape(1,-1)
    colorscale = [
        [0.0, 'rgb(8, 69, 148)'],    # dark blue - bad match
        [0.5, 'rgb(253, 187, 132)'], # yellow - medium match
        [1.0, 'rgb(65, 171, 93)']    # green - good match
    ]
    fig = go.Figure(data=go.Heatmap(
        z=gradient,
        colorscale=colorscale,
        showscale=False,
        hoverinfo='skip',
        xgap=0, ygap=0,
        zmin=0, zmax=1
    ))
    fig.update_layout(
        height=60,
        margin=dict(l=20,r=20,t=30,b=10),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        xaxis=dict(
            tickmode='array',
            tickvals=[0,150,299],
            ticktext=["Bad Match","Medium Match","Good Match"],
            showgrid=False,
            zeroline=False
        ),
        title="Residue Match Heatmap Color Gradient Legend"
    )
    st.plotly_chart(fig, use_container_width=True)

def trim_fasta_sequence(fasta_str):
    lines = fasta_str.splitlines()
    if not lines or not lines[0].startswith(">"):
        return "", "Invalid FASTA format."
    header = lines[0]
    sequence = "".join(lines[1:])
    if len(sequence) < 30:
        return "", "Sequence too short to trim."
    trimmed_seq = sequence[10:-10]
    trimmed_fasta = f"{header} trimmed\n{trimmed_seq}"
    explanation = (
        "The FASTA sequence was trimmed by removing 10 residues from the N- and C-termini "
        "to focus on the core antigenic region. This simulates removing non-relevant flanking "
        "regions that may affect epitope prediction."
    )
    return trimmed_fasta, explanation

def match_score_color(avg_match):
    """Map average match (0-1) to color hex for table indication."""
    if avg_match <= 0.33:
        return '#084594'  # Dark blue - bad match
    elif avg_match <= 0.66:
        return '#fdbb84'  # Yellow - medium match
    else:
        return '#41ab5d'  # Green - good match

def compute_avg_match(peptide, epitope):
    l = min(len(peptide), len(epitope))
    matches = [1 if peptide[i]==epitope[i] else 0 for i in range(l)]
    if matches:
        return sum(matches)/len(matches)
    return 0

# ----------------  ---------------- #
st.title(" Vaccine Candidate ")

organism_choice = st.selectbox("Select Organism", ORGANISMS)
organism = st.text_input("Enter organism name") if organism_choice == "Other..." else organism_choice

protein_term = st.text_input("Protein name or keyword")

st.markdown("""
**Epitope Prediction Threshold:** Determines the minimum score above which residues are considered part of epitopes.  
A lower threshold means **more sensitive** detection (more epitopes, more false positives),  
a higher threshold means **stricter** detection (fewer epitopes, more confident).  
""")

threshold = st.slider("Epitope prediction threshold", 0.5, 2.0, 1.0, 0.1)

st.markdown("""
**Sliding Window Size:** Number of neighboring residues averaged when scoring each position.  
Smaller windows detect fine local variation; larger windows produce smoother, broader epitope regions.
""")

window_size = st.slider("Sliding window size", 3, 15, 7, 2)

if st.button("Find & Predict Epitopes"):
    if not organism.strip() or not protein_term.strip():
        st.error("Please provide both organism and protein name")
        st.stop()
    res = fetch_protein_sequence(organism, protein_term)
    if "error" in res:
        st.error(res["error"])
    else:
        st.session_state["ncbi_result"] = res
        st.session_state["record"] = res["record"]
        st.session_state["trimmed"] = None
        st.session_state["show_all_epitopes"] = False

if "ncbi_result" in st.session_state:
    res = st.session_state["ncbi_result"]
    record = st.session_state["record"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Fetched Protein Information")
    st.markdown(f"**ID:** {record.id}")
    st.markdown(f"**Description:** {record.description}")
    st.markdown(f"**NCBI Record:** [View on NCBI]({res['ncbi_url']})")
    st.markdown('</div>', unsafe_allow_html=True)

    show_fasta = st.checkbox("Show FASTA sequence")
    if show_fasta:
        st.text_area("FASTA", res["FASTA"], height=200)
        if st.button("Trim FASTA sequence"):
            trimmed, explanation = trim_fasta_sequence(res["FASTA"])
            st.session_state["trimmed"] = (trimmed, explanation)

    if st.session_state.get("trimmed"):
        trimmed_seq, explanation = st.session_state["trimmed"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Trimmed FASTA Sequence and Explanation")
        st.markdown(explanation)
        st.text_area("Trimmed FASTA", trimmed_seq, height=180)
        st.markdown('</div>', unsafe_allow_html=True)

    seq_orig = str(record.seq)
    peptides = trypsin_digest(seq_orig)
    df_peps = analyse_peptides_with_epitopes(peptides, threshold, window_size)

    # 
    df_peps["HasEpitope"] = df_peps["Predicted Epitopes"].apply(lambda x: 0 if x.strip() == "-" else 1)
    df_peps = df_peps.sort_values(by="HasEpitope", ascending=False).drop(columns=["HasEpitope"]).reset_index(drop=True)

    # 
    cols_order = ["Select", "Predicted Epitopes", "Peptide", "Length", "MW",
                  "Instability", "GRAVY", "Hydrophobicity", "Aliphatic Index"]
    df_peps = df_peps[cols_order]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Peptide Biochemical Properties + Predicted Epitopes")
    edited_df = st.data_editor(df_peps,
                               column_config={"Select": st.column_config.CheckboxColumn()},
                               use_container_width=True,
                               num_rows="dynamic")
    st.markdown('</div>', unsafe_allow_html=True)

    selected_rows = edited_df[edited_df["Select"]]
    if not selected_rows.empty:
        sel_pep = selected_rows.iloc[0]
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f"### Details for Selected Peptide: `{sel_pep['Peptide']}`")
        cols = st.columns([1,1,1,1,1,1,2])
        cols[0].markdown(f"**Length:** {sel_pep['Length']}")
        cols[1].markdown(f"**MW:** {sel_pep['MW']} Da")
        cols[2].markdown(f"**Instability:** {sel_pep['Instability']}")
        cols[3].markdown(f"**GRAVY:** {sel_pep['GRAVY']}")
        cols[4].markdown(f"**Hydrophobicity:** {sel_pep['Hydrophobicity']}")
        cols[5].markdown(f"**Aliphatic Index:** {sel_pep['Aliphatic Index']}")
        cols[6].markdown(f"**Linked Protein:** [View on NCBI]({res['ncbi_url']})")

        st.markdown("""
            <div class="explanation-text">
            The plot below shows per-residue epitope scores computed with a <strong>BepiPred-like algorithm</strong> 
            using the Parker hydrophilicity scale.
            Residues with scores above the red dashed line (threshold) are predicted epitope residues.
            Continuous stretches of ≥4 such residues are predicted as epitopes.
            </div>
        """, unsafe_allow_html=True)

        theoretical_epitopes, residue_scores = bepi_pred_like(sel_pep['Peptide'], threshold, window_size)

        st.subheader("Residue Epitope Score Profile")
        plot_score_profile(sel_pep['Peptide'], residue_scores, threshold)

        if theoretical_epitopes:
            for i, epi in enumerate(theoretical_epitopes, 1):
                st.markdown(f"**Theoretical Epitope {i}:** `{epi}`")
                st.subheader(f"Residue Match Heatmap for Epitope {i}")
                st.markdown("""
                The heatmap below visualizes residue-by-residue matches between the peptide and theoretical epitope.
                The horizontal color gradient legend below explains colors:
                <ul>
                <li><span style='color:#084594;'>Dark Blue</span>: Bad Match (mismatched residue)</li>
                <li><span style='color:#fdbb84;'>Yellow</span>: Medium Match (partial similarity)</li>
                <li><span style='color:#41ab5d;'>Bright Green</span>: Good Match (identical residue)</li>
                </ul>
                """, unsafe_allow_html=True)
                plot_epitope_match_heatmap(sel_pep['Peptide'], epi)
                plot_color_gradient_legend()
        else:
            st.info("No predicted epitopes above threshold.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Show All Epitope Matches Toggle Button
    if st.button("Show All Epitope Matches"):
        st.session_state["show_all_epitopes"] = not st.session_state.get("show_all_epitopes", False)

    if st.session_state.get("show_all_epitopes", False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("All Peptides and Their Predicted Epitopes with Match Quality")

        # Extract peptides with predicted epitopes
        epitopes_display = df_peps[df_peps["Predicted Epitopes"] != "-"].copy()
        if epitopes_display.empty:
            st.info("No predicted epitopes found in any peptides above threshold.")
        else:
              # 
            def avg_match_and_color(row):
                peptide = row["Peptide"]
                # Use first predicted epitope for match calculation
                epitope = row["Predicted Epitopes"].split(", ")[0]
                avg_match = compute_avg_match(peptide, epitope)
                color = match_score_color(avg_match)
                return pd.Series([avg_match, color])

            epitopes_display[["AvgMatch", "ColorHex"]] = epitopes_display.apply(avg_match_and_color, axis=1)

              # 
            epitopes_display["MatchColorSquare"] = epitopes_display["ColorHex"].apply(
                lambda c: f'<div class="color-square" style="background-color:{c}"></div>'
            )

              # 
            display_cols = ["MatchColorSquare", "Peptide", "Predicted Epitopes"]
            display_df = epitopes_display[display_cols].copy()

              # 
            st.write(
                """
                <style>
                .color-square {
                    width: 20px;
                    height: 20px;
                    border-radius: 3px;
                    display: inline-block;
                    margin-right: 6px;
                    vertical-align: middle;
                }
                </style>
                """, unsafe_allow_html=True
            )

             # 
            for idx, row in display_df.iterrows():
                st.markdown(
                    f"{row['MatchColorSquare']} **Peptide:** `{row['Peptide']}`  |  **Predicted Epitopes:** {row['Predicted Epitopes']}",
                    unsafe_allow_html=True
                )

        st.markdown('</div>', unsafe_allow_html=True)
