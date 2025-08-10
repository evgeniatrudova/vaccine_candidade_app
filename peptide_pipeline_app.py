# Install Python3:    python3 --version  &  python --version
#  pip install biopython
#  pip install pandas
#  pip install numpy
#  pip install matplotlib
#  pip install streamlit
#  pip install py3dmol
#  pip install seaborn 
#  cd "C:\"
# Control Data :streamlit run peptide_pipeline_app.py


import streamlit as st
import pandas as pd
import time
import random
from Bio import Entrez, SeqIO
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# -------------------------------- CONFIG -------------------------------- #
Entrez.email = "your.email@example.com"  

# ---  Adjustable default list in search bar
PARASITE_LIST = [
    "Homo sapiens",
    "Plasmodium falciparum",
    "Toxocara canis",
    "Ascaris suum",
    "Schistosoma mansoni",
    "Fasciola hepatica",
    "Meloidogyne incognita",
    "Haemonchus contortus",
    "Ancylostoma caninum",
    "Trichuris trichiura",
    "Onchocerca volvulus",
    "Cysticercus cellulosae",
    "Phytophthora infestans"
]
kd_scale = ProtParamData.kd


# ----------------------Functions ------------------------- #

# --- 1.  explain_parameters, text explaining parameters 
# --- 2.  build_entrez_query, NCBI’s Entrez API/Taxonomy ID allowing user to search for other organisms
# --- 3.  calculate_protein_hydrophobicity Calculates average Kyte Doolittle score for found amino acid seq
# --- 4.  trypsin_digest  Simulates proteolytic cleavage by ttrypsin, cutting protein sequence after K or R residues unless followed by P.
# --- 5.  analyze_filter_peptides Filter or biochemical properties by length, hydrophobicity, GRAVY, molecular weight and instability index.
# --- 6.  compute_fitness  Propotype candidade score , from ideal length of 15aa
# --- 7.  peptide_epitope_similarity, Highest character by character match count between peptide and epitope
# --- 8.  fetch_proteome, Queries NCBI Protein DB for an organism, downloads FASTA sequences, calculates their hydrophobicity, stores them in session state, and returns the protein records.
# --- 9.  get_peptide_dataframe Generates a DataFrame

def explain_parameters():
    return {
        "Length": "Length of the peptide in amino acids.",
        "Instability Index": "Predicts protein stability; values <40 generally indicate stable peptides.",
        "GRAVY": "Grand Average of Hydropathy; positive values indicate hydrophobic peptides, negative values hydrophilic.",
        "Molecular Weight": "Molecular weight of the peptide in Daltons (Da).",
        "Peptide Hydrophobicity": "Average hydrophobicity score based on the Kyte-Doolittle scale for the peptide.",
        "Protein Hydrophobicity": "Average hydrophobicity of the full parent protein sequence.",
        "Best Epitope Match Score": "Max similarity score of the peptide aligned against experimental epitopes.",
        "Candidate Score": "Composite score combining instability, hydrophobicity, and length relative to ideal vaccine peptides."
    }

def build_entrez_query(organism_query, keywords=None):
    organism_query = organism_query.strip()
    if organism_query.lower().startswith("txid"):
        query = f"{organism_query}[Organism]"
    elif organism_query.isdigit():
        query = f"txid{organism_query}[Organism]"
    else:
        query = f"{organism_query}[Organism]"
    if keywords:
        query += f" {keywords}"
    return query

def calculate_protein_hydrophobicity(seq):
    values = [kd_scale.get(aa, 0) for aa in seq.upper()]
    return sum(values) / len(values) if values else 0

def trypsin_digest(seq):
    peptides, start = [], 0
    seq = str(seq)
    for i in range(len(seq) - 1):
        if seq[i] in ["K", "R"] and seq[i + 1] != "P":
            peptides.append(seq[start: i + 1])
            start = i + 1
    peptides.append(seq[start:])
    return peptides

def analyze_filter_peptides(peptides, min_len, max_len, min_gravy, max_gravy,
                           min_instab, max_instab, min_mw, max_mw):
    out = []
    for pep in peptides:
        pa = ProteinAnalysis(pep)
        if not (min_len <= len(pep) <= max_len):
            continue
        inst = pa.instability_index()
        gravy = pa.gravy()
        mw = pa.molecular_weight()
        pep_hydro = calculate_protein_hydrophobicity(pep)
        if (min_gravy <= gravy <= max_gravy and
            min_instab <= inst <= max_instab and
            min_mw <= mw <= max_mw):
            out.append({
                "Peptide": pep,
                "Length": len(pep),
                "Instability Index": round(inst, 2),
                "GRAVY": round(gravy, 2),
                "Molecular Weight": round(mw, 2),
                "Peptide Hydrophobicity": round(pep_hydro, 3)
            })
    return out

def compute_fitness(inst, gravy, length):
    return inst + abs(gravy) * 10 + abs(length - 15)

def peptide_epitope_similarity(peptide, epitopes):
    best_score = 0
    for epi in epitopes:
        score = sum(1 for a, b in zip(peptide, epi) if a == b)
        if score > best_score:
            best_score = score
    return best_score

def fetch_proteome(query, retmax):
    ids = Entrez.read(Entrez.esearch(db="protein", term=query, retmax=retmax), validate=False)["IdList"]
    proteins, seq_dict, hydros = [], {}, {}
    bar = st.progress(0)
    for i, pid in enumerate(ids):
        rec = SeqIO.read(Entrez.efetch(db="protein", id=pid, rettype="fasta", retmode="text"), "fasta")
        seq_dict[rec.id] = str(rec.seq)
        hydros[rec.id] = calculate_protein_hydrophobicity(str(rec.seq))
        proteins.append(rec)
        bar.progress((i + 1) / len(ids))
        time.sleep(0.1)
    st.session_state["protein_seq_dict"] = seq_dict
    st.session_state["protein_hydrophobicity"] = hydros
    return proteins

def get_peptide_dataframe(proteins, filters, exp_peptides):
    records = []
    hydros = st.session_state.get("protein_hydrophobicity", {})
    for prot in proteins:
        for pep in analyze_filter_peptides(trypsin_digest(prot.seq), **filters):
            pid = prot.id
            sim = peptide_epitope_similarity(pep["Peptide"].upper(), exp_peptides) if exp_peptides else 0
            records.append({
                **pep,
                "Protein ID": pid,
                "Protein Hydrophobicity": hydros.get(pid, 0),
                "Best Epitope Match Score": sim,
                "Candidate Score": compute_fitness(pep["Instability Index"], pep["GRAVY"], pep["Length"])
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(["Candidate Score", "Best Epitope Match Score"], ascending=[True, False])

# ------------------- Testing mode ---------------------- #
def generate_random_nucleotide_sequence(length=60):
    nucleotides = ['A', 'C', 'G', 'T']
    return ''.join(random.choices(nucleotides, k=length))

codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def translate_dna(sequence):
    sequence = sequence.upper()
    peptide = ''
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        aa = codon_table.get(codon, 'X')
        if aa == '_':
            break
        peptide += aa
    return peptide

def generate_theoretical_placeholder_peptides_with_placeholder(placeholder_peptide):
    base_peptides = []
    base = "ACDEFGHIKLMNPQRSTVWY"
    for length in range(8, 13):
        seq = base[:length]
        inst = ProteinAnalysis(seq).instability_index()
        gravy = ProteinAnalysis(seq).gravy()
        mw = ProteinAnalysis(seq).molecular_weight()
        pep_hydro = calculate_protein_hydrophobicity(seq)
        base_peptides.append(dict(
            Peptide=seq,
            Length=length,
            **{"Instability Index": round(inst, 2)},
            **{"GRAVY": round(gravy, 2)},
            **{"Molecular Weight": round(mw, 2)},
            **{"Peptide Hydrophobicity": round(pep_hydro, 3)},
            **{"Protein ID": f"Theoretical_{length}"},
            **{"Protein Hydrophobicity": round(gravy, 2)},
            **{"Best Epitope Match Score": 0},
            **{"Candidate Score": round(compute_fitness(inst, gravy, length), 2)}
        ))
    df = pd.DataFrame(base_peptides)
    if placeholder_peptide and len(placeholder_peptide) >= 6:
        inst = ProteinAnalysis(placeholder_peptide).instability_index()
        gravy = ProteinAnalysis(placeholder_peptide).gravy()
        mw = ProteinAnalysis(placeholder_peptide).molecular_weight()
        pep_hydro = calculate_protein_hydrophobicity(placeholder_peptide)
        placeholder_dict = dict(
            Peptide=placeholder_peptide,
            Length=len(placeholder_peptide),
            **{"Instability Index": round(inst, 2)},
            **{"GRAVY": round(gravy, 2)},
            **{"Molecular Weight": round(mw, 2)},
            **{"Peptide Hydrophobicity": round(pep_hydro, 3)},
            **{"Protein ID": "Placeholder_Peptide"},
            **{"Protein Hydrophobicity": round(gravy, 2)},
            **{"Best Epitope Match Score": 0},
            **{"Candidate Score": round(compute_fitness(inst, gravy, len(placeholder_peptide)), 2)}
        )
        df = pd.concat([pd.DataFrame([placeholder_dict]), df], ignore_index=True)
    return df

def plot_epitope_peptide_heatmap(peptides, epitopes):
    if not peptides or not epitopes:
        st.info("No peptides or epitopes to plot.")
        return
    sim_matrix = np.zeros((len(peptides), len(epitopes)))
    for i, pep in enumerate(peptides):
        for j, epi in enumerate(epitopes):
            sim_matrix[i, j] = sum(1 for a, b in zip(pep, epi) if a == b)
    df_heatmap = pd.DataFrame(sim_matrix, index=peptides, columns=epitopes)
    plt.figure(figsize=(max(8, len(epitopes)*0.5), max(6, len(peptides)*0.4)))
    sns.heatmap(df_heatmap, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Match Count'})
    st.pyplot(plt.gcf())
    plt.clf()

def show_peptide_detail(row, epitopes):
    st.header(f"Detail for Peptide: {row['Peptide']}")
    st.info(
        "The full untrimmed protein sequence is shown below; peptides are generated by simulating trypsin digestion "
        "at lysine (K) and arginine (R) residues, except when followed by proline (P), reflecting a standard proteomic breakdown step "
        "used for peptide candidate analysis."
    )
    prot_id = row.get("Protein ID", None)
    if prot_id and prot_id in st.session_state.get("protein_seq_dict", {}):
        full_seq = st.session_state["protein_seq_dict"][prot_id]
        st.subheader("Full Protein FASTA Sequence:")
        st.code(f">{prot_id}\n{full_seq}", language="fasta")
    else:
        st.warning("Full protein sequence not available for this peptide.")

    st.subheader("Calculated Parameters Explanation")
    explanations = explain_parameters()
    for key in explanations:
        if key in row:
            st.markdown(f"**{key}:** {row[key]}  \n*{explanations[key]}*")

    if prot_id:
        ncbi_link = f"https://www.ncbi.nlm.nih.gov/protein/{prot_id}"
        st.markdown(f"[View parent protein on NCBI]({ncbi_link})")

    if epitopes:
        best_epi = max(epitopes, key=lambda e: peptide_epitope_similarity(row['Peptide'], [e]))
        score = peptide_epitope_similarity(row['Peptide'], [best_epi])
        st.subheader(f"Best Epitope Match (Score: {score})")
        st.markdown("**Alignment:**")
        aligned_html = ""
        for p_aa, e_aa in zip(row['Peptide'], best_epi):
            color = 'green' if p_aa == e_aa else 'red'
            aligned_html += f"<span style='color:{color}; font-weight:bold'>{p_aa}</span>"
        st.markdown(aligned_html, unsafe_allow_html=True)

    plot_epitope_peptide_heatmap([row['Peptide']], epitopes)

  
    if row.get("Protein ID", "").startswith("Theoretical") or row.get("Protein ID", "") == "Placeholder_Peptide":
        st.subheader("Theoretical Fit Prototypes Comparison")
        prototypes = []
        base = "ACDEFGHIKLMNPQRSTVWY"
        for label, length, inst, gravy in [("Bad", 12, 55, -1.5), ("Medium", 15, 35, 0.0), ("Good", 15, 25, 0.8)]:
            seq = base[:length]
            mw = ProteinAnalysis(seq).molecular_weight()
            score = compute_fitness(inst, gravy, length)
            prototypes.append({
                "Peptide": seq,
                "Length": length,
                "Instability Index": inst,
                "GRAVY": gravy,
                "Molecular Weight": round(mw, 2),
                "Peptide Hydrophobicity": round(gravy, 3),
                "Protein ID": f"Prototype_{label}",
                "Protein Hydrophobicity": gravy,
                "Best Epitope Match Score": 0,
                "Candidate Score": round(score, 2),
                "Label": label
            })
        df_protos = pd.DataFrame(prototypes)
        st.table(df_protos[["Label", "Peptide", "Candidate Score", "Instability Index", "GRAVY", "Molecular Weight"]])

# ---------------------- Main ------------------------- #
def main():
    st.title("Vaccine Candidate Finder")
    parasite_option = st.selectbox("Select organism from list or choose 'Other' to enter manually:",
                                  PARASITE_LIST + ["Other (enter manually)"])
    if parasite_option == "Other (enter manually)":
        custom_organism = st.text_input("Enter organism name or taxonomy ID", placeholder="e.g. txid5833 or 5833 or Homo sapiens")
        organism_query = custom_organism.strip()
    else:
        organism_query = parasite_option

    col1, col2 = st.columns(2)
    if col1.button("Fetch Mode"):
        st.session_state["mode"] = "fetch"
    if col2.button("Testing Mode"):
        st.session_state["mode"] = "test"

    if "mode" not in st.session_state:
        st.stop()

    if st.session_state["mode"] == "fetch":
        st.header("Fetch Mode")
        fetch_keywords = st.text_input("Protein keywords (optional)")
        exp_peptides_input = st.text_area("Experimental/Reference peptides (one per line)", height=100)
        exp_peptides = [p.strip().upper() for p in exp_peptides_input.splitlines() if p.strip()]
        with st.expander("Peptide Filters"):
            min_len, max_len = st.slider("Length", 6, 30, (8, 20))
            min_gravy, max_gravy = st.slider("GRAVY", -2.0, 2.0, (-1.0, 1.0), 0.05)
            min_instab, max_instab = st.slider("Instability Index", 1.0, 120.0, (1.0, 40.0), 0.5)
            min_mw, max_mw = st.slider("Mol. Weight (Da)", 500.0, 5000.0, (700.0, 3000.0), 50.0)
        max_proteins = st.number_input("Max proteins to fetch", 10, 500, 50, 10)
        filters = dict(min_len=min_len, max_len=max_len,
                       min_gravy=min_gravy, max_gravy=max_gravy,
                       min_instab=min_instab, max_instab=max_instab,
                       min_mw=min_mw, max_mw=max_mw)

        if st.button("Run Fetch & Analyse"):
            fetch_query = build_entrez_query(organism_query, keywords=fetch_keywords)
            proteins = fetch_proteome(fetch_query, max_proteins)
            df = get_peptide_dataframe(proteins, filters, exp_peptides)
            st.session_state["result_df"] = df
            st.session_state["epitopes"] = exp_peptides
            st.session_state["protein_seq_dict"] = st.session_state.get("protein_seq_dict", {})

        if "result_df" in st.session_state and not st.session_state["result_df"].empty:
            df = st.session_state["result_df"]
            st.dataframe(df)

            st.subheader("Inspect a Peptide Candidate")
            selected_peptide = st.selectbox("Select peptide to show details", ["None"] + df["Peptide"].tolist())
            if selected_peptide != "None":
                sel_row = df[df["Peptide"] == selected_peptide].iloc[0]
                show_peptide_detail(sel_row, st.session_state.get("epitopes", []))

            plot_epitope_peptide_heatmap(df["Peptide"].tolist(), st.session_state.get("epitopes", []))

    elif st.session_state["mode"] == "test":
        st.header("Testing Mode")
        if st.button("Generate random DNA & peptides"):
            dna_seq = generate_random_nucleotide_sequence(60)
            pep_seq = translate_dna(dna_seq)
            df = generate_theoretical_placeholder_peptides_with_placeholder(pep_seq)
            epitopes = [p[:-1] + ("A" if p[-1] != "A" else "C") for p in df["Peptide"]]
            st.session_state["result_df"] = df
            st.session_state["epitopes"] = epitopes
            st.session_state["protein_seq_dict"] = {row["Protein ID"]: row["Peptide"] for _, row in df.iterrows()}
            st.success(f"Generated random DNA sequence: {dna_seq}")
            st.success(f"Translated peptide sequence: {pep_seq}")

        if "result_df" in st.session_state:
            df = st.session_state["result_df"]
            st.dataframe(df)

            st.subheader("Inspect a Peptide Candidate")
            selected_peptide = st.selectbox("Select peptide to show details", ["None"] + df["Peptide"].tolist())
            if selected_peptide != "None":
                sel_row = df[df["Peptide"] == selected_peptide].iloc[0]
                show_peptide_detail(sel_row, st.session_state.get("epitopes", []))

            plot_epitope_peptide_heatmap(df["Peptide"].tolist(), st.session_state.get("epitopes", []))


if __name__ == "__main__":
    main()
