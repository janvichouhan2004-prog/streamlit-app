import streamlit as st
import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, QED
import py3Dmol
from stmol import showmol
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

# ---------------- 1. PAGE CONFIG & SESSION STATE ----------------
st.set_page_config(page_title="BioChemFuse Pro", layout="wide")

if "start" not in st.session_state:
    st.session_state.start = False
if "ligand_df" not in st.session_state:
    st.session_state.ligand_df = None
if "pdb_data" not in st.session_state:
    st.session_state.pdb_data = None

# --- HELPER FUNCTIONS ---
def get_target_by_id(query_id):
    try:
        # Search for target and get the ChEMBL ID
        res = new_client.target.search(query_id)
        return res[0]['target_chembl_id'] if res else None
    except:
        return None

@st.dialog("📖 Detailed About BioChemFuse Pro")
def full_about_dialog():
    st.markdown("""
    ### 🧬 Welcome to BioChemFuse Pro
    **BioChemFuse Pro** is a comprehensive drug discovery suite developed by **Janvi Chouhan** under the guidance of **Dr. Kushagra Kashyap**.

    #### 🚀 Detailed Pipeline Stages:
    * **Data Retrieval**: Connects to the ChEMBL database to find known active compounds. It uses PDBe API to map PDB IDs to UniProt accessions for high-accuracy searching.
    * **3D Structure Viewer**: Uses the `py3Dmol` engine to render protein structures. This allows researchers to examine binding pockets and secondary structures.
    * **Ramachandran Plot**: A statistical map of $\phi$ (phi) and $\psi$ (psi) backbone dihedral angles. It is the gold standard for validating if a protein structure is physically realistic.
    * **ML Lead Ranking**: Uses the **Quantitative Estimate of Drug-likeness (QED)**. It calculates molecular weight, lipophilicity (LogP), and polar surface area to rank which molecules are most likely to succeed as oral drugs.
    * **Docking Prep**: Converts 2D SMILES strings into 3D coordinates. It uses the **MMFF94 force field** for energy minimization, ensuring the ligand is in its lowest energy "bioactive" shape.
    """)

# ---------------- 2. COVER PAGE ----------------
if not st.session_state.start:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #1a5276 0%, #117a65 100%); }
        .cover-box {
            background: rgba(255, 255, 255, 0.9); padding: 50px; border-radius: 25px;
            text-align: center; max-width: 800px; margin: auto; margin-top: 10vh;
        }
        </style>
        <div class="cover-box">
            <h1 style="color: #1a5276;">🧬 BioChemFuse Pro</h1>
            <p style="font-size: 1.2rem; color: #333;">Fusing Structural Biology & Cheminformatics</p>
            <hr>
        </div>
    """, unsafe_allow_html=True)
    
    _, col2, _ = st.columns([1,1,1])
    with col2:
        if st.button("🚀 Open Research Tool", use_container_width=True):
            st.session_state.start = True
            st.rerun()
        if st.button("📖 Detailed About BioChemFuse Pro", use_container_width=True):
            full_about_dialog()
    st.stop()

# ---------------- 3. MAIN APP ----------------
st.sidebar.title("🧬 Navigation")
page = st.sidebar.radio("Select Stage:", ["1. Data Retrieval", "2. 3D Structure Viewer", "3. Ramachandran Plot", "4. ML Lead Ranking", "5. Docking Prep"])

if page == "1. Data Retrieval":
    st.header("🔍 Data Retrieval")
    if st.button("ℹ️ About Data Retrieval"):
        st.info("This section queries the ChEMBL database. It maps PDB IDs to UniProt accessions via the PDBe API to ensure we find the correct biological target for bioactivity data retrieval.")

    input_type = st.radio("Input Type:", ["Protein Name", "PDB ID", "FASTA"])
    query = ""

    if input_type == "Protein Name":
        query = st.text_input("Protein Name", "EGFR")
    elif input_type == "PDB ID":
        pdb_id_input = st.text_input("Enter PDB ID", "1E8W").strip().lower()
        if pdb_id_input:
            url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id_input}"
            try:
                data = requests.get(url).json()
                query = list(data[pdb_id_input]["UniProt"].keys())[0]
                st.success(f"Mapped to UniProt ID: {query}")
            except: query = pdb_id_input
    elif input_type == "FASTA":
        file = st.file_uploader("Upload FASTA", type=["fasta", "fa"])
        if file:
            fasta = StringIO(file.getvalue().decode())
            record = next(SeqIO.parse(fasta, "fasta"))
            query = record.id.split("|")[1] if "|" in record.id else record.id

    if st.button("Search Ligands") and query:
        with st.spinner("Searching ChEMBL..."):
            target_id = get_target_by_id(query)
            if target_id:
                activities = new_client.activity.filter(target_chembl_id=target_id, standard_type="IC50").only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'target_organism'])
                df = pd.DataFrame(activities[:100])
                if not df.empty:
                    if 'target_organism' in df.columns:
                        df = df.rename(columns={'target_organism': 'organism'})
                    st.session_state.ligand_df = df
                    st.dataframe(df[['molecule_chembl_id','canonical_smiles','standard_value']])
                else: st.warning("No IC50 data found.")
            else: st.error("Target not found.")

elif page == "2. 3D Structure Viewer":
    st.header("🏟️ 3D Protein Viewer")
    if st.button("ℹ️ About 3D Viewer"):
        st.info("Uses the py3Dmol engine to render protein structures. This allows researchers to examine binding pockets and secondary structures (Helices/Sheets).")
    
    pdb_id = st.text_input("Enter PDB ID", "1E8W")
    if st.button("Fetch & View Structure"):
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
        if r.status_code == 200:
            st.session_state.pdb_data = r.text
            st.success(f"Loaded {pdb_id}")
        else: st.error("PDB not found.")

    if st.session_state.pdb_data:
        # DOWNLOAD BUTTON FOR PROTEIN
        st.download_button("📥 Download PDB File", data=st.session_state.pdb_data, file_name=f"{pdb_id}.pdb", mime="chemical/x-pdb")
        
        view = py3Dmol.view(width=800, height=500)
        view.addModel(st.session_state.pdb_data, "pdb")
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo(); showmol(view)

elif page == "3. Ramachandran Plot":
    st.header("📊 Protein Geometry Validation")
    if st.button("ℹ️ About Ramachandran Plot"):
        st.info("A statistical map of phi and psi backbone dihedral angles. It is the gold standard for validating if a protein structure is physically realistic.")
    
    if st.session_state.pdb_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode='w') as tmp:
            tmp.write(st.session_state.pdb_data)
            tmp_path = tmp.name
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("P", tmp_path)
        phi, psi = [], []
        for model in structure:
            for chain in model:
                for pp in PPBuilder().build_peptides(chain):
                    for a in pp.get_phi_psi_list():
                        if a[0] and a[1]:
                            phi.append(np.degrees(a[0])); psi.append(np.degrees(a[1]))
        if phi:
            total = len(phi)
            favoured = sum(1 for p, s in zip(phi, psi) if (-150 < p < -30 and -100 < s < 50) or (-160 < p < -50 and 100 < s < 175))
            c1, c2 = st.columns(2)
            c1.metric("Total Residues", total)
            c2.metric("Favoured Region %", f"{(favoured/total)*100:.1f}%")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_facecolor('mistyrose') 
            ax.add_patch(plt.Rectangle((-160, 100), 100, 75, color='honeydew', zorder=1))
            ax.add_patch(plt.Rectangle((-150, -90), 120, 140, color='honeydew', zorder=1))
            ax.axhline(0, color='black', lw=1.5, zorder=2)
            ax.axvline(0, color='black', lw=1.5, zorder=2)
            ax.scatter(phi, psi, s=12, color='darkgreen', edgecolors='white', linewidth=0.3, alpha=0.7, zorder=3)
            ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
            ax.set_xlabel(r"$\phi$ (Phi)"); ax.set_ylabel(r"$\psi$ (Psi)")
            st.pyplot(fig)
        os.remove(tmp_path)
    else: st.warning("Please load a PDB in Stage 2 first.")

elif page == "4. ML Lead Ranking":
    st.header("🤖 ML Lead Ranking")
    if st.button("ℹ️ About ML Ranking"):
        st.info("Uses the Quantitative Estimate of Drug-likeness (QED). It calculates molecular weight, lipophilicity (LogP), and polar surface area to rank lead molecules.")
    
    if st.session_state.ligand_df is not None:
        df = st.session_state.ligand_df.copy()
        df['QED'] = df['canonical_smiles'].apply(lambda x: round(QED.qed(Chem.MolFromSmiles(x))*100, 2) if (isinstance(x, str) and Chem.MolFromSmiles(x)) else 0)
        cols_to_show = ['molecule_chembl_id', 'organism', 'canonical_smiles', 'standard_value', 'QED']
        existing_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[existing_cols].sort_values(by='QED', ascending=False))
    else: st.warning("No data found. Please run Stage 1 first.")

elif page == "5. Docking Prep":
    st.header("🛠️ Ligand Preparation")
    if st.button("ℹ️ About Docking Prep"):
        st.info("Converts 2D SMILES into 3D coordinates. Uses the MMFF94 force field for energy minimization to ensure a bioactive shape.")
    
    smiles = st.text_input("Enter SMILES", "CC(=O)Oc1ccccc1C(=O)O")
    if st.button("Prepare 3D"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
            st.success("3D Conformer Optimized!")
            
            # DOWNLOAD BUTTON FOR LIGAND
            mol_block = Chem.MolToMolBlock(mol)
            st.download_button("📥 Download Optimized Ligand (.mol)", data=mol_block, file_name="optimized_ligand.mol", mime="chemical/x-mdl-molfile")
            
            view = py3Dmol.view(width=400, height=400)
            view.addModel(mol_block, 'mol'); view.setStyle({'stick':{}}); view.zoomTo(); showmol(view)