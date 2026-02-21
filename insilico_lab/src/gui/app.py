
import streamlit as st
import os
import sys
import time


# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.inference.property_engine import PropertyEngine
from src.admet.tox_engine import ToxicityEngine
from src.admet.druglikeness_engine import DrugLikenessEngine
from src.admet.risk_engine import RiskEngine
from src.gui.components import (
    render_molecule_3d,
    render_basic_properties,
    render_physicochemical_tab,
    render_permeability_tab,
    render_toxicity_tab,
    render_druglikeness_tab,
    render_risk_gauge,
    render_radar_chart,
    plot_radar_comparison
)
from src.chemistry.modifier_engine import generate_variants
from src.gui.comparison_utils import generate_comparison_table, generate_delta_summary
from src.gui.editor_tab import render_editor_tab
from src.gui.comparison_tab import render_comparison_tab
from src.gui.md_tab import render_md_tab
from src.gui.validation_tab import render_validation_tab
from src.synthesis.synthesis_engine import SynthesisEngine
# Removed FormulationEngine import


# Page config
st.set_page_config(
    page_title="Chopper - Drug Screening Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize engines
@st.cache_resource
def load_engines():
    """Load all prediction engines."""
    return {
        'property': PropertyEngine(),
        'toxicity': ToxicityEngine(),
        'druglikeness': DrugLikenessEngine(),
        'risk': RiskEngine()
    }

def name_to_smiles(name):
    """Convert common drug name to SMILES using a simple lookup."""
    from rdkit import Chem
    try:
        # Try to use RDKit's name parsing (limited)
        mol = Chem.MolFromSmiles(name)
        if mol:
            return Chem.MolToSmiles(mol)
    except:
        pass
    
    # Fallback dictionary for common drugs
    common_drugs = {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
        "diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "penicillin": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
        "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "dopamine": "NCCc1ccc(O)c(O)c1",
        "serotonin": "NCCc1c[nH]c2ccc(O)cc12"
    }
    
    return common_drugs.get(name.lower())

def run_full_prediction(smiles, engines):
    """Run complete ADMET prediction pipeline."""
    properties = engines['property'].predict_properties(smiles)
    toxicity = engines['toxicity'].predict(smiles)
    druglikeness = engines['druglikeness'].calculate(smiles)
    
    all_predictions = {
        'properties': properties,
        'toxicity': toxicity,
        'permeability': properties.get('permeability', {}),
        'druglikeness': druglikeness
    }
    
    risk = engines['risk'].calculate_risk(all_predictions)
    all_predictions['risk'] = risk
    
    return all_predictions

def main():
    # Header
    st.markdown('<div class="main-header"> Chopper</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visual Drug Screening & simulation Platform</div>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["🔬 Molecule Screening", "🧪 Molecule Modification", "⚗️ Advanced Editor", "📊 Compare Candidates", "🧊 MD Simulation", "📈 Model Validation", "🧪 Synthesis Intelligence"],
        horizontal=True
    )
    
    # Load engines once
    engines = load_engines()
    
    if mode == "🔬 Molecule Screening":
        render_screening_mode()
    elif mode == "🧪 Molecule Modification":
        render_modification_mode()
    elif mode == "⚗️ Advanced Editor":
        render_editor_tab(engines)
    elif mode == "📊 Compare Candidates":
        render_comparison_tab(engines)
    elif mode == "🧊 MD Simulation":
        render_md_tab()
    elif mode == "📈 Model Validation":
        render_validation_tab()
    elif mode == "🧪 Synthesis Intelligence":
        render_synthesis_tab()

def render_screening_mode():
    """Original screening interface."""
    # Sidebar
    with st.sidebar:
        st.header("📥 Input")
        
        input_method = st.radio(
            "Input method:",
            ["SMILES String", "Molecule Name"],
            help="Select how you want to input your molecule"
        )
        
        smiles_input = None
        
        if input_method == "SMILES String":
            smiles_input = st.text_input(
                "Enter SMILES:",
                value="CC(=O)Oc1ccccc1C(=O)O",
                help="Enter a valid SMILES string"
            )
        
        elif input_method == "Molecule Name":
            from src.gui.molecule_library import get_molecule_list
            
            # Load molecule library
            molecule_list = get_molecule_list()
            molecule_names = [name for name, _ in molecule_list]
            
            if molecule_names:
                selected = st.selectbox(
                    "Select molecule:",
                    options=[""] + molecule_names,
                    help="Choose from available molecules in datasets"
                )
                
                if selected:
                    smiles_input = dict(molecule_list)[selected]
                    st.success(f"✅ Selected: `{smiles_input[:50]}...`" if len(smiles_input) > 50 else f"✅ Selected: `{smiles_input}`")
            else:
                st.warning("No molecules loaded from datasets")
        
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        
    
    # Main content
    if run_button or smiles_input:
        with st.spinner("Running predictions..."):
            try:
                # Load engines
                engines = load_engines()
                
                # Run predictions
                properties = engines['property'].predict_properties(smiles_input)
                toxicity = engines['toxicity'].predict(smiles_input)
                druglikeness = engines['druglikeness'].calculate(smiles_input)
                
                # Aggregate predictions for risk calculation
                all_predictions = {
                    'properties': properties,
                    'toxicity': toxicity,
                    'permeability': properties.get('permeability', {}),
                    'druglikeness': druglikeness
                }
                
                risk = engines['risk'].calculate_risk(all_predictions)
                
                # Layout: Top section
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Molecule")
                    render_molecule_3d(smiles_input)
                    render_basic_properties(properties)
                
                with col2:
                    st.subheader("Overall Risk Assessment")
                    render_risk_gauge(risk)
                    render_radar_chart(all_predictions)
                
                # Tabs for detailed results
                st.markdown("---")
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Physicochemical",
                    "🧪 Permeability",
                    "☠️ Toxicity",
                    "💊 Drug-Likeness"
                ])
                
                with tab1:
                    render_physicochemical_tab(properties)
                
                with tab2:
                    render_permeability_tab(properties.get('permeability', {}))
                
                with tab3:
                    render_toxicity_tab(toxicity)
                
                with tab4:
                    render_druglikeness_tab(druglikeness)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
    else:
        st.info("👈 Enter a SMILES string in the sidebar and click 'Run Simulation' to begin.")

def render_modification_mode():
    """Molecule modification and comparison interface."""
    st.subheader("🧪 Virtual Drug Modification")
    st.markdown("Apply chemical modifications and compare ADMET properties")
    
    # Input method selection
    input_method = st.radio(
        "Input method:",
        ["SMILES String", "Select from Library"],
        horizontal=True,
        key="mod_input_method"
    )
    
    base_smiles = None
    
    if input_method == "SMILES String":
        # Manual SMILES input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            base_smiles = st.text_input(
                "Base Molecule (SMILES):",
                value="c1ccccc1",
                help="Enter SMILES for the molecule you want to modify",
                key="mod_smiles_input"
            )
        
        with col2:
            generate_btn = st.button("🔬 Generate Variants", type="primary", use_container_width=True)
    
    else:
        # Molecule library dropdown
        from src.gui.molecule_library import get_molecule_list
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            molecule_list = get_molecule_list()
            molecule_names = [name for name, _ in molecule_list]
            
            if molecule_names:
                selected = st.selectbox(
                    "Select molecule from library:",
                    options=[""] + molecule_names,
                    help="Choose from available molecules in datasets",
                    key="mod_molecule_select"
                )
                
                if selected:
                    base_smiles = dict(molecule_list)[selected]
                    st.success(f"✅ Selected: `{base_smiles[:50]}...`" if len(base_smiles) > 50 else f"✅ Selected: `{base_smiles}`")
            else:
                st.warning("No molecules loaded from datasets")
        
        with col2:
            generate_btn = st.button("🔬 Generate Variants", type="primary", use_container_width=True, disabled=not base_smiles)
    
    # Initialize session state
    if 'mod_variants' not in st.session_state:
        st.session_state.mod_variants = None
    if 'mod_base_smiles' not in st.session_state:
        st.session_state.mod_base_smiles = None
    if 'mod_base_results' not in st.session_state:
        st.session_state.mod_base_results = None
    
    # Generate variants on button click
    if generate_btn and base_smiles:
        with st.spinner("Generating variants and running predictions..."):
            try:
                # Load engines
                engines = load_engines()
                
                # Generate variants
                variants = generate_variants(base_smiles)
                
                # Run predictions for base
                base_results = run_full_prediction(base_smiles, engines)
                
                # Store in session state
                st.session_state.mod_variants = variants
                st.session_state.mod_base_smiles = base_smiles
                st.session_state.mod_base_results = base_results
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
    
    # Display variants if available
    if st.session_state.mod_variants is not None:
        variants = st.session_state.mod_variants
        base_smiles = st.session_state.mod_base_smiles
        base_results = st.session_state.mod_base_results
        
        st.markdown("---")
        st.subheader("Select a Modification")
        
        variant_names = [
            ("Original", "original"),
            ("+ Methyl (CH₃)", "methyl_variant"),
            ("+ Fluorine (F)", "fluoro_variant"),
            ("+ Hydroxyl (OH)", "hydroxyl_variant"),
            ("+ Chlorine (Cl)", "chloro_variant"),
            ("Extended Chain", "extended_chain")
        ]
        
        selected_variant = st.selectbox(
            "Choose modification:",
            options=[name for name, _ in variant_names],
            index=0
        )
        
        # Get selected SMILES
        variant_key = [key for name, key in variant_names if name == selected_variant][0]
        selected_smiles = variants.get(variant_key)
        
        if selected_smiles and selected_smiles != base_smiles:
            # Run predictions for modified
            engines = load_engines()
            mod_results = run_full_prediction(selected_smiles, engines)
            
            st.markdown("---")
            
            # Side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔵 Original")
                st.code(base_smiles, language="text")
                render_molecule_3d(base_smiles)
                render_basic_properties(base_results['properties'])
            
            with col2:
                st.markdown(f"### 🔴 {selected_variant}")
                st.code(selected_smiles, language="text")
                render_molecule_3d(selected_smiles)
                render_basic_properties(mod_results['properties'])
            
            # Comparison table
            st.markdown("---")
            st.subheader("📊 Property Comparison")
            
            comp_table = generate_comparison_table(base_results['properties'], mod_results['properties'])
            st.dataframe(comp_table, use_container_width=True)
            
            # Delta summary
            summary = generate_delta_summary(base_results['properties'], mod_results['properties'])
            st.info(f"**Summary:** {summary}")
            
            # Radar chart overlay
            st.markdown("---")
            st.subheader("📈 ADMET Profile Comparison")
            
            radar_fig = plot_radar_comparison(base_results['properties'], mod_results['properties'])
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning("Insufficient data for radar chart")
            
        elif selected_smiles is None:
            st.error("❌ Modification not chemically feasible")
        else:
            st.info("Select a modification to see comparison")
    
    elif not base_smiles and input_method == "SMILES String":
        st.info("👆 Enter a base molecule SMILES and click 'Generate Variants'")

@st.cache_resource
def _get_synthesis_engine():
    """Cached singleton – built once, reused every render."""
    from src.synthesis.synthesis_engine import SynthesisEngine
    return SynthesisEngine()

def render_synthesis_tab():
    """Render the Synthesis Intelligence tab."""
    from src.gui.components import render_molecule_3d
    from src.gui.molecule_library import get_molecule_list

    engine = _get_synthesis_engine()

    st.subheader("🧪 Synthesis Intelligence")
    st.markdown("Analyze molecular synthetic accessibility and retrosynthetic pathways.")

    # ── 1. Input row ────────────────────────────────────────────────────────
    molecule_list = get_molecule_list()
    molecule_names = [name for name, _ in molecule_list]

    col_input, col_3d = st.columns([3, 2])
    with col_input:
        selected_molecule = st.selectbox(
            "Select Molecule from Library:",
            options=["- Custom -"] + molecule_names,
            key="synthesis_mol_select"
        )
        default_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        if selected_molecule != "- Custom -":
            default_smiles = dict(molecule_list)[selected_molecule]

        smiles = st.text_input("Enter SMILES for Synthesis Analysis:", value=default_smiles)

        analysis_depth = st.radio(
            "Analysis Depth:",
            options=[1, 2],
            index=0,
            format_func=lambda x: f"{x}-Step Structural Analysis",
            horizontal=True,
            help="Depth 2 explores secondary disconnections of first-level precursors."
        )

    with col_3d:
        if smiles:
            st.markdown("**3D Structure Preview**")
            render_molecule_3d(smiles)

    if not smiles:
        return

    # ── 2. Full-width analysis ───────────────────────────────────────────────
    try:
        result = engine.analyze(smiles, depth=analysis_depth)
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        return

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SAS Score", f"{result['sas_score']}", help="1 (Easy) → 10 (Difficult)")
    m2.metric("Ring Strain", f"{result['ring_strain_score']}")
    m3.metric("Functional Penalty", f"{result['functional_penalty_score']}")
    color = "green" if result['classification'] == "Easy" else "orange" if result['classification'] == "Moderate" else "red"
    m4.markdown(
        f"**Feasibility:** <span style='color:{color}; font-size:1.4rem;'>{result['classification']}</span>"
        f"  <small>({result['feasibility_score']}%)</small>",
        unsafe_allow_html=True
    )

    # ── 3. Retrosynthesis section ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧬 Structural Decomposition (Precursors)")
    st.info(result['retrosynthesis']['analysis_label'])

    conf = result['retrosynthesis']['confidence']
    conf_label = "High" if conf >= 0.9 else "Moderate" if conf >= 0.6 else "Low"
    st.markdown(f"**Retrosynthesis Confidence:** {conf_label} ({conf})")

    precursors = result['retrosynthesis']['precursors']
    if precursors:
        for i in range(0, len(precursors), 3):
            chunk = precursors[i:i+3]
            p_cols = st.columns(len(chunk))
            for j, p_data in enumerate(chunk):
                with p_cols[j]:
                    st.markdown(f"**Disconnection {i+j+1}**")
                    st.caption(f"**Type:** {p_data['reaction_type']}")
                    st.caption(f"**Confidence:** {p_data['confidence']}")
                    p_smiles = p_data['smiles']
                    st.code(p_smiles, language="text")
                    render_molecule_3d(p_smiles)
    else:
        st.info("No single-step structural decompositions identified for this molecule.")

    st.warning("**Disclaimer:** Retrosynthetic suggestions are structural decompositions only. No laboratory conditions or instructions are provided.")

    # ── 4. Virtual Forward Synthesis ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 Virtual Forward Synthesis")
    st.markdown("Combine two precursors to generate new virtual products.")

    precursor_pairs = {
        "- Custom -": "CC(=O)O, CCO",
        "Esterification: Acetic Acid + Ethanol": "CC(=O)O, CCO",
        "Amidation: Acetic Acid + Ethylamine": "CC(=O)O, CCN",
        "Esterification: Salicylic Acid + Methanol": "Oc1ccccc1C(=O)O, CO",
        "Amidation: Benzene + Methylamine (Invalid/Safety Test)": "c1ccccc1, CN"
    }

    selected_pair = st.selectbox(
        "Quick-Select Precursor Pair:",
        options=list(precursor_pairs.keys()),
        key="precursor_pair_select"
    )
    default_precursors = precursor_pairs[selected_pair]
    precursor_input = st.text_input(
        "Enter Precursor SMILES (comma-separated):",
        value=default_precursors,
        help="e.g. Acetic Acid and Ethanol to form an ester."
    )

    if st.button("🚀 Generate Virtual Products"):
        with st.spinner("Simulating structural transformations..."):
            try:
                precursor_list = [s.strip() for s in precursor_input.split(",") if s.strip()]
                vs_result = engine.virtual_synthesis(precursor_list)
                products = vs_result.get("product_details", [])

                if products:
                    st.success(f"Generated {len(products)} virtual product(s).")
                    for i, prod in enumerate(products, start=1):
                        st.markdown("---")
                        v1, v2 = st.columns([3, 2])
                        with v1:
                            st.markdown(f"### Product {i}")
                            st.markdown(f"**SMILES:** `{prod['smiles']}`")
                            st.markdown(f"**Reaction Class:** {prod['reaction_name']}")

                            c_color = "green" if prod['reaction_confidence'] == "High" else "orange" if prod['reaction_confidence'] == "Moderate" else "red"
                            st.markdown(
                                f"**Base Reaction Confidence:** <span style='color:{c_color};'>{prod['reaction_confidence']}</span>",
                                unsafe_allow_html=True
                            )
                            st.write(f"**Synthetic Plausibility:** {prod['classification']}")
                            st.write(f"**Plausibility Score:** {prod['plausibility_score']} *(lower is better)*")

                            admet = prod['admet_preview']
                            st.markdown("**ADMET Screening (Ro5)**")
                            a1, a2 = st.columns(2)
                            a1.caption(f"MW: {admet['MW']}")
                            a1.caption(f"LogP: {admet['logP']}")
                            a2.caption(f"HBD: {admet['HBD']}")
                            a2.caption(f"HBA: {admet['HBA']}")

                            status = admet['Ro5_status']
                            s_color = "green" if status == "Pass" else "orange" if status == "Borderline" else "red"
                            st.markdown(
                                f"**Drug-likeness:** <span style='color:{s_color}; font-weight:bold;'>{status}</span>",
                                unsafe_allow_html=True
                            )

                            if st.button(f"🧊 MD Simulate Product #{i}", key=f"md_vs_{i}"):
                                st.session_state.vs_smiles = prod["smiles"]
                                st.info("Product copied! Go to 'MD Simulation' tab.")

                        with v2:
                            st.markdown("**3D Structure**")
                            render_molecule_3d(prod["smiles"])

                else:
                    st.info("No valid transformations identified for these precursors. Check structural compatibility.")

            except Exception as e:
                st.error(f"Virtual Synthesis Error: {str(e)}")

if __name__ == "__main__":
    main()
