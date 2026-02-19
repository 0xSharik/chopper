
import streamlit as st
import os
import sys

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

# Page config
st.set_page_config(
    page_title="InSilico Lab - Drug Screening Platform",
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
    st.markdown('<div class="main-header">🧬 InSilico Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visual In-Silico Drug Screening Platform</div>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Select Mode:",
        ["🔬 Molecule Screening", "🧪 Molecule Modification", "⚗️ Advanced Editor", "📊 Compare Candidates"],
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
    else:
        render_comparison_tab(engines)

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
        
        st.markdown("---")
        st.markdown("### 📚 Example Molecules")
        examples = {
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
            "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"
        }
        
        for name, smi in examples.items():
            if st.button(name, use_container_width=True):
                smiles_input = smi
                st.rerun()
    
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

if __name__ == "__main__":
    main()
