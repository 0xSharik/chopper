"""
Advanced Molecule Editor Tab
Provides controlled functional group attachment and free-form SMILES editing.
"""
import streamlit as st
import pandas as pd
from src.chemistry.group_library import list_groups
from src.chemistry.atom_tools import get_atom_summary, get_attachable_atoms
from src.chemistry.reaction_engine import attach_group
from src.chemistry.smiles_editor import validate_smiles, canonicalize_smiles, generate_edit_report
from src.chemistry.validation_engine import validate_modified_molecule
from src.chemistry.suggestion_engine import suggest_modifications
from src.gui.components import render_molecule_3d, render_basic_properties, plot_radar_comparison
from src.gui.comparison_utils import generate_comparison_table, generate_delta_summary

def render_editor_tab(engines):
    """
    Render advanced molecule editor interface.
    
    Args:
        engines: Dict of prediction engines
    """
    st.subheader("🔬 Advanced Molecule Editor")
    st.markdown("Professional medicinal chemistry workbench with atom-level control")
    
    # Mode selection
    editor_mode = st.radio(
        "Editor Mode:",
        ["Controlled Attachment", "Free-Form SMILES"],
        horizontal=True,
        help="Choose between guided functional group attachment or direct SMILES editing"
    )
    
    if editor_mode == "Controlled Attachment":
        render_controlled_mode(engines)
    else:
        render_freeform_mode(engines)

def render_controlled_mode(engines):
    """Controlled functional group attachment interface."""
    st.markdown("### 🎯 Controlled Functional Group Attachment")
    
    # Initialize session state
    if 'editor_analyzed_smiles' not in st.session_state:
        st.session_state.editor_analyzed_smiles = None
    if 'editor_atom_summary' not in st.session_state:
        st.session_state.editor_atom_summary = None
    if 'editor_attachable_atoms' not in st.session_state:
        st.session_state.editor_attachable_atoms = None
    
    # Input method selection
    input_method = st.radio(
        "Input method:",
        ["SMILES String", "Select from Library"],
        horizontal=True,
        key="editor_input_method"
    )
    
    base_smiles = None
    
    if input_method == "SMILES String":
        # Manual SMILES input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            base_smiles = st.text_input(
                "Base Molecule (SMILES):",
                value="c1ccccc1",
                help="Enter the molecule you want to modify",
                key="editor_base_smiles"
            )
        
        with col2:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    else:
        # Molecule library dropdown
        from src.gui.molecule_library import get_molecule_list
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            molecule_list = get_molecule_list()
            molecule_names = [name for name, _ in molecule_list]
            
            if molecule_names:
                selected = st.selectbox(
                    "Select molecule from library:",
                    options=[""] + molecule_names,
                    help="Choose from available molecules in datasets",
                    key="editor_molecule_select"
                )
                
                if selected:
                    base_smiles = dict(molecule_list)[selected]
                    st.success(f"✅ Selected: `{base_smiles[:50]}...`" if len(base_smiles) > 50 else f"✅ Selected: `{base_smiles}`")
            else:
                st.warning("No molecules loaded from datasets")
        
        with col2:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True, disabled=not base_smiles)
    
    # Analyze molecule on button click
    if analyze_btn and base_smiles:
        # Validate base molecule
        if not validate_smiles(base_smiles):
            st.error("❌ Invalid SMILES string")
        else:
            # Get atom information
            atom_summary = get_atom_summary(base_smiles)
            attachable_atoms = get_attachable_atoms(base_smiles)
            
            # Store in session state
            st.session_state.editor_analyzed_smiles = base_smiles
            st.session_state.editor_atom_summary = atom_summary
            st.session_state.editor_attachable_atoms = attachable_atoms
    
    # Display analysis if available
    if st.session_state.editor_analyzed_smiles is not None:
        analyzed_smiles = st.session_state.editor_analyzed_smiles
        atom_summary = st.session_state.editor_atom_summary
        attachable_atoms = st.session_state.editor_attachable_atoms
        
        st.markdown("---")
        
        # Display molecule and atom info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🧬 Base Molecule")
            render_molecule_3d(analyzed_smiles)
            st.code(analyzed_smiles, language="text")
        
        with col2:
            st.markdown("#### ⚛️ Atom Summary")
            
            if atom_summary:
                df = pd.DataFrame(atom_summary)
                # Highlight attachable atoms
                def highlight_attachable(row):
                    if row['index'] in attachable_atoms:
                        return ['background-color: lightgreen'] * len(row)
                    return [''] * len(row)
                
                styled_df = df.style.apply(highlight_attachable, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                st.caption("🟢 Green = Attachable atoms")
            else:
                st.warning("No atom information available")
        
        # Attachment controls
        if attachable_atoms:
            st.markdown("---")
            st.markdown("### 🔧 Attach Functional Group")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                groups = list_groups()
                selected_group = st.selectbox(
                    "Select Group:",
                    options=groups,
                    help="Choose functional group to attach"
                )
            
            with col2:
                selected_atom = st.selectbox(
                    "Attachment Point:",
                    options=attachable_atoms,
                    help="Select atom index for attachment"
                )
            
            with col3:
                st.write("")  # Spacing
                st.write("")
                attach_btn = st.button("➕ Attach Group", type="primary", use_container_width=True)
            
            if attach_btn:
                with st.spinner("Attaching group and running predictions..."):
                    # Perform attachment
                    modified_smiles = attach_group(analyzed_smiles, selected_group, selected_atom)
                    
                    if modified_smiles is None:
                        st.error("❌ Attachment failed. Invalid modification.")
                    else:
                        # Validate modified molecule
                        validation = validate_modified_molecule(modified_smiles)
                        
                        if not validation['valid']:
                            st.error(f"❌ Invalid molecule: {', '.join(validation['errors'])}")
                        else:
                            # Run predictions
                            base_results = run_full_prediction(analyzed_smiles, engines)
                            mod_results = run_full_prediction(modified_smiles, engines)
                            
                            # Display comparison
                            display_comparison(analyzed_smiles, modified_smiles, base_results, mod_results)
        else:
            st.warning("⚠️ No attachable atoms found in this molecule")

def render_freeform_mode(engines):
    """Free-form SMILES editing interface."""
    st.markdown("### ✏️ Free-Form SMILES Editor")
    
    # Input method for base molecule
    input_method = st.radio(
        "Select base molecule:",
        ["SMILES String", "Select from Library"],
        horizontal=True,
        key="freeform_input_method"
    )
    
    base_smiles = None
    
    if input_method == "SMILES String":
        base_smiles = st.text_input(
            "Original SMILES:",
            value="c1ccccc1",
            help="Original molecule",
            key="freeform_base"
        )
    else:
        from src.gui.molecule_library import get_molecule_list
        
        molecule_list = get_molecule_list()
        molecule_names = [name for name, _ in molecule_list]
        
        if molecule_names:
            selected = st.selectbox(
                "Select original molecule:",
                options=[""] + molecule_names,
                help="Choose from available molecules in datasets",
                key="freeform_base_select"
            )
            
            if selected:
                base_smiles = dict(molecule_list)[selected]
                st.success(f"✅ Selected: `{base_smiles[:50]}...`" if len(base_smiles) > 50 else f"✅ Selected: `{base_smiles}`")
        else:
            st.warning("No molecules loaded from datasets")
    
    # Modified SMILES input
    modified_smiles = st.text_input(
        "Modified SMILES:",
        value="",
        help="Enter your modified SMILES here",
        key="freeform_modified"
    )
    
    compare_btn = st.button("🔄 Compare", type="primary", use_container_width=True)
    
    if compare_btn and base_smiles and modified_smiles:
        # Validate both
        if not validate_smiles(base_smiles):
            st.error("❌ Invalid original SMILES")
            return
        
        if not validate_smiles(modified_smiles):
            st.error("❌ Invalid modified SMILES")
            return
        
        # Validate modified molecule
        validation = validate_modified_molecule(modified_smiles)
        
        if not validation['valid']:
            st.error("❌ Modified molecule validation failed:")
            for error in validation['errors']:
                st.error(f"  • {error}")
            return
        
        # Show detailed edit analysis
        st.markdown("---")
        st.subheader("🔍 Structural Changes Analysis")
        
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        # Get molecules
        old_mol = Chem.MolFromSmiles(base_smiles)
        new_mol = Chem.MolFromSmiles(modified_smiles)
        
        # Detailed comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Atoms",
                new_mol.GetNumAtoms(),
                delta=new_mol.GetNumAtoms() - old_mol.GetNumAtoms()
            )
        
        with col2:
            st.metric(
                "Bonds",
                new_mol.GetNumBonds(),
                delta=new_mol.GetNumBonds() - old_mol.GetNumBonds()
            )
        
        with col3:
            old_mw = Descriptors.MolWt(old_mol)
            new_mw = Descriptors.MolWt(new_mol)
            st.metric(
                "Molecular Weight",
                f"{new_mw:.2f}",
                delta=f"{new_mw - old_mw:+.2f}"
            )
        
        # Atom composition changes
        st.markdown("#### ⚛️ Atom Composition Changes")
        
        # Count atoms by element
        def count_atoms(mol):
            counts = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                counts[symbol] = counts.get(symbol, 0) + 1
            return counts
        
        old_atoms = count_atoms(old_mol)
        new_atoms = count_atoms(new_mol)
        
        # Get all unique elements
        all_elements = set(list(old_atoms.keys()) + list(new_atoms.keys()))
        
        atom_changes = []
        for element in sorted(all_elements):
            old_count = old_atoms.get(element, 0)
            new_count = new_atoms.get(element, 0)
            change = new_count - old_count
            
            if change != 0:
                atom_changes.append({
                    "Element": element,
                    "Original": old_count,
                    "Modified": new_count,
                    "Change": f"{change:+d}"
                })
        
        if atom_changes:
            df_changes = pd.DataFrame(atom_changes)
            st.dataframe(df_changes, use_container_width=True)
        else:
            st.info("No atom composition changes (isomer or conformer)")
        
        # Show edit report
        report = generate_edit_report(base_smiles, modified_smiles)
        st.info(f"**Summary:** {report}")
        
        with st.spinner("Running predictions..."):
            # Run predictions (engines already passed as parameter)
            base_results = run_full_prediction(base_smiles, engines)
            mod_results = run_full_prediction(modified_smiles, engines)
            
            # Display comparison
            display_comparison(base_smiles, modified_smiles, base_results, mod_results)

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

def display_comparison(base_smiles, modified_smiles, base_results, mod_results):
    """Display side-by-side comparison of molecules."""
    st.markdown("---")
    st.markdown("## 📊 Comparison Results")
    
    # Side-by-side 3D structures
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Original")
        st.code(base_smiles, language="text")
        render_molecule_3d(base_smiles)
        render_basic_properties(base_results['properties'])
    
    with col2:
        st.markdown("### 🔴 Modified")
        st.code(modified_smiles, language="text")
        render_molecule_3d(modified_smiles)
        render_basic_properties(mod_results['properties'])
    
    # Property comparison table
    st.markdown("---")
    st.subheader("📈 Property Changes")
    
    comp_table = generate_comparison_table(base_results['properties'], mod_results['properties'])
    st.dataframe(comp_table, use_container_width=True)
    
    # Delta summary
    summary = generate_delta_summary(base_results['properties'], mod_results['properties'])
    st.info(f"**Summary:** {summary}")
    
    # Radar chart overlay
    st.markdown("---")
    st.subheader("🎯 ADMET Profile Comparison")
    
    radar_fig = plot_radar_comparison(base_results['properties'], mod_results['properties'])
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Optimization suggestions
    st.markdown("---")
    st.subheader("💡 Optimization Suggestions")
    
    suggestions = suggest_modifications(mod_results)
    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")
