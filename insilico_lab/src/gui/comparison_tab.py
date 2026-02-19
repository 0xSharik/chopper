"""
Multi-Molecule Comparison Tab
Side-by-side comparison of drug candidates with automated ranking.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.analysis.ranking_engine import rank_molecules, generate_executive_summary
from src.gui.components import render_molecule_3d
from src.chemistry.smiles_editor import validate_smiles

def render_comparison_tab(engines):
    """
    Render multi-molecule comparison interface.
    
    Args:
        engines: Dict of prediction engines
    """
    st.subheader("📊 Multi-Molecule Comparison")
    st.markdown("Compare multiple drug candidates side-by-side with automated ranking")
    
    # Input section
    st.markdown("### 🧬 Input Molecules")
    
    num_molecules = st.number_input(
        "Number of molecules to compare:",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        help="Compare 2-10 molecules"
    )
    
    molecules = []
    
    for i in range(num_molecules):
        st.markdown(f"#### Molecule {i + 1}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input method
            input_method = st.radio(
                f"Input method for Molecule {i + 1}:",
                ["SMILES String", "Select from Library"],
                horizontal=True,
                key=f"comp_input_method_{i}"
            )
            
            smiles = None
            name = f"Molecule {i + 1}"
            
            if input_method == "SMILES String":
                smiles = st.text_input(
                    f"SMILES:",
                    value="",
                    key=f"comp_smiles_{i}",
                    help="Enter SMILES string"
                )
                name = st.text_input(
                    f"Name (optional):",
                    value=f"Molecule {i + 1}",
                    key=f"comp_name_{i}"
                )
            else:
                from src.gui.molecule_library import get_molecule_list
                
                molecule_list = get_molecule_list()
                molecule_names = [n for n, _ in molecule_list]
                
                if molecule_names:
                    selected = st.selectbox(
                        f"Select molecule:",
                        options=[""] + molecule_names,
                        key=f"comp_select_{i}"
                    )
                    
                    if selected:
                        smiles = dict(molecule_list)[selected]
                        name = selected
                        st.success(f"✅ {smiles[:40]}...")
            
            if smiles:
                molecules.append({"smiles": smiles, "name": name})
        
        with col2:
            if smiles:
                st.markdown("**Preview:**")
                render_molecule_3d(smiles)
    
    # Run comparison button
    st.markdown("---")
    compare_btn = st.button("🔬 Run Comparison", type="primary", use_container_width=True)
    
    if compare_btn:
        if len(molecules) < 2:
            st.error("❌ Please input at least 2 molecules to compare")
            return
        
        # Validate all molecules
        valid_molecules = []
        for mol in molecules:
            if validate_smiles(mol['smiles']):
                valid_molecules.append(mol)
            else:
                st.error(f"❌ Invalid SMILES for {mol['name']}: {mol['smiles']}")
        
        if len(valid_molecules) < 2:
            st.error("❌ Need at least 2 valid molecules to compare")
            return
        
        with st.spinner("Running predictions and ranking..."):
            # Run predictions for all molecules
            results_list = []
            
            for mol in valid_molecules:
                try:
                    properties = engines['property'].predict_properties(mol['smiles'])
                    toxicity = engines['toxicity'].predict(mol['smiles'])
                    druglikeness = engines['druglikeness'].calculate(mol['smiles'])
                    
                    all_predictions = {
                        'smiles': mol['smiles'],
                        'name': mol['name'],
                        'properties': properties,
                        'toxicity': toxicity,
                        'permeability': properties.get('permeability', {}),
                        'druglikeness': druglikeness
                    }
                    
                    risk = engines['risk'].calculate_risk(all_predictions)
                    all_predictions['risk'] = risk
                    
                    results_list.append(all_predictions)
                
                except Exception as e:
                    st.error(f"Error predicting {mol['name']}: {str(e)}")
            
            if len(results_list) < 2:
                st.error("❌ Failed to generate predictions for comparison")
                return
            
            # Rank molecules
            ranked = rank_molecules(results_list)
            
            # Display results
            display_comparison_results(ranked)

def display_comparison_results(ranked):
    """Display comparison results with visualizations."""
    
    st.markdown("---")
    
    # Executive Summary
    st.markdown("## 🏆 Executive Summary")
    summary = generate_executive_summary(ranked)
    st.info(summary)
    
    st.markdown("---")
    
    # Ranking Table
    st.markdown("## 📊 Ranking Results")
    
    ranking_data = []
    for mol in ranked:
        ranking_data.append({
            "Rank": f"#{mol['rank']}",
            "Molecule": mol['name'],
            "Score": f"{mol['final_score']:.1f}/100",
            "Strengths": ", ".join(mol.get('strengths', [])),
            "Weaknesses": ", ".join(mol.get('weaknesses', []))
        })
    
    df_ranking = pd.DataFrame(ranking_data)
    st.dataframe(df_ranking, use_container_width=True)
    
    st.markdown("---")
    
    # Property Comparison Table
    st.markdown("## 📈 Property Comparison")
    
    display_property_comparison_table(ranked)
    
    st.markdown("---")
    
    # Visual Comparisons
    st.markdown("## 🎯 Visual Comparisons")
    
    # Radar chart overlay
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Radar Chart Overlay")
        radar_fig = create_multi_radar_chart(ranked)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Comparison")
        risk_fig = create_risk_comparison_chart(ranked)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)
    
    # 3D Structure Grid
    st.markdown("---")
    st.markdown("## 🧬 Molecular Structures")
    
    cols = st.columns(len(ranked))
    for idx, (col, mol) in enumerate(zip(cols, ranked)):
        with col:
            st.markdown(f"**#{mol['rank']} {mol['name']}**")
            render_molecule_3d(mol['smiles'])
            st.caption(f"Score: {mol['final_score']:.1f}")

def display_property_comparison_table(ranked):
    """Create property comparison table with best value highlighting."""
    
    # Helper to extract value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction') or data.get('score')
        return data
    
    # Build comparison data
    properties_to_compare = [
        ('MW', 'properties', 'MW', 'lower'),
        ('LogP', 'properties', 'logP', 'optimal_2'),
        ('LogS', 'properties', 'logS', 'higher'),
        ('TPSA', 'properties', 'TPSA', 'optimal_90'),
        ('Drug-likeness', 'druglikeness', 'score', 'higher'),
        ('Risk Score', 'risk', 'overall_score', 'lower')
    ]
    
    table_data = []
    
    for prop_name, category, key, best_type in properties_to_compare:
        row = {"Property": prop_name}
        values = []
        
        for mol in ranked:
            if category in mol['raw_results']:
                val = get_val(mol['raw_results'][category].get(key))
                if val is not None:
                    row[mol['name']] = f"{val:.2f}"
                    values.append((val, mol['name']))
                else:
                    row[mol['name']] = "N/A"
            else:
                row[mol['name']] = "N/A"
        
        # Determine best value
        if values:
            if best_type == 'lower':
                best_name = min(values, key=lambda x: x[0])[1]
            elif best_type == 'higher':
                best_name = max(values, key=lambda x: x[0])[1]
            elif best_type == 'optimal_2':
                best_name = min(values, key=lambda x: abs(x[0] - 2))[1]
            elif best_type == 'optimal_90':
                best_name = min(values, key=lambda x: abs(x[0] - 90))[1]
            
            row['Best'] = f"✅ {best_name}"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def create_multi_radar_chart(ranked):
    """Create radar chart with all molecules overlaid."""
    
    # Helper to extract value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction')
        return data
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, mol in enumerate(ranked):
        props = mol['raw_results']['properties']
        
        categories = []
        values = []
        
        # Solubility (normalize: -8 to 0 → 0 to 100)
        logs = get_val(props.get('logS'))
        if logs is not None:
            categories.append('Solubility')
            values.append(min(100, max(0, (logs + 8) * 12.5)))
        
        # Lipophilicity (normalize: 0 to 5 → 0 to 100)
        logp = get_val(props.get('logP'))
        if logp is not None:
            categories.append('Lipophilicity')
            values.append(min(100, max(0, logp * 20)))
        
        # TPSA (normalize: 0 to 140 → 0 to 100)
        tpsa = get_val(props.get('TPSA'))
        if tpsa is not None:
            categories.append('Polarity')
            values.append(min(100, max(0, tpsa / 1.4)))
        
        # Drug-likeness
        dl = get_val(mol['raw_results']['druglikeness'].get('score'))
        if dl is not None:
            categories.append('Drug-likeness')
            values.append(min(100, max(0, dl)))
        
        if len(categories) >= 3:
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"#{mol['rank']} {mol['name']}",
                line=dict(color=colors[idx % len(colors)])
            ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=400
    )
    
    return fig

def create_risk_comparison_chart(ranked):
    """Create grouped bar chart for risk comparison."""
    
    # Helper to extract value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction') or data.get('score')
        return data
    
    fig = go.Figure()
    
    molecule_names = [mol['name'] for mol in ranked]
    
    # Overall Risk
    overall_risks = []
    for mol in ranked:
        risk = get_val(mol['raw_results']['risk'].get('overall_score'))
        overall_risks.append(risk if risk is not None else 0)
    
    fig.add_trace(go.Bar(
        name='Overall Risk',
        x=molecule_names,
        y=overall_risks,
        marker_color='indianred'
    ))
    
    # Drug-likeness (inverted to show as positive)
    dl_scores = []
    for mol in ranked:
        dl = get_val(mol['raw_results']['druglikeness'].get('score'))
        dl_scores.append(dl if dl is not None else 0)
    
    fig.add_trace(go.Bar(
        name='Drug-likeness',
        x=molecule_names,
        y=dl_scores,
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        barmode='group',
        yaxis_title='Score',
        height=400,
        showlegend=True
    )
    
    return fig
