
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import py3Dmol

def render_molecule_3d(smiles):
    """Render 3D molecule viewer using py3Dmol."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string")
            return
        
        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Convert to mol block
        mol_block = Chem.MolToMolBlock(mol)
        
        # Create 3D viewer
        view = py3Dmol.view(width=400, height=300)
        view.addModel(mol_block, 'sdf')  # Use 'sdf' format to avoid warnings
        view.setStyle({'stick': {}, 'sphere': {'radius': 0.3}})
        view.zoomTo()
        
        # Render in Streamlit
        st.components.v1.html(view._make_html(), height=300)
        
    except Exception as e:
        st.error(f"Failed to render molecule: {e}")

def render_basic_properties(properties):
    """Render basic molecular properties."""
    st.markdown("### Basic Properties")
    
    try:
        col1, col2 = st.columns(2)
        
        # Helper to extract value
        def get_val(prop_data):
            if isinstance(prop_data, dict):
                return prop_data.get('value') or prop_data.get('prediction')
            return prop_data
        
        with col1:
            mw = get_val(properties.get('MW'))
            st.metric("Molecular Weight", f"{mw:.2f}" if mw is not None else "N/A")
            
            logp = get_val(properties.get('logP'))
            st.metric("LogP", f"{logp:.2f}" if logp is not None else "N/A")
        
        with col2:
            tpsa = get_val(properties.get('TPSA'))
            st.metric("TPSA", f"{tpsa:.2f}" if tpsa is not None else "N/A")
            
            hbd = get_val(properties.get('HBD'))
            hba = get_val(properties.get('HBA'))
            st.metric("HBD/HBA", f"{hbd}/{hba}" if hbd is not None and hba is not None else "N/A")
    
    except Exception as e:
        st.error(f"Error rendering properties: {e}")

def render_physicochemical_tab(properties):
    """Render physicochemical properties tab."""
    st.markdown("### Solubility & Lipophilicity")
    
    # Helper to extract value
    def get_val(prop_data):
        if isinstance(prop_data, dict):
            return prop_data.get('value') or prop_data.get('prediction')
        return prop_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        logs = get_val(properties.get('logS'))
        st.metric("LogS (Solubility)", f"{logs:.2f}" if logs is not None else "N/A")
    
    with col2:
        logp = get_val(properties.get('logP'))
        st.metric("LogP (Lipophilicity)", f"{logp:.2f}" if logp is not None else "N/A")
    
    with col3:
        logd = get_val(properties.get('logD'))
        st.metric("LogD (pH 7.4)", f"{logd:.2f}" if logd is not None else "N/A")
    
    st.markdown("### Ionization")
    
    pka = properties.get('pKa', {})
    if isinstance(pka, dict):
        col1, col2 = st.columns(2)
        with col1:
            acidic = get_val(pka.get('acidic'))
            st.metric("pKa (Acidic)", f"{acidic:.2f}" if acidic is not None else "N/A")
        with col2:
            basic = get_val(pka.get('basic'))
            st.metric("pKa (Basic)", f"{basic:.2f}" if basic is not None else "N/A")

def render_permeability_tab(permeability):
    """Render permeability tab."""
    st.markdown("### Permeability Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Caco-2 Permeability")
        caco2 = permeability.get('Caco2', {}) if isinstance(permeability, dict) else {}
        
        # Check for prediction key
        logpapp = caco2.get('prediction') if isinstance(caco2, dict) else None
        
        if logpapp is not None:
            st.metric("LogPapp", f"{logpapp:.2f}")
            st.caption("cm/s")
            # Normalize to 0-1 scale: -8 (poor) to -4 (good)
            progress_val = min(1.0, max(0.0, (logpapp + 8) / 4))
            st.progress(progress_val)
            
            if logpapp > -5:
                st.success("High permeability")
            elif logpapp > -6:
                st.warning("Moderate permeability")
            else:
                st.error("Low permeability")
        else:
            st.info("Caco-2 prediction not available")
    
    with col2:
        st.markdown("#### BBB Permeability")
        bbb = permeability.get('BBB', {}) if isinstance(permeability, dict) else {}
        
        prob = bbb.get('probability') if isinstance(bbb, dict) else None
        
        if prob is not None:
            st.metric("BBB+ Probability", f"{prob:.2%}")
            st.progress(prob)
            
            if prob > 0.7:
                st.success("High BBB permeability")
            elif prob > 0.3:
                st.warning("Moderate BBB permeability")
            else:
                st.error("Low BBB permeability")
        else:
            st.info("BBB prediction not available")

def render_toxicity_tab(toxicity):
    """Render toxicity tab."""
    st.markdown("### Toxicity Predictions")
    
    clintox = toxicity.get('clintox', {})
    
    if clintox.get('probability') is not None:
        prob = clintox['probability']
        risk_level = clintox.get('risk_level', 'Unknown')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Clinical Toxicity (ClinTox)")
            st.metric("Toxicity Probability", f"{prob:.2%}")
            st.progress(prob)
        
        with col2:
            st.markdown("#### Risk Level")
            if risk_level == "Low":
                st.success(f"✅ {risk_level}")
            elif risk_level == "Moderate":
                st.warning(f"⚠️ {risk_level}")
            else:
                st.error(f"❌ {risk_level}")
    else:
        st.info("Toxicity prediction not available")

def render_druglikeness_tab(druglikeness):
    """Render drug-likeness tab."""
    st.markdown("### Drug-Likeness Assessment")
    
    score = druglikeness.get('score')
    if score is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric("Drug-Likeness Score", f"{score:.1f}/100")
            st.progress(score / 100)
        
        with col2:
            dl_class = druglikeness.get('class', 'Unknown')
            if dl_class == "Good":
                st.success(f"✅ {dl_class}")
            elif dl_class == "Moderate":
                st.warning(f"⚠️ {dl_class}")
            else:
                st.error(f"❌ {dl_class}")
        
        st.markdown("---")
        
        # Lipinski Rule of 5
        lipinski = druglikeness.get('lipinski', {})
        st.markdown("#### Lipinski Rule of 5")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mw = lipinski.get('mw')
            if mw:
                st.metric("MW", f"{mw:.0f}")
                st.caption(f"{'✅' if mw <= 500 else '❌'} ≤ 500")
        
        with col2:
            logp = lipinski.get('logp')
            if logp is not None:
                st.metric("LogP", f"{logp:.2f}")
                st.caption(f"{'✅' if logp <= 5 else '❌'} ≤ 5")
        
        with col3:
            hbd = lipinski.get('hbd')
            if hbd is not None:
                st.metric("HBD", f"{hbd}")
                st.caption(f"{'✅' if hbd <= 5 else '❌'} ≤ 5")
        
        with col4:
            hba = lipinski.get('hba')
            if hba is not None:
                st.metric("HBA", f"{hba}")
                st.caption(f"{'✅' if hba <= 10 else '❌'} ≤ 10")
        
        violations = lipinski.get('violations', 0)
        if violations <= 1:
            st.success(f"✅ Passes Lipinski Rule of 5 ({violations} violation{'s' if violations != 1 else ''})")
        else:
            st.error(f"❌ Fails Lipinski Rule of 5 ({violations} violations)")
        
        # Veber's Rule
        st.markdown("#### Veber's Rule")
        veber = druglikeness.get('veber', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            tpsa = veber.get('tpsa')
            if tpsa:
                st.metric("TPSA", f"{tpsa:.1f}")
                st.caption(f"{'✅' if tpsa <= 140 else '❌'} ≤ 140")
        
        with col2:
            rotatable = veber.get('rotatable_bonds')
            if rotatable is not None:
                st.metric("Rotatable Bonds", f"{rotatable}")
                st.caption(f"{'✅' if rotatable <= 10 else '❌'} ≤ 10")
        
        if veber.get('pass'):
            st.success("✅ Passes Veber's Rule")
        else:
            st.error("❌ Fails Veber's Rule")
    else:
        st.info("Drug-likeness assessment not available")

def render_risk_gauge(risk):
    """Render overall risk gauge."""
    score = risk.get('overall_score')
    
    if score is not None:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Development Risk Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': risk.get('risk_color', 'gray')},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level
        risk_level = risk.get('risk_level', 'Unknown')
        if risk_level == "Low":
            st.success(f"✅ {risk_level} Risk")
        elif risk_level == "Moderate":
            st.warning(f"⚠️ {risk_level} Risk")
        else:
            st.error(f"❌ {risk_level} Risk")
    else:
        st.info("Risk assessment not available")

def render_radar_chart(predictions):
    """Render radar chart for ADMET profile."""
    try:
        # Prepare data
        categories = []
        values = []
        
        # Solubility (inverse of risk)
        logs = predictions.get('properties', {}).get('logS')
        if logs is not None:
            sol_score = min(100, max(0, (logs + 8) * 12.5))  # -8 to 0 → 0 to 100
            categories.append('Solubility')
            values.append(sol_score)
        
        # Permeability (BBB)
        bbb_prob = predictions.get('permeability', {}).get('bbb', {}).get('probability')
        if bbb_prob is not None:
            categories.append('Permeability')
            values.append(bbb_prob * 100)
        
        # Drug-likeness
        dl_score = predictions.get('druglikeness', {}).get('score')
        if dl_score is not None:
            categories.append('Drug-Likeness')
            values.append(dl_score)
        
        # Safety (inverse of toxicity)
        tox_prob = predictions.get('toxicity', {}).get('clintox', {}).get('probability')
        if tox_prob is not None:
            categories.append('Safety')
            values.append((1 - tox_prob) * 100)
        
        if len(categories) >= 3:
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='ADMET Profile'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for radar chart")
    
    except Exception as e:
        st.error(f"Error rendering radar chart: {e}")

def plot_radar_comparison(base_results, modified_results):
    """
    Create overlay radar chart comparing base and modified molecules.
    
    Args:
        base_results: Property dict for base molecule
        modified_results: Property dict for modified molecule
    
    Returns:
        plotly.graph_objects.Figure
    """
    # Helper to extract value
    def get_val(data):
        if isinstance(data, dict):
            return data.get('value') or data.get('prediction')
        return data
    
    # Prepare data
    categories = []
    base_values = []
    mod_values = []
    
    # Solubility (normalize: -8 to 0 → 0 to 100)
    base_logs = get_val(base_results.get('logS'))
    mod_logs = get_val(modified_results.get('logS'))
    if base_logs is not None and mod_logs is not None:
        categories.append('Solubility')
        base_values.append(min(100, max(0, (base_logs + 8) * 12.5)))
        mod_values.append(min(100, max(0, (mod_logs + 8) * 12.5)))
    
    # Lipophilicity (normalize: 0 to 5 → 0 to 100)
    base_logp = get_val(base_results.get('logP'))
    mod_logp = get_val(modified_results.get('logP'))
    if base_logp is not None and mod_logp is not None:
        categories.append('Lipophilicity')
        base_values.append(min(100, max(0, base_logp * 20)))
        mod_values.append(min(100, max(0, mod_logp * 20)))
    
    # TPSA (normalize: 0 to 140 → 0 to 100)
    base_tpsa = get_val(base_results.get('TPSA'))
    mod_tpsa = get_val(modified_results.get('TPSA'))
    if base_tpsa is not None and mod_tpsa is not None:
        categories.append('Polarity')
        base_values.append(min(100, max(0, base_tpsa / 1.4)))
        mod_values.append(min(100, max(0, mod_tpsa / 1.4)))
    
    # BBB Permeability
    base_perm = base_results.get('permeability', {})
    mod_perm = modified_results.get('permeability', {})
    
    if isinstance(base_perm, dict) and isinstance(mod_perm, dict):
        base_bbb = base_perm.get('BBB', {}).get('probability')
        mod_bbb = mod_perm.get('BBB', {}).get('probability')
        
        if base_bbb is not None and mod_bbb is not None:
            categories.append('BBB Penetration')
            base_values.append(base_bbb * 100)
            mod_values.append(mod_bbb * 100)
    
    if len(categories) < 3:
        return None
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=base_values,
        theta=categories,
        fill='toself',
        name='Original',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=mod_values,
        theta=categories,
        fill='toself',
        name='Modified',
        line=dict(color='red'),
        opacity=0.7
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig
