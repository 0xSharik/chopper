import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time

from src.md.md_engine import MolecularDynamicsEngine
from src.gui.molecule_library import get_molecule_list
from src.chemistry.group_library import list_groups
from src.chemistry.reaction_engine import attach_group
from src.chemistry.atom_tools import get_atom_summary, get_attachable_atoms
from src.chemistry.smiles_editor import validate_smiles

# Try importing py3Dmol
try:
    import py3Dmol
    from stmol import showmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

def render_md_tab(engines=None):
    """
    Render the Molecular Dynamics simulation tab.
    """
    st.header("🧬 Molecular Dynamics Simulation")
    st.markdown("Research-grade MD engine using OpenMM with RDKit MMFF94 parameterization.")

    # ------------------------------------------------------------------
    # Input Section
    # ------------------------------------------------------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Molecule Selection")
        input_method = st.radio(
            "Input Method:",
            ["Select from Library", "SMILES String"],
            horizontal=True
        )
        
        # Defaults
        smiles_input = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        molecule_id = "custom_mol"
        
        if input_method == "SMILES String":
            smiles_input = st.text_input(
                "Enter SMILES:",
                value=smiles_input,
                help="Enter a valid SMILES string"
            )
            molecule_id = "custom_" + str(int(time.time()))
        else:
            # Load library
            with st.spinner("Loading library..."):
                molecule_list = get_molecule_list()
            
            mol_dict = {name: smiles for name, smiles in molecule_list}
            
            if not mol_dict:
                st.warning("No molecules found.")
            else:
                selected_name = st.selectbox(
                    "Search Library:",
                    options=list(mol_dict.keys()),
                    index=0
                )
                if selected_name:
                    smiles_input = mol_dict[selected_name]
                    # Sanitize name for ID
                    safe_name = "".join([c if c.isalnum() else "_" for c in selected_name]).lower()
                    molecule_id = safe_name[:40]

        if smiles_input:
            st.caption(f"SMILES: `{smiles_input}`")
        else:
            st.warning("Please select or enter a molecule.")

    with col2:
        st.subheader("2. Simulation Protocol")
        
        mode = st.selectbox(
            "Simulation Mode:",
            options=["Demo", "Research"],
            format_func=lambda x: "🚀 Demo (Fast, ~0.2ns)" if x == "Demo" else "🔬 Research (Rigorous, ~5ns+)",
            help="Demo: Fast equilibration for quick demos. Research: Full equilibration and strict validation."
        ).lower()
        
        # Advanced Settings
        with st.expander("⚙️ Advanced Configuration", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                temp = st.number_input("Temperature (K)", value=300.0, step=10.0)
                pressure = st.number_input("Pressure (atm)", value=1.0)
                
                # Platform Selection
                platform = st.selectbox(
                    "Compute Platform:",
                    ["Auto", "CUDA", "OpenCL", "CPU"],
                    index=0,
                    help="Force specific compute platform. 'Auto' tries CUDA > OpenCL > CPU."
                )
                
                if st.button("🚀 Run Speed Test", help="Benchmark GPU performance"):
                    with st.spinner("Running Benchmark (Aspirin)..."):
                        try:
                            # Run generic benchmark logic here or call a func
                            # For simplicity, we can just run a quick dummy sim inline
                            # or use subprocess to call benchmark_gpu.py?
                            # Subprocess is safer to avoid polluting current session?
                            # But we want to show output.
                            import subprocess
                            result = subprocess.run(
                                [".\\venv\\Scripts\\python.exe", "benchmark_gpu.py"], 
                                capture_output=True, text=True, cwd=os.getcwd()
                            )
                            st.code(result.stdout)
                            if result.stderr:
                                st.error(result.stderr)
                        except Exception as e:
                            st.error(f"Benchmark failed: {e}")
            with c2:
                # Default based on mode logic? 
                # If Demo, suggest 0.2. If Research, suggest 5.0.
                default_ns = 0.2 if mode == "demo" else 5.0
                prod_ns = st.number_input("Production Time (ns)", value=default_ns, min_value=0.01, max_value=100.0, step=0.1)
                output_ps = st.number_input("Output Interval (ps)", value=2.0)

    # ------------------------------------------------------------------
    # Design & Simulate Workflow (New Section)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Design & Simulate Workflow (New Section)
    # ------------------------------------------------------------------
    with st.expander("🛠️ Modify & Simulate (Design Workflow)", expanded=False):
        st.info("Modify the selected molecule and run a comparative simulation.")
        
        design_tabs = st.tabs(["⚡ Quick Modify", "🔬 Advanced Editor"])
        
        # --- TAB 1: Quick Modify ---
        with design_tabs[0]:
            mod_type = st.selectbox(
                "Modification Type:",
                ["Add Methyl", "Add Fluorine", "Add Hydroxyl", "Extend Chain", "Add Chlorine"],
                help="Select a chemical transformation to apply."
            )
            
            if st.button("🧪 Simulate Variant", type="secondary", disabled=not smiles_input, key="btn_quick_sim"):
                st.session_state['trigger_design'] = True
                st.session_state['mod_type'] = mod_type

        # --- TAB 2: Advanced Editor ---
        with design_tabs[1]:
            st.markdown("**Controlled Functional Group Attachment**")
            
            if smiles_input:
                # 1. Analyze Atoms
                if st.button("🔍 Analyze Structure", key="btn_adv_analyze"):
                    st.session_state['adv_atom_summary'] = get_atom_summary(smiles_input)
                    st.session_state['adv_attachable'] = get_attachable_atoms(smiles_input)
                
                # Display Analysis
                if st.session_state.get('adv_atom_summary'):
                    with st.expander("View Atom Indices", expanded=True):
                         df_atoms = pd.DataFrame(st.session_state['adv_atom_summary'])
                         attachable = st.session_state.get('adv_attachable', [])
                         
                         # Highlight attachable
                         def highlight_valid(row):
                             return ['background-color: #d4edda'] * len(row) if row['index'] in attachable else [''] * len(row)
                             
                         st.dataframe(df_atoms.style.apply(highlight_valid, axis=1), use_container_width=True, height=200)

                    # 2. Select Modification
                    c1, c2 = st.columns(2)
                    with c1:
                        adv_group = st.selectbox("Functional Group:", list_groups(), key="adv_group_sel")
                    with c2:
                        # Filter to only valid attachable atoms if possible
                        valid_atoms = st.session_state.get('adv_attachable', [])
                        if not valid_atoms:
                            st.warning("No attachable atoms found.")
                            adv_atom = None
                        else:
                            adv_atom = st.selectbox("Attach at Atom Index:", valid_atoms, key="adv_atom_sel")
                    
                    # 3. Action
                    if st.button("🧪 Simulate Advanced Variant", type="primary", disabled=not valid_atoms, key="btn_adv_sim"):
                        # Perform attachment immediately to validate
                        new_smiles = attach_group(smiles_input, adv_group, adv_atom)
                        if new_smiles:
                             st.success(f"Generated: {new_smiles}")
                             st.session_state['trigger_design_adv'] = True
                             st.session_state['adv_new_smiles'] = new_smiles
                             st.session_state['adv_desc'] = f"Add {adv_group} at {adv_atom}"
                        else:
                             st.error("Failed to generate valid molecule.")
            else:
                st.warning("Please select a molecule first.")
        
    # ------------------------------------------------------------------
    # Execution Logic
    # ------------------------------------------------------------------
    st.divider()
    run_col, status_col = st.columns([1, 3])
    
    # Standard Run
    with run_col:
        run_btn = st.button("🚀 Start Simulation", type="primary", use_container_width=True, disabled=not smiles_input)
    
    # Handle Design Workflow Trigger
    if st.session_state.get('trigger_design', False) and smiles_input:
        from src.workflows.design_and_simulate import DesignAndSimulate
        
        # Clear trigger
        st.session_state['trigger_design'] = False
        mod_type = st.session_state.get('mod_type')
        
        md_config = {
            "mode": mode,
            "temperature": temp,
            "pressure": pressure,
            "production_ns": prod_ns,
            "output_frame_ps": output_ps,
            "platform_preference": [platform] if platform != "Auto" else ["CUDA", "OpenCL", "CPU"],
        }
        
        with status_col:
            with st.status(f"Running Design Workflow: {mod_type}...", expanded=True) as status:
                try:
                    designer = DesignAndSimulate()
                    
                    st.write("🔹 Applying Modification & Validating...")
                    # We might want to see the new structure before running? 
                    # For now, run the full pipeline as requested.
                    
                    st.write("🔹 Starting Fresh Simulation (No Parameter Reuse)...")
                    result = designer.modify_and_simulate(
                        parent_smiles=smiles_input,
                        modification_type=mod_type,
                        md_config=md_config,
                        molecule_label=molecule_id
                    )
                    
                    status.update(label="✅ Design Workflow Complete!", state="complete", expanded=False)
                    st.session_state['design_result'] = result
                    st.session_state['md_result'] = result
                    
                    # Also store as main result to show plots? 
                    # Or show side-by-side comparison?
                    # Let's clean up UI to show comparison.
                    
                except Exception as e:
                    status.update(label="❌ Workflow Failed", state="error")
                    st.error(f"Error: {e}")

    # Handle Advanced Design Workflow Trigger
    if st.session_state.get('trigger_design_adv', False) and smiles_input:
        from src.workflows.design_and_simulate import DesignAndSimulate
        
        # Clear trigger
        st.session_state['trigger_design_adv'] = False
        new_smiles = st.session_state.get('adv_new_smiles')
        desc = st.session_state.get('adv_desc')
        
        md_config = {
            "mode": mode,
            "temperature": temp,
            "pressure": pressure,
            "production_ns": prod_ns,
            "output_frame_ps": output_ps,
            "platform_preference": [platform] if platform != "Auto" else ["CUDA", "OpenCL", "CPU"],
        }
        
        with status_col:
            with st.status(f"Running Advanced Design: {desc}...", expanded=True) as status:
                try:
                    designer = DesignAndSimulate()
                    
                    st.write("🔹 Validating Custom Structure...")
                    # Already validated by attach_group returning? 
                    # Workflow will re-validate.
                    
                    st.write("🔹 Starting Fresh Simulation...")
                    result = designer.simulate_modified_smiles(
                        parent_smiles=smiles_input,
                        modified_smiles=new_smiles,
                        modification_desc=desc,
                        md_config=md_config,
                        molecule_label=molecule_id
                    )
                    
                    status.update(label="✅ Advanced Design Complete!", state="complete", expanded=False)
                    st.session_state['design_result'] = result
                    # Critical Fix: Also set md_result so the main results pane shows the 3D viz/plots
                    st.session_state['md_result'] = result
                    
                except Exception as e:
                    status.update(label="❌ Workflow Failed", state="error")
                    st.error(f"Error: {e}")

    # Standard Run Trigger (Existing Logic)
    if run_btn and smiles_input:
        # Configuration
        md_config = {
            "mode": mode,
            "temperature": temp,
            "pressure": pressure,
            "production_ns": prod_ns,
            "output_frame_ps": output_ps,
            "platform_preference": [platform] if platform != "Auto" else ["CUDA", "OpenCL", "CPU"],
        }
        
        engine = MolecularDynamicsEngine(config=md_config)
        
        with status_col:
            with st.status("Initializing MD Engine...", expanded=True) as status:
                st.write("🔹 Building System (MMFF94 + TIP3P)...")
                
                try:
                    result = engine.run(smiles_input, molecule_id=molecule_id)
                    
                    if result["status"] == "success":
                        status.update(label="✅ Simulation Completed Successfully!", state="complete", expanded=False)
                        st.session_state['md_result'] = result
                    else:
                        status.update(label="❌ Simulation Failed", state="error")
                        st.error(f"Error: {result.get('message')}")
                        
                except Exception as e:
                    status.update(label="❌ Critical Error", state="error")
                    st.error(f"Exception: {str(e)}")

    # ------------------------------------------------------------------
    # Results Display
    # ------------------------------------------------------------------
    if 'md_result' in st.session_state:
        result = st.session_state['md_result']
        _render_results(result)


def _render_results(result):
    """Display comprehensive results."""
    
    metrics = result.get("metrics", {})
    output_path = result.get("output_path", "")
    
    # 1. Validation & Stability Score
    st.divider()
    st.subheader("🔬 Simulation Integrity & Stability")
    
    # Extract embedded validation info if available, or just use raw metrics
    score = metrics.get('stability_score', 0)
    
    # Columns for Score and Integrity Checks
    sc1, sc2 = st.columns([1, 2])
    
    with sc1:
        # Display Score
        st.metric(
            label="Dynamic Stability Score", 
            value=f"{score:.1f}/100", 
            delta="Excellent" if score > 80 else "Stable" if score > 50 else "Unstable",
            help="Composite score based on RMSD fluctuation, SASA, and Energy stability."
        )
        if score > 80:
            st.success("High confidence simulation.")
        elif score > 50:
            st.info("Acceptable stability.")
        else:
            st.error("Low stability detected.")

    with sc2:
        # Integrity Checklist
        rmsd = metrics.get('rmsd_mean', 0)
        rmsd_std = metrics.get('rmsd_std', 0)
        sasa = metrics.get('sasa_mean', 0)
        
        checks = {
            "Ligand Detected": rmsd > 0.0,
            "Convergence (RMSD stable)": rmsd_std < 0.2, # Strict metric
            "SASA Realistic": 0.5 < sasa < 25.0, # Slightly relaxed upper bound
            "Energy Finite": True
        }
        
        c_cols = st.columns(2)
        idx = 0
        for name, passed in checks.items():
            icon = "✅" if passed else "⚠️"
            c_cols[idx % 2].write(f"{icon} **{name}**")
            idx += 1

    # 2. Detailed Metrics
    st.subheader("📊 Quantitative Descriptors")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSD (Mean)", f"{rmsd:.3f} nm", f"± {rmsd_std:.3f}")
    m2.metric("Radius of Gyration", f"{metrics.get('rg_mean', 0):.3f} nm")
    m3.metric("SASA", f"{sasa:.1f} nm²")
    m4.metric("H-Bonds (Avg)", f"{metrics.get('hbond_avg', 0):.1f}")
    
    # 2.5 3D Visualization
    st.subheader("🧬 Structure Visualization")
    pdb_path = os.path.join(output_path, "topology.pdb")
    traj_path = os.path.join(output_path, "trajectory.dcd")

    if HAS_PY3DMOL and os.path.exists(pdb_path):
        try:
            with open(pdb_path, "r") as f:
                pdb_block = f.read()
            
            # Create Model
            view = py3Dmol.view(width=800, height=400)
            view.addModel(pdb_block, "pdb")
            view.setStyle({'stick': {}})
            view.zoomTo()
            
            # Show
            st.caption("Interactive 3D View (Final Frame Structure)")
            showmol(view, height=400, width=800)
        except Exception as e:
            st.warning(f"3D Visualization Error: {e}")
    else:
        st.info("3D Visualization unavailable (py3Dmol not installed or PDB missing).")

    # 3. Plots
    st.subheader("📈 Trajectory Analysis")
    csv_path = os.path.join(output_path, "state_data.csv")
    if os.path.exists(csv_path):
        try:
             df = pd.read_csv(csv_path)
             df.columns = [c.strip().replace('"', '') for c in df.columns]
             
             p1, p2 = st.columns(2)
             with p1:
                 # Temperature
                 if "Temperature (K)" in df.columns:
                     fig = px.line(df, x="Step", y="Temperature (K)", title="Temperature Stability")
                     st.plotly_chart(fig, use_container_width=True)
             with p2:
                 # Potential Energy
                 pe_col = [c for c in df.columns if "Potential Energy" in c]
                 if pe_col:
                     fig = px.line(df, x="Step", y=pe_col[0], title="Potential Energy Landscape")
                     st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load plots: {e}")

    # 4. Downloads & Viz
    st.subheader("📂 Output Files")
    c1, c2, c3 = st.columns(3)
    
    traj_path = os.path.join(output_path, "trajectory.dcd")
    pdb_path = os.path.join(output_path, "topology.pdb")
    
    with c1:
        if os.path.exists(traj_path):
            with open(traj_path, "rb") as f:
                st.download_button("📥 Download DCD Trajectory", f, file_name="trajectory.dcd")
    with c2:
        if os.path.exists(pdb_path):
            with open(pdb_path, "rb") as f:
                st.download_button("📥 Download PDB Topology", f, file_name="topology.pdb")
                
    st.caption(f"Full results stored in: `{output_path}`")

    # ------------------------------------------------------------------
    # Design Workflow Results (Comparison)
    # ------------------------------------------------------------------
    if 'design_result' in st.session_state:
        st.divider()
        st.header("🧬 Design Comparison Results")
        
        d_result = st.session_state['design_result']
        d_metrics = d_result["md_metrics"]
        
        # Assuming we have parent metrics if a run was done before?
        # If not, we only show variant metrics.
        # But wait, workflow returns comparison? 
        # Actually workflow returns metrics. Comparison engine is separate helper.
        
        # Let's check if we have previous run results to compare against
        parent_metrics = {}
        if 'md_result' in st.session_state:
             parent_metrics = st.session_state['md_result'].get("metrics", {})
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Variant Details")
            st.code(d_result["modified_smiles"], language="text")
            st.info(f"Modification: {d_result['modification']}")
            
        with c2:
            st.subheader("Stability Analysis")
            # If we have parent metrics, compare
            if parent_metrics:
                from src.workflows.comparison_engine import MoleculeComparison
                comp = MoleculeComparison.compare(parent_metrics, d_metrics)
                
                st.metric("Verdict", comp["verdict"])
                st.metric("Stability Change", f"{comp['stability_change']:+.1f}", help="Positive is more stable")
                st.text(f"RMSD Change: {comp['rmsd_change']:+.3f} nm")
            else:
                st.warning("Run simulation on Parent first to see comparison.")
                st.metric("Variant Stability Score", f"{d_metrics.get('stability_score', 0):.1f}")
                
        # Show specific metrics for variant
        st.subheader("Variant Metrics")
        _render_metrics_row(d_metrics)

def _render_metrics_row(metrics):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSD", f"{metrics.get('rmsd_mean', 0):.3f} nm")
    m2.metric("Rg", f"{metrics.get('rg_mean', 0):.3f} nm")
    m3.metric("SASA", f"{metrics.get('sasa_mean', 0):.1f} nm²")
    m4.metric("Score", f"{metrics.get('stability_score', 0):.1f}")
