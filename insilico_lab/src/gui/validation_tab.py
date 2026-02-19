
import streamlit as st
import pandas as pd
import os
from src.validation.benchmark_logp import LogPBenchmark

def render_validation_tab():
    """
    Render key model validation benchmarks.
    """
    st.header("📊 Model Validation Benchmark")
    st.markdown("Validate the internal property prediction engine against consensus literature values.")
    
    st.info("Benchmarks run against standard small molecule datasets (Aspirin, Caffeine, Ibuprofen).")

    if st.button("🚀 Run LogP Benchmark", type="primary"):
        with st.spinner("Running Benchmark vs Literature..."):
            try:
                # Run Benchmark
                bench = LogPBenchmark()
                results = bench.run_benchmark()
                
                # Extract Metrics
                mae = results['mae']
                rmse = results['rmse']
                df = pd.DataFrame(results['results'])
                
                # 1. Summary Metrics
                st.subheader("Model Performance Summary")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE (Mean Abs Error)", f"{mae:.3f} log units", 
                          delta="Excellent" if mae < 0.5 else "Acceptable" if mae < 1.0 else "Needs Improvement",
                          delta_color="inverse")
                c2.metric("RMSE (Root Mean Sq Error)", f"{rmse:.3f} log units")
                c3.metric("Molecules Tested", len(df))
                
                # Interpretation
                if mae < 0.5:
                    st.success("✅ **Excellent Agreement:** Model is highly accurate for screening.")
                elif mae < 1.0:
                    st.warning("⚠️ **Acceptable:** Good for rough screening, but check outliers.")
                else:
                    st.error("❌ **Poor Performance:** Retraining recommended.")
                
                # 2. Detailed Table
                st.subheader("Detailed Comparison")
                # Style the dataframe: Error column color gradient
                st.dataframe(
                    df[["name", "literature_logp", "predicted_logp", "error", "abs_error"]].style.background_gradient(subset=['abs_error'], cmap='Reds'),
                    use_container_width=True
                )
                
                # 3. Visualization
                st.subheader("Visual vs Literature")
                # Show the generated plot
                plot_path = os.path.join("artifacts", "benchmark_logp.png")
                if os.path.exists(plot_path):
                    st.image(plot_path, caption="Predicted vs Literature logP", use_container_width=True)
                else:
                    st.warning("Plot generation failed or file not found.")
                    
            except Exception as e:
                st.error(f"Benchmark Failed: {str(e)}")
