import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from path_utils import paths

MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "tencent": "hy3", 
    "minimax": "MiniMax-M3",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.7-plus",
    "mistral": "mistral-medium-latest",
    "sealion": "aisingapore/Gemma-SEA-LION-v4-27B-IT"
}

OUTCOMES = [
    "Hireability_Score",
    "Competence_Score",
    "Fit_Score",
    "Performance_Score",
    "Leadership_Avg",
    "Status_Score",
    "Warmth_Score"
]

def main():
    filtered_dir = paths.get_data_dir("filtered")
    output_dir = paths.get_output_dir("step_2a_interaction_effect", "draft2")
    
    regression_results = []
    print("=== Step 2A Revised Analysis: Staged M0, M1, M2 ===")
    
    for provider, model_id in MODELS.items():
        file_path = filtered_dir / f"results_{provider}_region_filtered.csv"
        
        if not file_path.exists():
            continue
            
        df = pd.read_csv(file_path)
        if df.empty:
            continue
            
        for outcome in OUTCOMES:
            if outcome in df.columns:
                df[outcome] = pd.to_numeric(df[outcome], errors='coerce')
                
        # Drop NA across all necessary categorical and clustering variables
        required_cols = ['CV_framing', 'Seniority_level', 'Industry', 'JD_framing', 'JD_ID', 'Frame_CV_ID']
        df = df.dropna(subset=[col for col in required_cols if col in df.columns])
        
        if df.empty:
            continue
            
        for outcome in OUTCOMES:
            df_out = df.dropna(subset=[outcome]).copy()
            if df_out.empty:
                continue
                
            # Define exact reference categories
            base_interact = "C(CV_framing, Treatment('Agentic')) * C(Seniority_level, Treatment('Entry-Level'))"
            
            # Staged Models
            stages = {
                "M0": f"{outcome} ~ {base_interact}",
                "M1": f"{outcome} ~ {base_interact} + C(Industry)",
                "M2": f"{outcome} ~ {base_interact} + C(Industry) + C(JD_framing)"
            }
            
            for stage, formula in stages.items():
                try:
                    groups = np.asarray(df_out[['JD_ID', 'Frame_CV_ID']])
                    model = smf.ols(formula, data=df_out).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': groups}
                    )
                    se_type = "Two-way clustered (JD_ID, Frame_CV_ID)"
                except Exception:
                    try:
                        model = smf.ols(formula, data=df_out).fit(
                            cov_type='cluster',
                            cov_kwds={'groups': df_out['JD_ID']}
                        )
                        se_type = "One-way clustered (JD_ID)"
                    except Exception:
                        model = smf.ols(formula, data=df_out).fit()
                        se_type = "Standard OLS (Clustering failed)"

                # Extract interaction term
                interaction_term = next((term for term in model.params.index 
                                         if ":" in term and "Communal" in term and "Senior-Level" in term), None)
                
                if interaction_term:
                    coef = model.params[interaction_term]
                    se = model.bse[interaction_term]
                    p_val = model.pvalues[interaction_term]
                    ci_lower, ci_upper = model.conf_int().loc[interaction_term]
                    
                    direction = "Positive" if coef > 0 else "Negative" if coef < 0 else "Zero"
                    
                    regression_results.append({
                        "Model_Provider": provider,
                        "Model_Name": model_id,
                        "Outcome": outcome,
                        "Stage": stage,
                        "Formula": formula,
                        "N": int(model.nobs),
                        "Interaction_Term": interaction_term,
                        "Coefficient": coef,
                        "Std_Error": se,
                        "T_Statistic": model.tvalues[interaction_term],
                        "Raw_P_Value": p_val,
                        "CI_Lower": ci_lower,
                        "CI_Upper": ci_upper,
                        "Interaction_Direction": direction,
                        "SE_Type": se_type,
                        "Ref_CV_Framing": "Agentic",
                        "Ref_Seniority": "Entry-Level"
                    })

    res_df = pd.DataFrame(regression_results)
    
    if res_df.empty:
        print("No regression results generated.")
        return
        
    # --- Benjamini-Hochberg Correction (FDR) ---
    res_df['BH_P_Value'] = np.nan
    res_df['Significant_Raw_0.05'] = res_df['Raw_P_Value'] < 0.05
    res_df['Significant_BH_0.05'] = False
    res_df['Correction_Family'] = ''
    
    primary_mask = res_df['Outcome'] == 'Hireability_Score'
    secondary_mask = res_df['Outcome'] != 'Hireability_Score'
    
    # 1. Correct Primary Family (Hireability across all stages)
    if primary_mask.sum() > 0:
        pvals = res_df.loc[primary_mask, 'Raw_P_Value']
        reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        res_df.loc[primary_mask, 'BH_P_Value'] = p_corr
        res_df.loc[primary_mask, 'Significant_BH_0.05'] = reject
        res_df.loc[primary_mask, 'Correction_Family'] = 'Primary (Hireability)'
        
    # 2. Correct Secondary Family (Other 6 outcomes across all stages)
    if secondary_mask.sum() > 0:
        pvals = res_df.loc[secondary_mask, 'Raw_P_Value']
        reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        res_df.loc[secondary_mask, 'BH_P_Value'] = p_corr
        res_df.loc[secondary_mask, 'Significant_BH_0.05'] = reject
        res_df.loc[secondary_mask, 'Correction_Family'] = 'Secondary (Other 6 Outcomes)'
        
    # Sort logically for M0 -> M1 -> M2 comparison
    res_df = res_df.sort_values(["Correction_Family", "Model_Provider", "Outcome", "Stage"]).reset_index(drop=True)
    
    # Outputs
    res_df.to_csv(output_dir / "step_2a_draft2_full_results.csv", index=False)
    
    summary_cols = [
        "Model_Provider", "Outcome", "Stage", "N", "Coefficient", 
        "CI_Lower", "CI_Upper", "Raw_P_Value", "BH_P_Value", 
        "Significant_Raw_0.05", "Significant_BH_0.05", "Interaction_Direction"
    ]
    summary_df = res_df[summary_cols].copy()
    summary_df.to_csv(output_dir / "step_2a_draft2_interaction_summary.csv", index=False)
    
    print(f"\nTotal analyses run: {len(res_df)}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()