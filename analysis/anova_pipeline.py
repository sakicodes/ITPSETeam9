import pandas as pd
import numpy as np
from scipy import stats
import logging
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
METRICS = [
    "Hireability_Score", "Competence_Score", "Fit_Score", 
    "Performance_Score", "Leadership_Avg", "Status_Score", "Warmth_Score"
]

CELL_COLS = [
    "Prompt_Version", "Industry", "JD_Region", 
    "JD_framing", "Seniority_level", "CV_framing"
]

MODEL_REGIONS = {
    # USA
    "openai": "USA", "anthropic": "USA", "google": "USA", 
    "openrouter": "USA", "groq": "USA", 
    # China
    "tencent": "China", "minimax": "China", "deepseek": "China", "qwen": "China",
    # EU & SEA (Will be skipped per instructions)
    "mistral": "EU", "sealion": "SEA"
}

EXPECTED_MODELS_PER_REGION = {
    "USA": 4,
    "China": 4
}

ALPHA = 0.05

# ==========================================
# LOGGING SETUP
# ==========================================
def setup_logging(base_dir):
    log_file = base_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("--- ANOVA Analysis Pipeline Started ---")

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================
def load_results(data_dir):
    """Loads all result CSVs, maps models to their geographic region, and combines them."""
    all_files = list(data_dir.glob("results_*.csv"))
    if not all_files:
        logging.error(f"No results_*.csv files found in {data_dir}")
        return pd.DataFrame()
    
    logging.info(f"Files Loaded: {len(all_files)}")
    
    df_list = []
    for file in all_files:
        try:
            # Determine provider from filename: results_{provider}_{prompt_version}.csv
            filename_parts = file.stem.split("_")
            provider = filename_parts[1]
            
            df = pd.read_csv(file)
            df['Provider'] = provider
            df['Model_Origin'] = MODEL_REGIONS.get(provider, "Unknown")
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error loading {file.name}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Clean numeric data (handle potential parsing errors or blanks)
    for metric in METRICS:
        combined_df[metric] = pd.to_numeric(combined_df[metric], errors='coerce')
        
    # Drop rows with missing crucial identifiers
    combined_df.dropna(subset=CELL_COLS, inplace=True)
    
    logging.info(f"Rows Loaded: {len(combined_df)}")
    return combined_df

def run_assumption_checks(model_groups):
    """
    Runs Shapiro-Wilk for normality on each group and Levene's for equal variance.
    """
    arrays = list(model_groups.values())
    
    shapiro_p_values = []
    for arr in arrays:
        if len(arr) < 3 or np.var(arr) == 0:
            shapiro_p_values.append(0.0)
        else:
            stat, p = stats.shapiro(arr)
            shapiro_p_values.append(p)
            
    min_shapiro_p = min(shapiro_p_values) if shapiro_p_values else 0.0
    
    if all(len(arr) >= 2 for arr in arrays) and len(arrays) >= 2:
        if all(np.var(arr) == 0 for arr in arrays):
            levene_p = 0.0
        else:
            stat, levene_p = stats.levene(*arrays)
    else:
        levene_p = 0.0
        
    assumptions_passed = (min_shapiro_p >= ALPHA) and (levene_p >= ALPHA)
    return min_shapiro_p, levene_p, assumptions_passed

def run_statistical_test(model_groups, assumptions_passed):
    """Routes to ANOVA or Kruskal-Wallis based on assumption checks."""
    arrays = list(model_groups.values())
    
    if assumptions_passed:
        stat, p_val = stats.f_oneway(*arrays)
        test_used = "ANOVA"
    else:
        try:
            stat, p_val = stats.kruskal(*arrays)
            test_used = "Kruskal-Wallis"
        except ValueError:
            stat, p_val = np.nan, np.nan
            test_used = "Failed"
            
    return stat, p_val, test_used

def calculate_effect_size(model_groups, test_used, stat):
    """Calculates Eta-squared (ANOVA) or Epsilon-squared (Kruskal-Wallis)."""
    arrays = list(model_groups.values())
    total_N = sum(len(a) for a in arrays)
    
    if total_N <= 1 or pd.isna(stat):
        return np.nan
        
    if test_used == "ANOVA":
        # Eta-squared = SS_between / SS_total
        overall_mean = np.mean([val for arr in arrays for val in arr])
        ss_between = sum(len(arr) * (np.mean(arr) - overall_mean)**2 for arr in arrays)
        ss_total = sum((val - overall_mean)**2 for arr in arrays for val in arr)
        return ss_between / ss_total if ss_total > 0 else 0.0
        
    elif test_used == "Kruskal-Wallis":
        # Epsilon-squared = H / (N - 1)
        epsilon_sq = stat / (total_N - 1)
        # Bounding between 0 and 1 for extreme edge cases
        return min(max(epsilon_sq, 0.0), 1.0)
        
    return np.nan

def main():
    base_dir = Path.cwd()
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        data_dir.mkdir()
        print("Created /data directory. Please place results CSVs inside and run again.")
        return

    setup_logging(base_dir)
    
    df = load_results(data_dir)
    if df.empty:
        logging.error("Empty dataframe. Exiting.")
        return

    # Filter out EU and SEA
    df = df[df['Model_Origin'].isin(['USA', 'China'])]
    
    anova_results = []
    assumption_results = []
    descriptive_results = []
    
    grouping_cols = CELL_COLS + ['Model_Origin']
    grouped = df.groupby(grouping_cols)
    
    logging.info(f"Number of Unique Cells (Conditions x Regions) identified: {len(grouped)}")
    logging.info(f"Number of Metrics to analyze per cell: {len(METRICS)}")
    
    anovas_completed = 0
    skipped_cells = 0

    for name, group_df in grouped:
        cell_info = dict(zip(grouping_cols, name))
        model_origin = cell_info['Model_Origin']
        
        unique_models = group_df['Provider'].unique()
        if len(unique_models) != EXPECTED_MODELS_PER_REGION[model_origin]:
            logging.warning(f"Skipping cell {name}: Expected {EXPECTED_MODELS_PER_REGION[model_origin]} models for {model_origin}, found {len(unique_models)} ({unique_models})")
            skipped_cells += 1
            continue

        for metric in METRICS:
            model_groups = {}
            for provider in unique_models:
                scores = group_df[group_df['Provider'] == provider][metric].dropna().values
                if len(scores) > 0:
                    model_groups[provider] = scores
                    
            if len(model_groups) != EXPECTED_MODELS_PER_REGION[model_origin]:
                logging.warning(f"Skipping {metric} in cell {name}: Missing valid numeric data for some models.")
                continue

            # 1. Descriptive Statistics
            for provider, scores in model_groups.items():
                descriptive_results.append({
                    **cell_info,
                    "Metric": metric,
                    "Provider": provider,
                    "Mean": np.round(np.mean(scores), 4),
                    "Std_Dev": np.round(np.std(scores, ddof=1), 4) if len(scores) > 1 else 0.0,
                    "Sample_Size": len(scores)
                })

            # 2. Assumption Checks
            shapiro_p, levene_p, passed = run_assumption_checks(model_groups)
            assumption_results.append({
                **cell_info,
                "Metric": metric,
                "Normality_p_value": shapiro_p,
                "Levene_p_value": levene_p,
                "Assumptions_Passed": passed
            })
            
            # 3. Run Test & Effect Size
            stat, p_val, test_used = run_statistical_test(model_groups, passed)
            effect_size = calculate_effect_size(model_groups, test_used, stat)
            
            if pd.isna(p_val):
                logging.warning(f"Statistical test failed mathematically for {metric} in cell {name}.")
                continue
                
            decision = "Reject H0 (Models Differ)" if p_val < ALPHA else "Fail to Reject H0 (Models Equivalent)"
            pooling = "Do Not Pool" if p_val < ALPHA else "Pool"
            
            anova_results.append({
                **cell_info,
                "Metric": metric,
                "Test_Used": test_used,
                "F_Statistic": stat,
                "P_Value": p_val,
                "Effect_Size": np.round(effect_size, 4),
                "Decision": decision,
                "Pooling_Recommendation": pooling
            })
            
            anovas_completed += 1

    logging.info(f"ANOVAs/Kruskal-Wallis tests successfully completed: {anovas_completed}")
    logging.info(f"Cells skipped due to incompleteness: {skipped_cells}")

    # ==========================================
    # WRITE OUTPUTS
    # ==========================================
    if not anova_results:
        logging.error("No valid ANOVA results generated. Check data completeness.")
        return

    pd.DataFrame(anova_results).to_csv(base_dir / "anova_results.csv", index=False)
    pd.DataFrame(assumption_results).to_csv(base_dir / "assumption_checks.csv", index=False)
    pd.DataFrame(descriptive_results).to_csv(base_dir / "descriptive_statistics.csv", index=False)
    
    summary_data = []
    results_df = pd.DataFrame(anova_results)
    for region in results_df['Model_Origin'].unique():
        region_df = results_df[results_df['Model_Origin'] == region]
        total = len(region_df)
        significant = len(region_df[region_df['Pooling_Recommendation'] == 'Do Not Pool'])
        non_significant = total - significant
        pct_sig = (significant / total) * 100 if total > 0 else 0
        
        global_rec = "Do Not Pool" if significant > 0 else "Pool"
        
        summary_data.append({
            "Region": region,
            "Number_of_ANOVAs": total,
            "Number_Significant": significant,
            "Number_Non_significant": non_significant,
            "Percentage_Significant": f"{pct_sig:.2f}%",
            "Global_Recommendation": global_rec
        })
        
    pd.DataFrame(summary_data).to_csv(base_dir / "anova_summary.csv", index=False)
    
    logging.info("Outputs written: anova_results.csv, anova_summary.csv, assumption_checks.csv, descriptive_statistics.csv")
    logging.info("--- ANOVA Analysis Pipeline Finished ---")

if __name__ == "__main__":
    main()