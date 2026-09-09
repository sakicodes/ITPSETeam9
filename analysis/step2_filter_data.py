import pandas as pd
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

ORIGIN_MAPPING = {
    "openai": "US",
    "anthropic": "US",
    "google": "US",
    "openrouter": "US",
    "tencent": "China",
    "minimax": "China",
    "deepseek": "China",
    "qwen": "China",
    "mistral": "Europe",
    "sealion": "Singapore"
}

def get_allowed_regions(origin):
    if origin == "US":
        return ["USA", "US", "United States"]
    elif origin == "China":
        return ["China"]
    elif origin == "Europe":
        return ["Europe", "France", "European", "EU"]
    elif origin == "Singapore":
        return ["Singapore", "SEA", "Singapore hiring context"]
    return []

def main():
    # Use dynamic path class
    data_path = paths.data
    filtered_dir = paths.get_data_dir("filtered")
    
    total_input = 0
    total_region = 0
    total_home = 0
    
    print("=== Step 2 Filtering: Home-Context Region Prompts ===")
    
    for provider, model_name in MODELS.items():
        origin = ORIGIN_MAPPING.get(provider)
        allowed_region_kws = get_allowed_regions(origin)
        
        file_path = data_path / f"results_{provider}_region.csv"
        
        if not file_path.exists():
            print(f"\n[!] File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        initial_rows = len(df)
        total_input += initial_rows
        
        if "Prompt_Version" in df.columns:
            df_region = df[df["Prompt_Version"].str.contains("Region", case=False, na=False)].copy()
        else:
            df_region = df.copy()
            
        region_rows = len(df_region)
        total_region += region_rows
        
        if "JD_Region" in df_region.columns:
            def is_home_region(r_str):
                return any(kw.lower() in str(r_str).lower() for kw in allowed_region_kws)
                
            mask = df_region["JD_Region"].apply(is_home_region)
            df_home = df_region[mask].copy()
        else:
            df_home = pd.DataFrame()
            
        home_rows = len(df_home)
        total_home += home_rows
        
        print(f"\nModel: {provider} (Origin: {origin})")
        print(f"  Input rows: {initial_rows}")
        print(f"  Region-prompt rows: {region_rows}")
        print(f"  Home-context rows retained: {home_rows}")
        print(f"  Rows removed: {initial_rows - home_rows}")
        
        if home_rows > 0:
            out_file = filtered_dir / f"results_{provider}_region_filtered.csv"
            df_home.to_csv(out_file, index=False)
            
    print("\n=== Filtering Summary ===")
    print(f"Total Input Rows: {total_input}")
    print(f"Total Home-Context Rows Retained: {total_home}")

if __name__ == '__main__':
    main()