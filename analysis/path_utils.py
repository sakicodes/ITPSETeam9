from pathlib import Path

BASE_PATH = Path.cwd()

DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "outputs"
STEP1_OUTPUTS = OUTPUT_PATH / "step_1_anova_analysis"
STEP2A_OUTPUTS = OUTPUT_PATH / "step_2a_interaction_effect"
STEP2A_DRAFT1 = STEP2A_OUTPUTS / "draft1"

def open_new_path(parent_dir, child_path):
	new_path = parent_dir / child_path
	if not new_path.exists():
		new_path.mkdir(parents=True, exist_ok=True)
	return new_path