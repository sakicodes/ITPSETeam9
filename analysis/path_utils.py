from pathlib import Path

class ProjectPaths:
    def __init__(self, base_path=None):
        # Dynamically set the base to wherever the script is executed from
        self.base = Path(base_path) if base_path else Path.cwd()
        
        # Core directories created immediately upon initialization
        self.data = self._ensure_exists(self.base / "data")
        self.outputs = self._ensure_exists(self.base / "outputs")

    def _ensure_exists(self, path: Path) -> Path:
        """Silently creates the directory structure if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_output_dir(self, *subdirs) -> Path:
        """
        Pass any number of folder names. It builds the path inside /outputs, 
        creates the folders, and returns the usable Path object.
        """
        new_path = self.outputs.joinpath(*subdirs)
        return self._ensure_exists(new_path)

    def get_data_dir(self, *subdirs) -> Path:
        """Same behavior, but builds inside the /data directory."""
        new_path = self.data.joinpath(*subdirs)
        return self._ensure_exists(new_path)

# Instantiate the object so other scripts can simply import `paths`
paths = ProjectPaths()