from pathlib import Path
from svs_raw_api import load_config, RawToDng, DngToJpg

raw_path = Path("/90daydata/dash_agir/semifield-upload/MD_2025-09-09/MD_1757432311.RAW")
jpg_path = Path("/90daydata/dash_agir/semifield-developed-images/MD_2025-09-09/MD_1757432311.jpg")
jpg_path.parent.mkdir(parents=True, exist_ok=True)

config_path = "config/scinet.yaml"
config = load_config(config_path)
converter = RawToDng(config)
dng = converter.convert(raw_path)

developer = DngToJpg(config)
jpg = developer.develop(dng, jpg_path)