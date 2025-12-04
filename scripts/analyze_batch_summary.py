from pathlib import Path
import re
import pandas as pd

def find_latest_summary_file(directory: Path, pattern: str = "batch_summary_*.csv") -> Path:
    """Find the latest batch summary CSV file in the given directory using the time stamp in the file name."""
    summary_files = list(directory.glob(pattern))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {directory} matching pattern {pattern}")

    # Extract timestamps and sort files
    def extract_timestamp(file: Path) -> str:
        # time stamp is in the format YYYYMMDD_HHMMSS
        return file.stem.split('_')[-2] + '_' + file.stem.split('_')[-1]

    summary_files.sort(key=extract_timestamp, reverse=True)
    return summary_files[0]

log_dir = Path("/project/dash_agir/logs")

latest_summary_file = find_latest_summary_file(log_dir)
print(f"Latest summary file found: {latest_summary_file}")

# Load the CSV file into a DataFrame
df = pd.read_csv(latest_summary_file)

# Remove rows whose batch_id does not start with "MD_" or "TX_", or "NC_"
df = df[df['batch_id'].str.startswith(('MD_', 'TX_', 'NC_'))]

# Use regex to filter batch_id that match the pattern "^(MD|TX|NC)_\d{4}-\d{2}-\d{2}$"
df = df[df['batch_id'].str.match(r'^(MD|TX|NC)_\d{4}-\d{2}-\d{2}$')]

# Create a datetime column from the batch_id
df['date'] = pd.to_datetime(df['batch_id'].str.extract(r'_(\d{4}-\d{2}-\d{2})$')[0])

# Create a state column from the batch_id
df['state'] = df['batch_id'].str.extract(r'^(MD|TX|NC)_')[0]

# Create a second column called source2 based on the source column. If source does not contains 'JUNO', set source2 to 'NCSU', else set it to SCINET.
df['source2'] = df['source'].apply(lambda x: 'SCINET' if 'JUNO' in x else 'NCSU')

"""
           batch_id       kind            source  raw_number  jpg_number  metadata_json_number
0     MD_2022-09-12     upload         GROW_DATA           0           0                     0
1     MD_2022-08-12     upload         GROW_DATA           0           0                     0
2     MD_2022-08-29     upload         GROW_DATA           0           0                     0
3     MD_2022-08-01     upload         GROW_DATA           0           0                     0
4     MD_2022-09-14     upload         GROW_DATA           0           0                     0
...             ...        ...               ...         ...         ...                   ...
2325  TX_2025-07-23  developed  longterm_images2           0         652                   652
2326  TX_2025-07-30  developed  longterm_images2           0         659                   659
2327  TX_2025-08-18  developed  longterm_images2           0         307                   307
2328  TX_2025-07-21  developed  longterm_images2           0         652                   652
2329  TX_2025-07-28  developed  longterm_images2           0         649                   649
"""
print("Batch Summary Data:")
print(df.head())

# Compare the total number of upload batches vs developed batches
upload_count = len(df[df['kind'] == 'upload']['batch_id'].unique())
developed_count = len(df[df['kind'] == 'developed']['batch_id'].unique())
print(f"Total upload batches vs developed batches: {upload_count} vs {developed_count}")

# Get the number of unique upload batches
unique_upload_batches = df[df['kind'] == 'upload']['batch_id'].nunique()
print(f"Number of unique upload batches: {unique_upload_batches}")

# Get the number of unique developed batches
unique_developed_batches = df[df['kind'] == 'developed']['batch_id'].nunique()
print(f"Number of unique developed batches: {unique_developed_batches}")

# Get the number of developed batches with jpg_number > 0
developed_batches_with_jpgs = df[(df['kind'] == 'developed') & (df['jpg_number'] > 0)]['batch_id'].nunique()
print(f"Number of developed batches with jpg_number > 0: {developed_batches_with_jpgs}")

# Get the number of upload batches that don't have a matching developed batch
upload_batches = set(df[df['kind'] == 'upload']['batch_id'])
developed_batches = set(df[df['kind'] == 'developed']['batch_id'])
upload_without_developed = upload_batches - developed_batches
print(f"Number of upload batches without a matching developed batch: {len(upload_without_developed)}")

# Get the number of developed batches in JUNO that are not in SCINET
developed_juno_batches = set(df[(df['kind'] == 'developed') & (df['source2'] == 'SCINET')]['batch_id'])
developed_ncsu_batches = set(df[(df['kind'] == 'developed') & (df['source2'] == 'NCSU')]['batch_id'])
juno_not_in_scinet = developed_juno_batches - developed_ncsu_batches
print(f"Number of developed batches in JUNO not in SCINET: {len(juno_not_in_scinet)}")

# Get the number of developed batches in SCINET that are not in JUNO
scinet_not_in_juno = developed_ncsu_batches - developed_juno_batches
print(f"Number of developed batches in SCINET not in JUNO: {len(scinet_not_in_juno)}")  

# Get the number of upload batches in JUNO that are not in SCINET
upload_juno_batches = set(df[(df['kind'] == 'upload') & (df['source2'] == 'SCINET')]['batch_id'])
upload_ncsu_batches = set(df[(df['kind'] == 'upload') & (df['source2'] == 'NCSU')]['batch_id'])
upload_juno_not_in_scinet = upload_juno_batches - upload_ncsu_batches
print(f"Number of upload batches in JUNO not in SCINET: {len(upload_juno_not_in_scinet)}")

# Get the number of upload batches in SCINET that are not in JUNO
upload_scinet_not_in_juno = upload_ncsu_batches - upload_juno_batches
print(f"Number of upload batches in SCINET not in JUNO: {len(upload_scinet_not_in_juno)}")


# Get the rows where upload batches don't have a matching developed batch
# missing_developed_batches = df[(df['kind'] == 'upload') & (df['batch_id'].isin(upload_without_developed))]
# print("Upload batches without a matching developed batch:")
# print(missing_developed_batches)

# Get the developed batches that don't have a matching upload batch
# developed_batches = df[df['kind'] == 'developed']
# developed_without_upload = developed_batches[(~developed_batches['batch_id'].isin(upload_batches)) & (developed_batches['jpg_number'] > 25)]
# print("Developed batches without a matching upload batch:")
# print(developed_without_upload.drop_duplicates(subset=['batch_id']))


# Find the batches that need to be developed (upload batches without a matching developed batch)
# batches_to_develop = df[(df['kind'] == 'upload') & (df['batch_id'].isin(upload_without_developed))]
# print("Batches that need to be developed:")
# print(batches_to_develop)


# Find developed batches without any metadta files
developed_without_metadata = df[(df['kind'] == 'developed') & (df['metadata_json_number'] == 0) & (df['jpg_number'] > 50)]
print("Developed batches without metadata files:")
print(developed_without_metadata.drop_duplicates(subset=['batch_id']))

# group developed_without_metadata by year and month
# create a quarter year column
developed_without_metadata['quarter_year'] = developed_without_metadata['date'].dt.to_period('Q')
grouped = developed_without_metadata.groupby(['state','quarter_year']).size()
print("Developed batches without metadata files grouped by year and month:")
print(grouped)


