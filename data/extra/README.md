# Extra Datasets Directory

Place any external or out-of-domain CSV datasets here (e.g., Indian news dataset like IFND).

### Expected Columns
The dataset should contain columns for:
- **Title**: `title`, `headline`, or `subject`
- **Text / Body**: `text`, `body`, `content`, `article`, or `story`
- **Label**: `label`, `target`, `veracity`, or `class` with values `REAL` / `FAKE` (or `0` / `1`).

### Automatic Processing
When `run_pipeline.py` or `data_loader.py` runs:
1. All `*.csv` files in this directory are automatically discovered and standardized.
2. The `source_dataset` column is tagged with the file name.
3. Partitions are stratified jointly by `label` AND `source_dataset`.
4. Deduplication across datasets is strictly enforced prior to train/val/calib/test splitting to guarantee zero leakage.
5. In evaluation, per-source accuracy/F1/ECE and cross-domain transfer metrics are computed and written to `outputs/metrics/`.
