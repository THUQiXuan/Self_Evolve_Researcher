"""SER — Competition loading, HCE data split, and grading."""

import json
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Optional

import csv as _csv_mod
import numpy as np
import pandas as pd

from config import MLEBENCH_DATA_DIR as DATA_DIR, TRAIN_RATIO, SEARCH_RATIO, VAL_RATIO


def _robust_read_csv(path, **kwargs) -> pd.DataFrame:
    """Read CSV with fallback for multiline-quoted fields (e.g. jigsaw comment_text).

    Tries pandas first; if it fails with a tokenize/parse error, falls back to
    Python's csv.DictReader which handles embedded newlines correctly.
    """
    try:
        return pd.read_csv(path, on_bad_lines='skip', **kwargs)
    except Exception:
        pass
    # Fallback: Python csv module (handles multiline quoted fields)
    try:
        with open(path, newline='', encoding='utf-8', errors='replace') as f:
            reader = _csv_mod.DictReader(f)
            rows = list(reader)
        if rows:
            df = pd.DataFrame(rows)
            # DictReader puts extra fields in a None column — drop it
            if None in df.columns:
                df = df.drop(columns=[None])
            return df
    except Exception:
        pass
    return pd.DataFrame()

logger = logging.getLogger(__name__)


class CompetitionManager:
    """Manages MLE-bench competition data, HCE splits, and grading."""

    def __init__(self, competition_id: str, work_dir: Path, data_dir: Path = DATA_DIR):
        self.competition_id = competition_id
        self.data_dir = data_dir
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Load competition via mlebench registry
        from mlebench.registry import registry
        reg = registry.set_data_dir(data_dir)
        self.competition = reg.get_competition(competition_id)

        # Paths
        self.public_dir = self.competition.public_dir
        self.private_dir = self.competition.private_dir
        self.sample_submission_path = self.competition.sample_submission
        self.answers_path = self._resolve_answers_path(self.competition.answers)
        # Override competition.answers so grade_submission uses the resolved path
        object.__setattr__(self.competition, 'answers', self.answers_path)
        self.leaderboard_path = self._resolve_leaderboard_path(self.competition.leaderboard)

        # HCE split paths
        self.hce_dir = self.work_dir / "hce"
        self.hce_dir.mkdir(parents=True, exist_ok=True)
        self.dtrain_dir = self.hce_dir / "train"
        self.dsearch_dir = self.hce_dir / "search"
        self.dval_dir = self.hce_dir / "val"

        # Metadata
        self.description = self.competition.description
        self.is_lower_better = self._check_is_lower_better()

    @staticmethod
    def _resolve_answers_path(original_path) -> Path:
        """If answers.csv.bak exists alongside answers.csv, use .bak (original unmodified version).
        Copies to a local .csv file since mlebench load_answers requires .csv extension."""
        original_path = Path(original_path)
        bak_path = original_path.parent / (original_path.name + ".bak")
        if bak_path.exists():
            # Copy .bak to local dir with .csv extension (load_answers needs .csv)
            local_dir = Path.home() / ".ser_answers_fix"
            local_dir.mkdir(parents=True, exist_ok=True)
            # Extract competition name from path:
            # .../data/<comp>/prepared/private/answers.csv → parent x3 = <comp>
            comp_name = original_path.parent.parent.parent.name
            local_csv = local_dir / f"{comp_name}_answers.csv"
            if not local_csv.exists():
                import shutil
                shutil.copy2(bak_path, local_csv)
                logger.warning(f"Copied original answers from {bak_path} to {local_csv}")
            else:
                logger.info(f"Using cached original answers: {local_csv}")
            return local_csv
        return original_path

    @staticmethod
    def _resolve_leaderboard_path(original_path) -> Path:
        """If the leaderboard file is a Git LFS pointer, use the real copy from ~/mle-bench-lfs/."""
        original_path = Path(original_path)
        try:
            with open(original_path) as f:
                first_line = f.readline()
            if not first_line.startswith("version https://git-lfs"):
                return original_path  # Real CSV, use as-is
        except Exception:
            return original_path

        # Try local LFS-resolved copy
        # original: /usr/local/.../mlebench/competitions/<comp>/leaderboard.csv
        # local:    ~/mle-bench-lfs/mlebench/competitions/<comp>/leaderboard.csv
        comp_name = original_path.parent.name
        local_path = Path.home() / "mle-bench-lfs" / "mlebench" / "competitions" / comp_name / "leaderboard.csv"
        if local_path.exists():
            with open(local_path) as f:
                first_line = f.readline()
            if not first_line.startswith("version https://git-lfs"):
                logger.info(f"Using local leaderboard for {comp_name}: {local_path}")
                return local_path
        logger.warning(f"Leaderboard for {comp_name} is LFS pointer and no local copy found")
        return original_path

    def _check_is_lower_better(self) -> bool:
        """Check if the competition metric is 'lower is better'."""
        try:
            grader = self.competition.grader
            if hasattr(grader, "is_lower_better"):
                import pandas as pd
                lb = pd.read_csv(self.leaderboard_path)
                return grader.is_lower_better(lb)
        except Exception:
            pass
        # Fallback: check from leaderboard sorting or grader config
        lower_better_keywords = ["loss", "rmse", "mae", "error", "rmsle", "pinball"]
        desc_lower = self.description.lower() if self.description else ""
        for kw in lower_better_keywords:
            if kw in desc_lower:
                return True
        return False

    def prepare_hce_split(self, seed: Optional[int] = None) -> dict:
        """
        Perform HCE (Hidden Consistent Evaluation) data split.
        Split the training data into D_train (80%), D_search (10%), D_val (10%).

        Returns dict with paths to split directories.
        """
        if seed is None:
            # Deterministic seed from competition id
            seed = int(hashlib.md5(self.competition_id.encode()).hexdigest()[:8], 16) % (2**31)

        rng = np.random.RandomState(seed)

        # Find the main training data file
        train_file = self._find_train_file()
        if train_file is None:
            logger.warning(f"No train file found for {self.competition_id}, using full data dir as train")
            result = self._symlink_full_as_train()
            self._check_and_fix_leakage()
            return result

        logger.info(f"Splitting {train_file} with seed={seed}")

        # Check file size — for very large files (>500MB), use sampling approach
        file_size_mb = train_file.stat().st_size / (1024 * 1024)
        MAX_FILE_SIZE_MB = 500

        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.info(f"Large file ({file_size_mb:.0f}MB), using line-count based sampling")
            result = self._split_large_file(train_file, rng)
            self._check_and_fix_leakage()
            return result

        # Read and split
        if train_file.suffix == ".csv":
            df = pd.read_csv(train_file)
        elif train_file.suffix == ".jsonl":
            df = pd.read_json(train_file, lines=True)
        else:
            logger.warning(f"Unsupported train file format: {train_file.suffix}")
            result = self._symlink_full_as_train()
            self._check_and_fix_leakage()
            return result

        n = len(df)
        indices = rng.permutation(n)
        n_train = int(n * TRAIN_RATIO)
        n_search = int(n * SEARCH_RATIO)

        train_idx = indices[:n_train]
        search_idx = indices[n_train:n_train + n_search]
        val_idx = indices[n_train + n_search:]

        # Save splits
        for split_name, split_dir, split_idx in [
            ("train", self.dtrain_dir, train_idx),
            ("search", self.dsearch_dir, search_idx),
            ("val", self.dval_dir, val_idx),
        ]:
            split_dir.mkdir(parents=True, exist_ok=True)
            split_df = df.iloc[split_idx]
            out_path = split_dir / train_file.name
            if train_file.suffix == ".csv":
                split_df.to_csv(out_path, index=False)
            else:
                split_df.to_json(out_path, orient="records", lines=True)

            # Symlink other public files (description, sample_submission, test, images, etc.)
            self._symlink_public_files(split_dir, exclude={train_file.name})

            logger.info(f"  {split_name}: {len(split_idx)} samples -> {out_path}")

        self._check_and_fix_leakage()
        return {
            "train": self.dtrain_dir,
            "search": self.dsearch_dir,
            "val": self.dval_dir,
            "train_size": len(train_idx),
            "search_size": len(search_idx),
            "val_size": len(val_idx),
        }

    def _split_large_file(self, train_file: Path, rng: np.random.RandomState) -> dict:
        """For large files: chunked pandas 80/10/10 split.

        Uses pandas chunked reading to correctly handle CSV quoting (multi-line
        fields, embedded commas). Two passes: count rows, then split.
        Caps search/val at MAX_SEARCH_VAL_ROWS each.
        """
        MAX_SEARCH_VAL_ROWS = 50000  # Cap search/val to keep grading fast
        CHUNK_SIZE = 50000

        # Pass 1: count rows (pandas handles multi-line CSV correctly)
        logger.info("Large file: counting rows...")
        n = 0
        for chunk in pd.read_csv(train_file, chunksize=CHUNK_SIZE, usecols=[0]):
            n += len(chunk)
        logger.info(f"Large file has {n} rows, using chunked pandas split")

        n_search = min(int(n * SEARCH_RATIO), MAX_SEARCH_VAL_ROWS)
        n_val = min(int(n * VAL_RATIO), MAX_SEARCH_VAL_ROWS)

        # Pre-compute row assignments: 0=train, 1=search, 2=val
        assignments = np.zeros(n, dtype=np.int8)
        all_indices = rng.permutation(n)
        assignments[all_indices[:n_search]] = 1
        assignments[all_indices[n_search:n_search + n_val]] = 2

        # Prepare output dirs
        for d in [self.dtrain_dir, self.dsearch_dir, self.dval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        train_path = self.dtrain_dir / train_file.name
        search_path = self.dsearch_dir / train_file.name
        val_path = self.dval_dir / train_file.name

        # Pass 2: read chunks and split
        train_first = search_first = val_first = True
        train_count = search_count = val_count = 0
        row_offset = 0

        for chunk in pd.read_csv(train_file, chunksize=CHUNK_SIZE, low_memory=False):
            chunk_size = len(chunk)
            chunk_assignments = assignments[row_offset:row_offset + chunk_size]

            train_mask = chunk_assignments == 0
            search_mask = chunk_assignments == 1
            val_mask = chunk_assignments == 2

            if train_mask.any():
                chunk[train_mask].to_csv(train_path, mode='w' if train_first else 'a',
                                         header=train_first, index=False)
                train_first = False
                train_count += train_mask.sum()

            if search_mask.any():
                chunk[search_mask].to_csv(search_path, mode='w' if search_first else 'a',
                                          header=search_first, index=False)
                search_first = False
                search_count += search_mask.sum()

            if val_mask.any():
                chunk[val_mask].to_csv(val_path, mode='w' if val_first else 'a',
                                       header=val_first, index=False)
                val_first = False
                val_count += val_mask.sum()

            row_offset += chunk_size

        # Symlink other public files into each split dir
        for split_dir in [self.dtrain_dir, self.dsearch_dir, self.dval_dir]:
            self._symlink_public_files(split_dir, exclude={train_file.name})

        logger.info(f"  train: {train_count} samples -> {train_path}")
        logger.info(f"  search: {search_count} samples -> {search_path}")
        logger.info(f"  val: {val_count} samples -> {val_path}")

        return {
            "train": self.dtrain_dir,
            "search": self.dsearch_dir,
            "val": self.dval_dir,
            "train_size": int(train_count),
            "search_size": int(search_count),
            "val_size": int(val_count),
        }

    def _find_train_file(self) -> Optional[Path]:
        """Find the main training data file in the public directory."""
        candidates = [
            "train.csv", "training_data.csv", "train_labels.csv",
            "train.jsonl", "train_data.csv", "labels.csv",
            "training.csv", "data.csv", "train_set.csv",
        ]
        for name in candidates:
            path = self.public_dir / name
            if path.exists():
                return path
        # Fallback: look for any CSV with 'train' in name
        for f in sorted(self.public_dir.glob("*train*.csv")):
            return f
        return None

    def _symlink_public_files(self, target_dir: Path, exclude: set[str] = None) -> None:
        """Symlink public files (except excluded) into target directory."""
        exclude = exclude or set()
        for item in self.public_dir.iterdir():
            if item.name in exclude:
                continue
            link = target_dir / item.name
            if not link.exists():
                link.symlink_to(item)

    def _symlink_full_as_train(self) -> dict:
        """When no splittable file found, symlink entire public dir as train."""
        for split_dir in [self.dtrain_dir, self.dsearch_dir, self.dval_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            self._symlink_public_files(split_dir)
        return {
            "train": self.dtrain_dir,
            "search": self.dsearch_dir,
            "val": self.dval_dir,
            "train_size": -1,
            "search_size": -1,
            "val_size": -1,
        }

    # -------------------------------------------------------------------------
    # Leakage detection and sanitization
    # -------------------------------------------------------------------------

    def _check_and_fix_leakage(self) -> None:
        """Scan agent-visible files for test-label leakage and sanitize them.

        Runs once per competition (guarded by .leakage.done marker in hce_dir).
        Handles:
        - CSV files that contain test IDs paired with ground-truth answer values
        - .tar.gz / .tgz archives containing nested ZIP/MAT files with embedded labels
        - .zip archives containing CSV/MAT files with embedded labels
        - .json / .jsonl archives (7z or zip) with answer columns

        Files that are found to leak test labels are either:
        - Removed (if the file has NO safe content the agent needs), or
        - Replaced by a sanitized version (with labels stripped)
        """
        import os as _os
        leakage_done = self.hce_dir / ".leakage.done"
        if leakage_done.exists():
            return

        if not self.answers_path.exists():
            leakage_done.touch()
            return

        try:
            answers = _robust_read_csv(self.answers_path)
            if len(answers.columns) < 2:
                leakage_done.touch()
                return
            id_col = answers.columns[0]
            answer_cols = answers.columns[1:].tolist()
            test_ids = set(answers[id_col].astype(str))

            if not test_ids:
                leakage_done.touch()
                return

            # Also include HCE eval IDs (search + val splits) in the protected set.
            # These come from the original competition's validation data and may appear
            # in validation archives with ground-truth labels — a leakage path that
            # bypasses the test-only check.
            eval_ids: set = set()
            eval_info_path = self.hce_dir / "eval" / "eval_info.json"
            if eval_info_path.exists():
                try:
                    import json as _json
                    eval_info = _json.loads(eval_info_path.read_text())
                    for key in ["search_answers_path", "val_answers_path"]:
                        ans_path = Path(eval_info.get(key, ""))
                        if ans_path.exists():
                            edf = _robust_read_csv(ans_path)
                            if len(edf.columns) >= 1:
                                eval_ids.update(edf[edf.columns[0]].astype(str))
                except Exception:
                    pass

            all_protected_ids = test_ids | eval_ids
            if eval_ids:
                logger.info(f"Leakage scan: {len(test_ids)} test IDs + {len(eval_ids)} eval IDs protected, answer cols: {answer_cols}")
            else:
                logger.info(f"Leakage scan: {len(test_ids)} test IDs, answer cols: {answer_cols}")

            # Gather unique real paths from one split dir (symlinks all point to same source)
            checked_names: set = set()
            leaky_names: list = []
            sanitized_cache: dict = {}  # real_path → sanitized_path (expensive, cache)

            for item in list(self.dtrain_dir.iterdir()):
                if item.name in checked_names:
                    continue
                checked_names.add(item.name)

                real = item.resolve() if item.is_symlink() else item
                name_lower = item.name.lower()

                # --- CSV files ---
                if name_lower.endswith('.csv') or name_lower.endswith('.tsv'):
                    if self._csv_leaks_labels(real, all_protected_ids, answer_cols, answers_df=answers):
                        logger.warning(f"Leakage: CSV {item.name} contains test/eval labels → removing")
                        self._remove_from_all_splits(item.name)
                        leaky_names.append(item.name)

                # --- Compressed archives ---
                elif any(name_lower.endswith(e) for e in
                         ['.tar.gz', '.tgz', '.tar.bz2', '.tar', '.zip']):
                    sanitized = self._sanitize_archive(real, all_protected_ids, answer_cols,
                                                       cache=sanitized_cache)
                    if sanitized is not None:
                        logger.warning(f"Leakage: archive {item.name} contained test/eval labels → replaced with sanitized version")
                        self._replace_in_all_splits(item.name, sanitized)
                        leaky_names.append(item.name)

                # --- JSON/7z archives (harder to handle generically) ---
                elif any(name_lower.endswith(e) for e in ['.json.7z', '.jsonl.gz']):
                    # These are less common; log a warning if suspected
                    logger.warning(
                        f"Cannot auto-sanitize {item.name} — manual review recommended"
                    )

            if leaky_names:
                logger.info(f"Leakage scan complete: fixed {len(leaky_names)} files: {leaky_names}")
            else:
                logger.info("Leakage scan complete: no leakage found")

        except Exception as exc:
            logger.error(f"Leakage check failed with exception: {exc}", exc_info=True)

        # --- Nested-directory ground-truth supplemental file removal ---
        # Some competitions (e.g. smartphone-decimeter-2022) use a nested
        # trips/phones/files layout where the HCE test sub-directories contain
        # supplemental files that ARE the ground truth (e.g. span_log.nmea is the
        # NovAtel SPAN reference GPS used to compute ground_truth.csv).  The flat
        # leakage scan above only looks at top-level files in dtrain_dir and cannot
        # reach these.  We handle them here with a competition-specific blocklist.
        self._remove_nested_gt_supplementals()

        leakage_done.touch()

    # Competition-specific supplemental files inside nested test sub-directories that
    # are derived from (or ARE) the ground truth, and must be hidden from the agent.
    # Format: { competition_id: [relative glob patterns to match under each test trip dir] }
    _NESTED_GT_SUPPLEMENTALS: dict = {
        # span_log.nmea is the NovAtel SPAN reference GPS — ground_truth.csv is built
        # directly from it.  gnss_rinex.20o is the carrier-phase RINEX derived from the
        # same receiver and can also be used to reconstruct precise positions.
        "smartphone-decimeter-2022": ["supplemental/span_log.nmea",
                                      "supplemental/gnss_rinex.20o"],
    }

    def _remove_nested_gt_supplementals(self) -> None:
        """Remove competition-specific ground-truth supplemental files from HCE test dirs.

        These are files that exist in test/<trip>/<phone>/supplemental/ and are either
        the ground truth or directly derivable from it.  They are invisible to the flat
        leakage scan but can be exploited by agents reading raw data files.
        """
        patterns = self._NESTED_GT_SUPPLEMENTALS.get(self.competition_id)
        if not patterns:
            return

        removed = []
        # Walk all HCE split test directories (hce/train/test, hce/val/test, hce/search/test)
        for split_root in [self.dtrain_dir, self.dval_dir, self.dsearch_dir]:
            test_dir = split_root / "test"
            if not test_dir.exists():
                continue
            for trip_dir in test_dir.iterdir():
                if not trip_dir.is_dir():
                    continue
                for phone_dir in trip_dir.iterdir():
                    if not phone_dir.is_dir():
                        continue
                    for rel_pattern in patterns:
                        target = phone_dir / rel_pattern
                        if target.exists() or target.is_symlink():
                            try:
                                target.unlink(missing_ok=True)
                                removed.append(str(target))
                                logger.info(f"Nested GT supplemental removed: {target}")
                            except OSError as e:
                                logger.debug(f"Could not remove {target}: {e}")

        if removed:
            logger.warning(
                f"Removed {len(removed)} nested ground-truth supplemental file(s) "
                f"from test splits: {patterns}"
            )
        else:
            logger.debug("No nested GT supplemental files found to remove.")

    # Files that are safe-by-name: templates, feature-only files agents always need
    _SAFE_CSV_NAMES = frozenset([
        'sample_submission.csv', 'sample_submission.tsv',
        'test.csv', 'test.tsv',  # public test.csv has features only, not labels
    ])

    def _csv_leaks_labels(self, path: Path, test_ids: set, answer_cols: list,
                          answers_df: "Optional[pd.DataFrame]" = None) -> bool:
        """Return True if CSV contains test IDs WITH CORRECT ground-truth label values.

        Uses 4-criterion check to avoid false positives:
        1. Test ID overlap
        2. Answer column name present
        3. High variance values (not placeholder)
        4. >=50% value match against private ground truth (requires answers_df)

        If answers_df is not provided, falls back to structural heuristic only
        (may have false positives for low-cardinality columns).
        """
        if path.name.lower() in self._SAFE_CSV_NAMES:
            return False
        try:
            df = pd.read_csv(path, nrows=2000, low_memory=False)
            if df.empty or len(df.columns) < 2:
                return False
            id_col_candidate = df.columns[0]
            file_ids = set(df[id_col_candidate].astype(str))
            overlap = file_ids & test_ids
            if not overlap:
                return False
            # Check if file has answer column(s) present
            common_answer_cols = [c for c in answer_cols if c in df.columns]
            if not common_answer_cols:
                return False

            # Criterion 3: variance check — skip if all values are constant (placeholder)
            has_variance = False
            for col in common_answer_cols:
                series = df[col].dropna()
                if len(series) > 0 and series.nunique() > 1:
                    has_variance = True
                    break
            if not has_variance:
                return False

            # Criterion 4: value-matching against private ground truth (if available)
            if answers_df is not None and not answers_df.empty:
                ans_id_col = answers_df.columns[0]
                sample_ids = list(overlap)[:20]
                file_rows = df[df[id_col_candidate].astype(str).isin(sample_ids)]
                gt_rows = answers_df[answers_df[ans_id_col].astype(str).isin(sample_ids)]
                matches, total = 0, 0
                for sid in sample_ids:
                    fr = file_rows[file_rows[id_col_candidate].astype(str) == sid]
                    gr = gt_rows[gt_rows[ans_id_col].astype(str) == sid]
                    for col in common_answer_cols:
                        if col in answers_df.columns and not fr.empty and not gr.empty \
                                and col in fr.columns and col in gr.columns:
                            total += 1
                            if str(fr[col].iloc[0]) == str(gr[col].iloc[0]):
                                matches += 1
                if total > 0:
                    match_rate = matches / total
                    if match_rate < 0.50:
                        logger.debug(f"CSV {path.name}: value match {match_rate:.0%} < 50% — not leaky")
                        return False
                    logger.warning(f"CSV {path.name}: value match {match_rate:.0%} ≥ 50% — LEAKY")
                    return True
                # No comparable rows found — fall through to structural heuristic

            # Structural heuristic (no answers_df available for comparison)
            logger.debug(f"CSV {path.name}: {len(overlap)} test IDs, answer cols present, no value comparison available")
            return True
        except Exception:
            return False

    def _sanitize_archive(
        self,
        archive_path: Path,
        test_ids: set,
        answer_cols: list,
        cache: dict,
    ) -> Optional[Path]:
        """Check and sanitize a compressed archive for test-label leakage.

        Returns path to the sanitized archive (in a temp dir inside hce_dir),
        or None if no leakage was found or sanitization isn't possible.
        """
        import tarfile as _tarfile
        import zipfile as _zipfile

        if archive_path in cache:
            return cache[archive_path]

        name = archive_path.name.lower()
        result = None

        try:
            if name.endswith('.tar.gz') or name.endswith('.tgz') or name.endswith('.tar.bz2') or name.endswith('.tar'):
                result = self._sanitize_tar(archive_path, test_ids, answer_cols)
            elif name.endswith('.zip'):
                result = self._sanitize_zip_archive(archive_path, test_ids, answer_cols)
        except Exception as e:
            logger.warning(f"Archive sanitization failed for {archive_path.name}: {e}")
            result = None

        cache[archive_path] = result
        return result

    @staticmethod
    def _extract_numeric_id_from_name(name: str) -> Optional[str]:
        """Extract a numeric ID from an archive member name.

        Examples:
          'Sample00300.zip'  → '300'
          './Sample00300.zip' → '300'
          'train_300.csv'    → '300'
          'image_0123456.jpg' → '123456'
        Returns the last sequence of digits found, or None.
        """
        import re as _re
        digits = _re.findall(r'\d+', name.split('/')[-1])
        if digits:
            # Return last run of digits with leading zeros stripped
            return str(int(digits[-1]))
        return None

    def _sanitize_tar(
        self, tar_path: Path, test_ids: set, answer_cols: list
    ) -> Optional[Path]:
        """Sanitize a .tar.gz archive: strip test labels from nested files.

        Only sanitizes members whose numeric ID (from filename) is in test_ids.
        This prevents false-positive sanitization of training/validation archives.

        Handles the gesture competition pattern:
          test.tar.gz → Sample00XXX.zip → Sample00XXX_data.mat (has 'Labels' field)
        """
        import tarfile as _tarfile
        import zipfile as _zipfile
        import io as _io

        mode = 'r:gz' if tar_path.name.lower().endswith('.gz') else (
               'r:bz2' if tar_path.name.lower().endswith('.bz2') else 'r:')
        leaky = False

        # First pass: detect leakage (only check members whose ID is in test_ids)
        try:
            with _tarfile.open(tar_path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    mname = member.name.lower()
                    member_id = self._extract_numeric_id_from_name(member.name)

                    if mname.endswith('.csv') or mname.endswith('.tsv'):
                        f = tf.extractfile(member)
                        if f and self._csv_leaks_labels_from_bytes(f.read(), test_ids, answer_cols):
                            leaky = True
                            break

                    elif mname.endswith('.zip'):
                        # Only check if this zip corresponds to a test ID
                        if member_id is not None and member_id not in test_ids:
                            continue
                        f = tf.extractfile(member)
                        if f:
                            inner_bytes = f.read()
                            if self._zip_bytes_has_leakage(inner_bytes, test_ids, answer_cols):
                                leaky = True
                                break

                    elif mname.endswith('.mat'):
                        # Only check if this mat corresponds to a test ID
                        if member_id is not None and member_id not in test_ids:
                            continue
                        f = tf.extractfile(member)
                        if f:
                            mat_bytes = f.read()
                            if self._mat_bytes_has_labels(mat_bytes, test_ids):
                                leaky = True
                                break
        except Exception as e:
            logger.debug(f"Tar leakage scan failed for {tar_path.name}: {e}")
            return None

        if not leaky:
            return None

        # Second pass: rebuild sanitized archive
        logger.info(f"Sanitizing {tar_path.name} (removing test labels from nested files)...")
        sanitized_dir = self.hce_dir / ".sanitized"
        sanitized_dir.mkdir(exist_ok=True)
        out_path = sanitized_dir / tar_path.name
        write_mode = 'w:gz' if tar_path.name.lower().endswith('.gz') else (
                     'w:bz2' if tar_path.name.lower().endswith('.bz2') else 'w:')

        try:
            with _tarfile.open(tar_path, mode) as src_tf, \
                 _tarfile.open(out_path, write_mode) as dst_tf:
                for member in src_tf.getmembers():
                    if not member.isfile():
                        dst_tf.addfile(member)
                        continue
                    mname = member.name.lower()
                    f = src_tf.extractfile(member)
                    if f is None:
                        dst_tf.addfile(member)
                        continue
                    raw = f.read()

                    member_id = self._extract_numeric_id_from_name(member.name)
                    is_test_member = (member_id is None or member_id in test_ids)

                    if mname.endswith('.zip') and is_test_member:
                        # Sanitize inner zip (only for test-ID members)
                        sanitized_bytes = self._sanitize_zip_bytes(raw, test_ids, answer_cols)
                        new_raw = sanitized_bytes if sanitized_bytes else raw
                    elif mname.endswith('.mat') and is_test_member:
                        # Sanitize mat file (only for test-ID members)
                        sanitized_bytes = self._sanitize_mat_bytes(raw)
                        new_raw = sanitized_bytes if sanitized_bytes else raw
                    elif (mname.endswith('.csv') or mname.endswith('.tsv')) and is_test_member:
                        sanitized_bytes = self._sanitize_csv_bytes(raw, test_ids, answer_cols)
                        new_raw = sanitized_bytes if sanitized_bytes else raw
                    else:
                        new_raw = raw

                    info = _tarfile.TarInfo(name=member.name)
                    info.size = len(new_raw)
                    info.mode = member.mode
                    info.mtime = member.mtime
                    dst_tf.addfile(info, _io.BytesIO(new_raw))

            logger.info(f"Sanitized archive saved: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to rebuild sanitized tar {tar_path.name}: {e}")
            out_path.unlink(missing_ok=True)
            return None

    def _sanitize_zip_archive(
        self, zip_path: Path, test_ids: set, answer_cols: list
    ) -> Optional[Path]:
        """Sanitize a top-level .zip archive."""
        import zipfile as _zipfile
        import io as _io

        leaky = False
        try:
            with _zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    n = name.lower()
                    if n.endswith('.csv') or n.endswith('.tsv'):
                        data = zf.read(name)
                        if self._csv_leaks_labels_from_bytes(data, test_ids, answer_cols):
                            leaky = True
                            break
                    elif n.endswith('.mat'):
                        data = zf.read(name)
                        if self._mat_bytes_has_labels(data, test_ids):
                            leaky = True
                            break
        except Exception:
            return None

        if not leaky:
            return None

        sanitized_dir = self.hce_dir / ".sanitized"
        sanitized_dir.mkdir(exist_ok=True)
        out_path = sanitized_dir / zip_path.name

        try:
            with _zipfile.ZipFile(zip_path, 'r') as src_zf, \
                 _zipfile.ZipFile(out_path, 'w', compression=_zipfile.ZIP_DEFLATED) as dst_zf:
                for name in src_zf.namelist():
                    raw = src_zf.read(name)
                    n = name.lower()
                    if n.endswith('.mat'):
                        sanitized = self._sanitize_mat_bytes(raw)
                        raw = sanitized if sanitized else raw
                    elif n.endswith('.csv') or n.endswith('.tsv'):
                        sanitized = self._sanitize_csv_bytes(raw, test_ids, answer_cols)
                        raw = sanitized if sanitized else raw
                    dst_zf.writestr(name, raw)
            return out_path
        except Exception as e:
            logger.error(f"Failed to sanitize zip {zip_path.name}: {e}")
            out_path.unlink(missing_ok=True)
            return None

    def _zip_bytes_has_leakage(
        self, zip_bytes: bytes, test_ids: set, answer_cols: list
    ) -> bool:
        """Check if a ZIP stored as bytes contains test-label data."""
        import zipfile as _zipfile
        import io as _io
        try:
            with _zipfile.ZipFile(_io.BytesIO(zip_bytes)) as zf:
                for name in zf.namelist():
                    n = name.lower()
                    if n.endswith('.csv') or n.endswith('.tsv'):
                        data = zf.read(name)
                        if self._csv_leaks_labels_from_bytes(data, test_ids, answer_cols):
                            return True
                    elif n.endswith('.mat'):
                        data = zf.read(name)
                        if self._mat_bytes_has_labels(data, test_ids):
                            return True
        except Exception:
            pass
        return False

    def _sanitize_zip_bytes(
        self, zip_bytes: bytes, test_ids: set, answer_cols: list
    ) -> Optional[bytes]:
        """Sanitize a ZIP stored as bytes; return sanitized bytes or None."""
        import zipfile as _zipfile
        import io as _io

        if not self._zip_bytes_has_leakage(zip_bytes, test_ids, answer_cols):
            return None

        buf = _io.BytesIO()
        try:
            with _zipfile.ZipFile(_io.BytesIO(zip_bytes)) as src, \
                 _zipfile.ZipFile(buf, 'w', compression=_zipfile.ZIP_DEFLATED) as dst:
                for name in src.namelist():
                    raw = src.read(name)
                    n = name.lower()
                    if n.endswith('.mat'):
                        sanitized = self._sanitize_mat_bytes(raw)
                        raw = sanitized if sanitized else raw
                    elif n.endswith('.csv') or n.endswith('.tsv'):
                        sanitized = self._sanitize_csv_bytes(raw, test_ids, answer_cols)
                        raw = sanitized if sanitized else raw
                    dst.writestr(name, raw)
            return buf.getvalue()
        except Exception:
            return None

    _LABEL_FIELD_NAMES = frozenset(['Labels', 'labels', 'Label', 'label',
                                     'GT', 'gt', 'GroundTruth', 'groundtruth', 'ground_truth'])

    def _mat_bytes_has_labels(self, mat_bytes: bytes, test_ids: set) -> bool:
        """Check if a .mat file contains a 'Labels' field (top-level or nested in struct)."""
        try:
            import scipy.io as _sio
            import io as _io
            mat = _sio.loadmat(_io.BytesIO(mat_bytes))
            for key in mat:
                if key.startswith('_'):
                    continue
                if key in self._LABEL_FIELD_NAMES:
                    return True
                # Check if value is a structured array with Labels sub-field
                val = mat[key]
                if hasattr(val, 'dtype') and val.dtype.names:
                    if any(f in self._LABEL_FIELD_NAMES for f in val.dtype.names):
                        return True
                # Check val[0,0] if shape is (1,1) (common MATLAB struct pattern)
                if (hasattr(val, 'shape') and len(val.shape) == 2
                        and val.shape[0] == 1 and val.shape[1] == 1):
                    try:
                        elem = val[0, 0]
                        if (hasattr(elem, 'dtype') and elem.dtype.names
                                and any(f in self._LABEL_FIELD_NAMES for f in elem.dtype.names)):
                            return True
                    except Exception:
                        pass
        except Exception:
            pass
        return False

    def _sanitize_mat_bytes(self, mat_bytes: bytes) -> Optional[bytes]:
        """Return .mat file bytes with 'Labels' field removed (top-level or nested).

        Handles the MATLAB struct pattern: mat['Video'][0,0] has sub-fields
        including 'Labels'.
        """
        import scipy.io as _sio
        import io as _io
        import numpy as _np

        try:
            mat = _sio.loadmat(_io.BytesIO(mat_bytes))
            changed = False

            # Remove top-level label fields
            for k in list(mat.keys()):
                if k in self._LABEL_FIELD_NAMES:
                    del mat[k]
                    changed = True

            # Remove nested label fields from MATLAB structs
            for key in list(mat.keys()):
                if key.startswith('_'):
                    continue
                val = mat[key]
                if not (hasattr(val, 'dtype') and val.dtype.names):
                    continue
                label_fields = [f for f in val.dtype.names if f in self._LABEL_FIELD_NAMES]
                if not label_fields:
                    # Check one level deeper
                    if (hasattr(val, 'shape') and len(val.shape) == 2
                            and val.shape[0] == 1 and val.shape[1] == 1):
                        try:
                            elem = val[0, 0]
                            if hasattr(elem, 'dtype') and elem.dtype.names:
                                label_fields = [f for f in elem.dtype.names
                                                if f in self._LABEL_FIELD_NAMES]
                                if label_fields:
                                    val = elem  # work on the inner struct
                                    # Rebuild inner struct without label fields
                                    keep = [(n, elem.dtype[n]) for n in elem.dtype.names
                                            if n not in self._LABEL_FIELD_NAMES]
                                    new_dtype = _np.dtype(keep)
                                    vals = tuple(elem[n] for n, _ in keep)
                                    new_inner = _np.array([vals], dtype=new_dtype).reshape(1, 1)
                                    mat[key] = new_inner
                                    changed = True
                        except Exception:
                            pass
                    continue

                if label_fields:
                    # Rebuild struct without label fields
                    keep = [(n, val.dtype[n]) for n in val.dtype.names
                            if n not in self._LABEL_FIELD_NAMES]
                    new_dtype = _np.dtype(keep)
                    # Rebuild each element
                    new_val = _np.empty(val.shape, dtype=new_dtype)
                    for idx in _np.ndindex(val.shape):
                        elem = val[idx]
                        new_val[idx] = tuple(elem[n] for n, _ in keep)
                    mat[key] = new_val
                    changed = True

            if not changed:
                return None

            buf = _io.BytesIO()
            _sio.savemat(buf, mat, do_compression=True)
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"mat sanitization error: {e}")
            return None

    def _csv_leaks_labels_from_bytes(
        self, data: bytes, test_ids: set, answer_cols: list
    ) -> bool:
        """Check if CSV bytes contain test IDs with answer columns."""
        try:
            import io as _io
            df = pd.read_csv(_io.BytesIO(data), nrows=500, low_memory=False)
            return self._csv_leaks_labels_from_df(df, test_ids, answer_cols)
        except Exception:
            return False

    def _csv_leaks_labels_from_df(
        self, df: "pd.DataFrame", test_ids: set, answer_cols: list
    ) -> bool:
        if df.empty or len(df.columns) < 2:
            return False
        id_col = df.columns[0]
        file_ids = set(df[id_col].astype(str))
        if not (file_ids & test_ids):
            return False
        return any(c in df.columns for c in answer_cols)

    def _sanitize_csv_bytes(
        self, data: bytes, test_ids: set, answer_cols: list
    ) -> Optional[bytes]:
        """Return CSV bytes with answer columns removed for test IDs."""
        try:
            import io as _io
            df = pd.read_csv(_io.BytesIO(data), low_memory=False)
            if not self._csv_leaks_labels_from_df(df, test_ids, answer_cols):
                return None
            cols_to_drop = [c for c in answer_cols if c in df.columns]
            df = df.drop(columns=cols_to_drop)
            buf = _io.StringIO()
            df.to_csv(buf, index=False)
            return buf.getvalue().encode()
        except Exception:
            return None

    def _remove_from_all_splits(self, filename: str) -> None:
        """Remove a file (by name) from all three split directories."""
        for split_dir in [self.dtrain_dir, self.dsearch_dir, self.dval_dir]:
            p = split_dir / filename
            if p.exists() or p.is_symlink():
                p.unlink()

    def _replace_in_all_splits(self, filename: str, sanitized_path: Path) -> None:
        """Replace a symlink (by name) in all splits with a sanitized copy."""
        for split_dir in [self.dtrain_dir, self.dsearch_dir, self.dval_dir]:
            p = split_dir / filename
            if p.exists() or p.is_symlink():
                p.unlink()
            # Hard-link or copy sanitized version
            try:
                import os as _os
                _os.link(sanitized_path, p)
            except Exception:
                shutil.copy2(sanitized_path, p)

    def _find_txt_zip_train(self) -> Optional[Path]:
        """Find a .txt.zip training file in the public directory (e.g., billion-word-imputation)."""
        for f in self.public_dir.glob("train*.txt.zip"):
            return f
        for f in self.public_dir.glob("*train*.txt.zip"):
            return f
        return None

    def _prepare_eval_from_txt_zip(
        self, txt_zip: Path, eval_dir: Path, answers_dir: Path, info_path: Path
    ) -> Optional[dict]:
        """Create synthetic eval data from a .txt.zip training file.

        Used for competitions like billion-word-imputation where training data
        is raw text lines. Mimics the competition's test format:
        - Randomly inserts an extra word into each sentence (creating eval_input.csv)
        - Original sentences are the ground-truth answers (eval_answers.csv)
        """
        import zipfile as _zipfile
        import random as _random

        N_EVAL = 5000  # samples for eval (2500 search + 2500 val)
        SEED = int(hashlib.md5(self.competition_id.encode()).hexdigest()[:8], 16) % (2**31)
        rng_obj = _random.Random(SEED)

        logger.info(f"Preparing eval from txt.zip: {txt_zip}")
        try:
            # Sample N_EVAL lines from the zip without reading everything into memory
            with _zipfile.ZipFile(txt_zip) as z:
                inner_name = z.namelist()[0]
                # Reservoir sampling
                reservoir = []
                with z.open(inner_name) as f:
                    for i, raw in enumerate(f):
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        if len(reservoir) < N_EVAL:
                            reservoir.append(line)
                        else:
                            j = rng_obj.randint(0, i)
                            if j < N_EVAL:
                                reservoir[j] = line

            if not reservoir:
                logger.warning("txt.zip train file is empty, skipping eval prep")
                return None

            # Create corrupted (input) sentences: randomly insert a word at another position
            original_sentences = []
            corrupted_sentences = []
            for sent in reservoir:
                words = sent.split()
                if len(words) < 3:
                    # Too short to corrupt meaningfully
                    original_sentences.append(sent)
                    corrupted_sentences.append(sent)
                    continue
                # Pick a word to duplicate/move: pick random index, insert at another position
                pick_idx = rng_obj.randint(0, len(words) - 1)
                word = words[pick_idx]
                insert_idx = rng_obj.randint(0, len(words))  # can be same position
                new_words = list(words)
                new_words.insert(insert_idx, word)
                original_sentences.append(sent)
                corrupted_sentences.append(" ".join(new_words))

            n = len(reservoir)
            ids = list(range(n))

            # Split 50/50 search/val
            split_at = n // 2
            search_ids = ids[:split_at]
            val_ids = ids[split_at:]

            # eval_input: corrupted sentences (what agent sees)
            eval_input = pd.DataFrame({"id": ids, "sentence": corrupted_sentences})
            eval_input_path = eval_dir / "eval_input.csv"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_input.to_csv(eval_input_path, index=False)

            # answers: original sentences
            all_answers = pd.DataFrame({"id": ids, "sentence": original_sentences})
            search_answers = all_answers.iloc[search_ids].reset_index(drop=True)
            val_answers = all_answers.iloc[val_ids].reset_index(drop=True)

            search_answers_path = answers_dir / "eval_search_answers.csv"
            val_answers_path = answers_dir / "eval_val_answers.csv"
            answers_dir.mkdir(parents=True, exist_ok=True)
            search_answers.to_csv(search_answers_path, index=False)
            val_answers.to_csv(val_answers_path, index=False)

            info = {
                "eval_input_path": str(eval_input_path),
                "search_answers_path": str(search_answers_path),
                "val_answers_path": str(val_answers_path),
                "id_col": "id",
                "n_search": len(search_answers),
                "n_val": len(val_answers),
                "n_total": n,
            }
            info_path.write_text(json.dumps(info))
            self._eval_info = info
            logger.info(
                f"Eval data prepared from txt.zip: {n} total, "
                f"{len(search_answers)} search, {len(val_answers)} val"
            )
            return info
        except Exception as e:
            logger.error(f"txt.zip eval prep failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def prepare_eval_data(self, answers_dir: Path = None) -> Optional[dict]:
        """Prepare eval dataset from D_search + D_val for independent evaluation.

        Creates:
        - {hce_dir}/eval/eval_input.csv: features from D_search + D_val (no labels)
        - {answers_dir}/eval_search_answers.csv: grader-format answers for D_search
        - {answers_dir}/eval_val_answers.csv: grader-format answers for D_val
        - {hce_dir}/eval/eval_info.json: metadata

        Args:
            answers_dir: Directory to store answer CSVs. Defaults to hce/eval/.
                         When called by grading_service, this is a private dir
                         inaccessible to the agent.

        The agent predicts on eval_input.csv → eval_submission.csv.
        We grade that against the held-out answer files for search/val scores.

        Returns dict with paths and metadata, or None if preparation fails.
        """
        eval_dir = self.hce_dir / "eval"
        info_path = eval_dir / "eval_info.json"
        # Default answers_dir to eval_dir (original behavior)
        if answers_dir is None:
            answers_dir = eval_dir
        else:
            answers_dir = Path(answers_dir)
            answers_dir.mkdir(parents=True, exist_ok=True)

        # Check if already prepared — verify private answer files exist too (may be missing after machine restart)
        if info_path.exists():
            info = json.loads(info_path.read_text())
            search_ok = Path(info.get("search_answers_path", "")).exists()
            val_ok = Path(info.get("val_answers_path", "")).exists()
            if search_ok and val_ok:
                self._eval_info = info
                logger.info(f"Eval data already prepared: {info.get('n_total', '?')} samples")
                return info
            else:
                logger.warning(
                    f"eval_info.json exists but private answer files missing "
                    f"(search={search_ok}, val={val_ok}) — re-preparing eval data"
                )

        eval_dir.mkdir(parents=True, exist_ok=True)

        # Find train file in D_search directory
        train_file = self._find_train_file_in(self.dsearch_dir)
        if train_file is None:
            # Check for .txt.zip format (e.g., billion-word-imputation)
            txt_zip = self._find_txt_zip_train()
            if txt_zip is not None:
                return self._prepare_eval_from_txt_zip(txt_zip, eval_dir, answers_dir, info_path)
            logger.warning("No train file found in D_search, eval data preparation skipped")
            return None

        try:
            # Read D_search and D_val train data
            if train_file.suffix == ".csv":
                search_df = _robust_read_csv(self.dsearch_dir / train_file.name, low_memory=False)
                val_df = _robust_read_csv(self.dval_dir / train_file.name, low_memory=False)
            elif train_file.suffix in (".jsonl", ".json"):
                search_df = pd.read_json(self.dsearch_dir / train_file.name, lines=True)
                val_df = pd.read_json(self.dval_dir / train_file.name, lines=True)
            else:
                logger.warning(f"Unsupported format for eval: {train_file.suffix}")
                return None

            # Determine feature columns (what the agent sees as input)
            test_file = self._find_test_file()
            if test_file is not None:
                test_df = pd.read_csv(test_file, nrows=5)
                test_cols = list(test_df.columns)
            else:
                # No test.csv (e.g., image tasks) — infer feature cols from answers
                # Feature cols = train cols minus answer target cols (excluding id)
                answers_df_tmp = _robust_read_csv(self.answers_path, nrows=5)
                ans_target_cols = set(answers_df_tmp.columns[1:])  # non-ID answer cols
                test_cols = [c for c in search_df.columns if c not in ans_target_cols]
                logger.info(f"No test.csv found, inferred feature cols: {test_cols}")

            # Read real answers to get answer format
            answers_df = _robust_read_csv(self.answers_path)
            id_col = answers_df.columns[0]

            # Combine D_search + D_val
            combined = pd.concat([search_df, val_df], ignore_index=True)

            # Create eval_input: keep only columns that appear in test file
            feature_cols = [c for c in test_cols if c in combined.columns]
            if not feature_cols:
                logger.warning("No matching feature columns between test and train")
                return None

            eval_input = combined[feature_cols]
            eval_input_path = eval_dir / "eval_input.csv"
            eval_input.to_csv(eval_input_path, index=False)

            # Create eval answers in the same format as real answers
            eval_answers = self._create_eval_answers(combined, answers_df, test_cols)
            if eval_answers is None:
                # Fallback 1: try to enrich train data with image dimensions for segmentation tasks
                eval_answers = self._create_eval_answers_with_image_dims(
                    combined, answers_df, test_cols,
                    split_dirs=[self.dsearch_dir, self.dval_dir]
                )
            if eval_answers is None:
                logger.warning("Could not create eval answers, eval data preparation skipped")
                return None

            # Split answers by D_search / D_val IDs
            search_ids = set(search_df[id_col].astype(str).tolist())
            val_ids = set(val_df[id_col].astype(str).tolist())

            # Handle ID overlap (can occur with corrupted CSV splits for large files)
            overlap_ids = search_ids & val_ids
            if overlap_ids:
                logger.warning(f"ID overlap between search and val: {len(overlap_ids)} IDs, "
                               f"removing from val to ensure disjoint sets")
                val_ids -= overlap_ids

            eval_answers[id_col] = eval_answers[id_col].astype(str)
            search_answers = eval_answers[eval_answers[id_col].isin(search_ids)].reset_index(drop=True)
            val_answers = eval_answers[eval_answers[id_col].isin(val_ids)].reset_index(drop=True)

            search_answers_path = answers_dir / "eval_search_answers.csv"
            val_answers_path = answers_dir / "eval_val_answers.csv"
            search_answers.to_csv(search_answers_path, index=False)
            val_answers.to_csv(val_answers_path, index=False)

            # Symlink non-tabular data (images, audio) into eval dir
            self._symlink_eval_data(eval_dir, exclude={train_file.name, "eval_input.csv"})

            info = {
                "eval_input_path": str(eval_input_path),
                "search_answers_path": str(search_answers_path),
                "val_answers_path": str(val_answers_path),
                "id_col": id_col,
                "n_search": len(search_answers),
                "n_val": len(val_answers),
                "n_total": len(eval_input),
            }
            info_path.write_text(json.dumps(info))
            self._eval_info = info

            logger.info(f"Eval data prepared: {len(eval_input)} total, "
                        f"{len(search_answers)} search, {len(val_answers)} val")
            return info
        except Exception as e:
            logger.error(f"Eval data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_eval_answers(self, train_df: pd.DataFrame, answers_df: pd.DataFrame,
                             test_cols: list) -> Optional[pd.DataFrame]:
        """Create eval answers from train data in the same format as real answers.

        Handles three cases:
        1. Direct mapping: all answer columns exist in train (cassava, us-patent)
        2. One-hot encoding: a categorical train column maps to binary answer columns (spooky)
        3. Column rename: train column name differs from answer column name (jigsaw)
        """
        answer_cols = answers_df.columns.tolist()
        train_cols = train_df.columns.tolist()
        id_col = answer_cols[0]

        # Check if all answer columns exist in train
        missing_cols = [c for c in answer_cols if c not in train_cols]

        if not missing_cols:
            # Case 1: Direct mapping
            return train_df[answer_cols].copy()

        # Find unmapped train columns (in train, not in test, not in answers)
        test_col_set = set(test_cols)
        answer_col_set = set(answer_cols)
        unmapped_train_cols = [c for c in train_cols
                               if c not in test_col_set and c not in answer_col_set]

        # Start building result with directly mappable columns
        direct_cols = [c for c in answer_cols if c in train_cols]
        result = train_df[direct_cols].copy()

        # Case 2: One-hot encoding detection
        for train_col in unmapped_train_cols:
            unique_vals = set(str(v) for v in train_df[train_col].dropna().unique())
            if set(missing_cols).issubset(unique_vals):
                for val in missing_cols:
                    result[val] = (train_df[train_col].astype(str) == val).astype(float)
                logger.info(f"Eval answers: one-hot encoded '{train_col}' → {missing_cols}")
                return result[answer_cols]

        # Case 3: Column rename (1-to-1 mapping between missing and unmapped)
        if len(missing_cols) <= len(unmapped_train_cols) and len(missing_cols) > 0:
            for answer_c, train_c in zip(missing_cols, unmapped_train_cols):
                result[answer_c] = train_df[train_c]
                logger.info(f"Eval answers: renamed '{train_c}' → '{answer_c}'")
            try:
                return result[answer_cols]
            except KeyError:
                pass  # Fall through to failure

        # Case 4: Concatenated ID column (e.g. osic: Patient_Week = Patient + "_" + Weeks)
        # Detect: if any missing column name looks like "<col1>_<col2>" where both cols exist in train
        for missing_col in list(missing_cols):
            for sep in ['_', '-', ' ']:
                parts = missing_col.split(sep)
                if len(parts) >= 2:
                    # Try all prefix combinations
                    for split_i in range(1, len(parts)):
                        left  = sep.join(parts[:split_i])
                        right = sep.join(parts[split_i:])
                        if left in train_df.columns and right in train_df.columns:
                            result[missing_col] = (train_df[left].astype(str)
                                                   + sep + train_df[right].astype(str))
                            logger.info(f"Eval answers: concat '{left}' + '{sep}' + '{right}' → '{missing_col}'")
                            missing_cols = [c for c in missing_cols if c != missing_col]
                            break
                    if missing_col not in missing_cols:
                        break

        # Case 5: Fill remaining missing cols with constant from answers_df (e.g. Confidence=100)
        for missing_col in list(missing_cols):
            if missing_col in answers_df.columns:
                # Use the most common value in answers as the constant
                const_val = answers_df[missing_col].mode().iloc[0] if len(answers_df) > 0 else None
                if const_val is not None:
                    result[missing_col] = const_val
                    logger.info(f"Eval answers: filled '{missing_col}' = {const_val!r} (constant from answers)")
                    missing_cols = [c for c in missing_cols if c != missing_col]

        if not missing_cols:
            try:
                return result[answer_cols]
            except KeyError as e:
                logger.warning(f"Eval answers: KeyError after Case 4/5: {e}")

        logger.warning(f"Cannot map train columns to answers format. "
                       f"Missing: {missing_cols}, Unmapped train: {unmapped_train_cols}")
        return None

    def _create_eval_answers_with_image_dims(
        self,
        train_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        test_cols: list,
        split_dirs: list = None,
    ) -> Optional[pd.DataFrame]:
        """Fallback: add image_width / image_height from actual image files, then retry."""
        import glob as _glob

        dim_cols = [c for c in answers_df.columns if c in ('image_width', 'image_height')]
        if not dim_cols:
            return None  # No image dims needed, this fallback doesn't apply

        # Build id → (width, height) from image files in split directories
        dims: dict = {}
        for split_dir in (split_dirs or []):
            img_root = split_dir / 'train'
            if not img_root.exists():
                continue
            for fpath in _glob.glob(str(img_root / '**' / '*.png'), recursive=True):
                try:
                    parts = fpath.split('/')
                    fname    = parts[-1]    # slice_0013_360_310_1.50_1.50.png
                    case_day = parts[-3]    # case116_day0
                    fp = fname.split('_')
                    slice_id = f'slice_{fp[1]}'
                    row_id = f'{case_day}_{slice_id}'
                    if row_id not in dims:
                        dims[row_id] = (int(fp[2]), int(fp[3]))
                except Exception:
                    pass

        if not dims:
            return None

        id_col = answers_df.columns[0]
        if id_col not in train_df.columns:
            return None

        # Enrich train_df with image dims
        enriched = train_df.copy()
        if 'image_width' not in enriched.columns:
            enriched['image_width']  = enriched[id_col].map(lambda x: dims.get(str(x), (None, None))[0])
        if 'image_height' not in enriched.columns:
            enriched['image_height'] = enriched[id_col].map(lambda x: dims.get(str(x), (None, None))[1])

        # Retry _create_eval_answers with enriched data
        return self._create_eval_answers(enriched, answers_df, test_cols)

    def _find_train_file_in(self, directory: Path) -> Optional[Path]:
        """Find the main training data file in a given directory."""
        candidates = [
            "train.csv", "training_data.csv", "train_labels.csv",
            "train.jsonl", "train_data.csv", "labels.csv",
            "training.csv", "data.csv", "train_set.csv",
            "train.json",  # e.g. stanford-covid-vaccine
            "simplified-nq-train.jsonl",  # tensorflow2-question-answering
        ]
        for name in candidates:
            path = directory / name
            if path.exists():
                return path
        for f in sorted(directory.glob("*train*.csv")):
            return f
        for f in sorted(directory.glob("*train*.jsonl")):
            return f
        return None

    def _find_test_file(self) -> Optional[Path]:
        """Find the main test data file in the public directory."""
        candidates = ["test.csv", "test_data.csv", "test_set.csv"]
        for name in candidates:
            path = self.public_dir / name
            if path.exists():
                return path
        for f in sorted(self.public_dir.glob("*test*.csv")):
            if "sample" not in f.name.lower():
                return f
        return None

    def _symlink_eval_data(self, eval_dir: Path, exclude: set[str] = None) -> None:
        """Symlink public data files/directories into eval directory for image/audio access."""
        exclude = exclude or set()
        for item in self.public_dir.iterdir():
            if item.name in exclude:
                continue
            link = eval_dir / item.name
            if not link.exists():
                try:
                    link.symlink_to(item)
                except OSError:
                    pass

    def grade_eval_submission(self, eval_submission_path: Path, split: str,
                             answers_dir: Path = None) -> Optional[float]:
        """Grade an eval submission against held-out D_search or D_val answers.

        Args:
            eval_submission_path: Path to the agent's eval predictions CSV
            split: 'search' or 'val'
            answers_dir: Directory containing answer CSVs. Defaults to hce/eval/.
                         When called by grading_service, this is a private dir.

        Returns score or None if grading fails.
        """
        eval_submission_path = Path(eval_submission_path)
        if not eval_submission_path.exists():
            return None

        if answers_dir is None:
            answers_dir = self.hce_dir / "eval"
        else:
            answers_dir = Path(answers_dir)

        if split == "search":
            answers_path = answers_dir / "eval_search_answers.csv"
        else:
            answers_path = answers_dir / "eval_val_answers.csv"

        if not answers_path.exists():
            logger.warning(f"Eval answers not found at {answers_path}")
            return None

        try:
            sub_df = pd.read_csv(eval_submission_path)
            ans_df = pd.read_csv(answers_path)
            id_col = ans_df.columns[0]

            # Determine composite key: find the minimal prefix of shared columns
            # that already forms a unique key in ans_df.
            # For single-key:  sub=['contact_id','contact']     → key=['contact_id']
            # For multi-key:   sub=['id','class','predicted']   → key=['id','class']
            shared_cols = [c for c in sub_df.columns if c in ans_df.columns]
            key_cols = [shared_cols[0]] if shared_cols else [id_col]
            for col in shared_cols[1:]:
                if ans_df[key_cols].duplicated().any():
                    key_cols.append(col)
                else:
                    break  # current key_cols already uniquely identifies rows

            # Stringify key columns for consistent matching
            for c in key_cols:
                sub_df[c] = sub_df[c].astype(str)
                ans_df[c] = ans_df[c].astype(str)

            if len(key_cols) > 1:
                # Multi-column key: match on composite key tuple
                ans_key_set = set(zip(*[ans_df[c] for c in key_cols]))
                sub_key = list(zip(*[sub_df[c] for c in key_cols]))
                sub_filtered = sub_df[[k in ans_key_set for k in sub_key]].reset_index(drop=True)
            else:
                # Single-column key
                answer_ids = set(ans_df[id_col])
                sub_filtered = sub_df[sub_df[id_col].isin(answer_ids)].reset_index(drop=True)

            if len(sub_filtered) == 0:
                logger.warning(f"No matching IDs in eval submission for split={split} "
                               f"(submission has {len(sub_df)} rows, answer set has {len(ans_df)} rows)")
                return None

            # Require at least 50% coverage of expected eval IDs.
            coverage = len(sub_filtered) / len(ans_df)
            if coverage < 0.5:
                logger.warning(
                    f"Eval submission coverage too low for split={split}: "
                    f"{len(sub_filtered)}/{len(ans_df)} rows ({coverage:.1%}) — returning None"
                )
                return None
            if coverage < 1.0:
                logger.warning(
                    f"Eval submission partial coverage for split={split}: "
                    f"{len(sub_filtered)}/{len(ans_df)} rows ({coverage:.1%})"
                )

            # Sort both by composite key for consistent row alignment
            sub_filtered = sub_filtered.sort_values(key_cols).reset_index(drop=True)
            if len(key_cols) > 1:
                present_keys = set(zip(*[sub_filtered[c] for c in key_cols]))
                ans_mask = [tuple(r) in present_keys for r in ans_df[key_cols].values]
                ans_df = ans_df[ans_mask].sort_values(key_cols).reset_index(drop=True)
            else:
                present_ids = set(sub_filtered[id_col])
                ans_df = ans_df[ans_df[id_col].isin(present_ids)].sort_values(id_col).reset_index(drop=True)

            grader = self.competition.grader
            # Fill NaN/empty strings to avoid ZeroDivisionError in metrics like Jaccard
            for col in sub_filtered.columns:
                if sub_filtered[col].dtype == object:
                    sub_filtered[col] = sub_filtered[col].fillna("").astype(str)
            for col in ans_df.columns:
                if ans_df[col].dtype == object:
                    ans_df[col] = ans_df[col].fillna("").astype(str)

            try:
                score = grader.grade_fn(sub_filtered, ans_df)
            except Exception as e:
                logger.warning(f"Eval grading metric computation failed for split={split}: {e}")
                return None

            if score is None or (isinstance(score, float) and np.isnan(score)):
                logger.warning(f"Eval grading returned {score} for split={split}")
                return None

            logger.info(f"Eval grading ({split}): score={score} "
                        f"({len(sub_filtered)}/{len(sub_df)} rows)")
            return score
        except Exception as e:
            logger.warning(f"Eval grading failed for split={split}: {e}")
            return None

    def grade_submission(self, submission_path: Path) -> Optional[float]:
        """Grade a submission CSV using mlebench grader. Returns score or None."""
        from mlebench.utils import load_answers, read_csv

        submission_path = Path(submission_path)
        if not submission_path.exists():
            logger.error(f"Submission file not found: {submission_path}")
            return None

        try:
            submission_df = read_csv(submission_path)
            answers = load_answers(self.competition.answers)
            score = self.competition.grader(submission_df, answers)
            logger.info(f"Graded {submission_path}: score={score}")
            if score is None:
                logger.warning(f"Score is None — likely invalid submission format. "
                             f"Check sample_submission for expected format.")
                return None
            # Handle NaN scores (treat as invalid)
            if isinstance(score, float) and np.isnan(score):
                logger.warning(f"Score is NaN — likely invalid predictions. "
                             f"Check for NaN values in your predictions.")
                return None
            return score
        except Exception as e:
            logger.error(f"Grading failed for {submission_path}: {e}")
            return None

    def compute_percentile_rank(self, score: float) -> float:
        """
        Compute percentile rank: P = (N - R) / (N - 1) * 100
        where R is the rank (1-indexed) of the score in the leaderboard.
        Returns -1.0 if leaderboard is unavailable (e.g. LFS pointer).
        """
        try:
            # Handle NaN/None scores
            if score is None or (isinstance(score, float) and np.isnan(score)):
                logger.warning(f"Cannot compute percentile for NaN/None score")
                return -1.0

            # Detect Git LFS pointer files (not real CSV data)
            with open(self.leaderboard_path) as f:
                first_line = f.readline()
            if first_line.startswith("version https://git-lfs"):
                logger.warning(f"Leaderboard is a Git LFS pointer, cannot compute PR")
                return -1.0

            lb = pd.read_csv(self.leaderboard_path)
            scores = lb["score"].dropna().values
            N = len(scores)
            if N <= 1:
                return 50.0

            if self.is_lower_better:
                # Lower is better: rank = number of scores <= this score
                R = np.sum(scores <= score)
            else:
                # Higher is better: rank = number of scores >= this score
                R = np.sum(scores >= score)

            percentile = (N - R) / (N - 1) * 100
            return max(0.0, min(100.0, percentile))
        except Exception as e:
            logger.error(f"Percentile rank computation failed: {e}")
            return -1.0

    def get_sample_submission(self) -> pd.DataFrame:
        """Load sample submission."""
        return pd.read_csv(self.sample_submission_path)

    def check_submission_format(self, submission_path: Path) -> Optional[str]:
        """Compare submission.csv against sample_submission.csv.

        Returns a human-readable error string listing all issues found,
        or None if the format looks correct.  Runs quickly (no model inference).
        """
        submission_path = Path(submission_path)
        issues = []

        # --- Load submission ---
        try:
            sub = pd.read_csv(submission_path)
        except Exception as e:
            return f"CANNOT READ submission.csv: {e}"

        # --- Load sample ---
        try:
            sample = pd.read_csv(self.sample_submission_path)
        except Exception:
            return None  # Can't check without a valid sample

        sam_cols = list(sample.columns)
        sub_cols = list(sub.columns)

        # Column names / order
        if sub_cols != sam_cols:
            missing = [c for c in sam_cols if c not in sub_cols]
            extra   = [c for c in sub_cols if c not in sam_cols]
            if missing:
                issues.append(f"Missing columns: {missing}  (expected: {sam_cols})")
            if extra:
                issues.append(f"Unexpected extra columns: {extra}")
            if not missing and not extra:
                issues.append(f"Column order wrong: got {sub_cols}, expected {sam_cols}")

        # Row count
        if len(sub) != len(sample):
            issues.append(
                f"Row count: got {len(sub)}, expected {len(sample)}  "
                f"({'too few' if len(sub) < len(sample) else 'too many'} rows)"
            )

        # ID column checks (first column)
        id_col = sam_cols[0] if sam_cols else None
        if id_col and id_col in sub_cols:
            sample_ids = sample[id_col].astype(str)
            sub_ids    = sub[id_col].astype(str)

            # Float-formatted IDs ("123.0" instead of "123")
            float_fmt = sub_ids[sub_ids.str.match(r'^\d+\.0$')].head(3)
            if len(float_fmt) > 0:
                issues.append(
                    f"ID column '{id_col}' has float format: {list(float_fmt)}  "
                    f"— use .astype(int).astype(str) or proper formatting"
                )

            # ID set mismatch (only when row counts match, to avoid noise)
            if len(sub) == len(sample):
                expected_ids = set(sample_ids)
                actual_ids   = set(sub_ids)
                missing_ids  = expected_ids - actual_ids
                extra_ids    = actual_ids   - expected_ids
                if missing_ids:
                    ex = sorted(missing_ids)[:3]
                    issues.append(f"Missing {len(missing_ids)} IDs from sample_submission (e.g. {ex})")
                if extra_ids:
                    ex = sorted(extra_ids)[:3]
                    issues.append(f"{len(extra_ids)} IDs not in sample_submission (e.g. {ex})")

        # Value-column checks
        for col in sub_cols[1:]:
            if col not in sub.columns:
                continue
            # NaN
            n_nan = sub[col].isna().sum()
            if n_nan > 0:
                issues.append(f"Column '{col}' has {n_nan} NaN values")
            # Degenerate (all identical)
            if sub[col].nunique(dropna=False) == 1:
                issues.append(
                    f"Column '{col}' has only one unique value ({sub[col].iloc[0]!r})  "
                    f"— model may not be predicting correctly"
                )

        if not issues:
            return None  # Format OK

        header = "FORMAT ISSUES DETECTED — fix these before resubmitting:"
        body   = "\n".join(f"  • {i}" for i in issues)
        return f"{header}\n{body}"

    def get_context_for_agent(self) -> str:
        """Build context string for the ReAct agent."""
        # Read sample submission header
        sample_sub = pd.read_csv(self.sample_submission_path, nrows=3)
        sample_str = sample_sub.to_string(index=False)

        # List available data files
        try:
            data_files = [f.name for f in self.dtrain_dir.iterdir() if f.is_file()]
            data_files_str = ", ".join(sorted(data_files)[:10])
        except Exception:
            data_files_str = "(check with ls)"

        # Eval data section (if prepared)
        eval_section = ""
        if hasattr(self, '_eval_info') and self._eval_info:
            eval_path = self._eval_info["eval_input_path"]
            n_eval = self._eval_info["n_total"]
            eval_section = f"""
### Eval Data (REQUIRED — for model selection)
After training your model and creating submission.csv for the test set, you MUST also
predict on the eval dataset and save as `./eval_submission.csv`:
- Eval input: {eval_path}  ({n_eval} samples)
- Format: same columns as test data, but these are held-out training samples
- Your eval predictions should be in the SAME format as submission.csv
- For image/audio tasks: eval images are in the TRAINING data directory (not test)
- Save as `./eval_submission.csv` in your current working directory

This eval prediction is used for model selection and does NOT affect your leaderboard score.
"""

        ctx = f"""## Competition: {self.competition_id}

### Description
{self.description}

### Data Directory
Training data: {self.dtrain_dir}
Files: {data_files_str}
Test data / sample submission are in the same directory.
(Use ONLY this data for training your model)

### Sample Submission Format
{sample_str}

Full sample submission: {self.dtrain_dir}/sample_submission.csv

### Submission Requirements
- Save as `./submission.csv` in your current working directory
- Must match the sample submission format exactly (same columns, same number of rows)
- When ready: Action: submit(submission.csv)
{eval_section}
### Environment
- GPU: {self._get_gpu_info()} available (use CUDA for training)
- Python packages: torch, sklearn, xgboost, lightgbm, transformers, timm, etc.
"""
        return ctx

    @staticmethod
    def _get_gpu_info() -> str:
        """Report single GPU per instance."""
        return "1x NVIDIA L20Z 81GB"
