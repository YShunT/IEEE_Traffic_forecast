"""全タスクの答えを提出キーに結合し、提出ファイル（Parquet・zstd）を書く。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb

from core.config import SUBMISSION_ROWS, out, work_path
from core.runtime import log, section
from task1_3.part import state_file
from task2.ongoing.ensemble import NAME as ONGOING
from task2.onset_mix import NAME as ONSET

FILE_NAME = "submission.parquet"
SPLITS = ("validation", "private")
KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id"],
}
VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sql(path):
    return "'" + str(path).replace("'", "''") + "'"


def assemble(c, files, path):
    key = Path(c["data"]) / "submission_key.csv"
    using = {task: ",".join(columns) for task, columns in KEYS.items()}
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='4GB'")
        con.execute(f"CREATE TABLE keys AS SELECT * FROM read_csv({sql(key)},all_varchar=true)")
        counts = {}
        for task, paths in files.items():
            con.execute(
                f"CREATE TABLE {task} AS SELECT {','.join(KEYS[task] + VALUES[task])} "
                f"FROM read_parquet([{','.join(sql(p) for p in paths)}])"
            )
            duplicated = con.execute(
                f"SELECT count(*) FROM (SELECT {using[task]} FROM {task} GROUP BY ALL HAVING count(*) > 1)"
            ).fetchone()[0]
            if duplicated:
                raise ValueError(f"{task}: {duplicated} duplicate keys")
            counts[task] = con.execute(f"SELECT count(*) FROM {task}").fetchone()[0]
        con.execute(f"""CREATE TABLE submission AS SELECT * FROM (
            SELECT CAST(k.submission_id AS BIGINT) submission_id,k.task,s.speed_kmh::DOUBLE speed_kmh,s.flow_vph::DOUBLE flow_vph,
                   0.0::DOUBLE queue_pred,0.0::DOUBLE path_flow
              FROM keys k LEFT JOIN state s USING({using["state"]}) WHERE k.task='state'
            UNION ALL SELECT CAST(k.submission_id AS BIGINT),k.task,0.0,0.0,q.queue_pred::DOUBLE,0.0
              FROM keys k LEFT JOIN queue q USING({using["queue"]}) WHERE k.task='queue'
            UNION ALL SELECT CAST(k.submission_id AS BIGINT),k.task,0.0,0.0,0.0,o.path_flow::DOUBLE
              FROM keys k LEFT JOIN odme o USING({using["odme"]}) WHERE k.task='odme') ORDER BY submission_id""")
        rows, ids, blank, queue, path_flow = con.execute("""SELECT count(*),count(DISTINCT submission_id),
            count(*) FILTER (WHERE speed_kmh IS NULL OR flow_vph IS NULL OR queue_pred IS NULL OR path_flow IS NULL),
            count(*) FILTER (WHERE task='queue' AND queue_pred NOT IN (0,1)),
            count(*) FILTER (WHERE task='odme' AND (NOT isfinite(path_flow) OR path_flow < 0)) FROM submission""").fetchone()
        by_task = dict(con.execute("SELECT task,count(*) FROM keys GROUP BY task").fetchall())
        if rows != SUBMISSION_ROWS or ids != rows or blank or queue or path_flow or any(counts[t] != by_task[t] for t in counts):
            raise ValueError(
                f"submission rejected: rows {rows}, distinct ids {ids}, blank {blank}, queue not 0/1 {queue}, "
                f"bad path_flow {path_flow}, answers {counts} against keys {by_task}"
            )
        temp = path.with_suffix(path.suffix + ".tmp")
        con.execute(f"COPY submission TO {sql(temp)} (FORMAT parquet,COMPRESSION zstd)")
        temp.replace(path)
    return rows, counts


def run(c, run_id):
    section("submit")
    files = {
        "state": [state_file(c, run_id, s) for s in SPLITS],
        "queue": [work_path(c, run_id, f"official_{ONSET}.parquet"), work_path(c, run_id, f"official_{ONGOING}.parquet")],
        "odme": [work_path(c, run_id, f"path_flow_{s}.parquet") for s in SPLITS],
    }
    path = out(c, FILE_NAME)
    rows, counts = assemble(c, files, path)
    log(
        f"  [submit]  {path} | rows: {rows:,} | answers: "
        + ", ".join(f"{k} {v:,}" for k, v in counts.items())
        + f" | sha256 {sha256(path)[:16]}"
    )
