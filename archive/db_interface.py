"""
Database interface for svs-raw-api integration with agir-db.

This module provides functions to:
- Query batches that need processing
- Update processing status
- Log processing events
- Track file inventory
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras


@dataclass
class DBConfig:
    """Database connection configuration."""
    host: str
    port: int
    dbname: str
    user: str

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("PGHOST"),
            port=int(os.environ.get("PGPORT", 5432)),
            dbname=os.environ.get("PGDATABASE", "agir"),
            user=os.environ.get("PGUSER", os.environ.get("USER")),
        )


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    
    Usage:
        with get_db_connection() as conn:
            # do work
            conn.commit()
    """
    cfg = DBConfig.from_env()
    conn = psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
    )
    try:
        yield conn
    finally:
        conn.close()


# ============================================================
# BATCH INVENTORY & STATUS
# ============================================================

def sync_batch_inventory(conn, batch_id: str) -> Dict:
    """
    Update batch file counts from source.globus_file_index.
    
    Returns the updated batch record.
    """
    sql = """
    WITH file_counts AS (
        SELECT 
            batch_id,
            batch_state,
            batch_date,
            SUM(CASE WHEN file_ext IN ('raw','RAW') AND data_state = 'upload_raw' THEN 1 ELSE 0 END) as raw_count,
            SUM(CASE WHEN file_ext = 'dng' AND data_state = 'developed_jpg' THEN 1 ELSE 0 END) as dng_count,
            SUM(CASE WHEN file_ext IN ('jpg','jpeg') AND data_state = 'developed_jpg' THEN 1 ELSE 0 END) as jpg_count,
            SUM(CASE WHEN file_ext = 'json' AND data_state = 'developed_jpg' THEN 1 ELSE 0 END) as json_count,
            MAX(site) as primary_site,
            MAX(lts_root) as primary_lts_root
        FROM source.globus_file_index
        WHERE batch_id = %s AND entry_type = 'file'
        GROUP BY batch_id, batch_state, batch_date
    )
    INSERT INTO processed.batch_processing_status (
        batch_id, batch_state, batch_date,
        raw_file_count, dng_file_count, jpg_file_count, json_file_count,
        primary_site, primary_lts_root
    )
    SELECT 
        batch_id, batch_state, batch_date,
        raw_count, dng_count, jpg_count, json_count,
        primary_site, primary_lts_root
    FROM file_counts
    ON CONFLICT (batch_id) DO UPDATE SET
        raw_file_count = EXCLUDED.raw_file_count,
        dng_file_count = EXCLUDED.dng_file_count,
        jpg_file_count = EXCLUDED.jpg_file_count,
        json_file_count = EXCLUDED.json_file_count,
        primary_site = EXCLUDED.primary_site,
        primary_lts_root = EXCLUDED.primary_lts_root,
        updated_at = now()
    RETURNING *;
    """
    
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (batch_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_batches_for_processing(conn, pipeline_stage: str, limit: int = 10) -> List[Dict]:
    """
    Get batches ready for processing.
    
    Args:
        pipeline_stage: 'raw_to_dng' or 'dng_to_jpg'
        limit: Maximum number of batches to return
    
    Returns:
        List of batch records
    """
    if pipeline_stage == 'raw_to_dng':
        view = 'processed.v_batches_ready_for_dng'
    elif pipeline_stage == 'dng_to_jpg':
        view = 'processed.v_batches_ready_for_jpg'
    else:
        raise ValueError(f"Invalid pipeline_stage: {pipeline_stage}")
    
    sql = f"SELECT * FROM {view} LIMIT %s;"
    
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return [dict(row) for row in cur.fetchall()]


def get_batch_files(conn, batch_id: str, data_state: str, file_ext: str = None) -> List[Dict]:
    """
    Get file list for a batch from source.globus_file_index.
    
    Args:
        batch_id: Batch identifier
        data_state: 'upload_raw' or 'developed_jpg'
        file_ext: Optional file extension filter (e.g., 'raw', 'jpg')
    
    Returns:
        List of file records with paths
    """
    sql = """
    SELECT 
        file_id,
        file_name,
        rel_path,
        root_path,
        size_bytes,
        endpoint,
        site,
        lts_root
    FROM source.globus_file_index
    WHERE 
        batch_id = %s
        AND data_state = %s
        AND entry_type = 'file'
    """
    
    params = [batch_id, data_state]
    
    if file_ext:
        sql += " AND LOWER(file_ext) = LOWER(%s)"
        params.append(file_ext)
    
    sql += " ORDER BY file_name;"
    
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


# ============================================================
# STATUS UPDATES
# ============================================================

def start_batch_processing(
    conn, 
    batch_id: str, 
    pipeline_stage: str,
    job_id: str = None
) -> None:
    """
    Mark a batch as starting processing.
    
    Args:
        batch_id: Batch identifier
        pipeline_stage: 'raw_to_dng' or 'dng_to_jpg'
        job_id: Optional SLURM job ID
    """
    if pipeline_stage == 'raw_to_dng':
        sql = """
        UPDATE processed.batch_processing_status
        SET 
            raw_to_dng_status = 'in_progress',
            raw_to_dng_started = %s,
            raw_to_dng_job_id = %s,
            updated_at = now()
        WHERE batch_id = %s;
        """
    elif pipeline_stage == 'dng_to_jpg':
        sql = """
        UPDATE processed.batch_processing_status
        SET 
            dng_to_jpg_status = 'in_progress',
            dng_to_jpg_started = %s,
            dng_to_jpg_job_id = %s,
            updated_at = now()
        WHERE batch_id = %s;
        """
    else:
        raise ValueError(f"Invalid pipeline_stage: {pipeline_stage}")
    
    with conn.cursor() as cur:
        cur.execute(sql, (datetime.now(timezone.utc), job_id, batch_id))


def complete_batch_processing(
    conn,
    batch_id: str,
    pipeline_stage: str,
    success: bool = True,
    error_message: str = None
) -> None:
    """
    Mark a batch as completed (or failed).
    
    Args:
        batch_id: Batch identifier
        pipeline_stage: 'raw_to_dng' or 'dng_to_jpg'
        success: Whether processing succeeded
        error_message: Optional error message if failed
    """
    status = 'completed' if success else 'failed'
    now = datetime.now(timezone.utc)
    
    if pipeline_stage == 'raw_to_dng':
        sql = """
        UPDATE processed.batch_processing_status
        SET 
            raw_to_dng_status = %s,
            raw_to_dng_completed = %s,
            last_error = %s,
            retry_count = CASE WHEN %s THEN 0 ELSE retry_count + 1 END,
            updated_at = now()
        WHERE batch_id = %s;
        """
    elif pipeline_stage == 'dng_to_jpg':
        sql = """
        UPDATE processed.batch_processing_status
        SET 
            dng_to_jpg_status = %s,
            dng_to_jpg_completed = %s,
            last_error = %s,
            retry_count = CASE WHEN %s THEN 0 ELSE retry_count + 1 END,
            updated_at = now()
        WHERE batch_id = %s;
        """
    else:
        raise ValueError(f"Invalid pipeline_stage: {pipeline_stage}")
    
    with conn.cursor() as cur:
        cur.execute(sql, (status, now, error_message, success, batch_id))


# ============================================================
# EVENT LOGGING
# ============================================================

def log_processing_event(
    conn,
    batch_id: str,
    file_name: str,
    pipeline_stage: str,
    status: str,
    input_path: str = None,
    output_path: str = None,
    processing_time_sec: float = None,
    error_message: str = None,
    job_id: str = None,
    node_name: str = None
) -> None:
    """
    Log a processing event for a single file.
    
    Args:
        batch_id: Batch identifier
        file_name: Name of the file processed
        pipeline_stage: 'raw_to_dng' or 'dng_to_jpg'
        status: 'success', 'failed', or 'skipped'
        input_path: Input file path
        output_path: Output file path
        processing_time_sec: Processing duration in seconds
        error_message: Error message if failed
        job_id: SLURM job ID
        node_name: Compute node name
    """
    sql = """
    INSERT INTO logs.image_processing_events (
        batch_id, file_name, pipeline_stage, status,
        input_path, output_path, processing_time_sec,
        error_message, job_id, node_name
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, (
            batch_id, file_name, pipeline_stage, status,
            input_path, output_path, processing_time_sec,
            error_message, job_id, node_name
        ))


def log_batch_events(conn, events: List[Dict]) -> None:
    """
    Bulk insert processing events.
    
    Args:
        events: List of event dictionaries
    """
    if not events:
        return
    
    sql = """
    INSERT INTO logs.image_processing_events (
        batch_id, file_name, pipeline_stage, status,
        input_path, output_path, processing_time_sec,
        error_message, job_id, node_name
    ) VALUES (
        %(batch_id)s, %(file_name)s, %(pipeline_stage)s, %(status)s,
        %(input_path)s, %(output_path)s, %(processing_time_sec)s,
        %(error_message)s, %(job_id)s, %(node_name)s
    );
    """
    
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, events)


# ============================================================
# REPORTING
# ============================================================

def get_processing_summary(conn) -> Dict:
    """
    Get overall processing summary statistics.
    
    Returns:
        Dictionary with summary statistics
    """
    sql = "SELECT * FROM processed.v_batch_processing_summary;"
    
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        result = cur.fetchone()
        return dict(result) if result else {}


def get_batch_status(conn, batch_id: str) -> Optional[Dict]:
    """
    Get processing status for a specific batch.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        Batch status record or None if not found
    """
    sql = "SELECT * FROM processed.batch_processing_status WHERE batch_id = %s;"
    
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (batch_id,))
        result = cur.fetchone()
        return dict(result) if result else None
