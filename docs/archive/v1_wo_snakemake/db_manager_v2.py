#!/usr/bin/env python3
"""
Database Manager for SVS RAW Image Processing Pipeline
Enhanced with NCSU → JUNO → Ceres tracking
Tracks batches through three-tier storage system
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


class BatchDatabase:
    """Manages SQLite database for batch and file tracking"""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Main batches table with NCSU tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                date TEXT NOT NULL,
                
                -- Storage paths
                ncsu_path TEXT,
                juno_path TEXT NOT NULL,
                ceres_path TEXT,
                output_path TEXT,
                
                -- Status tracking
                ncsu_sync_status TEXT DEFAULT 'unknown',  -- unknown, needed, syncing, synced, failed
                transfer_status TEXT DEFAULT 'pending',   -- pending, transferring, transferred, failed
                processing_status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
                
                -- Metadata
                file_count INTEGER DEFAULT 0,
                total_size_bytes INTEGER DEFAULT 0,
                globus_task_id TEXT,
                
                -- Timestamps
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ncsu_synced_at TIMESTAMP,
                transferred_at TIMESTAMP,
                processing_started_at TIMESTAMP,
                processing_completed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,
                
                -- Status
                ncsu_exists BOOLEAN DEFAULT 0,
                juno_exists BOOLEAN DEFAULT 0,
                ceres_exists BOOLEAN DEFAULT 0,
                transfer_status TEXT DEFAULT 'pending',
                processing_status TEXT DEFAULT 'pending',
                
                -- Processing outputs
                dng_created BOOLEAN DEFAULT 0,
                dng_path TEXT,
                jpg_created BOOLEAN DEFAULT 0,
                jpg_path TEXT,
                
                -- Metadata
                file_size_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id),
                UNIQUE(batch_id, filename)
            )
        ''')
        
        # NCSU sync history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ncsu_sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                globus_task_id TEXT,
                status TEXT NOT NULL,
                bytes_transferred INTEGER,
                files_transferred INTEGER,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            )
        ''')
        
        # Transfer history (JUNO → Ceres)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transfer_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                globus_task_id TEXT,
                status TEXT NOT NULL,
                bytes_transferred INTEGER,
                files_transferred INTEGER,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            )
        ''')
        
        # Processing history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                job_id TEXT,
                status TEXT NOT NULL,
                files_processed INTEGER DEFAULT 0,
                files_failed INTEGER DEFAULT 0,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            )
        ''')
        
        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_status ON batches(ncsu_sync_status, transfer_status, processing_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_state ON batches(state)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_batch ON files(batch_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_status ON files(transfer_status, processing_status)')
        
        self.conn.commit()
    
    def add_batch(self, batch_id: str, juno_path: str, ncsu_path: Optional[str] = None,
                  ceres_path: Optional[str] = None, state: Optional[str] = None) -> bool:
        """
        Add a new batch to database
        
        Args:
            batch_id: Batch identifier (e.g., 'MD_2025-10-22')
            juno_path: Path on JUNO storage
            ncsu_path: Optional path on NCSU storage
            ceres_path: Optional path on Ceres
            state: State code (extracted from batch_id if not provided)
            
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        # Extract state and date from batch_id if not provided
        if not state:
            state = batch_id.split('_')[0]
        
        date = batch_id.split('_')[1] if '_' in batch_id else None
        
        # Determine NCSU sync status
        ncsu_sync_status = 'synced' if not ncsu_path else 'unknown'
        
        try:
            cursor.execute('''
                INSERT INTO batches (batch_id, state, date, ncsu_path, juno_path, ceres_path, 
                                   ncsu_sync_status, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (batch_id, state, date, ncsu_path, juno_path, ceres_path, ncsu_sync_status))
            
            self.conn.commit()
            return True
        
        except sqlite3.IntegrityError:
            # Batch already exists, update paths if provided
            updates = []
            params = []
            
            if ncsu_path:
                updates.append('ncsu_path = ?')
                params.append(ncsu_path)
            if juno_path:
                updates.append('juno_path = ?')
                params.append(juno_path)
            if ceres_path:
                updates.append('ceres_path = ?')
                params.append(ceres_path)
            
            if updates:
                updates.append('updated_at = CURRENT_TIMESTAMP')
                params.append(batch_id)
                
                cursor.execute(f'''
                    UPDATE batches 
                    SET {', '.join(updates)}
                    WHERE batch_id = ?
                ''', params)
                
                self.conn.commit()
            
            return True
    
    def update_batch_status(self, batch_id: str, ncsu_sync_status: Optional[str] = None,
                          transfer_status: Optional[str] = None,
                          processing_status: Optional[str] = None,
                          globus_task_id: Optional[str] = None,
                          file_count: Optional[int] = None) -> bool:
        """
        Update batch status
        
        Args:
            batch_id: Batch identifier
            ncsu_sync_status: NCSU sync status
            transfer_status: Transfer status (JUNO → Ceres)
            processing_status: Processing status
            globus_task_id: Globus task ID
            file_count: Number of files in batch
            
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if ncsu_sync_status:
            updates.append('ncsu_sync_status = ?')
            params.append(ncsu_sync_status)
            if ncsu_sync_status == 'synced':
                updates.append('ncsu_synced_at = CURRENT_TIMESTAMP')
        
        if transfer_status:
            updates.append('transfer_status = ?')
            params.append(transfer_status)
            if transfer_status == 'transferred':
                updates.append('transferred_at = CURRENT_TIMESTAMP')
        
        if processing_status:
            updates.append('processing_status = ?')
            params.append(processing_status)
            if processing_status == 'processing':
                updates.append('processing_started_at = CURRENT_TIMESTAMP')
            elif processing_status == 'completed':
                updates.append('processing_completed_at = CURRENT_TIMESTAMP')
        
        if globus_task_id:
            updates.append('globus_task_id = ?')
            params.append(globus_task_id)
        
        if file_count is not None:
            updates.append('file_count = ?')
            params.append(file_count)
        
        if not updates:
            return False
        
        updates.append('updated_at = CURRENT_TIMESTAMP')
        params.append(batch_id)
        
        cursor.execute(f'''
            UPDATE batches 
            SET {', '.join(updates)}
            WHERE batch_id = ?
        ''', params)
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get_batches_needing_sync(self, state_filter: Optional[str] = None) -> List[Dict]:
        """
        Get batches that need to be synced from NCSU to JUNO
        
        Args:
            state_filter: Optional state code filter
            
        Returns:
            List of batch dictionaries
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT * FROM batches 
            WHERE ncsu_sync_status IN ('unknown', 'needed', 'failed')
            AND ncsu_path IS NOT NULL
        '''
        
        params = []
        if state_filter:
            query += ' AND state = ?'
            params.append(state_filter)
        
        query += ' ORDER BY date DESC, batch_id'
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_batches_ready_for_transfer(self, state_filter: Optional[str] = None) -> List[Dict]:
        """
        Get batches ready to transfer from JUNO to Ceres
        
        Args:
            state_filter: Optional state code filter
            
        Returns:
            List of batch dictionaries
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT * FROM batches 
            WHERE (ncsu_sync_status = 'synced' OR ncsu_path IS NULL)
            AND transfer_status IN ('pending', 'failed')
        '''
        
        params = []
        if state_filter:
            query += ' AND state = ?'
            params.append(state_filter)
        
        query += ' ORDER BY date DESC, batch_id'
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_batches_by_status(self, ncsu_sync_status: Optional[str] = None,
                             transfer_status: Optional[str] = None,
                             processing_status: Optional[str] = None,
                             state_filter: Optional[str] = None) -> List[Dict]:
        """
        Get batches filtered by status
        
        Args:
            ncsu_sync_status: Filter by NCSU sync status
            transfer_status: Filter by transfer status
            processing_status: Filter by processing status
            state_filter: Filter by state code
            
        Returns:
            List of batch dictionaries
        """
        cursor = self.conn.cursor()
        
        conditions = []
        params = []
        
        if ncsu_sync_status:
            conditions.append('ncsu_sync_status = ?')
            params.append(ncsu_sync_status)
        
        if transfer_status:
            conditions.append('transfer_status = ?')
            params.append(transfer_status)
        
        if processing_status:
            conditions.append('processing_status = ?')
            params.append(processing_status)
        
        if state_filter:
            conditions.append('state = ?')
            params.append(state_filter)
        
        query = 'SELECT * FROM batches'
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY date DESC, batch_id'
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def record_ncsu_sync(self, batch_id: str, globus_task_id: str,
                        status: str, bytes_transferred: Optional[int] = None,
                        files_transferred: Optional[int] = None,
                        error_message: Optional[str] = None) -> bool:
        """
        Record NCSU sync event in history
        
        Args:
            batch_id: Batch identifier
            globus_task_id: Globus task ID
            status: Status (started, completed, failed)
            bytes_transferred: Bytes transferred
            files_transferred: Files transferred
            error_message: Error message if failed
            
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        timestamp_field = 'started_at' if status == 'started' else 'completed_at'
        
        cursor.execute(f'''
            INSERT INTO ncsu_sync_history 
            (batch_id, globus_task_id, status, bytes_transferred, files_transferred, 
             error_message, {timestamp_field})
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (batch_id, globus_task_id, status, bytes_transferred, files_transferred, error_message))
        
        self.conn.commit()
        return True
    
    def add_file(self, batch_id: str, filename: str, file_path: Optional[str] = None,
                ncsu_exists: bool = False, juno_exists: bool = False,
                ceres_exists: bool = False) -> bool:
        """Add file to database"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO files (batch_id, filename, file_path, ncsu_exists, juno_exists, ceres_exists)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (batch_id, filename, file_path, ncsu_exists, juno_exists, ceres_exists))
            
            self.conn.commit()
            return True
        
        except sqlite3.IntegrityError:
            # File already exists, update it
            cursor.execute('''
                UPDATE files 
                SET file_path = ?, ncsu_exists = ?, juno_exists = ?, ceres_exists = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE batch_id = ? AND filename = ?
            ''', (file_path, ncsu_exists, juno_exists, ceres_exists, batch_id, filename))
            
            self.conn.commit()
            return True
    
    def get_batch_summary(self) -> Dict:
        """
        Get overall pipeline summary
        
        Returns:
            Dictionary with summary statistics
        """
        cursor = self.conn.cursor()
        
        # Overall counts
        cursor.execute('SELECT COUNT(*) as total FROM batches')
        total_batches = cursor.fetchone()['total']
        
        # NCSU sync status
        cursor.execute('''
            SELECT ncsu_sync_status, COUNT(*) as count 
            FROM batches 
            WHERE ncsu_path IS NOT NULL
            GROUP BY ncsu_sync_status
        ''')
        ncsu_status = {row['ncsu_sync_status']: row['count'] for row in cursor.fetchall()}
        
        # Transfer status
        cursor.execute('''
            SELECT transfer_status, COUNT(*) as count 
            FROM batches 
            GROUP BY transfer_status
        ''')
        transfer_status = {row['transfer_status']: row['count'] for row in cursor.fetchall()}
        
        # Processing status
        cursor.execute('''
            SELECT processing_status, COUNT(*) as count 
            FROM batches 
            GROUP BY processing_status
        ''')
        processing_status = {row['processing_status']: row['count'] for row in cursor.fetchall()}
        
        return {
            'total_batches': total_batches,
            'ncsu_sync': ncsu_status,
            'transfer': transfer_status,
            'processing': processing_status
        }
    
    def export_to_json(self, output_file: str):
        """Export all data to JSON file"""
        cursor = self.conn.cursor()
        
        data = {
            'exported_at': datetime.now().isoformat(),
            'batches': [],
            'summary': self.get_batch_summary()
        }
        
        cursor.execute('SELECT * FROM batches ORDER BY batch_id')
        for batch_row in cursor.fetchall():
            batch = dict(batch_row)
            
            # Get files for this batch
            cursor.execute('SELECT * FROM files WHERE batch_id = ?', (batch['batch_id'],))
            batch['files'] = [dict(row) for row in cursor.fetchall()]
            
            data['batches'].append(batch)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """CLI for database operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management for SVS RAW pipeline')
    parser.add_argument(
        '--db',
        default='/project/dash_agir/matthew.kutugata/pipeline_tracking.db',
        help='Database path'
    )
    parser.add_argument(
        'command',
        choices=['summary', 'list', 'export', 'init'],
        help='Command to execute'
    )
    parser.add_argument('--state', help='Filter by state')
    parser.add_argument('--output', help='Output file for export')
    
    args = parser.parse_args()
    
    db = BatchDatabase(args.db)
    
    if args.command == 'summary':
        summary = db.get_batch_summary()
        print("\n=== PIPELINE SUMMARY ===")
        print(f"Total batches: {summary['total_batches']}")
        print(f"\nNCSU Sync Status:")
        for status, count in summary['ncsu_sync'].items():
            print(f"  {status}: {count}")
        print(f"\nTransfer Status (JUNO → Ceres):")
        for status, count in summary['transfer'].items():
            print(f"  {status}: {count}")
        print(f"\nProcessing Status:")
        for status, count in summary['processing'].items():
            print(f"  {status}: {count}")
    
    elif args.command == 'list':
        batches = db.get_batches_by_status(state_filter=args.state)
        print(f"\n{'Batch ID':<20} {'NCSU Sync':<12} {'Transfer':<12} {'Processing':<12}")
        print("="*60)
        for batch in batches:
            print(f"{batch['batch_id']:<20} {batch['ncsu_sync_status']:<12} "
                  f"{batch['transfer_status']:<12} {batch['processing_status']:<12}")
    
    elif args.command == 'export':
        output_file = args.output or 'pipeline_data.json'
        db.export_to_json(output_file)
        print(f"✅ Exported to {output_file}")
    
    elif args.command == 'init':
        print("✅ Database initialized")
    
    db.close()


if __name__ == '__main__':
    main()
