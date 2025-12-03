#!/usr/bin/env python3
"""
Enhanced Globus Transfer Manager with NCSU → JUNO → Ceres Pipeline
Manages three-tier data flow:
1. Discover batches in NCSU NFS storage
2. Sync missing batches from NCSU → JUNO
3. Transfer batches from JUNO → Ceres (existing functionality)
"""

import subprocess
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse

# Import database manager
try:
    from db_manager import BatchDatabase
except ImportError:
    print("Warning: db_manager.py not found. Database functionality will be limited.")
    BatchDatabase = None


class GlobusTransferManager:
    """Manages Globus transfers with three-tier storage system"""
    
    # Globus Endpoint IDs
    JUNO_ENDPOINT = "904c2108-90cf-11e8-9672-0a6d4e044368"
    CERES_ENDPOINT = "f45a24f8-09ba-11ec-b342-1feaf93e3729"
    NCSU_ENDPOINT = "PLACEHOLDER_NCSU_ENDPOINT_ID"  # TODO: User needs to provide this
    
    # Storage paths
    JUNO_BASE_PATH = "/project/dash_agir/semifield-upload"
    CERES_BASE_PATH = "/90daydata/dash_agir/data/semifield-upload"
    NCSU_BASE_PATH = "PLACEHOLDER_NCSU_PATH"  # TODO: User needs to provide this (e.g., /rsstu/users/group/semifield-upload)
    
    # Batch name pattern: STATE_YYYY-MM-DD
    BATCH_PATTERN = re.compile(r'^([A-Z]{2})_(\d{4})-(\d{2})-(\d{2})$')
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Globus manager
        
        Args:
            db_path: Path to SQLite database for tracking
        """
        self.db = None
        if db_path and BatchDatabase:
            self.db = BatchDatabase(db_path)
    
    def _run_globus_cmd(self, cmd: List[str]) -> Tuple[bool, str, str]:
        """
        Run a Globus CLI command
        
        Args:
            cmd: Command to run as list of strings
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check_globus_login(self) -> bool:
        """Check if user is logged into Globus CLI"""
        success, stdout, _ = self._run_globus_cmd(['globus', 'whoami'])
        return success and len(stdout.strip()) > 0
    
    def list_ncsu_batches(self, state_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all batches available in NCSU storage
        
        Args:
            state_filter: Optional state code to filter (e.g., 'MD', 'NC')
            
        Returns:
            List of batch dictionaries with 'batch_id', 'state', 'date', 'path'
        """
        print(f"🔍 Scanning NCSU storage: {self.NCSU_BASE_PATH}")
        
        cmd = [
            'globus', 'ls',
            f'{self.NCSU_ENDPOINT}:{self.NCSU_BASE_PATH}',
            '--format', 'json'
        ]
        
        success, stdout, stderr = self._run_globus_cmd(cmd)
        
        if not success:
            print(f"❌ Error listing NCSU batches: {stderr}")
            return []
        
        try:
            items = json.loads(stdout)
        except json.JSONDecodeError:
            print("❌ Failed to parse Globus output")
            return []
        
        batches = []
        for item in items:
            if item['type'] == 'dir':
                name = item['name']
                match = self.BATCH_PATTERN.match(name)
                if match:
                    state, year, month, day = match.groups()
                    if state_filter and state != state_filter:
                        continue
                    
                    batches.append({
                        'batch_id': name,
                        'state': state,
                        'date': f"{year}-{month}-{day}",
                        'path': f"{self.NCSU_BASE_PATH}/{name}"
                    })
        
        batches.sort(key=lambda x: x['batch_id'])
        return batches
    
    def list_juno_batches(self, state_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all batches available in JUNO storage
        
        Args:
            state_filter: Optional state code to filter
            
        Returns:
            List of batch dictionaries
        """
        print(f"🔍 Scanning JUNO archive: {self.JUNO_BASE_PATH}")
        
        cmd = [
            'globus', 'ls',
            f'{self.JUNO_ENDPOINT}:{self.JUNO_BASE_PATH}',
            '--format', 'json'
        ]
        
        success, stdout, stderr = self._run_globus_cmd(cmd)
        
        if not success:
            print(f"❌ Error listing JUNO batches: {stderr}")
            return []
        
        try:
            items = json.loads(stdout)
        except json.JSONDecodeError:
            print("❌ Failed to parse Globus output")
            return []
        
        batches = []
        for item in items:
            if item['type'] == 'dir':
                name = item['name']
                match = self.BATCH_PATTERN.match(name)
                if match:
                    state, year, month, day = match.groups()
                    if state_filter and state != state_filter:
                        continue
                    
                    batches.append({
                        'batch_id': name,
                        'state': state,
                        'date': f"{year}-{month}-{day}",
                        'path': f"{self.JUNO_BASE_PATH}/{name}"
                    })
        
        batches.sort(key=lambda x: x['batch_id'])
        return batches
    
    def find_missing_batches(self, state_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Find batches that exist in NCSU but not in JUNO
        
        Args:
            state_filter: Optional state code to filter
            
        Returns:
            List of batch dictionaries that need to be synced to JUNO
        """
        print("\n" + "="*60)
        print("COMPARING NCSU vs JUNO STORAGE")
        print("="*60)
        
        ncsu_batches = self.list_ncsu_batches(state_filter)
        juno_batches = self.list_juno_batches(state_filter)
        
        ncsu_ids = {b['batch_id'] for b in ncsu_batches}
        juno_ids = {b['batch_id'] for b in juno_batches}
        
        missing_ids = ncsu_ids - juno_ids
        missing_batches = [b for b in ncsu_batches if b['batch_id'] in missing_ids]
        
        print(f"\n📊 Summary:")
        print(f"   NCSU batches:  {len(ncsu_batches)}")
        print(f"   JUNO batches:  {len(juno_batches)}")
        print(f"   Missing in JUNO: {len(missing_batches)}")
        
        if missing_batches:
            print(f"\n📦 Batches in NCSU but not in JUNO:")
            for batch in missing_batches:
                print(f"   • {batch['batch_id']} ({batch['date']})")
        
        return missing_batches
    
    def sync_ncsu_to_juno(self, batch_id: str, label: Optional[str] = None) -> Tuple[bool, str]:
        """
        Sync a batch from NCSU to JUNO
        
        Args:
            batch_id: Batch ID to sync (e.g., 'MD_2025-10-22')
            label: Optional label for the transfer
            
        Returns:
            Tuple of (success, task_id or error_message)
        """
        if not self.BATCH_PATTERN.match(batch_id):
            return False, f"Invalid batch ID format: {batch_id}"
        
        source_path = f"{self.NCSU_BASE_PATH}/{batch_id}/"
        dest_path = f"{self.JUNO_BASE_PATH}/{batch_id}/"
        
        if not label:
            label = f"NCSU→JUNO: {batch_id}"
        
        print(f"\n📤 Syncing batch to JUNO: {batch_id}")
        print(f"   Source: {source_path}")
        print(f"   Dest:   {dest_path}")
        
        cmd = [
            'globus', 'transfer',
            '--recursive',
            '--sync-level', 'checksum',
            '--preserve-timestamp',
            '--label', label,
            '--format', 'json',
            f'{self.NCSU_ENDPOINT}:{source_path}',
            f'{self.JUNO_ENDPOINT}:{dest_path}'
        ]
        
        success, stdout, stderr = self._run_globus_cmd(cmd)
        
        if not success:
            print(f"❌ Transfer submission failed: {stderr}")
            return False, stderr
        
        try:
            result = json.loads(stdout)
            task_id = result.get('task_id')
            
            if task_id:
                print(f"✅ Transfer submitted: {task_id}")
                print(f"   Monitor: globus task show {task_id}")
                
                # Update database if available
                if self.db:
                    self.db.add_batch(
                        batch_id=batch_id,
                        juno_path=dest_path,
                        ncsu_path=source_path
                    )
                    self.db.update_batch_status(
                        batch_id=batch_id,
                        ncsu_sync_status='syncing',
                        globus_task_id=task_id
                    )
                
                return True, task_id
            else:
                print(f"❌ No task ID in response: {stdout}")
                return False, "No task ID returned"
        
        except json.JSONDecodeError:
            print(f"❌ Failed to parse transfer response: {stdout}")
            return False, "Invalid JSON response"
    
    def submit_juno_to_ceres_transfer(self, batch_id: str, label: Optional[str] = None) -> Tuple[bool, str]:
        """
        Submit transfer from JUNO to Ceres (existing functionality)
        
        Args:
            batch_id: Batch ID to transfer
            label: Optional label for the transfer
            
        Returns:
            Tuple of (success, task_id or error_message)
        """
        if not self.BATCH_PATTERN.match(batch_id):
            return False, f"Invalid batch ID format: {batch_id}"
        
        source_path = f"{self.JUNO_BASE_PATH}/{batch_id}/"
        dest_path = f"{self.CERES_BASE_PATH}/{batch_id}/"
        
        if not label:
            label = f"JUNO→Ceres: {batch_id}"
        
        print(f"\n📤 Transferring to Ceres: {batch_id}")
        print(f"   Source: {source_path}")
        print(f"   Dest:   {dest_path}")
        
        cmd = [
            'globus', 'transfer',
            '--recursive',
            '--sync-level', 'checksum',
            '--preserve-timestamp',
            '--label', label,
            '--format', 'json',
            f'{self.JUNO_ENDPOINT}:{source_path}',
            f'{self.CERES_ENDPOINT}:{dest_path}'
        ]
        
        success, stdout, stderr = self._run_globus_cmd(cmd)
        
        if not success:
            print(f"❌ Transfer submission failed: {stderr}")
            return False, stderr
        
        try:
            result = json.loads(stdout)
            task_id = result.get('task_id')
            
            if task_id:
                print(f"✅ Transfer submitted: {task_id}")
                print(f"   Monitor: globus task show {task_id}")
                
                # Update database
                if self.db:
                    self.db.update_batch_status(
                        batch_id=batch_id,
                        transfer_status='transferring',
                        globus_task_id=task_id
                    )
                
                return True, task_id
            else:
                return False, "No task ID returned"
        
        except json.JSONDecodeError:
            return False, "Invalid JSON response"
    
    def check_transfer_status(self, task_id: str) -> Dict[str, str]:
        """Check status of a Globus transfer"""
        cmd = ['globus', 'task', 'show', task_id, '--format', 'json']
        success, stdout, stderr = self._run_globus_cmd(cmd)
        
        if not success:
            return {'status': 'ERROR', 'message': stderr}
        
        try:
            result = json.loads(stdout)
            return {
                'status': result.get('status', 'UNKNOWN'),
                'bytes_transferred': result.get('bytes_transferred', 0),
                'files': result.get('files', 0),
                'completion_time': result.get('completion_time', '')
            }
        except json.JSONDecodeError:
            return {'status': 'ERROR', 'message': 'Invalid JSON response'}
    
    def wait_for_transfer(self, task_id: str, timeout: int = 3600) -> bool:
        """Wait for a transfer to complete"""
        cmd = ['globus', 'task', 'wait', task_id, '--timeout', str(timeout)]
        success, stdout, stderr = self._run_globus_cmd(cmd)
        return success


def main():
    """CLI interface for Globus manager"""
    parser = argparse.ArgumentParser(
        description='Manage three-tier Globus transfers: NCSU → JUNO → Ceres'
    )
    parser.add_argument(
        'action',
        choices=['check-missing', 'sync-to-juno', 'transfer-to-ceres', 'full-sync', 'status'],
        help='Action to perform'
    )
    parser.add_argument(
        '--batch-id',
        help='Batch ID for specific operations (e.g., MD_2025-10-22)'
    )
    parser.add_argument(
        '--state',
        help='Filter by state code (e.g., MD, NC)'
    )
    parser.add_argument(
        '--db',
        default='/project/dash_agir/matthew.kutugata/pipeline_tracking.db',
        help='Path to database file'
    )
    parser.add_argument(
        '--task-id',
        help='Globus task ID to check status'
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = GlobusTransferManager(db_path=args.db)
    
    # Check Globus login
    if not manager.check_globus_login():
        print("❌ Not logged into Globus CLI. Run: globus login")
        sys.exit(1)
    
    # Check endpoint configuration
    if manager.NCSU_ENDPOINT == "PLACEHOLDER_NCSU_ENDPOINT_ID":
        print("⚠️  WARNING: NCSU endpoint not configured!")
        print("   Edit NCSU_ENDPOINT and NCSU_BASE_PATH in this script")
        print("   Find your endpoint: globus endpoint search 'NC State'")
        sys.exit(1)
    
    # Execute action
    if args.action == 'check-missing':
        missing = manager.find_missing_batches(state_filter=args.state)
        if not missing:
            print("\n✅ All NCSU batches are already in JUNO")
        sys.exit(0)
    
    elif args.action == 'sync-to-juno':
        if not args.batch_id:
            print("❌ --batch-id required for sync-to-juno")
            sys.exit(1)
        
        success, result = manager.sync_ncsu_to_juno(args.batch_id)
        sys.exit(0 if success else 1)
    
    elif args.action == 'transfer-to-ceres':
        if not args.batch_id:
            print("❌ --batch-id required for transfer-to-ceres")
            sys.exit(1)
        
        success, result = manager.submit_juno_to_ceres_transfer(args.batch_id)
        sys.exit(0 if success else 1)
    
    elif args.action == 'full-sync':
        """Sync all missing batches from NCSU → JUNO"""
        missing = manager.find_missing_batches(state_filter=args.state)
        
        if not missing:
            print("\n✅ Nothing to sync")
            sys.exit(0)
        
        print(f"\n🔄 Will sync {len(missing)} batches from NCSU → JUNO")
        response = input("Continue? [y/N]: ")
        
        if response.lower() != 'y':
            print("Cancelled")
            sys.exit(0)
        
        for batch in missing:
            print(f"\n{'='*60}")
            success, result = manager.sync_ncsu_to_juno(batch['batch_id'])
            if not success:
                print(f"❌ Failed to sync {batch['batch_id']}: {result}")
    
    elif args.action == 'status':
        if not args.task_id:
            print("❌ --task-id required for status check")
            sys.exit(1)
        
        status = manager.check_transfer_status(args.task_id)
        print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
