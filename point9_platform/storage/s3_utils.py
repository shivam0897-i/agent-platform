"""
S3 Storage Utilities
====================

Handles file uploads, downloads, and presigned URLs for Point9 agents.

Usage:
    from point9_platform.storage import S3Storage, get_s3_storage
    
    # Using singleton
    s3 = get_s3_storage()
    s3.upload_file("local.pdf", "inputs/session-123/doc.pdf")
    
    # Or create custom instance
    s3 = S3Storage(bucket_name="my-bucket")

Environment Variables:
    S3_BUCKET_NAME: Target S3 bucket
    AWS_ACCESS_KEY_ID: AWS credentials
    AWS_SECRET_ACCESS_KEY: AWS credentials
    AWS_REGION: AWS region (default: us-east-1)
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Storage:
    """
    S3 storage client for Point9 agents.
    
    Handles:
    - File uploads (local files, bytes, JSON)
    - File downloads
    - Presigned URL generation
    - File listing and deletion
    """
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        """
        Initialize S3 client.
        
        Args:
            bucket_name: S3 bucket name (or from env S3_BUCKET_NAME)
            region: AWS region (or from env AWS_REGION)
            access_key: AWS access key (or from env AWS_ACCESS_KEY_ID)
            secret_key: AWS secret key (or from env AWS_SECRET_ACCESS_KEY)
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "point9-storage")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        self.client = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
    
    # ==================== Upload Methods ====================
    
    def upload_file(
        self,
        file_path: str,
        s3_key: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a local file to S3.
        
        Args:
            file_path: Local file path
            s3_key: S3 object key (e.g., 'inputs/session-123/image.jpg')
            content_type: MIME type (auto-detected if not provided)
            
        Returns:
            Dict with success, s3_key, bucket (or error)
        """
        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            
            self.client.upload_file(
                file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            
            return {
                "success": True,
                "s3_key": s3_key,
                "bucket": self.bucket_name
            }
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return {"success": False, "error": str(e)}
    
    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        content_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Upload bytes directly to S3.
        
        Args:
            data: Bytes to upload
            s3_key: S3 object key
            content_type: MIME type
            
        Returns:
            Dict with success, s3_key, bucket (or error)
        """
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type
            )
            
            logger.info(f"Uploaded bytes to s3://{self.bucket_name}/{s3_key}")
            
            return {
                "success": True,
                "s3_key": s3_key,
                "bucket": self.bucket_name
            }
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return {"success": False, "error": str(e)}
    
    def upload_json(self, data: Dict, s3_key: str) -> Dict[str, Any]:
        """
        Upload JSON data to S3.
        
        Args:
            data: Dictionary to serialize as JSON
            s3_key: S3 object key
            
        Returns:
            Dict with success, s3_key, bucket (or error)
        """
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        return self.upload_bytes(json_bytes, s3_key, "application/json")
    
    # ==================== Download Methods ====================
    
    def download_file(self, s3_key: str, local_path: str) -> Dict[str, Any]:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local destination path
            
        Returns:
            Dict with success, local_path (or error)
        """
        try:
            self.client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return {"success": True, "local_path": local_path}
            
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return {"success": False, "error": str(e)}
    
    def download_bytes(self, s3_key: str) -> Optional[bytes]:
        """
        Download file content as bytes.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            File content as bytes, or None if failed
        """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return None
    
    # ==================== URL Generation ====================
    
    def get_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        operation: str = "get_object"
    ) -> Optional[str]:
        """
        Generate a presigned URL for S3 object.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration in seconds (default 1 hour)
            operation: 'get_object' for download, 'put_object' for upload
            
        Returns:
            Presigned URL or None if failed
        """
        try:
            url = self.client.generate_presigned_url(
                operation,
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    # ==================== File Management ====================
    
    def delete_file(self, s3_key: str) -> bool:
        """Delete a file from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False
    
    def list_files(self, prefix: str) -> List[Dict[str, Any]]:
        """
        List files with given prefix.
        
        Args:
            prefix: S3 key prefix (e.g., 'inputs/session-123/')
            
        Returns:
            List of file info dicts with key, size, last_modified
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get("Contents", []):
                files.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat()
                })
            
            return files
            
        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
            return []
    
    def file_exists(self, s3_key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False


# Singleton instance
_s3_instance: Optional[S3Storage] = None


def get_s3_storage() -> S3Storage:
    """Get or create S3 storage singleton instance."""
    global _s3_instance
    if _s3_instance is None:
        _s3_instance = S3Storage()
    return _s3_instance
