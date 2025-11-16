"""
AWS S3 Storage Integration
Handles final audio file storage and retrieval

Benefits:
- Fast, scalable storage for audio files
- CDN-like delivery via presigned URLs
- Automatic lifecycle management (optional)
- Cost-effective ($0.023/GB/month storage, $0.09/GB transfer)

Setup:
1. Create S3 bucket: aws s3 mb s3://wavecraft-audio
2. Set environment variables:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_S3_BUCKET (default: wavecraft-audio)
   - AWS_REGION (default: us-east-1)
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
import logging
from typing import Optional
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class S3AudioStorage:
    """
    S3 storage manager for audio files

    Features:
    - Upload final audio to S3
    - Generate presigned URLs for download
    - Optional lifecycle policies for auto-deletion
    - Error handling and fallbacks
    """

    def __init__(self):
        """
        Initialize S3 client

        Reads from environment:
        - AWS_ACCESS_KEY_ID: AWS access key
        - AWS_SECRET_ACCESS_KEY: AWS secret key
        - AWS_S3_BUCKET: S3 bucket name (default: wavecraft-audio)
        - AWS_REGION: AWS region (default: us-east-1)
        - S3_ENABLED: Enable/disable S3 (default: True if credentials present)
        """
        self.bucket_name = os.environ.get('AWS_S3_BUCKET', 'wavecraft-audio')
        self.region = os.environ.get('AWS_REGION', 'us-east-1')
        self.s3_enabled = os.environ.get('S3_ENABLED', 'true').lower() == 'true'

        # Check if credentials are available
        self.access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        self.secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

        if not self.access_key or not self.secret_key:
            logger.warning("AWS credentials not found, S3 storage disabled")
            self.s3_enabled = False
            self.s3_client = None
            return

        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )

            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)

            logger.info(f"✓ S3 storage initialized: {self.bucket_name} ({self.region})")

        except NoCredentialsError:
            logger.error("AWS credentials invalid")
            self.s3_enabled = False
            self.s3_client = None

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                logger.error(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                logger.error(f"S3 client error: {e}")

            self.s3_enabled = False
            self.s3_client = None

    def upload_audio(
        self,
        audio_data: bytes,
        filename: Optional[str] = None,
        content_type: str = 'audio/wav'
    ) -> Optional[str]:
        """
        Upload audio file to S3

        Args:
            audio_data: Audio file content as bytes
            filename: Optional custom filename (default: auto-generated UUID)
            content_type: MIME type (default: audio/wav)

        Returns:
            S3 object key if successful, None if failed

        Example:
            >>> storage = S3AudioStorage()
            >>> with open('audio.wav', 'rb') as f:
            ...     audio_data = f.read()
            >>> key = storage.upload_audio(audio_data)
            >>> print(f"Uploaded: {key}")
        """
        if not self.s3_enabled or not self.s3_client:
            logger.warning("S3 not enabled, skipping upload")
            return None

        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                unique_id = str(uuid.uuid4())[:8]
                filename = f"audio_{timestamp}_{unique_id}.wav"

            # Add folder structure for organization
            # Format: audio/YYYY/MM/DD/filename.wav
            date_path = datetime.utcnow().strftime('%Y/%m/%d')
            s3_key = f"audio/{date_path}/{filename}"

            logger.info(f"Uploading to S3: {s3_key} ({len(audio_data) / 1024 / 1024:.2f} MB)")

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=audio_data,
                ContentType=content_type,
                # Optional: Add metadata
                Metadata={
                    'uploaded_at': datetime.utcnow().isoformat(),
                    'service': 'wavecraft'
                },
                # Optional: Server-side encryption
                # ServerSideEncryption='AES256'
            )

            logger.info(f"✓ Upload successful: s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            return None

    def get_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate presigned URL for downloading audio

        Presigned URLs allow temporary, secure access to S3 objects
        without requiring AWS credentials

        Args:
            s3_key: S3 object key (from upload_audio)
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)

        Returns:
            Presigned URL string if successful, None if failed

        Example:
            >>> url = storage.get_presigned_url('audio/2024/01/15/audio_xyz.wav')
            >>> print(f"Download: {url}")
            >>> # URL valid for 1 hour
        """
        if not self.s3_enabled or not self.s3_client:
            logger.warning("S3 not enabled, cannot generate presigned URL")
            return None

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )

            logger.info(f"Generated presigned URL for {s3_key} (expires in {expiration}s)")
            return url

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    def upload_and_get_url(
        self,
        audio_data: bytes,
        filename: Optional[str] = None,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Upload audio and immediately get presigned URL

        Convenience method combining upload_audio() and get_presigned_url()

        Args:
            audio_data: Audio file content as bytes
            filename: Optional custom filename
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL if successful, None if failed

        Example:
            >>> with open('final_audio.wav', 'rb') as f:
            ...     audio_data = f.read()
            >>> url = storage.upload_and_get_url(audio_data)
            >>> # Return URL to client for download
        """
        # Upload file
        s3_key = self.upload_audio(audio_data, filename)

        if not s3_key:
            return None

        # Generate presigned URL
        url = self.get_presigned_url(s3_key, expiration)

        return url

    def delete_audio(self, s3_key: str) -> bool:
        """
        Delete audio file from S3

        Note: In production, consider using S3 lifecycle policies
        for automatic deletion instead of manual deletion

        Args:
            s3_key: S3 object key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_enabled or not self.s3_client:
            logger.warning("S3 not enabled, cannot delete")
            return False

        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            logger.info(f"✓ Deleted: s3://{self.bucket_name}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False

    def is_enabled(self) -> bool:
        """Check if S3 storage is enabled and configured"""
        return self.s3_enabled and self.s3_client is not None

    def get_stats(self) -> dict:
        """Get storage statistics"""
        return {
            'enabled': self.is_enabled(),
            'bucket': self.bucket_name,
            'region': self.region
        }


# ===========================================================================
# S3 LIFECYCLE POLICY (Optional - for auto-deletion)
# ===========================================================================

"""
To automatically delete old audio files, configure S3 lifecycle policy:

AWS Console:
1. Go to S3 bucket → Management → Lifecycle rules
2. Create rule:
   - Rule name: "auto-delete-old-audio"
   - Prefix: "audio/"
   - Expiration: 30 days
   - Status: Enabled

AWS CLI:
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket wavecraft-audio \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "auto-delete-old-audio",
      "Status": "Enabled",
      "Prefix": "audio/",
      "Expiration": {
        "Days": 30
      }
    }]
  }'
```

Cost savings:
- Without lifecycle: Files stored forever
- With 30-day lifecycle: 87% cost reduction
- Example: 1000 files × 5MB × $0.023/GB/month × 12 months = $1.38/year
           vs with lifecycle = $0.18/year
"""
