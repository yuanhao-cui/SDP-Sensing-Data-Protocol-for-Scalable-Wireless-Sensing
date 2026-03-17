"""Tests for download module."""
import os
import pytest
from unittest.mock import patch, MagicMock
from wsdp.download import _resolve_s3_region, download


class TestS3RegionRedirect:
    def test_no_redirect(self):
        """URL with no redirect should return unchanged."""
        with patch('wsdp.download.requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}
            mock_req.head.return_value = mock_resp
            result = _resolve_s3_region("https://example.com/file.zip", "file.zip")
            assert result == "https://example.com/file.zip"

    def test_301_redirect(self):
        """301 redirect should return new URL."""
        with patch('wsdp.download.requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 301
            mock_resp.headers = {'Location': 'https://correct-region.s3.amazonaws.com/file.zip'}
            mock_req.head.return_value = mock_resp
            result = _resolve_s3_region("https://wrong-region.s3.amazonaws.com/file.zip", "file.zip")
            assert "correct-region" in result

    def test_302_redirect(self):
        """302 redirect should return new URL."""
        with patch('wsdp.download.requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 302
            mock_resp.headers = {'Location': 'https://new-url.com/file.zip'}
            mock_req.head.return_value = mock_resp
            result = _resolve_s3_region("https://old-url.com/file.zip", "file.zip")
            assert result == "https://new-url.com/file.zip"


class TestDownloadAuth:
    def test_download_with_token(self):
        """download() should use token when provided."""
        with patch('wsdp.download.load_mapping', return_value="test.zip"), \
             patch('wsdp.download._download_without_aws', side_effect=Exception("force SDP")), \
             patch('wsdp.download.load_api', return_value="https://api.test/auth"), \
             patch('wsdp.download.requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "https://download.url/file.zip"
            mock_req.post.return_value = mock_resp

            with patch('wsdp.download._download_file_from_url') as mock_dl:
                download("widar", "/tmp", token="my-jwt-token")
                # Verify token was used
                call_args = mock_req.post.call_args
                headers = call_args[1]['headers']
                assert 'Authorization' in headers
                assert headers['Authorization'] == 'Bearer my-jwt-token'

    def test_download_with_email_password(self):
        """download() should use email/password when no token."""
        with patch('wsdp.download.load_mapping', return_value="test.zip"), \
             patch('wsdp.download._download_without_aws', side_effect=Exception("force SDP")), \
             patch('wsdp.download.load_api', return_value="https://api.test/auth"), \
             patch('wsdp.download.requests') as mock_req:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "https://download.url/file.zip"
            mock_req.post.return_value = mock_resp

            with patch('wsdp.download._download_file_from_url'):
                download("widar", "/tmp", email="user@test.com", password="secret")
                call_args = mock_req.post.call_args
                payload = call_args[1]['json']
                assert payload['email'] == "user@test.com"
                assert payload['password'] == "secret"
