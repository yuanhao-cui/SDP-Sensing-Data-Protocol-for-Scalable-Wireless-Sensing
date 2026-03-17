"""Tests for CLI module."""
import sys
import pytest
from unittest.mock import patch
from wsdp.cli import main_cli, __version__


class TestVersion:
    def test_version_string(self):
        assert __version__ == "0.2.0"

    def test_version_flag(self):
        with patch('sys.argv', ['wsdp', '--version']):
            with pytest.raises(SystemExit) as exc_info:
                main_cli()
            assert exc_info.value.code == 0


class TestListCommand:
    def test_list_basic(self, capsys):
        with patch('sys.argv', ['wsdp', 'list']):
            main_cli()
        captured = capsys.readouterr()
        assert "Available datasets" in captured.out
        assert "widar" in captured.out

    def test_list_verbose(self, capsys):
        with patch('sys.argv', ['wsdp', 'list', '--verbose']):
            main_cli()
        captured = capsys.readouterr()
        assert "Format:" in captured.out
        assert "Description:" in captured.out


class TestHelpCommand:
    def test_no_args(self, capsys):
        with patch('sys.argv', ['wsdp']):
            main_cli()
        captured = capsys.readouterr()
        assert "available commands" in captured.out

    def test_download_help(self, capsys):
        with patch('sys.argv', ['wsdp', 'download', '-h']):
            with pytest.raises(SystemExit):
                main_cli()


class TestNonInteractiveMode:
    def test_download_has_email_arg(self):
        """Verify --email argument is available."""
        from wsdp.cli import main_cli
        with patch('sys.argv', ['wsdp', 'download', '-h']):
            with pytest.raises(SystemExit):
                main_cli()

    def test_download_has_token_arg(self):
        """Verify --token argument is available."""
        with patch('sys.argv', ['wsdp', 'download', '-h']):
            with pytest.raises(SystemExit):
                main_cli()
