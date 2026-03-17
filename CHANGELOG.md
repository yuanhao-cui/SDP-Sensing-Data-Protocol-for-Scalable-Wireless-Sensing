# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of WSDP (Wi-Fi Sensing Data Processing)
- Multi-dataset support: Widar, Gait, XRF55, ElderAL, ZTE
- Intelligent preprocessing: wavelet denoising, phase calibration, signal resizing
- Deep learning pipeline with CNN + Transformer architecture
- CLI interface with authentication support (JWT, email/password)
- Visualization tools: heatmaps, denoising comparison, phase calibration plots
- Inference API for deployment
- 53 unit tests covering core modules

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.2.0] - 2026-03-16

### Added
- JWT Token authentication support
- S3 region auto-detection and fix
- Non-interactive mode for CLI (--email, --password, --token)
- --version flag
- list command with --verbose option
- Custom model injection support

### Changed
- Improved README with bilingual support
- Enhanced CLI help messages

## [0.1.0] - 2026-02-14

### Added
- Initial prototype
- Basic readers for Widar and Gait datasets
- Simple preprocessing pipeline
