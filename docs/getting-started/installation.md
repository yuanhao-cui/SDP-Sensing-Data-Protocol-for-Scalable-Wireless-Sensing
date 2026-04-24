# Installation

## Requirements

- Python 3.9 or higher
- PyTorch 1.10 or higher
- CUDA (optional, for GPU acceleration)

## Install from PyPI (Coming Soon)

```bash
pip install wsdp
```

## Install from Source

### Clone the Repository

```bash
git clone https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing.git
cd SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing
```

### Install in Development Mode

```bash
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

This will install additional dependencies for testing:
- pytest
- pytest-cov

## Verify Installation

```bash
# Check CLI
wsdp --version

# Run tests
pytest tests/ -v
```

## Create SDP8.org Account

Before downloading datasets, create a free account at **[SDP8.org](https://sdp8.org)**.

Your SDP8.org credentials are used for `wsdp download` authentication:

```bash
# Option 1: Email/password
wsdp download elderAL ./data --email you@example.com --password yourpassword

# Option 2: JWT token (from SDP8.org dashboard)
wsdp download elderAL ./data --token YOUR_JWT_TOKEN
```

## Docker Installation

### Build Docker Image

```bash
docker build -t wsdp .
```

### Run with Docker

```bash
# Run pipeline
docker run -v /data:/data -v /output:/output wsdp run /data /output widar

# Download dataset
docker run -v /data:/data wsdp download widar /data
```

## Troubleshooting

### Import Error

If you encounter `ModuleNotFoundError`, ensure you're in the correct environment:

```bash
# If using virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### CUDA Issues

For GPU support, ensure you have the correct PyTorch version:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, WSDP will fall back to CPU mode automatically.
