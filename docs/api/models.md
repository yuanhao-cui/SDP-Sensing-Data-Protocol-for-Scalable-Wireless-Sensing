# API Reference - Models

See [Full API Reference](../API_REFERENCE.md) for complete documentation.

## CSIModel

CNN + Transformer architecture for CSI classification.

```python
from wsdp.models import CSIModel

model = CSIModel(num_classes=6, input_shape=(200, 30, 3))
```

## Custom Models

Create `custom_model.py`:
```python
class YourModel(nn.Module):
    def forward(self, x):
        # x: (Batch, Timestamp, Frequency, Antenna)
        return output

model = YourModel
```
