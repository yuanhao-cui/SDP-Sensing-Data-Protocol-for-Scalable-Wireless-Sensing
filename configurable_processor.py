from functools import partial
from concurrent.futures import ProcessPoolExecutor
from wsdp.algorithms import execute_pipeline
import numpy as np

class ConfigurableProcessor:
  """支持自定义算法pipeline的Processor"""
  def __init__(self, pipeline_steps):
      """
      Args:
          pipeline_steps: dict, 例如：
              {'denoise': {'method': 'wavelet'},
               'calibrate': {'method': 'stc'}}
      """
      self.pipeline_steps = pipeline_steps

  def process(self, data_list, **kwargs):
      dataset = kwargs.get('dataset', '')
      all_data, all_labels, all_groups = [], [], []

      worker_func = partial(
          _process_single_csi_configurable,
          dataset=dataset,
          pipeline_steps=self.pipeline_steps
      )

      with ProcessPoolExecutor(max_workers=4) as executor:
          results = executor.map(worker_func, data_list)
          for csi, label, group in results:
              if csi is not None:
                  all_data.append(csi)
                  all_labels.append(label)
                  all_groups.append(group)
      return all_data, all_labels, all_groups

def _process_single_csi_configurable(csi_data, dataset, pipeline_steps):
      """支持配置算法的单文件处理"""
      from wsdp.processors.base_processor import _parse_file_info_from_filename, _selector

      res = _parse_file_info_from_filename(csi_data.file_name, dataset)
      label, group = _selector(res, dataset)

      # 构建CSI tensor
      sorted_frames = sorted(csi_data.frames, key=lambda f: f.timestamp)
      frame_tensors = [f.csi_array for f in sorted_frames]

      if not frame_tensors:
          return None, None, None

      whole_csi = np.stack(frame_tensors, axis=0)
      if whole_csi.ndim == 2:
          whole_csi = np.expand_dims(whole_csi, -1)
      if whole_csi.shape[0] < 2:
          return None, None, None

      # 使用配置的算法pipeline
      cleaned_csi = execute_pipeline(whole_csi, pipeline_steps)

      return cleaned_csi, label, group

if __name__ == '__main__':
    from wsdp import readers
    import os

    # 示例：使用 ConfigurableProcessor 处理 CSI 数据
    input_path = "data/xrf55"  # 修改为你的数据路径
    dataset_name = "xrf55"

    if not os.path.isdir(input_path):
        print(f"示例数据路径不存在: {input_path}")
        print("请将 input_path 修改为实际的数据目录后重新运行")
        exit(1)

    # 定义自定义算法 pipeline
    pipeline_steps = {
        'denoise': {'method': 'wavelet'},
        'calibrate': {'method': 'stc'},
        'normalize': {'method': 'z-score'},
    }

    print(f"正在加载数据: {input_path} ...")
    csi_data_list = readers.load_data(input_path, dataset_name)
    print(f"共加载 {len(csi_data_list)} 个 CSI 样本")

    processor = ConfigurableProcessor(pipeline_steps)
    all_data, all_labels, all_groups = processor.process(csi_data_list, dataset=dataset_name)

    print(f"处理完成:")
    print(f"  数据样本数: {len(all_data)}")
    print(f"  标签样本数: {len(all_labels)}")
    print(f"  分组样本数: {len(all_groups)}")
    if all_data:
        print(f"  单个样本形状: {all_data[0].shape}")
