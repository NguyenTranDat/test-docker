# Model analyzer

- Tìm cấu hình tốt nhất cho công việc và phần cứng dự kiến 
- Tóm tắt các phát hiện về độ trễ, thông lượng, mức sử dụng tài nguyên GPU, mức tiêu thụ điện năng và nhiều thông tin khác, với các báo cáo, số liệu và biểu đồ chi tiết. Các báo cáo này giúp so sánh hiệu suất giữa các cấu hình thiết lập khác nhau.
- Triển khai mô hình tùy chỉnh để đáp ứng các yêu cầu về Chất lượng dịch vụ của người dùng như giới hạn độ trễ p99 cụ thể, mức sử dụng bộ nhớ GPU và thông lượng tối thiểu!

## Các lệnh quan trọng
- những lệnh này có thể được tạo bằng flag khi chạy cli hoặc tạo trong file .yaml

### objectives

- sắp xếp kết quả dựa trên mục tiêu triển khai, thông lượng, độ trễ hoặc tùy chỉnh theo các hạn chế về tài nguyên cụ thể

| **Tên tùy chọn**              | **Sự miêu tả**                                                                    |
|-------------------------------|------------------------------------------------------------------------------------|
| `perf_throughput`              | Sử dụng thông lượng làm mục tiêu.                                                  |
| `perf_latency_p99`             | Sử dụng độ trễ làm mục tiêu.                                                       |
| `gpu_used_memory`              | Sử dụng bộ nhớ GPU được mô hình sử dụng làm mục tiêu.                              |
| `gpu_free_memory`              | Sử dụng bộ nhớ GPU không được mô hình sử dụng làm mục tiêu.                        |
| `gpu_utilization`              | Sử dụng mức sử dụng GPU làm mục tiêu.                                              |
| `cpu_used_ram`                 | Sử dụng RAM được mô hình sử dụng làm mục tiêu.                                     |
| `cpu_free_ram`                 | Sử dụng RAM không được mô hình sử dụng làm mục tiêu.                               |
| `output_token_throughput`      | Sử dụng thông lượng mã thông báo đầu ra làm mục tiêu.                              |
| `inter_token_latency_p99`      | Sử dụng độ trễ giữa các mã thông báo làm mục tiêu.                                 |
| `time_to_first_token_p99`      | Sử dụng thời gian để độ trễ của mã thông báo đầu tiên làm mục tiêu.                |

```
objectives:
  - perf_latency_p99
  - perf_throughput
```

### constraints
- Người dùng cũng có thể chọn giới hạn việc lựa chọn quét theo các yêu cầu cụ thể về thông lượng, độ trễ hoặc sử dụng bộ nhớ GPU

|**Tên tùy chọn**         |**Mô tả**               |
|-------------------------|--------------------|
|`perf_throughput`        |throughput tối thiểu|
|`perf_latency_p99`       |latency tối đa có thể chấp nhận được|
|`output_token_throughput`|Chỉ định thông lượng mã thông báo đầu ra mong muốn tối thiểu|
|`inter_token_latency_p99`|Chỉ định độ trễ giữa các mã thông báo tối đa có thể chấp nhận được|
|`time_to_first_token_p99`|Chỉ định thời gian tối đa có thể chấp nhận được cho độ trễ của mã thông báo đầu tiên|
|`gpu_used_memory`        |bộ nhớ GPU tối đa mô hình sử dụng|

```
constraints:
	gpu_used_memory:
		max: 200
	perf_latency_p99:
		max: 100
```

## các lệnh chính

### profile
- dùng để thử nghiệm các lần chạy. Khởi tạo các tham số chi tiết như số lượng phiên bản, max_batch_size, ...
- ghi lại hiệu suất của từng cấu hình và lưu các lần chạy là điểm kiểm tra

```
model-analyzer profile --output-model-repository-path /path/to/output -f <path to config file>
```

### report
- Biểu đồ phác thảo thông lượng và độ trễ trên số lượng yêu cầu đồng thời ngày càng tăng được gửi đến máy chủ (chi tiết)
- Biểu đồ Bộ nhớ GPU so với Độ trễ và Mức sử dụng GPU so với Độ trễ (chi tiết)
- Bảng phác thảo độ trễ p99, các thành phần khác nhau của độ trễ, thông lượng, Sử dụng GPU và sử dụng bộ nhớ GPU cho tối đa số lượng yêu cầu đồng thời được chọn trong bước lập hồ sơ (mặc định là 1024) (chi tiết)
- Biểu đồ thông lượng so với độ trễ, biểu đồ bộ nhớ GPU so với độ trễ và bảng chứa thông tin chi tiết cấp cao so sánh thông tin này giữa các cấu hình hàng đầu và cấu hình mặc định do người dùng chọn. (tóm tắt)

```
model-analyzer report --report-model-configs text_recognition_config_4,text_recognition_config_5,text_recognition_config_6 --export-path /workspace --config-file perf.yaml
```

## config
- Sử dụng tệp cấu hình yaml hoặc cli

| **Tham số**                                        | **Mô tả**                       |
|----------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `model_repository: <string>`                                 | Đường dẫn đến repository chứa các mô hình Triton|
| `profile_models: <comma-delimited-string-list>`                                   | Danh sách các mô hình cần được phân tích hiệu năng (cách nhau bằng dấu phẩy)|
| `cpu_only_composing_models <comma-delimited-string-list>`                        | Danh sách các mô hình thành phần chỉ sử dụng CPU|
| `[override_output_model_repository]: <boolean> \| default: false`               | Cho phép ghi đè nội dung của repository đầu ra của mô hình|
| `[concurrency]: <comma-delimited-string \|list \|range>`                                    | Các giá trị concurrency sử dụng |
| `[batch_sizes]: <comma-delimited-string \|list \|range> \| default: 1`                                    | Các giá trị batch size sử dụng|
| `[collect_cpu_metrics]: <bool> \| default: false`                            | Thu thập số liệu CPU |
| `[client_protocol]: <string> \| default: grpc`                                | Giao thức sử dụng để giao tiếp với Triton Inference Server (chỉ cho phép 'http' và 'grpc') |
| `[perf_output]: <bool> \| default: false`                                    | Bật hoặc tắt việc ghi kết quả từ perf_analyzer ra file hoặc stdout |
| `[perf_output_path]: <str>`                               | Đường dẫn để ghi kết quả perf_analyzer nếu tùy chọn perf_output được bật |
| `[perf_analyzer_max_auto_adjusts]: <int> \| default: 10`                 | Số lần tối đa perf_analyzer tự điều chỉnh để phân tích hiệu năng mô hình |
| `[triton_docker_image]: <string> \| default: nvcr.io/nvidia/tritonserver:24.09-py3`           | Tag của hình ảnh Docker Triton sử dụng khi khởi chạy trong chế độ Docker |
| `[triton_docker_mounts]: <list of strings>`                           | Danh sách các đường dẫn được gắn kết vào container Docker của Triton |
| `[triton_docker_shm_size]: <string>`                         | Kích thước /dev/shm cho container Docker của Triton |
| `[triton_launch_mode]: <string> \| default: 'local'`                             | Chế độ khởi chạy Triton: "docker", "local", "remote" hoặc "c_api" |
| `[gpus]: <string \|comma-delimited-list-string> \| default: 'all'` | Danh sách GPU UUID để sử dụng cho việc phân tích hiệu năng|
| `[run_config_search_mode]: <string> \| default: brute` | Chế độ tìm kiếm cấu hình: "brute", "quick", hoặc "optuna"|
| `[run_config_search_min_concurrency]: <int> \| default: 1 `              | Concurrency tối thiểu sử dụng trong quá trình tìm kiếm cấu hình|
| `[run_config_search_max_concurrency]: <int> \| default: 1024`              | Concurrency tối đa sử dụng trong quá trình tìm kiếm cấu hình|
| `[run_config_search_min_model_batch_size]: <int> \| default: 1`         | Batch size tối thiểu cho mô hình sử dụng trong quá trình tìm kiếm cấu hình|
| `[run_config_search_max_model_batch_size]: <int> \| default: 128`         | Batch size tối đa cho mô hình sử dụng trong quá trình tìm kiếm cấu hình|
| `[run_config_search_min_instance_count]: <int> \| default: 1`           | Số lượng instance tối thiểu trong quá trình tìm kiếm cấu hình  |
| `[run_config_search_max_instance_count]: <int> \| default: 5`           | Số lượng instance tối đa trong quá trình tìm kiếm cấu hình  |
| `[run_config_search_max_binary_search_steps]: <int> \| default: 5`      | Số bước tối đa trong quá trình tìm kiếm nhị phân  |
| `[run_config_search_disable]: <bool> \| default: false`                      | Vô hiệu hóa việc tự động tìm kiếm cấu hình  |
| `[run_config_profile_models_concurrently_enable]: <bool> \| default: false`  | Cho phép phân tích hiệu năng đồng thời tất cả các mô hình cung cấp |
| `[min_percentage_of_search_space]: <int> \| default: 5`                 | Tỷ lệ phần trăm tối thiểu của không gian tìm kiếm khi sử dụng Optuna |
| `[always_report_gpu_metrics]: <bool> \| default: false`                      | Luôn báo cáo các số liệu GPU, ngay cả khi mô hình chỉ sử dụng CPU |
| `[skip_summary_reports]: <bool> \| default: false`                           | Bỏ qua việc tạo báo cáo tóm tắt  |
| `[skip_detailed_reports]: <bool> \| default: false`                          | Bỏ qua việc tạo báo cáo chi tiết  |
| `[model_type]: <string> \| default: generic`                                     | Loại mô hình được phân tích: generic hoặc LLM |
| `[filename_model_inference]: <string> \| default: metrics-model-inference.csv`                       | Tên file chứa kết quả suy luận mô hình |
| `[filename_model_gpu]: <string> \| default: metrics-model-gpu.csv`                             | Tên file chứa kết quả số liệu GPU của mô hình |
| `[filename_server_only]: <string> \| default: metrics-server-only.csv`                           | Tên file chứa kết quả số liệu chỉ của server |
| `[inference_output_fields]: <comma-delimited-string-list>`| Các trường cần thiết cho bảng số liệu suy luận mô hình |
| `[gpu_output_fields]: <comma-delimited-string-list>`| Các trường cần thiết cho bảng số liệu GPU |
| `[server_only_output_fields]: <comma-delimited-string-list>`                      | Các trường cần thiết cho bảng số liệu chỉ của server |

- một số cấu hình chỉ cấu hình trong file .yaml

| **Tham số**                                        | **Mô tả**                                                                                         |
|----------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `profile_models: <comma-delimited-string-list|list|profile_model>` | Danh sách các mô hình cần phân tích hiệu năng (dạng chuỗi phân tách bằng dấu phẩy hoặc danh sách)  |
| `cpu_only_composing_models: <comma-delimited-string-list>`      | Danh sách các mô hình thành phần chỉ phân tích hiệu năng bằng CPU                                  |
| `[constraints: <constraint>]`                                  | Danh sách các giới hạn được đặt lên kết quả tìm kiếm cấu hình                                      |
| `[objectives: <objective \|list>]`                               | Danh sách các mục tiêu mà người dùng muốn sắp xếp kết quả tìm kiếm                                  |
| `[weighting: <int>]`                                           | Trọng số được sử dụng để ưu tiên các mục tiêu của mô hình (so với các mô hình khác) trong chế độ đa mô hình đồng thời |
| `[triton_server_flags: <dict>]`                                | Các cờ tùy chỉnh để truyền cho các phiên bản Triton được khởi chạy bởi Model Analyzer              |
| `[perf_analyzer_flags: <dict>]`                                | Các cờ tùy chỉnh để cấu hình perf_analyzer sử dụng bởi Model Analyzer                              |
| `[genai_perf_flags: <dict>]`                                   | Các cờ tùy chỉnh để cấu hình GenAI-perf sử dụng bởi Model Analyzer                                 |
| `[triton_server_environment: <dict>]`                          | Các biến môi trường tùy chỉnh cho các phiên bản Tritonserver được khởi chạy bởi Model Analyzer      |
| `[triton_docker_labels: <dict>]`                               | Danh sách các cặp name=value chứa metadata cho container Docker của Triton được kh

### parameter
- cấu hình cho từng mô hình hoặc cấu hình toàn cục
`concurrency`, `request_rate`, `batch_sizes`

example:
```
profile_models:
  model_1:
    parameters:
      concurrency:
        start: 2
        stop: 64
        step: 8
      batch_sizes: 1,2,3
```

### model-config-parameters
- chỉ được cấu hình cho mỗi mô hình, không được cấu hình toàn cục
`dynamic_batching`, `instance_group`, `max_batch_size`
example:
```
model_config_parameters:
  max_batch_size: [6, 8]
  dynamic_batching:
    max_queue_delay_microseconds: [200, 300]
  instance_group:
    - kind: KIND_GPU
      count: [1, 2]
```

### triton-server-flags
```
triton_server_flags:
  strict_model_config: False
  log_verbose: True
```

### report-model-config
```
report_model_configs:
  model_config_default:
    plots:
      throughput_v_latency:
        title: Title
        x_axis: perf_latency_p99
        y_axis: perf_throughput
        monotonic: True
  model_config_0:
    plots:
      gpu_mem_v_latency:
        title: Title
        x_axis: perf_latency_p99
        y_axis: gpu_used_memory
        monotonic: False
```

# OpenVINO
- tối ưu hóa ô hình chạy trên CPU
- bọc mô hình bằng openvino runtime

#  Polygraphy
- giống như Perf Analyzer
- Chuyển đổi các mô hình sang nhiều định dạng khác nhau
- Xem thông tin về nhiều loại mô hình khác nhau

# model dali
- ý tưởng: bỏ luôn tiền xử lý data vào model
- vì data sau khi bị giải nén(như ảnh) thì sẽ nặng hơn data chưa giải nén 
