# MonkeyOCRv2 Text Detection Phase 3

Một lệnh chạy trọn bộ thí nghiệm trên HPC UIT: checkpoint gốc, suy luận ảnh toàn cảnh kết hợp ô 2×2, fine-tune clean (E1), fine-tune clean + degraded (E2), rồi xuất bảng so sánh trên cùng nguồn ảnh test. Người phụ trách: Tuấn.

```bash
python run_all.py
```

Nếu chạy lệnh ở login node có Slurm, chương trình **tự submit đúng một job `normal`** và in job ID cùng đường dẫn logs. Khi job được cấp GPU, toàn bộ các bước bên dưới chạy tự động trong allocation đó. Nếu đã ở trong Slurm GPU allocation, chương trình chạy trực tiếp. Không cần gọi riêng setup, audit, inference, train hay evaluation.

**Trạng thái bàn giao:** đã kiểm thử CPU, geometry, split, resume và controller bằng GPU workers giả trong test tạm. Chưa chạy training/inference Phase 3 thật trên H200. Không có kết quả cải thiện được tạo sẵn hoặc bảo đảm trước.

## Lấy code và khai báo dữ liệu

```bash
git clone https://github.com/ncat218/MonKeyOCR-.git
cd MonKeyOCR-
```

Đã điền đường dẫn server Tuấn cung cấp trong `run_all.py`:

```python
DATASET_PARENT = "/datastore/dungmt/textvqa/data/benchmark_v1/raw/"
```

Đây là **thư mục cha**, không phải đường dẫn gốc trên máy Windows hoặc mount `/kaggle/input`. Cấu hình mặc định khoanh vùng đúng ba thư mục dưới đây, rồi tìm file nhận dạng bên trong, kể cả khi file còn nằm trong nhiều cấp thư mục sau giải nén:

| Thư mục con trên server | Dataset | File nhận dạng |
|---|---|---|
| `textvqa_clean` | TextVQA clean | `TextVQA_0.5.1_train.json` và ảnh nguồn |
| `textocr_0_1_train_json` | TextOCR ground truth | `TextOCR_0.1_train.json` |
| `textvqa_degraded_10k` | Bộ suy thoái 10K đã tạo | `main/image_manifest.csv`, `config/frozen_degradation_config.json`, `metadata/locked_eval_manifest.csv`, ảnh MAIN |

Thư mục `textvqa_degraded` bị loại khỏi tìm kiếm. Nếu thiếu bản `textvqa_degraded_10k`, code dừng thay vì lấy bộ cũ để chạy tiếp. Việc này được kiểm thử riêng. Không cần điền lại đường dẫn khi clone repo về server đúng layout đã cung cấp.

Ví dụ cấu trúc bên trong (các cấp giải nén chưa được kiểm tra trực tiếp trên server):

```text
/datastore/dungmt/textvqa/data/benchmark_v1/raw/
├── textvqa_clean/
│   └── textvqa/
│       ├── TextVQA_0.5.1_train.json
│       └── train_images/...
├── textocr_0_1_train_json/
│   └── TextOCR_0.1_train.json
└── textvqa_degraded_10k/
    └── textvqa_realistic_v2_10k/
        ├── main/image_manifest.csv
        ├── main/images/realistic_mix/L2/...
        ├── config/frozen_degradation_config.json
        └── metadata/...
```

Ảnh clean mặc định nằm dưới thư mục chứa TextVQA JSON; degraded root được suy từ `main/image_manifest.csv`. Nếu server tách JSON và ảnh sang các ổ khác nhau, dùng trường đường dẫn tương ứng trong `configs/phase3_h200_normal.json`. Không thay ID hay sửa manifest nguồn để vá đường dẫn.

Nếu sau này chuyển dữ liệu, đổi dòng `DATASET_PARENT`; nếu đổi tên các thư mục con, sửa `data.dataset_subdirectories` trong config. Cũng có thể dùng biến `TEXTDET_DATA_PARENT` hoặc đối số `--dataset-parent`; **không bắt buộc** dùng các cách này. Nếu có nhiều bản dataset trong cùng vùng đã chọn, code liệt kê candidates và dừng; không tự chọn bản đầu tiên. Code kiểm tra hash ảnh và metadata khi ánh xạ đường dẫn cũ sang server. Chỉ khi không có parent được khai báo qua bất kỳ cách nào thì code thử `/datastore/<username>` và `/datastore/shared`.

Repo **không tải hoặc tạo lại ba dataset**. Người dùng đưa chúng lên server trước. Không upload ảnh, annotation JSON gốc, token Kaggle, checkpoints hoặc kết quả thí nghiệm vào repo public.

## Một job H200 normal cho toàn bộ suite

Runner tìm toolkit qua `environment.cuda_home`, các biến `CUDA_HOME`/`CUDA_PATH`/`CONDA_PREFIX`, `PATH` và những vị trí CUDA/HPC thông dụng. Nếu không tìm được toolkit 12.x phù hợp, mặc định dùng Conda có sẵn để cài **`cuda-toolkit=12.8.1`** từ `nvidia/label/cuda-12.8.1` vào `<workspace>/environment_and_models__phase1_verified_stack/cuda_toolkit_12_8_1__private_nvidia_conda_environment`. Không cài vào Conda base, không thay driver và không cần sudo. Cache Conda cũng nằm trong datastore. Có thể tắt bằng `environment.auto_install_private_cuda128_if_missing=false`.

Lần đầu bổ sung toolkit cần thêm thời gian tải, dung lượng datastore và truy cập kênh NVIDIA/conda-forge. Runner compile một file CUDA nhỏ để kiểm tra toolkit/host compiler trước khi tải stack lớn, rồi vẫn kiểm tra CUDA ops forward/backward. Dòng CUDA Version của `nvidia-smi` chỉ phản ánh driver; không dùng nó để chọn toolkit 13.x hay đổi Torch cu128 sang cu130.

Lệnh khởi động cần Python ≥3.8; môi trường chạy model Python 3.11 được tạo tự động bên trong job.

Mặc định của `configs/phase3_h200_normal.json`:

| Tham số Slurm | Giá trị |
|---|---|
| Account / partition | `uit` / `normal` |
| QoS | `mig2-35` theo danh sách quyền tài khoản server đã cung cấp |
| GPU MIG | `gpu:2g.35gb:1` |
| CPU / RAM hệ thống | 8 CPU / 40 GB |
| Thời gian yêu cầu | 68 giờ, dưới giới hạn 72 giờ/job trong hướng dẫn |

Đây là **một MIG instance H200**, không phải H200 nguyên chiếc và không phải hai GPU để DDP. Cả suite dùng cùng allocation `normal`, kể cả những bước inference vốn có thể vừa `small`. Không đổi partition giữa các thí nghiệm và không cần giữ nhiều lượt truy cập H200.

Code tự tạo Python 3.11 venv, tải nguồn/model đã pin, build MMCV cho compute capability GPU đang được cấp, chạy CUDA forward/backward và kiểm tra checkpoint. Môi trường, HF cache, source, dataset snapshots và outputs nằm trong datastore workspace; không đặt khối dữ liệu lớn ở Home.

Cần Internet ở compute node trong lần setup đầu hoặc một environment/cache đã chuẩn bị tương ứng; cần `git`, compiler C++ và driver tương thích PyTorch cu128. Toolkit `nvcc` CUDA 12.x (ưu tiên 12.8) có thể lấy từ hệ thống/module đã load hoặc từ phương án cài riêng tự động bên trên. Nếu tắt fallback hoặc không có Conda, cung cấp toolkit qua module HPC hoặc `environment.cuda_home`. Không thể suy ra `nvcc` từ dòng CUDA Version của `nvidia-smi`. Khác minor CUDA 12.x được ghi rõ là môi trường adapted và phải qua phép thử CUDA; không tự dùng toolkit 13.x hay cài driver.

Workspace mặc định là `/datastore/<username>/_workspace_monkeyocr_text_detection_phase3` nếu datastore cá nhân tồn tại; nếu không thì là `_workspace_monkeyocr_text_detection_phase3` cạnh repo. Có thể đặt `workspace_root` trước lượt đầu. Không đặt raw datasets trong workspace/cache do chương trình quản lý.

Không có benchmark H200 Phase 3 để dự đoán thời gian hoặc bảo đảm batch size 2 vừa VRAM. Probe training thực hiện **trước inference dài** để bắt lỗi forward/backward/optimizer. Nếu không đủ VRAM, giảm micro batch trong **một run_name mới**, giữ effective batch bằng accumulation và áp dụng giống nhau cho E1/E2. Không tự giảm cấu hình của chỉ một thí nghiệm giữa chừng.

## Thí nghiệm chạy tự động

1. Tìm đủ ba input, xác minh metadata/source hashes và dựng lại master manifest bằng code công khai Phase 2.
2. Mặc định kiểm tra SHA256 thực tế của manifest 9.898 cặp: chấp nhận bản Phase 2 gốc hoặc bản số thực HPC đã được đối chiếu toàn bộ và đăng ký hash riêng. Audit dữ liệu, protocol và evaluator vẫn được khóa; manifest chưa được duyệt sẽ dừng. Không dùng phép so sánh gần đúng để tự chấp nhận manifest lạ. Xem [đối chiếu job 7845](docs/MANIFEST_IDENTITY_REVIEW_JOB7845.md).
3. Freeze split theo source ID **và SHA256 ảnh nguồn**: 70% train, 15% validation, phần còn lại test, seed 42. Không để hai bản của cùng nguồn rơi vào hai split. Mọi condition dùng chung split.
4. Xuất training targets và exact validation GT. Chạy probe loss/backward/optimizer trên một minibatch, không ghi đè checkpoint gốc.
5. Chạy E0 global trên validation và test. Đây là baseline trên cùng H200/pipeline và cùng test của Phase 3.
6. Chạy global + bốn ô 2×2 trên tối đa 200 nguồn lấy cố định từ **validation**. Overlap được định nghĩa bằng 20% chiều của ô; inference từng ô tuần tự. Loại ứng viên chạm đường cắt nội bộ và dùng polygon NMS để xử lý trùng.
7. Chọn NMS IoU trong `[0.3, 0.5]` bằng trung bình Hmean clean/degraded trên development subset. Test không tham gia chọn tham số. Tái sử dụng global predictions đã tính để tránh thêm forward toàn ảnh.
8. Chạy phương án 2×2 đã cố định trên test, chấm cùng protocol.
9. E1 fine-tune clean từ checkpoint gốc; chọn checkpoint theo validation, chấm clean và degraded test bằng inference toàn ảnh.
10. E2 fine-tune clean + degraded từ **cùng checkpoint gốc**, không tiếp tục E1; chọn và chấm tương tự.
11. Xuất Markdown, CSV và JSON so sánh E0 global, E0+2×2, E1 global, E2 global; kèm bootstrap chênh lệch với E0 theo cùng source test.

**Chưa triển khai E3**, restoration, recognition hoặc Qwen. Không tự kết hợp tiling vào E1/E2 để tránh gộp hai thay đổi trong đối chứng đầu tiên. Tiling không đọc GT để quyết định cắt ảnh hay chọn vùng khi inference.

## Ngân sách E1 và E2

| Thiết lập | Mặc định chung |
|---|---|
| Khởi tạo | Checkpoint DBNet + MonkeyOCRv2-AS đã xác minh Phase 1 |
| Optimizer | AdamW, learning rate `2e-5`, weight decay `1e-4` |
| Micro / effective batch | 2 / 4, accumulation 2 |
| Optimizer updates | 3.000 mỗi thí nghiệm = 6.000 micro iterations = 12.000 ảnh đầu vào |
| Validation | Mỗi 1.000 optimizer updates, cả clean/degraded |
| Checkpoint resume | Mỗi 250 optimizer updates và cuối lượt |
| Chọn model tốt nhất | `0.5 × (Hmean validation clean + Hmean validation degraded)` |
| Augmentation / model | Pipeline training chính thức dùng giống nhau; không thay backbone/head/postprocessing |

E2 có số entries clean/degraded bằng nhau trong dataset và shuffled InfiniteSampler. Không ép từng minibatch thành một cặp; tỷ lệ ở một prefix training hữu hạn có thể xê dịch quanh 1:1. E1 và E2 có cùng pool source IDs, tổng ảnh xử lý và optimizer-update budget; E1 được xem nhiều clean hơn, E2 phân bổ lượt xem cho hai condition. Model weights được cập nhật toàn bộ; không dùng AMP wrapper mới, giữ BF16 backbone của cấu hình chính thức.

Training format của MMOCR chỉ nhận polygon đơn. Với GT valid bị phân mảnh, training dùng convex hull và ghi số lượng trong conversion audit. Ignored polygons và vùng ngoài ROI được phân rã mà không lấp lỗ vùng hợp lệ. **Evaluation luôn dùng exact polygon/MultiPolygon và ROI của manifest Phase 2**, không dùng bản training xấp xỉ để chấm.

## Input output và logs có tên rõ ràng

Trong `<workspace>/<run_name>/`:

```text
inputs/
  01_dataset_discovery__actual_server_locations.json
  02_shared_master_manifest__textvqa_clean_degraded_textocr_word_gt/
  03_frozen_source_id_splits__train70_validation15_test15/
  04_mmocr_training_annotations__E1_clean_E2_clean_degraded/
  05_model_and_training_configurations__pinned_phase1_detector/
outputs/
  E0_original_checkpoint__global__manifest__test__.../
  E0_original_checkpoint__global_plus_2x2__manifest__test__.../
  E1_clean_only__training_from_original_checkpoint/
  E2_clean_and_degraded_balanced__training_from_original_checkpoint/
  E1_clean_only__global__manifest__test__.../
  E2_clean_and_degraded_balanced__global__manifest__test__.../
  trained_models/
    detector_weights__E1_clean_only__best_balanced_validation_hmean__seed42.pth
    detector_weights__E2_clean_and_degraded_balanced__best_balanced_validation_hmean__seed42.pth
  metrics/<experiment>__test/
  FINAL_COMPARISON__E0_global_E0_2x2_E1_clean_E2_clean_degraded/
logs/
  slurm_all_experiments/<UTC>__all_phase3_experiments__job_<ID>.stdout.log
  slurm_all_experiments/<UTC>__all_phase3_experiments__job_<ID>.stderr.log
  allocation_<JOB_ID>__<UTC>/experiment_workers/<UTC>__<experiment_and_stage>.log
state/
  <stage_name>__stage_status.json
COMPLETE__all_phase3_experiments_finished.json
```

Files trong thư mục metric giữ tên chuẩn `metrics.json`, `per_image.jsonl`, `paired_source_metrics.csv`; thư mục cha định danh rõ thí nghiệm và split. Logs nội bộ MMEngine có timestamp riêng trong thư mục training E1/E2, ngoài log subprocess có tên thí nghiệm đầy đủ.

Bảng cuối: `RESULTS__phase3_dbnet_monkeyocrv2_as__global_2x2_E1_E2.md` và `comparison__all_experiments__precision_recall_hmean_small_text_recall.csv`. JSON chứa metric đầy đủ, clean/degraded gap CI và CI cải thiện Hmean từng phương pháp so với E0. Tiled latency ghi rõ khi cộng thời gian global đã cache với thời gian các ô mới đo.

## Resume và gián đoạn

**Khôi phục sau job 7845:** phiên bản này đổi run name mặc định từ `seed42_v1` sang `seed42_v2`, vì code kiểm tra manifest đã thay đổi. Sau khi job cũ kết thúc, `git pull --ff-only` rồi chạy lại lệnh cũ với cùng `--workspace` và `--dataset-parent`. Log/input của v1 được giữ nguyên; cache môi trường/model hợp lệ trong cùng workspace được tái sử dụng. Bản sửa không chỉnh GT hoặc tạo lại ảnh. File `phase2_manifest_identity_verification.json` ghi rõ bản manifest nào được chấp nhận. Khi gửi cho Paddle, dùng đúng manifest đã freeze của lượt v2.

Trước khi submit, runner chỉ kiểm tra vị trí các file metadata trong ba thư mục đã chọn. Nếu thiếu metadata của `textvqa_degraded_10k`, nó dừng ngay tại login node và không gửi job GPU. Chưa load annotation JSON, hash toàn ảnh hay chạy model ở bước kiểm tra tên file này. Audit nội dung và toàn bộ tính toán vẫn ở trong allocation.

Chạy lại **cùng lệnh** `python run_all.py`. Nếu ở login node, nó submit một allocation mới; tuân thủ quy định tần suất submit của HPC. Không có vòng lặp tự submit vô hạn. Trong một run_name, chương trình:

- Kiểm tra hashes và bỏ qua stage đã hoàn tất; không tự tin tưởng file marker mà bỏ qua kiểm tra artifacts.
- Giữ từng prediction đã ghi. Chỉ sửa fragment JSONL cuối bị ngắt, có giữ lại fragment làm bằng chứng; lỗi giữa file sẽ dừng.
- Resume training từ checkpoint gồm optimizer, scheduler, RNG và iteration. SIGUSR1/SIGTERM yêu cầu checkpoint ở biên accumulation khi còn thời gian; hard kill vẫn dựa vào checkpoint định kỳ.
- Không chạy đồng thời hai controller cùng run_name. Không trộn cấu hình, code hay model khác vào run cũ.

Thay budget/split/logic thí nghiệm cần đặt `run_name` mới. Không đổi code giữa một lượt đang chạy. Việc sửa đường dẫn dữ liệu nên hoàn tất trước lần thực thi đầu; không coi một bộ outputs đã di chuyển tùy ý là tự động resumable.

## Kiểm tra CPU và phạm vi kết luận

Không cần GPU để xem kế hoạch:

```bash
python run_all.py --plan-only
```

Kiểm thử cho người phát triển/CI (không phải các bước người chạy phải gọi):

```bash
python -m pip install -r requirements-controller.txt
python -m unittest discover -s tests -v
```

CI còn cài `requirements-config-tests.txt` và `mmengine==0.10.7 --no-deps` để kiểm tra việc sinh/đọc lại cấu hình bằng MMEngine thật mà không cần Torch. Nếu không cài nhóm tùy chọn này, riêng phép thử Config sẽ được skip; các phép thử CPU còn lại vẫn chạy.

Tests dùng vài ảnh fixture rất nhỏ trong temporary directories; không tái tạo bộ suy thoái, không tải model và không báo metric giả làm kết quả nghiên cứu. Có end-to-end controller test bằng fake GPU workers để kiểm tra các stage E0/tiling/E1/E2, khởi tạo độc lập và resume; test này không xác nhận CUDA training.

Metric là polygon IoU ≥0,5 theo protocol external TextOCR của đồ án, không phải official ICDAR2015 Hmean. Dữ liệu là synthetic realistic_mix L2, severity_basis image_relative_fallback. TextOCR train có trong nguồn pretraining Monkey; không tuyên bố unseen. Kết quả toàn master đã được xem ở Phase 2; test Phase 3 là held out khỏi fine-tuning/chọn tham số, không phải một bộ test chưa từng phân tích trong lịch sử.

## Nguồn

- [MonkeyOCRv2 chính thức](https://github.com/Yuliang-Liu/MonkeyOCRv2): source/model revisions và checkpoint hashes khóa trong `configs/verified_phase1_phase2_sources.json`.
- [MMOCR v1.0.1](https://github.com/open-mmlab/mmocr/tree/v1.0.1), [MMEngine v0.10.7](https://github.com/open-mmlab/mmengine/tree/v0.10.7).
- [Repo tham khảo cách tổ chức](https://github.com/ncat218/textvqa-qwen-baseline): tham khảo tách config/code/tests và lưu metadata; không sử dụng Qwen, dữ liệu 1.000 câu hỏi hay evaluator VQA của repo đó.
- Manifest/evaluator công khai lấy nguyên bytes từ implementation Phase 2 đã kiểm chứng của Tuấn; pinned evaluator SHA256 `ea25c8d9fc240ec10dc6eb45883b163c8e92952ee885423985e6d013f026fcdc`.
