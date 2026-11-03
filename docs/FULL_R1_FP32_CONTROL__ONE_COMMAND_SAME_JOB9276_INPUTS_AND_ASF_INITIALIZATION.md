# FULL_R1 FP32 control — giữ ASF, kiểm tra precision trước

## Thí nghiệm này trả lời câu hỏi gì?

Job9276 hoàn tất nhưng ASF kém DBNet. Saved backbone BF16 chỉ có khoảng15% phần tử
khác lúc khởi tạo; riêng post_trunk_norm chỉ1/512 phần tử đổi. Đây là dấu hiệu
cần kiểm tra precision, không phải bằng chứng BF16 đã gây toàn bộ mức tụt điểm.

Lượt này chạy hai nhánh:

- A: DBNet không ASF, trọng số train/inference FP32.
- B: DBNet+ASF chính thức, trọng số train/inference FP32.

Giữ đúng checkpoint gốc của A và **chính file khởi tạo ASF đã dùng trong job9276**
của B. Không rút lại random initialization, không identity init, LR multiplier,
augmentation/sampling mới hoặc thêm updates. AdamW2e-5, microbatch2, effective4,
10.000 optimizer updates, final weights only giống job9276.

So A/B mới để kiểm tra ASF trong điều kiện precision này. So với job9276 là thay
đổi chế độ precision **cả training và inference**, không riêng hiệu ứng rounding
khi training. Backend TF32 flags được ghi lại; FP32 parameter storage không được
tự đồng nhất với mọi chi tiết số học trong CUDA kernel. Chưa có cam kết tăng điểm.

## Một lệnh chạy trên HPC

Giữ thư mục run job9276. Đặc biệt không xóa `inputs`, `state`, completion marker,
`run_configuration__FULL_R1_two_architectures_same_budget.json` và ASF initial
trong `inputs/06_initialized_checkpoints`. Code tham chiếu các input lớn theo kiểu
chỉ đọc để không nhân đôi manifest/GT. Giữ báo cáo cũ để có bảng đối chiếu tự động.

```bash
cd /datastore/dungmt/textvqa/Tuan/MonKeyOCR-
git pull --ff-only origin main

python run_all.py \
  --config configs/phase4_full_textvqa_dbnet_vs_asf_FP32_control_job9276_h200_normal.json \
  --dataset-parent /datastore/dungmt/textvqa/data/benchmark_v1/raw \
  --workspace /datastore/dungmt/textvqa/Tuan/monkeyocr_workspace
```

Lệnh tự submit **một job normal / mig2-35**. Không gọi riêng các bước bên dưới,
không cài môi trường mới hoặc xóa stack cũ. Stack CUDA/Python đã xác minh được tái
sử dụng nếu vẫn nguyên identity.

Run mới có tên:

```text
phase4_FULL_R1_FP32_DBNet_vs_ASF_same_job9276_initialization_10000updates_final_only_seed42_v1
```

Trên Windows có thể kiểm tra cấu hình, không đụng dữ liệu/GPU:

```bash
python run_all.py --config configs/phase4_full_textvqa_dbnet_vs_asf_FP32_control_job9276_h200_normal.json --plan-only
```

## Các kiểm tra tự động trước lượt train dài

1. Kiểm hash reference master, bốn split manifests, protocol, initial ASF và
   receipts của annotation export. Kiểm lại source metadata và từng ảnh clean/
   degraded trong21.747 source hợp lệ. Dùng nguyên geometry/targets cũ, không tạo
   ảnh suy thoái hay xuất lại annotation lớn.
2. Build A/B, chuyển parameter storage FP32 trước strict loading, kiểm shared
   values bằng checkpoint gốc và mọi giá trị B bằng frozen initial sau dtype cast.
3. Mỗi nhánh thực hiện **một optimizer update thật**: hai microiterations, đúng
   OptimWrapper, warmup LR, accumulation và clipping của cấu hình train. Kiểm
   parameter FP32, gradient hữu hạn/khác0 theo module, actual weight movement,
   Adam moment FP32 và cùng sample IDs/LR giữa A/B. Bỏ model probe sau kiểm tra,
   không ghi checkpoint probe và không dùng các weights đã update để train thật.
4. Chạy A/B initial trên cùng16 source validation (32ảnh mỗi nhánh), native mask.
   Subset gồm source có canvas sau resize/padding lớn nhất trong validation để
   kiểm tra nhu cầu bộ nhớ inference FP32 trước khi tốn thời gian train dài.
   Ghi initial Hmean và attention/feature RMS thực tế của B. Đây là subset chẩn
   đoán cố định theo metadata, không đại diện toàn tập, không phải test hoặc một phép chọn hyperparameter.

Chỉ khi cả hai probe thành công mới bắt đầu train A. Correctness checks thất bại
thì dừng sớm và giữ log; không tự thay LR/precision để lách lỗi. Low initial Hmean
hay attention saturation được ghi làm bằng chứng, không phải một quy tắc tự
chọn kiến trúc hoặc thay kế hoạch sau khi xem kết quả.

## Training và đánh giá

- Train A rồi B, mỗi nhánh10.000 optimizer updates. Không periodic/best/optimizer
  checkpoint; chỉ weights cuối, đặt tên chứa `FULL_TextVQA_FP32` và tên nhánh.
- Validation mỗi2.000 updates, scalar bước được ghi theo microiteration thay vì
  `step=0`; chia2 nếu cần đổi sang optimizer updates.
- Tự đo initial→final **named parameters** theo module, không gộp BN buffers.
- Giữ native mask0,30 và sweep0,30/0,20/0,15/0,10/0,05 trên validation. Một ngưỡng
  chung clean/degraded, tiêu chí mean Hmean như cũ. Test chỉ native và ngưỡng đã chọn.
- Ghi attention thực tế của B khi suy luận. Báo cáo initial/final dùng **đúng cùng
  16 source validation**, tránh so hai tập ảnh khác nhau. Stats tính toàn padded
  FPN map, gồm background/padding; không được diễn giải thành thống kê riêng GT chữ nhỏ.
- Inference timing có overhead hook ở B, không dùng để tuyên bố ASF overhead chuẩn.

## Output và gửi kết quả

Root run có `READ_ME_FIRST__FULL_R1_DBNet_ASF_run_outputs_and_logs.json` trỏ đến
training, validation selection, final comparison và completion marker.

Khi hoàn tất, chương trình tự tạo:

```text
RESULTS_AND_DIAGNOSTICS__FULL_R1_FP32_DBNet_ASF__final_weights_only.zip
```

ZIP nằm ngay trong root run mới. Terminal in `RESULTS_AND_DIAGNOSTICS_ZIP:` và
đường dẫn đầy đủ. Chỉ cần tải ZIP đó bằng WinSCP; không phải chạy lại lệnh collect.

Gói gồm bảng clean/degraded, size/corruption, FP/FN, source-bootstrap, validation
selection, cấu hình, initialization receipts, actual first-update probes, lịch
validation, final parameter audits và initial/final attention trên cùng subset.
Có bảng native BF16 job9276 vs FP32 mới nếu báo cáo cũ còn và khớp hash đã nhận.
Không đóng gói ảnh, checkpoint, dataset manifests hoặc raw full predictions.

Xác nhận hoàn tất bằng cả Slurm `COMPLETED 0:0`, marker và delivery ZIP. Nếu bị
ngắt, chạy cùng lệnh/config/run_name; stage hoàn tất được kiểm hash và tái dùng,
còn training chưa hoàn tất chạy lại từ frozen initialization. Không `git pull`
thay source giữa một run đang chạy/resume.

## Kiểm thử trước khi phát hành

Đã kiểm thử CPU/config/controller với fixture nhỏ: exact frozen-input reuse,
mutation detection, exact-init reuse path, preflight-before-training order,
threshold scope, completion/resume và delivery không gồm weights/data.
Đã đọc encoder đúng revision/hash: attention implementation được pin là SDPA.
Chưa có môi trường GPU local để xác nhận thực thi FP32; real-update probe và
native-P-map equivalence là runtime gates đầu job, không giả làm evidence đã chạy.
