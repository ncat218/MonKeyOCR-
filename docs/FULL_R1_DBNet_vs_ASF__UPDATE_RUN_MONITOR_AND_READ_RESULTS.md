# Full TextVQA: đối chứng DBNet / DBNet+ASF

Tuấn chốt ngày 03/11/2026: dùng giao có GT trong **official TextVQA train**, chia detection train/validation/test **70/15/15 theo source và SHA256 clean**. Hai model cùng MonkeyOCRv2-AS, cùng mixed clean/degraded entries, cùng optimizer và budget. Đây là nhánh DBNet+ASF theo hướng DBNet++, không phải reproduce toàn bộ DBNet++ với backbone của paper.

## Cập nhật và chạy trên HPC

Chạy ở terminal SSH; entrypoint sẽ tự submit **một job normal / mig2-35**. Không chạy riêng training trên login node.

```bash
cd /datastore/dungmt/textvqa/Tuan/MonKeyOCR-
git pull --ff-only origin main

python run_all.py \
  --config configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json \
  --dataset-parent /datastore/dungmt/textvqa/data/benchmark_v1/raw \
  --workspace /datastore/dungmt/textvqa/Tuan/monkeyocr_workspace
```

**Phải có `--config` mới.** Lệnh không có `--config` vẫn chọn suite Phase3 cũ để giữ tương thích. Không sửa/xóa run7957 hoặc đổi tên nó thành run mới. Nếu gián đoạn, chạy lại đúng lệnh và đúng run_name để resume. Không submit thêm khi job cùng run còn pending/running.

Không cần cài package vào base Conda hoặc tải lại checkpoint trước. Bootstrap giữ nguyên, tái dùng private Python/CUDA/MMOCR/checkpoints đã xác minh ở workspace. Nếu cache không có, luồng setup cũ sẽ tạo nó trong allocation. Đây chưa phải một kết quả chạy ASF trên H200; compatibility forward/backward thật sẽ được kiểm tra trong job.

Xem cấu hình mà không submit hoặc đọc dataset:

```bash
python run_all.py --config configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json --plan-only
```

## Đúng dữ liệu nào?

Các thư mục được tìm bên dưới dataset parent:

| Nguồn | Thư mục nhận dạng |
|---|---|
| TextVQA train JSON và ảnh clean | `textvqa_clean` |
| TextOCR train word polygons | `textocr_0_1_train_json` |
| MAIN degraded phiên bản mới | `full_textvqa_degraded` |

Log Tuấn cung cấp xác nhận MAIN có 28.408 nguồn: train21.953, val3.166, test3.289, một realistic_mix/L2 mỗi nguồn. **Đây không phải số cặp có GT dùng train/test detection.** Adapter đếm và ghi lại giao thực tế, các nguồn thiếu GT, GT không usable và ảnh ngoài phạm vi.

Schema mới là `TEXTVQA_REALISTIC_DEGRADATION_V2_FULL_R1`. `metadata/locked_eval_manifest.csv` là QA index (45.336 câu hỏi), có nhiều câu hỏi/image; không dùng số dòng này như số ảnh hay word GT. TextOCR là GT duy nhất cho định vị chữ. Official val/test, calibration và ablation không vào benchmark theo quyết định hiện tại. Không lấy các nhãn review `simulated_random` làm human legibility/answerability labels.

Trước submit, code kiểm tra file, frozen protocol/config hashes, MAIN/QA IDs và official split/count consistency. Trong allocation, code kiểm tra SHA256 từng cặp ảnh có GT, pixel dimensions, geometry/homography/common FOV, rồi khóa manifest. Nếu metadata không khớp thì dừng và báo lỗi, không tự chuyển sang bộ10K cũ, không tạo lại ảnh, không nới kiểm tra identity để chạy tiếp.

Nếu `locked_eval_manifest.csv` chỉ là bản serialize khác của QA gốc, adapter tìm CSV dưới `metadata` khớp `frozen.source_manifest_sha256`, rồi kiểm tra cùng split/question/image keys. Nếu không có, cần cung cấp metadata gốc đúng hash; không thay frozen hash bằng một hash mới.

## Hai lần train, bốn hàng đánh giá

| Nhánh | Training | Hậu xử lý test |
|---|---|---|
| A0 | DBNet noASF, mixed | Native mask0.30 |
| A1 | Cùng checkpoint A0 | Mask chọn từ validation |
| B0 | DBNet+ASF, mixed | Native mask0.30 |
| B1 | Cùng checkpoint B0 | Mask chọn từ validation |

Hai model đều bắt đầu từ **original detector**, không lấy E2 cũ rồi chỉ bật ASF hoặc train thêm riêng nhánh ASF. Dùng `FPNC.asf_cfg.attention_type=ScaleChannelSpatial` có sẵn trong pinned MMOCR. Chỉ `neck.asf_conv.*`/`neck.asf_attn.*` được khởi tạo mới; mọi key/shape chung và giá trị sau load đều được kiểm tra, sau đó strict-load toàn state. Giữ nguyên cách init ASF chính thức; không thêm identity/residual-init biến thể.

Mỗi nhánh: AdamW lr2e-5, weight_decay1e-4, microbatch2, accumulation2, effective batch4, **10.000 optimizer updates =20.000 micro-iterations =40.000 lượt student inputs**, validation mỗi2.000 updates. Train640×640, cùng augmentation, backbone unfrozen; chọn best checkpoint theo mean clean/degraded validation Hmean với hậu xử lý native. Budget đủ đi qua toàn danh sách mixed train một lượt; best checkpoint có thể được chọn sớm hơn. Không khẳng định đã hội tụ.

Budget mới lớn hơn Phase3 (3.000 updates). Hai nhánh mới công bằng với nhau; **không quy chênh lệch so với E2/job7957 thành gain ASF** khi dữ liệu, budget và split đều đã đổi.

Trên best validation checkpoint, thử mask `[0.30,0.20,0.15,0.10,0.05]` trên **toàn validation**. Một ngưỡng chung cho clean/degraded, chọn max mean Hmean, hòa thì ưu tiên0.30 theo thứ tự khai báo. Cố định min_text_score0.30, min_text_width5, unclip1.5 và quad/global inference. Không T-map, không tiling, không thay loss/shrink/hard-negative sampling.

Mỗi ảnh dùng chung P-map cho các ngưỡng, không chạy backbone năm lần và không lưu P-map toàn tập lên đĩa. Mỗi launch kiểm tra đường dùng chung P-map khớp native predict trên một ảnh. Test chỉ chạy ngưỡng native và ngưỡng validation đã khóa. Nếu chọn0.30 thì hai hàng tương ứng dùng cùng predictions, không chạy GPU lại cho hàng trùng.

Giữ score floor0.30 có thể hạn chế recall khi giảm mask: thí nghiệm này kiểm tra tác động riêng của mask threshold, không tuyên bố đã tối ưu toàn bộ hậu xử lý. Kết quả không tăng không chứng minh feature collapse. Không dùng GT/suy thoái thực tế để chọn ngưỡng riêng từng ảnh lúc test.

## Xem tiến độ và hoàn tất

```bash
squeue -u dungmt
```

Sau khi có job ID, dùng `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` và logs có tên mục đích. Thư mục run mới:

```text
/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/
  phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_mixed_10000updates_mask_validation_seed42_v1/
```

Đặt biến terminal để đọc nhanh:

```bash
RUN_DIR=/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_mixed_10000updates_mask_validation_seed42_v1
tail -n 60 "$RUN_DIR"/logs/slurm_all_experiments/*.log
```

Slurm stdout/stderr có tên `FULL_R1_DBNet_vs_ASF__mixed_training_validation_masks_test__job_...`; worker logs ghi rõ nhánh A/B và initialize/probe/train/validation/test. Logs bootstrap môi trường vẫn nằm trong `allocation_JOBID__timestamp` bên trong run mới, không lẫn với run cũ.

Hoàn tất phải có cả Slurm `COMPLETED 0:0` và:

```bash
cat "$RUN_DIR/COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json"
```

Nếu thiếu marker thì suite chưa hoàn tất, dù đã có một checkpoint hoặc metrics trung gian. Nếu code/config đổi trong lúc resume, controller sẽ từ chối trộn trạng thái; dùng run_name mới thay vì xóa lock/state cũ.

## Đọc và gửi kết quả

Mở `READ_ME_FIRST__FULL_R1_DBNet_ASF_run_outputs_and_logs.json` ở root run. Các thư mục chính:

```text
inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT/
inputs/03_frozen_detection_source_splits__official_train_intersection__70_15_15/
inputs/05_model_configs__only_FPNC_ASF_differs/
inputs/06_initialized_checkpoints/
outputs/training/A_DBNet_noASF__mixed_clean_degraded/
outputs/training/B_DBNet_plus_ASF__mixed_clean_degraded/
outputs/validation_selection/
outputs/global_predictions/ <A/B> / <validation/test> / mask_0pXX/
outputs/test_metrics/ <A/B> / mask_0pXX/
outputs/FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks/
```

Final comparison gồm Markdown, CSV và JSON: clean/degraded P/R/H, FP/FN, small recall; 0–4/4–8/8–12/12–16/16–32/>=32px; small clean-TP retention; corruption groups có chứa từng operation; bốn contrast với paired-source bootstrap CI. Không tự gán FP vào size của một GT không match để báo “small precision”. Single-seed CI không đo độ ổn định giữa training seeds.

Sau khi hoàn tất, đóng gói báo cáo gọn để gửi, không gửi ảnh/checkpoints:

```bash
python -m zipfile -c \
  /datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/RESULTS__FULL_R1_DBNet_vs_ASF__mixed_10000updates__native_vs_validation_masks__seed42.zip \
  "$RUN_DIR/outputs/FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks" \
  "$RUN_DIR/outputs/validation_selection" \
  "$RUN_DIR/inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT/data_audit__FULL_R1_counts_GT_intersection_and_exclusions.json" \
  "$RUN_DIR/inputs/03_frozen_detection_source_splits__official_train_intersection__70_15_15/split_policy_and_counts.json" \
  "$RUN_DIR/COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json"
```

Chỉ khi thấy B0 hơn A0 mới có bằng chứng trong thí nghiệm này về ASF với native threshold; B1 so A1 cho kết quả sau cơ hội calibration ngang nhau. Xem clean loss, degraded gain, small-word survival và FP cùng lúc; không gọi thu hẹp gap do clean giảm là giải quyết suy thoái.
