# E0 Monkey gốc và A′ FP32: một job H200 để inference và EDA

Ngày 05/11/2026. Đây là phần bổ sung baseline còn thiếu, trước khi chốt cơ chế cải tiến.

- **E0:** checkpoint detection Monkey gốc, SHA256 `4b13e26ad5c2e1b13fef45ac608298254d083062d2b9cafba706064bd8e2cf82`, không fine-tune TextVQA. Chạy inference FP32 để cùng chế độ với A′.
- **A′:** nhánh no-ASF đã fine-tune mixed clean/degraded 10.000 optimizer updates, FP32, weights cuối của job9793. SHA checkpoint được ghi trong cấu hình; lượt này không cần nạp lại weights A′ vì sử dụng raw predictions đã kiểm chứng.
- Cùng MonkeyOCRv2-AS/ViTAEv2-S, FPNC không ASF, DBHead, độ phân giải global, protocol polygon TextOCR và split đã đóng băng.
- Không train, không thêm ASF, không tạo suy thoái, không tải lại bộ ảnh và không thay split/GT. Không sửa output của các job cũ.

## Cập nhật và chạy trên HPC

Trong SSH terminal, từ repo cũ:

```bash
cd /datastore/dungmt/textvqa/Tuan/MonKeyOCR-
git pull --ff-only origin main
python run_all.py \
  --config configs/phase5_full_r1_E0_vs_Aprime_FP32_inference_EDA_h200_normal.json \
  --dataset-parent /datastore/dungmt/textvqa/data/benchmark_v1/raw \
  --workspace /datastore/dungmt/textvqa/Tuan/monkeyocr_workspace
```

Code submit **một job normal, MIG 2g.35gb, 8 CPU, 40 GiB RAM**. Giới hạn Slurm 12 giờ là trần thời gian, không phải dự báo runtime. Không chạy inference trên login node.

Nếu thích notebook, mở `H200__E0_original_vs_Aprime_FP32__FULL_R1_inference_EDA.ipynb` trong repo trên HPC. Cell duy nhất gọi đúng lệnh trên. Nó là launcher cho Slurm, không phải notebook Kaggle. **Chọn một cách chạy; không chạy đồng thời cả terminal và notebook.**

`--plan-only` chỉ đọc cấu hình, không submit, cài đặt hoặc chạm dataset. Sau khi job dừng giữa chừng, chạy lại cùng lệnh và giữ run_name để kiểm chứng/tái sử dụng stage đã xong. Không submit bản trùng khi job cũ còn pending/running.

## Những bằng chứng cũ phải giữ

Trong workspace:

1. Frozen inputs của `phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_mixed_10000updates_final_weights_only_seed42_v2` (job9276), nhất là master manifest, protocol và ba detection split.
2. Với `phase4_FULL_R1_FP32_DBNet_vs_ASF_same_job9276_initialization_10000updates_final_only_seed42_v1` (job9793): completion marker, root config, reference receipt, inference config A, validation selection, final comparison, **state**, và raw predictions A tại `outputs/global_predictions/A_DBNet_noASF__mixed_clean_degraded/{validation,test}`.
3. Dataset FULL_R1 và clean đúng byte; environment/model gốc đã cài.

Không cần weights/training targets/ASF initialization cho *lượt inference mới này*, nhưng điều này **không phải lệnh xóa** bằng chứng cũ. Small ZIP gửi trước đây không chứa full raw predictions. Nếu thiếu file, chương trình báo đường dẫn và dừng, không âm thầm thay dữ liệu hoặc chạy lại training.

Trước submit chỉ kiểm tra metadata, file nhỏ và sự tồn tại của predictions. Trong allocation sẽ kiểm tra SHA frozen folds, tách source/clean SHA, hash **13.050 ảnh validation+test**, artifact hashes của A và toàn bộ record identity/sample ID. Không cần hash lại ảnh train vì không train trong lượt này.

## Quy trình và metric

Frozen detection folds trong giao official TextVQA train/TextOCR train:

| Fold | Source | Ảnh clean + degraded |
|---|---:|---:|
| Train, chỉ kiểm tra split, không inference | 15.222 | 30.444 |
| Validation | 3.262 | 6.524 |
| Test | 3.263 | 6.526 |

E0 chạy một network forward chính/ảnh global, cộng phép đối chiếu native predict ở ảnh đầu mỗi lượt. Trên validation, cùng P map được decode qua mask `[0.30, 0.20, 0.15, 0.10, 0.05]`; chọn một mask chung cho clean/degraded bằng trung bình Hmean validation, ưu tiên native khi hòa. Test chỉ chạy native và mask đã khóa. A′ dùng lại kết quả đã có với cùng lưới validation (selected = 0.20). Không quét mask trên test.

Luôn báo **cả native 0.30 và validation-selected**. Không gộp mask threshold với `min_text_score`: giữ nguyên score 0.30, min width 5, unclip 1.5, quad. Worker kiểm tra decode P-map trùng native model.predict trên ảnh đầu, strict checkpoint load sau cast FP32 và ghi runtime TF32 flags; flags E0 phải khớp A′.

Protocol không đổi: matching polygon IoU ≥0,5 theo maximum-cardinality matching; common ROI và ignore giữ nguyên. Báo Precision, Recall, Hmean, FP/FN, coverage phụ, recall theo chiều cao GT gốc. Small = bbox height <16 px. Hmean là harmonic mean của P/R; không dùng coverage thay detection Hmean.

A′ được tính lại metric từ raw predictions và phải khớp counts/Hmean đã gửi trong job9793. Báo CI chênh lệch Hmean A′−E0 bằng paired bootstrap theo source; CI này không đo phương sai training seed.

## Các bảng EDA trả về

1. Clean/degraded P/R/H, small recall, FP/FN ở hai điểm vận hành.
2. Theo từng GT ID và các bin `<4, 4–8, 8–12, 12–16, 16–32, ≥32 px`:
   - clean TP → degraded TP/FN; clean FN → degraded TP/FN, riêng từng model;
   - E0 → A′: sửa được, làm mất, cả hai TP, cả hai FN, riêng clean/degraded.
3. FN native chia loại loại trừ nhau: không có kept prediction IoU≥0,1; có overlap nhưng dưới ngưỡng match; có candidate nhưng xung đột matching. Các tín hiệu gộp từ/tách mảnh/được phủ ≥nửa vùng nhưng chưa match là **nhóm chồng lấp**.
4. FP native chia geometry invalid, partial overlap, không có valid GT IoU≥0,1; histogram confidence TP/FP theo bin cố định. Những invalid FP không có score histogram hợp lệ vẫn được tính trong FP metric.
5. Nhóm có chứa từng phép suy thoái, kèm clean đối ứng. Các nhóm chồng lấp vì MAIN trộn nhiều phép; không kết luận tác động nhân quả riêng từng phép.
6. Tối đa 32 word crops có cặp clean/degraded và overlay GT xanh lá, E0 cyan, A′ cam, ignored GT xám. Hiển thị được giới hạn cạnh 640 px/crop và ghi scale; không dùng ảnh hiển thị để thay dữ liệu đánh giá. Chọn mẫu phân tầng có chủ đích, không suy ra tỷ lệ lỗi từ mẫu này.

Taxonomy overlap/merge/fragment tương thích về ý nghĩa với EDA ICDAR2015 trước đó; **matching/score/ignore của hai bộ khác nhau**. Snapshot ICDAR 500 ảnh, Hmean lịch sử 88,993%, được kèm làm điểm neo có nguồn/hash. Không chạy lại ICDAR FP32, không ghép điểm hai dataset, không coi small recall ICDAR là bằng chứng cho toàn bộ chữ cực nhỏ (nhiều GT ignored).

## Output rõ tên và gói cần gửi

Run mới:

```text
phase5_FULL_R1_E0_original_vs_Aprime_FP32_job9793__inference_EDA_native_selected_seed42_v1/
  inputs/REFERENCE__verified_frozen_folds_and_Aprime_raw_predictions__read_only.json
  inputs/SELECTED_MASK__E0__validation_mean_clean_degraded_Hmean__locked_before_test.json
  outputs/E0_ORIGINAL_CHECKPOINT__global_predictions/{validation,test}/
  outputs/METRICS__unchanged_external_TextOCR_word_protocol/
  outputs/FINAL_EDA__E0_original_vs_Aprime_FP32__same_FULL_R1_test/
    REPORT__Monkey_original_baseline_errors_and_what_FP32_finetuning_repairs.md
    TABLE__E0_original_vs_Aprime_FP32__clean_degraded_Hmean_small_recall_FP_FN.csv
    COMPARISON_AND_EDA__original_Monkey_vs_mixed_finetuning__native_selected.json
    DIAGNOSIS__native_E0_Aprime__word_size_geometry_and_paired_transitions.json
    WORDS__native_mask__E0_vs_Aprime__same_GT_ID_clean_degraded.csv
  outputs/REVIEW_CROPS__native_GT_E0_Aprime__clean_degraded_pairs/
  logs/slurm_all_experiments/...
```

Raw predictions và full per-word CSV ở HPC. ZIP tự tạo chỉ chứa báo cáo, chỉ mục, metadata/config và crop nhỏ, giới hạn **64 MiB uncompressed**. Không đóng gói ảnh full, full predictions, per-word CSV hoặc weights.

Khi xong, tải bằng WinSCP file nằm **ngay trong run mới**:

```text
RESULTS_AND_EDA__FULL_R1_E0_original_vs_Aprime_FP32__native_selected_word_errors_review_crops.zip
```

Kiểm tra (thay `JOB_ID` bằng ID được submit):

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
RUN_DIR="/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/phase5_FULL_R1_E0_original_vs_Aprime_FP32_job9793__inference_EDA_native_selected_seed42_v1"
tail -n 60 "$RUN_DIR"/logs/slurm_all_experiments/*job_JOB_ID*.log
cat "$RUN_DIR/COMPLETE__E0_original_Aprime_FP32_FULL_R1_inference_and_automatic_EDA_finished.json"
ls -lh "$RUN_DIR"/RESULTS_AND_EDA*.zip
```

Chỉ kết luận job xong khi Slurm `COMPLETED 0:0`, có completion marker và ZIP. Marker xác nhận **phần tự động**, không nói rằng đã xem tay xong hoặc xác định nguyên nhân. Sau khi đọc kết quả và gán nhãn crop, mới quyết định cần audit probability map/target nhỏ nào và chọn một hướng cải tiến có đối chứng. Không tự mặc định low-resolution, KD hay ASF là lời giải.
