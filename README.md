# Vietnamese Word Segmentation Wrapper

This is a word segmentation package for Vietnamese. The code basically wraps around the [Vietnamese Word Segmentation model](https://huggingface.co/NlpHUST/vi-word-segmentation).

The requirements are:

- `python` >= 3.10
- `torch` >= 2.0
- `transformers`

Install with:

```bash
pip install git+https://github.com/hnthap/vietnamese-word-segment
```

Example code:

```python
from vwsegment import WordSegments


examples = [
    'Tất cả mọi người đều có quyền sống, quyền tự chủ và quyền an toàn cho cá nhân.',
    'Không ai có thể bị bắt làm nô lệ hoặc nô dịch; chế độ nô lệ và việc buôn nô lệ bị cấm dưới mọi hình thức.',
    'Không ai có thể bị tra tấn hoặc bạo hành, hoặc phải nhận sự đối xử hoặc trừng phạt một cách vô nhân tính và hèn hạ.',
]

segments_list = WordSegments(examples, case=False, device='cuda', batch_size=128)
for segments in segments_list:
    print(' '.join(segments))
```

The output would be:

```text
tất_cả mọi người đều có quyền sống , quyền tự_chủ và quyền an_toàn cho cá_nhân .
không ai có_thể bị bắt làm nô_lệ hoặc nô_dịch ; chế_độ nô_lệ và việc buôn nô_lệ bị cấm dưới mọi hình_thức .
không ai có_thể bị tra_tấn hoặc bạo_hành , hoặc phải nhận sự đối_xử hoặc trừng_phạt một_cách vô_nhân_tính và hèn_hạ .
```
