# -*- coding: utf-8 -*-
"""验证：坏 json 回退 zip（用完整 8MB zip 源）。"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, r"E:\信息\论文\医疗诊断\多目标压力\论文\EthicalGuard\src")

from ethicalguard.data.medethicsqa import MedEthicsQAAdapter

# 完整 zip 源（确认 8MB 有效）
src_zip = r"E:\信息\论文\医疗诊断\多目标压力\论文\数据集\MedEthicsQA\MedEthicsQA_open.zip"
assert os.path.getsize(src_zip) == 8081829, "zip 源不完整"

tmp = tempfile.mkdtemp()
d = os.path.join(tmp, "MedEthicsQA")
os.makedirs(d)
# 坏 json（截断）
with open(os.path.join(d, "MedEthicsQA_open.json"), "w", encoding="utf-8") as f:
    f.write('{"context": "truncated')
# 完整 zip
shutil.copy(src_zip, os.path.join(d, "MedEthicsQA_open.zip"))

ad = MedEthicsQAAdapter(tmp)
items = ad._open_json()
assert len(items) == 5351, f"回退 zip 应得 5351 条，实际 {len(items)}"
print("1) 坏 json 回退完整 zip 成功：5351 条 ✓")

# 好 json 优先
with open(os.path.join(d, "MedEthicsQA_open.json"), "w", encoding="utf-8") as f:
    f.write('[]')
items2 = MedEthicsQAAdapter(tmp)._open_json()
assert items2 == [], "好 json 应优先"
print("2) 好 json 优先读取成功 ✓")

shutil.rmtree(tmp)
print("ALL OK")
