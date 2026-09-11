import os
import time
from huggingface_hub import snapshot_download

# 环境变量必须放在import之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"   # Windows关闭高速传输，规避10060超时
os.environ["HF_HUB_TIMEOUT"] = "120"

repo_id = "sensenova/SenseNova-Vision-Corpus-50M"
local_dir = r".\SenseNova-Vision-Corpus-50M"

max_attempt = 8
for attempt in range(1, max_attempt + 1):
    try:
        print(f"\n===== 第 {attempt}/{max_attempt} 次尝试下载 =====")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=2,      # Windows降低并发，防止连接超时
            etag_timeout=120
        )
        print("\n✅ 下载完成！")
        break
    except Exception as e:
        print(f"❌ 下载异常: {e}")
        if attempt >= max_attempt:
            print("❌ 达到最大重试次数，退出")
            raise
        sleep_sec = min(2 ** attempt, 30)
        print(f"⏳ {sleep_sec} 秒后自动重试...")
        time.sleep(sleep_sec)
