import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# 数据
# ------------------------
vggt_mem = {
    50: 23123.640625,
    100: 34234.62939453125,
    150: 45693.91552734375,
    200: 57150.3505859375,
    250: 68608.83984375,
}
stream_mem = {
    50: 17813.65283203125,
    100: 23909.75048828125,
    150: 29993.69921875,
    200: 36079.35009765625,
    250: 42181.326171875,
    300: 80000
}
prune_mem = {
    50: 12824.93310546875,
    100: 13406.8369140625,
    150: 13984.544921875,
    200: 14573.02587890625,
    250: 15152.15478515625,
    300: 15726.00244140625,
}

# ------------------------
# 添加 OOM 点（300）并保持趋势延长
# ------------------------

def extend_to_300(d):
    xs = np.array(sorted(d.keys()))
    ys = np.array([d[k] for k in xs])
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return ys[-1] + slope * (300 - xs[-1])

vggt_mem[300] = extend_to_300(vggt_mem)
# stream_mem[300] = extend_to_300(stream_mem)

# pruning 无 OOM，但画到 300 即可
# ------------------------

# 聚合为曲线
def xy_from_dict(dic):
    xs = sorted(dic.keys())
    ys = [dic[x] for x in xs]
    return xs, ys

vggt_x, vggt_y = xy_from_dict(vggt_mem)
stream_x, stream_y = xy_from_dict(stream_mem)
prune_x, prune_y = xy_from_dict(prune_mem)

# ------------------------
# 绘图
# ------------------------
plt.figure(figsize=(9,6))

plt.plot(vggt_x, vggt_y, marker='o', label="VGGT")
plt.plot(stream_x, stream_y, marker='o', label="StreamVGGT")
plt.plot(prune_x, prune_y, marker='o', label="streamVGGT + pruning")

# ------------------------
# 80GB GPU memory limit
# ------------------------
gpu_limit_mb = 80000  # 80 GB
plt.axhline(gpu_limit_mb, linestyle='--', linewidth=1.5, color='gray')
plt.text(prune_x[-1]-100, gpu_limit_mb+500, "80GB GPU Memory Limit", fontsize=11, color='gray')

# ------------------------
# OOM 标记（强制贴到虚线上）
# ------------------------
oom_x = 300

plt.scatter(oom_x, gpu_limit_mb, marker='x', s=180, linewidths=3, color='C0')
plt.annotate("OOM", (oom_x, gpu_limit_mb), textcoords="offset points",
             xytext=(0,10), ha='center', color='C0', fontsize=14)

plt.scatter(oom_x, gpu_limit_mb, marker='x', s=180, linewidths=3, color='C1')
plt.annotate("OOM", (oom_x, gpu_limit_mb), textcoords="offset points",
             xytext=(0,-18), ha='center', color='C1', fontsize=14)

# ------------------------
# 美化外观
# ------------------------
plt.xlabel("Input Frames")
plt.ylabel("Peak Memory (MB)")
plt.title("Frame Count vs Peak Memory Usage")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("./frame_count_vs_memory.png", dpi=300)
plt.show()
