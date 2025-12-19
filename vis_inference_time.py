import matplotlib.pyplot as plt
import numpy as np

# 原始 inference times (秒/帧)
vggt = {
    50: 0.07136554896831512,
    100: 0.09004076659679414,
    150: 0.11659627278645833,
    200: 0.14629506066441536,
    250: 0.17264177465438843
}

stream_vggt = {
    50: 0.16047167479991914,
    100: 0.23494215130805968,
    150: 0.3160662480195363,
    200: 0.39826048001646996,
    250: 0.48400409376621245
}

fast_vggt = {
    50: 0.10212173700332641,
    100: 0.08883439570665361,
    150: 0.08825760205586751,
    200: 0.08906651318073273,
    250: 0.08805071806907654
}

# 转 FPS
def to_fps(d): return {k: 1/v for k, v in d.items()}

vggt_fps = to_fps(vggt)
stream_fps = to_fps(stream_vggt)
fast_fps = to_fps(fast_vggt)

# 只延伸到 300（OOM 点）
extend_frames_oom = [300]

def extend_to_300(dfps):
    ks = np.array(sorted(dfps.keys()))
    vs = np.array([dfps[k] for k in ks])
    slope = (vs[-1] - vs[-2]) / (ks[-1] - ks[-2])
    last_x = ks[-1]
    last_v = vs[-1]
    return {300: last_v + slope * (300 - last_x)}

vggt_ext = extend_to_300(vggt_fps)
stream_ext = extend_to_300(stream_fps)

# Pruning 版本继续画到 400
fast_ext = {300: np.mean(list(fast_fps.values())),
            350: np.mean(list(fast_fps.values())),
            400: np.mean(list(fast_fps.values()))}

def combine(orig, ext):
    merged = {**orig, **ext}
    xs = sorted(merged.keys())
    ys = [merged[x] for x in xs]
    return xs, ys

vggt_x, vggt_y = combine(vggt_fps, vggt_ext)
stream_x, stream_y = combine(stream_fps, stream_ext)
fast_x, fast_y = combine(fast_fps, fast_ext)

# 绘图
plt.figure(figsize=(9,6))

plt.plot(vggt_x, vggt_y, marker='o', label="VGGT")
plt.plot(stream_x, stream_y, marker='o', label="StreamVGGT")
plt.plot(fast_x, fast_y, marker='o', label="streamVGGT + pruning")

# 只在 300 标注 OOM
oom_x = 300
plt.scatter(oom_x, vggt_ext[oom_x], marker='x', s=120, linewidths=2, color='C0')
plt.annotate("OOM", (oom_x, vggt_ext[oom_x]), textcoords="offset points",
             xytext=(0,5), ha='center', color='C0', fontsize=16)

plt.scatter(oom_x, stream_ext[oom_x], marker='x', s=120, linewidths=2, color='C1')
plt.annotate("OOM", (oom_x, stream_ext[oom_x]), textcoords="offset points",
             xytext=(0,5), ha='center', color='C1', fontsize=16)

plt.xlabel("Input Frames")
plt.ylabel("FPS (frames per second)")
plt.title("Frame Count vs FPS")
plt.grid(True)
plt.legend()

plt.savefig("./frame_count_vs_fps.png", dpi=300, bbox_inches='tight')
plt.show()



# import matplotlib.pyplot as plt

# # Data
# vggt = {
#     50: 0.07136554896831512,
#     100: 0.09004076659679414,
#     150: 0.11659627278645833,
#     200: 0.14629506066441536,
#     250: 0.17264177465438843
# }

# stream_vggt = {
#     50: 0.16047167479991914,
#     100: 0.23494215130805968,
#     150: 0.3160662480195363,
#     200: 0.39826048001646996,
#     250: 0.48400409376621245
# }

# fast_vggt = {
#     50: 0.10212173700332641,
#     100: 0.08883439570665361,
#     150: 0.08825760205586751,
#     200: 0.08906651318073273,
#     250: 0.08805071806907654
# }

# plt.figure(figsize=(8,6))
# plt.plot(list(vggt.keys()), list(vggt.values()), marker='o', label="VGGT")
# plt.plot(list(stream_vggt.keys()), list(stream_vggt.values()), marker='o', label="StreamVGGT")
# plt.plot(list(fast_vggt.keys()), list(fast_vggt.values()), marker='o', label="streamVGGT + pruning")

# plt.xlabel("Input Frames")
# plt.ylabel("Inference Time per Frame (s)")
# plt.title("Frame Count vs Inference Time per Frame")
# plt.legend()
# plt.grid(True)

# plt.show()
# plt.savefig("./frame_count_vs_inference_time.png")
