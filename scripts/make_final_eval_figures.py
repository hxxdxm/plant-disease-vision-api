import json, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
"""최종 모델 평가 그림 2장 생성 (학습·평가 재실행 없이 reports/data 기록만 사용)

사용법 (프로젝트 루트에서):
    python scripts/make_final_eval_figures.py reports/data reports/figures
출력:
    confusion_matrix_e0_e4.png  - E0 / E4 Test 혼동행렬
    model_compare_val_curve.png - E0~E4 epoch별 Validation 정확도
"""
D = sys.argv[1] if len(sys.argv) > 1 else "reports/data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reports/figures"
# 한글 글꼴: 설치된 것 중 하나를 사용 (WSL: sudo apt install fonts-nanum 또는 fonts-noto-cjk)
for name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if any(f.name == name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
else:
    import glob
    for fp in glob.glob("/usr/share/fonts/**/*CJK*Regular*", recursive=True) + glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID = "#1f2328", "#59636e", "#e6e6e3"
LAB = ["각진잎반점병", "녹병", "정상"]

# 1) confusion matrices
cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#2a78d6", "#184f95"])
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=200)
for ax, tag, title in [(axes[0], "e0", "E0 제공 코드 그대로"), (axes[1], "e4", "E4 개선 모델 (최종)")]:
    d = json.load(open(f"{D}/eval_test_{tag}.json"))
    m = d["confusion_matrix"]["rows_true_cols_pred"]
    ax.imshow(m, cmap=cmap, vmin=0, vmax=43)
    for i in range(3):
        for j in range(3):
            v = m[i][j]; tot = sum(m[i])
            col = "white" if v > 25 else INK
            ax.text(j, i - 0.08, str(v), ha="center", va="center", fontsize=17, color=col, fontweight="bold")
            ax.text(j, i + 0.25, f"{v/tot*100:.0f}%", ha="center", va="center", fontsize=10, color=col)
    ax.set_xticks(range(3), LAB, fontsize=11, color=INK); ax.set_yticks(range(3), LAB, fontsize=11, color=INK)
    ax.set_xlabel("모델 예측", fontsize=11, color=INK2); ax.set_ylabel("정답 (폴더 라벨)", fontsize=11, color=INK2)
    ax.set_title(f"{title}\nTest 정확도 {d['accuracy']:.4f} ({d['correct']}/{d['total']}) · Macro F1 {d['macro_f1']:.4f}",
                 fontsize=12.5, color=INK, pad=10)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([x + .5 for x in range(2)], minor=True); ax.set_yticks([y + .5 for y in range(2)], minor=True)
    ax.grid(which="minor", color="white", linewidth=3); ax.tick_params(which="both", length=0)
fig.suptitle("Test 128장 혼동행렬 — 대각선이 정답, 칸 안 %는 정답 클래스별 비율(재현율)", fontsize=13, color=INK, y=0.995)
fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix_e0_e4.png", facecolor="white"); plt.close(fig)

# 2) all models: val acc per epoch
S = [("E0 Head-only · lr 2e-4", "training_report.json", "#9aa5b1", "o"),
     ("E2 Head-only · lr 1e-3 · 10ep", "training_report_e2.json", "#1baf7a", "s"),
     ("E3 Full · lr 2e-4", "training_report_e3.json", "#e87ba4", "D"),
     ("E1 Full · lr 5e-5", "training_report_full.json", "#2a78d6", "^"),
     ("E4 LP-FT + 증강 + 코사인", "training_report_e4.json", "#eb6834", "o")]
fig, ax = plt.subplots(figsize=(12, 5.6), dpi=200)
ends = []
for name, f, c, mk in S:
    h = json.load(open(f"{D}/{f}"))["history"]
    x = [e["epoch"] for e in h]; y = [e["val_accuracy"] for e in h]
    lw = 2.6 if name.startswith("E4") else 2
    ax.plot(x, y, color=c, lw=lw, marker=mk, ms=6.5, mec="white", mew=1.5, zorder=3)
    rep = json.load(open(f"{D}/{f}"))
    if "best_epoch" in rep:   # finetune_v2: 동률이면 val loss 낮은 epoch 저장
        bi = x.index(rep["best_epoch"])
    else:                     # 원본 finetune.py: 처음 최고값 epoch 저장 (동률이면 갱신 안 함)
        bi = max(range(len(y)), key=lambda i: (y[i], -i))
    ax.scatter([x[bi]], [y[bi]], s=170, facecolor="none", edgecolor=c, lw=2, zorder=4)
    ends.append([x[-1], y[-1], name, c, y[bi], x[bi]])
from matplotlib.lines import Line2D
hs=[Line2D([],[],color=e[3],marker=mk,lw=2.4,ms=7,mec='white') for e,(_,_,_,mk) in zip(ends,S)]
ax.legend(hs,[f"{e[2]}   최고 {e[4]:.4f} (ep {e[5]})" for e in ends],loc='lower right',fontsize=10.5,frameon=True,framealpha=1,edgecolor=GRID,title='실험 · 저장된 최고 Val',title_fontsize=10.5)
ax.axvspan(0.6, 2.5, color="#fdf0ea", zorder=0); ax.text(1.5, 0.905, "E4 처음 2 epoch:\n분류층만 학습(LP)", ha="center", va="top", fontsize=9, color=INK2)
ax.set_xlim(0.6, 15.5); ax.set_ylim(0.63, 1.0); ax.set_xticks(range(1, 16))
ax.set_xlabel("epoch", color=INK2); ax.set_ylabel("Validation 정확도 (133장)", color=INK2)
ax.grid(axis="y", color=GRID); [ax.spines[s].set_visible(False) for s in ("top", "right")]
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2)
ax.set_title("모든 실험의 epoch별 Validation 정확도 — 동그라미가 저장된(최고) epoch", fontsize=13, color=INK, loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/model_compare_val_curve.png", facecolor="white"); plt.close(fig)
print("ok")
