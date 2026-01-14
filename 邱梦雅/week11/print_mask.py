
import matplotlib.pyplot as plt
import random


def plot_mask(mask, s1_len):
    # 初始化调用次数（首次调用时设置）
    if not hasattr(plot_mask, 'call_count'):
        plot_mask.call_count = 0

    if plot_mask.call_count >= 5:
        return

    # random.choice(mask.numpy())
    idx = random.randint(0, mask.size()[0] - 1)
    mmat = mask.numpy()[idx]

    plt.figure(figsize=(50, 50))
    plt.imshow(mmat, cmap='viridis', interpolation='nearest')  # binary: 0=white, 1=black
    cbar = plt.colorbar()

    # 调整 colorbar 刻度字体大小
    cbar.ax.tick_params(labelsize=50)

    # 在 y = s1_len-1 处画一条水平线
    plt.axhline(y=s1_len-1, color='green', linestyle='--', linewidth=10, label='y = %d' % (s1_len-1))
    # 在 x = s1_len-1 处画一条垂直线
    plt.axvline(x=s1_len-1, color='red', linestyle='--', linewidth=10, label='x = %d' % (s1_len-1))

    plt.legend(prop={'size': 75})
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.title("mask at batch %d" % idx, fontsize=75, fontweight='bold')

    # plt.show()
    plt.savefig("./figs/mask at batch %d.png" % idx, bbox_inches='tight')
    plt.close()
    plot_mask.call_count += 1