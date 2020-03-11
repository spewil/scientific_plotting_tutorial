import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('seaborn-dark')

ax1 = plt.subplot(2, 2, 1)
ax1.set_xticks([]), ax1.set_yticks([])
ax1.text(
    0.5, 0.5, 'subplot(2,2,1)', ha='center', va='center', size=20, alpha=.5)

ax2 = plt.subplot(2, 2, 2)
ax2.set_xticks([]), ax2.set_yticks([])
ax2.text(
    0.5, 0.5, 'subplot(2,2,2)', ha='center', va='center', size=20, alpha=.5)

ax3 = plt.subplot(2, 2, 3)
ax3.set_xticks([]), ax3.set_yticks([])
ax3.text(
    0.5, 0.5, 'subplot(2,2,3)', ha='center', va='center', size=20, alpha=.5)

ax4 = plt.subplot(2, 2, 4)
ax4.set_xticks([]), ax4.set_yticks([])
ax4.text(
    0.5, 0.5, 'subplot(2,2,4)', ha='center', va='center', size=20, alpha=.5)

ax1.set_title("Hello, I'm a title!")
ax4.set_title("Nobody puts plot 4 in the corner...")

plt.show()

del ax1, ax2, ax3, ax4

# Exercise -- can you do this in a loop?

plt.show()
