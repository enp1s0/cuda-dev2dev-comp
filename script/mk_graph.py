import matplotlib.pyplot as plt
import pandas as pd

offset_list = [
        0, -1, 1
        ]

# Output
output_file_name = "result.pdf"
fig_block_m = len(offset_list)
fig_block_n = 1

# The list of `type` in the input csv file
type_list = ['cudaMemcpy', 'copy_kernel']
color_table = {
        'cudaMemcpy'  : 'red',
        'copy_kernel' : 'blue',
        }

# Load input data
df = pd.read_csv("data.csv", encoding="UTF-8")

# Create graph
fig, axs = plt.subplots(fig_block_m, fig_block_n, figsize=(10, 6))
fig.subplots_adjust(hspace=0.7)

type_index = 0
line_list = []
label_list = []
for j in range(fig_block_m):
    offset = offset_list[j]
    df_t = df.query('size_offset==' + str(offset))
    # Plot
    axs[j].set_xscale('log', base=2)
    axs[j].set_xticks([2**i for i in range(0, 33, 4)])
    axs[j].set_xlabel('$n$')
    axs[j].set_ylabel('Throughput [GB/s]')
    axs[j].set_title('Data size=$n {:+d}$ [Byte]'.format(offset))
    axs[j].grid(True)
    for t in type_list:
        l = axs[j].plot(
                [2** i for i in df_t['n']],
                df_t[t + '_bw'] * 1e-9,
                markersize=4,
                marker="*",
                color=color_table[t])
        if j == 0:
            line_list += [l]
            label_list += [t]

        # inc type_index
        type_index += 1

# Legend config
fig.legend(line_list,
        labels=label_list,
        loc='upper center',
        ncol=len(type_list))

# Save to file
fig.savefig(output_file_name, bbox_inches="tight")
