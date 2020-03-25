# for d and g: 4, 7, 10 is step Dloss Gloss
# for l      : 6, 9 is step Lloss
import matplotlib.pyplot as plt


def draw_log(sticks, values, name,title, path):
    plt.plot(sticks, values, label=name)
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(path)
    plt.close()

def draw_2_log(sticks, values1, values2, name1, name2,title, path):
    plt.plot(sticks, values2, label=name2, alpha=0.5)
    plt.plot(sticks, values1, label=name1, alpha=0.5)
    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(path)
    plt.close()


def read_log_file(dg_path: str, l_path: str):
    return read_log_file_dg(dg_path), read_log_file_l(l_path)


def read_log_file_dg(dg_path:str):
    f = open(dg_path, "r")
    steps_dg = []
    d_losses = []
    g_losses = []
    for line in f:
        step, d_loss, g_loss = extract_info_d_g(input_log_line=line)
        steps_dg.append(step)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
    return steps_dg, d_losses, g_losses


def read_log_file_l(l_path:str):
    f = open(l_path, "r")
    steps_l = []
    l_losses = []
    for line in f:
        step, l_loss = extract_info_l(input_log_line=line)
        steps_l.append(step)
        l_losses.append(l_loss)
    return steps_l, l_losses

def read_log_file_l2(l_path:str):
    f = open(l_path, "r")
    steps_l = []
    l_losses = []
    count = 0
    for line in f:
        step, l_loss = extract_info_l(input_log_line=line)
        steps_l.append(count)
        l_losses.append(l_loss)
        count+=1
    return steps_l, l_losses


def extract_info_d_g(input_log_line):
    list_features = input_log_line.split()
    step = int(list_features[4].split(",")[0])
    d_loss = float(list_features[7].split(",")[0])
    g_loss = float(list_features[10].split(",")[0])
    return step, d_loss, g_loss
    pass

def extract_info_l(input_log_line):
    list_features = input_log_line.split()
    step = int(list_features[6].split(",")[0])
    l_loss = float(list_features[9].split(",")[0])
    return step, l_loss
    pass

