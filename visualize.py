from modules.visualize import *



if __name__ == '__main__':
    file_dg= "output/mnist_metagan6_metagan_log_batch_size_64_percent_100_n_steps_300000_lambda_gp_0.00/mnist.txt"
    file_l = "output/mnist_metagan6_metagan_log_batch_size_64_percent_100_n_steps_300000_lambda_gp_0.00/mnist_lb.txt"
    a = read_log_file_dg(file_dg)
    b = read_log_file_l2(file_l)
    c = 5
    draw_log(a[0], a[1], "disc","Discriminator loss", "dloss.png")
    draw_log(a[0], a[1], "gen", "Generator loss", "gloss.png")
    draw_2_log(a[0], a[1], a[2], "dis", "gen", "DGLoss", "dgloss.png")
    draw_log(b[0], b[1] , "lab", "Label Generator loss", "lloss.png")