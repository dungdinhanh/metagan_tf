from modules.visualize import *



if __name__ == '__main__':
    file_dg= "output/mnist_metagan6_metagan_log_batch_size_64_percent_100_n_steps_300000_lambda_gp_0.00/mnist.txt"
    file_l = "output/mnist_metagan6_metagan_log_batch_size_64_percent_100_n_steps_300000_lambda_gp_0.00/mnist_lb.txt"

    file_fid_dcgan = "plot_data/mnist_dcgan5_dcgan_log_batch_size_64_percent_100_n_steps_400000_lambda_gp_0.00_fid_0_400000.txt"
    file_fid_metagan = "plot_data/mnist_metagan10_metagan_log_batch_size_64_percent_100_n_steps_400000_lambda_gp_0.00_fid_0_400000.txt"
    file_fid_metagan_noaux = "plot_data/mnist_metagan_test_nofakeaux2_metagan_log_batch_size_64_percent_100_n_steps_400000_lambda_gp_0.00_fid_0_400000.txt"
    dcgan_steps, dcgan_fid =  read_fid_file(file_fid_dcgan)
    metagan_steps, metagan_fid =  read_fid_file(file_fid_metagan)
    metagantest_steps, metagantest_fid =  read_fid_file(file_fid_metagan_noaux)
    c = 5
    # draw_log(a[0], a[1], "disc","Discriminator loss", "dloss.png")
    # draw_log(a[0], a[1], "gen", "Generator loss", "gloss.png")
    draw_3_fid_log(dcgan_steps, dcgan_fid, metagan_fid, metagantest_fid,"DCGAN", "METAGAN", "METAGAN_NOFAKE", "FID compare","fid.png")
    # draw_log(b[0], b[1] , "lab", "Label Generator loss", "lloss.png")