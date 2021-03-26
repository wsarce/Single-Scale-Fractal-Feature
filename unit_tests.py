import arce_pierce_velcsov_dimension as apv
import glob
import os
from PIL import Image
import numpy as np
import arce_pierce_velcsov_dimension as apv
import matplotlib.pyplot as plt
import sys
import image_utilities as iu
import timeit


def known_fractals(sier_dir, srcp_dir):
    """

    :param bmrk_dir:
    :param sier_dir:
    :param srcp_dir:
    :return:
    """
    print("Collecting images from the directory provided:", sier_dir, "&", srcp_dir)
    sier_files = []
    sier_r, sier_g, sier_b, sier_c = [], [], [], []
    srcp_files = []
    srcp_r, srcp_g, srcp_b, srcp_c = [], [], [], []
    sier_len = len(list(glob.iglob(sier_dir + '**/*', recursive=True)))
    srcp_len = len(list(glob.iglob(srcp_dir + '**/*', recursive=True)))
    current_file = 0
    for filename in glob.iglob(sier_dir + '**/*', recursive=True):
        if filename.endswith(".png"):
            current_file += 1
            sier_files.append(filename)
            print('\t', current_file, "/", sier_len, ":", filename)
    current_file = 0
    for filename in glob.iglob(srcp_dir + '**/*', recursive=True):
        if filename.endswith(".png"):
            current_file += 1
            srcp_files.append(filename)
            print('\t', current_file, "/", srcp_len, ":", filename)
    for file in sier_files:
        dim = apv.apv_dimension(file, image_flag=-1, bit_depth=255, invert=True, alpha_channel=True, alpha_index=None)
        print(os.path.basename(file)[:-4], '\t', dim[2], '\t', dim[0], '\t', dim[1], '\t', dim[3])
        sier_b.append(dim[0])
        sier_g.append(dim[1])
        sier_r.append(dim[2])
        sier_c.append(dim[3])
    for file in srcp_files:
        dim = apv.apv_dimension(file, image_flag=-1, bit_depth=255, invert=True, alpha_channel=True, alpha_index=None)
        print(os.path.basename(file)[:-4], '\t', dim[2], '\t', dim[0], '\t', dim[1], '\t', dim[3])
        srcp_b.append(dim[0])
        srcp_g.append(dim[1])
        srcp_r.append(dim[2])
        srcp_c.append(dim[3])
    return


def ivanovici_color_fractal(ivanovici_dir):
    """

    :param ivanovici_dir:
    :return:
    """
    iv_cfd = [3.8286, 3.9134, 3.9113, 3.6373, 3.2692, 2.8623, 2.5673, 2.3573, 2.2372]
    print("Collecting images from the directory provided:", ivanovici_dir)
    iv_files = []
    iv_b, iv_g, iv_r, iv_c = [], [], [], []
    files = len(list(glob.iglob(ivanovici_dir + '**/*', recursive=True)))
    current_file = 0
    for filename in glob.iglob(ivanovici_dir + '**/*', recursive=True):
        if filename.endswith(".bmp"):
            current_file += 1
            iv_files.append(filename)
            print('\t', current_file, "/", files, ":", filename)
    for file in iv_files:
        dim = apv.apv_dimension(file, image_flag=-1, bit_depth=255, invert=True, alpha_channel=False, alpha_index=None)
        print(os.path.basename(file)[:-4], dim)
        iv_b.append(dim[0])
        iv_g.append(dim[1])
        iv_r.append(dim[2])
        iv_c.append(dim[3])
    delta_iv = []
    for i in range(0, len(iv_cfd) - 1):
        delta_iv.append(iv_cfd[i + 1] - iv_cfd[i])
    delta_b, delta_g, delta_r, delta_c = [], [], [], []
    for i in range(0, len(iv_files) - 1):
        delta_b.append(iv_b[i + 1] - iv_b[i])
        delta_g.append(iv_g[i + 1] - iv_g[i])
        delta_r.append(iv_r[i + 1] - iv_r[i])
        delta_c.append(iv_c[i + 1] - iv_c[i])
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    t = np.arange(0, len(iv_files) - 1)
    hurst = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    plt.subplot(2, 2, 1)
    plt.title("Ivanovici CFD")
    plt.xlabel("Hurst Parameter: 0.1 - 0.9")
    plt.ylabel("Fractal Dimension")
    line, = plt.plot(hurst, iv_cfd, color='black')
    line.set_label("Ivanovici CFD")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.title("Three Channel CFD")
    plt.xlabel("Hurst Parameter: 0.1 - 0.9")
    plt.ylabel("Fractal Dimension")
    line, = plt.plot(hurst, iv_c, color='black')
    line.set_label("All Channels")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title("Composite CFD Variation")
    plt.xlabel("Variation Between Successive Hurst Parameter Settings")
    plt.ylabel("Variation")
    line, = plt.plot(t, delta_c, color='black')
    line.set_label("All Channels Variation")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.title("Ivanovici CFD Variation")
    plt.xlabel("Variation Between Successive Hurst Parameter Settings")
    plt.ylabel("Variation")
    line, = plt.plot(t, delta_iv, color='black')
    line.set_label("Ivanovici Variation")
    plt.legend()
    plt.grid()

    plt.show()

    plt.subplot(1, 2, 1)
    plt.title("Individual Color Channel CFD")
    plt.xlabel("Hurst Parameter: 0.1 - 0.9")
    plt.ylabel("Fractal Dimension")
    line, = plt.plot(hurst, iv_r, color='red')
    line.set_label("Red Channel")
    plt.legend()

    line, = plt.plot(hurst, iv_g, color='green')
    line.set_label("Green Channel")
    plt.legend()

    line, = plt.plot(hurst, iv_b, color='blue')
    line.set_label("Blue Channel")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("Individual Color Channel CFD Variation")
    plt.xlabel("Variation Between Successive Hurst Parameter Settings")
    plt.ylabel("Variation")
    line, = plt.plot(t, delta_b, color='blue')
    line.set_label("Blue Variation")
    plt.legend()

    line, = plt.plot(t, delta_g, color='green')
    line.set_label("Green Variation")
    plt.legend()

    line, = plt.plot(t, delta_r, color='red')
    line.set_label("Red Variation")
    plt.legend()
    plt.grid()

    plt.show()

    return


def color_image_testing(generated_dir):
    """

    :param generated_dir:
    :return:
    """
    # Initialize our results containers
    r, g, b, c = [], [], [], []
    # Generate a gradient image
    img = iu.get_gradation_3d(256, 256, (0, 255, 128), (255, 0, 128), (True, False, False))
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "GradientTestD2.png"), "PNG")
    # Run the fractal dimension estimator on the gradient image
    gradient_dim = apv.apv_dimension(os.path.join(generated_dir, "GradientTestD2.png"))
    print("Gradient D = 2", '\t', gradient_dim[2], '\t', gradient_dim[0], '\t', gradient_dim[1], '\t', gradient_dim[3], '\n')
    # Generate a randomized image
    img[:, :, 2] = np.round(np.random.rand(256, 256) * 255)
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "BlueRandomTest.png"), "PNG")
    # Run the fractal dimension estimator on the randomized image
    random_dim = apv.apv_dimension(os.path.join(generated_dir, "BlueRandomTest.png"))
    print("Blue Random Test", '\t', random_dim[2], '\t', random_dim[0], '\t', random_dim[1], '\t', random_dim[3], '\n')
    # Generate a randomized image
    img[:, :, 1] = np.round(np.random.rand(256, 256) * 255)
    img[:, :, 2] = np.round(np.random.rand(256, 256) * 255)
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "GreenBlueRandomTest.png"), "PNG")
    # Run the fractal dimension estimator on the randomized image
    random_dim = apv.apv_dimension(os.path.join(generated_dir, "GreenBlueRandomTest.png"))
    print("Green Blue Random Test", '\t', random_dim[2], '\t', random_dim[0], '\t', random_dim[1], '\t', random_dim[3], '\n')
    # Generate a randomized image
    img[:, :, 0] = np.round(np.random.rand(256, 256) * 255)
    img[:, :, 1] = np.round(np.random.rand(256, 256) * 255)
    img[:, :, 2] = np.round(np.random.rand(256, 256) * 255)
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "FullRandomTest.png"), "PNG")
    # Run the fractal dimension estimator on the randomized image
    random_dim = apv.apv_dimension(os.path.join(generated_dir, "FullRandomTest.png"))
    print("Full Random Test", '\t', random_dim[2], '\t', random_dim[0], '\t', random_dim[1], '\t', random_dim[3], '\n')
    # Get our minimum cell brightness
    l_min = apv.get_l_min(img, 255)
    # Populate an RGB image with full luminosity in the red channel
    img[:, :, 0].fill(255)
    img[:, :, 1].fill(l_min)
    img[:, :, 2].fill(l_min)
    # Run the fractal dimension estimator on the red image
    random_dim = apv.apv_dimension(img, image_flag='arr', invert=False)
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "RedTest.png"), "PNG")
    print("Red Test", '\t', random_dim[2], '\t', random_dim[0], '\t', random_dim[1], '\t', random_dim[3], '\n')
    # Populate an RGB image with full luminosity in the green channel
    img[:, :, 0].fill(l_min)
    img[:, :, 1].fill(255)
    img[:, :, 2].fill(l_min)
    # Run the fractal dimension estimator on the green image
    random_dim = apv.apv_dimension(img, image_flag='arr', invert=False)
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "GreenTest.png"), "PNG")
    print("Green Test", '\t', random_dim[2], '\t', random_dim[0], '\t', random_dim[1], '\t', random_dim[3], '\n')
    # Populate an RGB image with full luminosity in the blue channel
    img[:, :, 0].fill(l_min)
    img[:, :, 1].fill(l_min)
    img[:, :, 2].fill(255)
    # Run the fractal dimension estimator on the blue image
    random_dim = apv.apv_dimension(img, image_flag='arr', invert=False)
    Image.fromarray(np.uint8(img)).save(os.path.join(generated_dir, "BlueTest.png"), "PNG")
    print("Blue Test", '\t', random_dim[2], '\t', random_dim[0], '\t', random_dim[1], '\t', random_dim[3], '\n')
    # Create an empty four channel image, to be used in our boundary testing
    img = np.empty((256, 256, 4))
    # Specify our dimension for each channel of the empty image
    dims = [2.0, 3.0, 4.0, 5.0]
    # Perform boundary testing from l_min to 255 in increments of
    for i in np.arange(l_min, 255.1, 0.001):
        if i > 255.0:
            break
        img[:, :, 0].fill(i)
        img[:, :, 1].fill(i)
        img[:, :, 2].fill(i)
        img[:, :, 3].fill(i)
        dim = apv.apv_dimension(img, image_flag='arr', dimensions=dims)
        # print("Fill Test", i, '\t', dim[2], '\t', dim[0], '\t', dim[1], '\t', dim[3])
        b.append(dim[0])
        g.append(dim[1])
        r.append(dim[2])
        c.append(dim[3])
    t = np.arange(0, len(b))
    r_delta, b_delta, g_delta, c_delta = [], [], [], []
    for i in range(0, len(r) - 1):
        r_delta.append(r[i + 1] - r[i])
        b_delta.append(b[i + 1] - b[i])
        g_delta.append(g[i + 1] - g[i])
        c_delta.append(c[i + 1] - c[i])

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30

    #plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.subplot(1, 2, 1)
    plt.title("Color Fill Testing")
    plt.xlabel("Color Fill from Min to Max Luminosity")
    plt.ylabel("Fractal Dimension")
    plt.grid()

    line, = plt.plot(t, c)
    line.set_label("Dmax = 5.0")
    plt.legend()

    line, = plt.plot(t, r)
    line.set_label("Dmax = 4.0")
    plt.legend()

    line, = plt.plot(t, g)
    line.set_label("Dmax = 3.0")
    plt.legend()

    line, = plt.plot(t, b)
    line.set_label("Dmax = 2.0")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("FD Change Per Iteration")
    plt.xlabel("Iteration of Color Fill")
    plt.ylabel("Variation in Each Iteration of Color Fill")
    plt.grid()

    t = np.arange(0, len(b_delta))
    line, = plt.plot(t, c_delta)
    line.set_label("Dmax = 5.0 Variation")
    plt.legend()

    line, = plt.plot(t, r_delta)
    line.set_label("Dmax = 4.0 Variation")
    plt.legend()

    line, = plt.plot(t, g_delta)
    line.set_label("Dmax = 3.0 Variation")
    plt.legend()

    line, = plt.plot(t, b_delta)
    line.set_label("Dmax = 2.0 Variation")
    plt.legend()

    plt.show()


def segmented_virus_images(virus_dir):
    """

    :param virus_dir:
    :return:
    """
    print("Collecting images from the directory provided:", virus_dir)
    vrs_files = []
    files = len(list(glob.iglob(virus_dir + '**/*', recursive=True)))
    current_file = 0
    for filename in glob.iglob(virus_dir + '**/*', recursive=True):
        if filename.endswith(".png"):
            current_file += 1
            vrs_files.append(filename)
            print('\t', current_file, "/", files, ":", filename)
    print('*' * 20)
    for file in vrs_files:
        dim = apv.apv_dimension(file, image_flag=-1, bit_depth=255, invert=False, alpha_channel=True, alpha_index=None)
        print((os.path.basename(file)[:-4] + "\t{0}\t".format(dim,)))
    return


def timing_tests():
    """

    :return:
    """
    time = []
    time.append(timeit.timeit(timed_function_256, number=1000))
    time.append(timeit.timeit(timed_function_512, number=1000))
    time.append(timeit.timeit(timed_function_768, number=1000))
    time.append(timeit.timeit(timed_function_1024, number=1000))
    time.append(timeit.timeit(timed_function_1280, number=1000))
    print(time)
    t = ['256x256', '512x512', '768x768', '1024x1024', '1280x1280']
    plt.plot(t, time)
    plt.title("Execution Time With Increasing Image Size at 1000 Iterations")
    plt.xlabel("Image Size (px)")
    plt.ylabel("Execution Time (ms)")
    plt.grid()
    plt.show()
    return


def timed_function_256():
    img = np.empty((256, 256, 3))
    img[:, :, 0] = np.round(np.random.rand(256, 256) * 255)
    img[:, :, 1] = np.round(np.random.rand(256, 256) * 255)
    img[:, :, 2] = np.round(np.random.rand(256, 256) * 255)
    return apv.apv_dimension(img, image_flag='arr')


def timed_function_512():
    img = np.empty((512, 512, 3))
    img[:, :, 0] = np.round(np.random.rand(512, 512) * 255)
    img[:, :, 1] = np.round(np.random.rand(512, 512) * 255)
    img[:, :, 2] = np.round(np.random.rand(512, 512) * 255)
    return apv.apv_dimension(img, image_flag='arr')


def timed_function_768():
    img = np.empty((768, 768, 3))
    img[:, :, 0] = np.round(np.random.rand(768, 768) * 255)
    img[:, :, 1] = np.round(np.random.rand(768, 768) * 255)
    img[:, :, 2] = np.round(np.random.rand(768, 768) * 255)
    return apv.apv_dimension(img, image_flag='arr')


def timed_function_1024():
    img = np.empty((1024, 1024, 3))
    img[:, :, 0] = np.round(np.random.rand(1024, 1024) * 255)
    img[:, :, 1] = np.round(np.random.rand(1024, 1024) * 255)
    img[:, :, 2] = np.round(np.random.rand(1024, 1024) * 255)
    return apv.apv_dimension(img, image_flag='arr')


def timed_function_1280():
    img = np.empty((1280, 1280, 3))
    img[:, :, 0] = np.round(np.random.rand(1280, 1280) * 255)
    img[:, :, 1] = np.round(np.random.rand(1280, 1280) * 255)
    img[:, :, 2] = np.round(np.random.rand(1280, 1280) * 255)
    return apv.apv_dimension(img, image_flag='arr')


SIER_files = r'./SIER/'
SRCP_files = r'./SRCP/'
color_fractals = r'./CFIICC/'
gen_images = r'./GENIMG/'
virus_images = r'./VRSTR/'

# timing_tests()
# known_fractals(SIER_files, SRCP_files)
# ivanovici_color_fractal(color_fractals)
color_image_testing(gen_images)
# segmented_virus_images(virus_images)
