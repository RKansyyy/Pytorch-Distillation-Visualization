from visualization import *
from utils import *
from KdDistill import KdDistill
from StudentModel import *
import numpy as np
from Imagenet64 import Imagenet64
from AlexNet import AlexNet
import random
import torchfunc
import pathlib
import shutil
import lucent.modelzoo.util
from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.lines as mlines


seed = 13
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 45
    learning_rate = 0.005
    decay_epoch = 3
    alpha = 0.1
    temperature = 3
    batch_size = 32
    learning_rate_factor = 0.9
    default_factor_epoch = 45
    mode = 'train'

    class_labels_40 = [(22, 'killer_whale'), (41, 'dalmatian'), (44, 'skunk'),
                    (80, 'zebra'), (81, 'ram'), (280, 'garbage_truck'),
                    (281, 'pickup'), (282, 'tow_truck'), (450, 'goldfish'),
                    (456, 'sturgeon'), (231, 'warplane'), (230, 'airliner'),
                    (545, 'crane'), (441, 'albatross'), (399, 'vulture'),
                    (224, 'ant'), (237, 'speedboat'), (239, 'canoe'),
                    (243, 'container_ship'), (234, 'space_shuttle'),
                    (606, 'garden_spider'),
                    (614, 'rock_crab'), (629, 'fly'), (630, 'bee'),
                    (632, 'cricket'), (687, 'library'), (690, 'church'),
                    (693, 'planetarium'), (922, 'maze'), (417, 'toucan'),
                    (190, 'lion'), (217, 'platypus'), (232, 'airship'),
                    (233, 'balloon'), (319, 'orange'), (320, 'lemon'), (322, 'pineapple'),
                    (415, 'hummingbird'), (442, 'great_white_shark'), (682, 'viaduct')]

    class_indices_40 = [22, 41, 44,
                     80, 81, 280,
                     281, 282, 450,
                     456, 231, 230,
                     545, 441, 399,
                     224, 237, 239,
                     243, 234, 606,
                     614, 629, 630,
                     632, 687, 690,
                     693, 922, 417,
                     190, 217, 232,
                     233, 319, 320,
                     322, 415, 442, 682]

    class_labels_50 = [(9, 'ibex'), (12, 'gazelle'),
                       (17, 'great_dane'), (18, 'walker_hound'), (241, 'catamaran'),
                       (22, 'killer_whale'), (41, 'dalmatian'), (44, 'skunk'),
                        (80, 'zebra'), (81, 'ram'), (280, 'garbage_truck'),
                        (281, 'pickup'), (282, 'tow_truck'), (450, 'goldfish'),
                        (456, 'sturgeon'), (231, 'warplane'), (230, 'airliner'),
                        (545, 'crane'), (441, 'albatross'), (399, 'vulture'),
                        (224, 'ant'), (237, 'speedboat'), (239, 'canoe'),
                        (243, 'container_ship'), (234, 'space_shuttle'),
                        (606, 'garden_spider'),
                        (614, 'rock_crab'), (629, 'fly'), (630, 'bee'),
                        (632, 'cricket'), (687, 'library'), (690, 'church'),
                        (693, 'planetarium'), (922, 'maze'), (417, 'toucan'),
                        (190, 'lion'), (217, 'platypus'), (232, 'airship'),
                        (233, 'balloon'), (319, 'orange'), (320, 'lemon'), (322, 'pineapple'),
                        (415, 'hummingbird'), (442, 'great_white_shark'), (682, 'viaduct'),
                        (342, 'cello'), (343, 'violin'), (291, 'tricycle'), (292, 'unicycle'), (277, 'moped')]

    class_indices_50 = [9, 12, 17, 18, 241,
                        22, 41, 44,
                        80, 81, 280,
                        281, 282, 450,
                        456, 231, 230,
                        545, 441, 399,
                        224, 237, 239,
                        243, 234, 606,
                        614, 629, 630,
                        632, 687, 690,
                        693, 922, 417,
                        190, 217, 232,
                        233, 319, 320,
                        322, 415, 442,
                        682, 342, 343, 291, 292, 277]

    class_labels = class_labels_50
    num_classes = len(class_labels)
    indices = class_indices_50

    #train_loader = get_imagenet64_dataloaders_train(batch_size, indices)
    #val_loader = get_imagenet64_dataloaders_val(1, indices)

    vgg_model, vgg_model_name = get_vgg19_bn(num_classes, True, False)
    #raw_6_model, raw_6_model_name = get_6_conv_2_fc(num_classes, True, False)
    dist_6_model, dist_6_model_name = get_6_conv_2_fc(num_classes, True, True)
    #raw_3_model, raw_3_model_name = get_3_conv_2_fc(num_classes, True, False, flag=False)
    dist_3_model, dist_3_model_name = get_3_conv_2_fc(num_classes, True, True, flag=False)

    print(vgg_model)
    path = '../generated/test/all_2048/'
    neuron_list = []

    print(([1 + (i - 5) / 50.0 for i in range(11)]))
    print((list(range(-10, 11)) + 5 * [0]))

    visualize_filter_fc(vgg_model, 6, [get_index_class('ant', class_labels)], path, class_labels, device)

    #print(lucent.modelzoo.util.get_model_layers(dist_6_model))
    for i in range(13, 21):
        neuron_list.append(['features_18', i])
        neuron_list.append(['features_11', i])
    #visualize_multiple_neurons(dist_6_model, neuron_list, path, device)

    # print(vgg_model)
    #img = Image.open("../generated/gazelle/gazelle_striped3.jpg")


    #train_acc, train_loss, val_acc, val_loss = get_train_plot_from_file(teacher_model_name, teacher_model_name, 'train')

    #draw_plot(train_acc, val_acc)
    #print(train_acc)
    #print(val_acc)
    #generate_activations_fc(student_model, student_model_name, class_labels_19, device)
    #generate_activations_fc(teacher_model, teacher_model_name, class_labels_19, device)

    #generate_model_compare()

    #generate_max_indices_img_dir(student_model, device, path)
    #visualize_multiple_neurons(teacher_model, neuron_list, path, device)
    #model, _ = get_3conv_2_fc(40, pretrained=True, distilled=False)
    #generate_model_compare()
    #
    #generate_max_indices_img_dir(model, device, path)
    #filters = [i for i in range(40)]
    #visualize_filter_fc(student_model, 6, filters, path, device)
    #generate_max_indices_img_dir(student_model, device, path)
    #generate_max_indices_img(student_model, image_path, dir_path, device, idx=0)
    #generate_max_indices_img(student_model, image_path2, dir_path, device, idx=1)

    #generate_max_indices(student_model, device, dir_path)


    #path = '../generated/test/'

    #visualize_filter(student_model, 49, filters[49], path, device)

    #filters = [i for i in range(256)]
    #visualize_filter_fc(teacher_model, 0, filters, f'../generated/3_conv_2_fc_40cl/fc_{0}', device)
    #visualize_diversity(student_model, 8, 10, f'../generated/3_conv_2_fc_40cl/features_8', 4, device)
    #obj = "features_8:105"
    #visualize_diversity(student_model, 8, 105, path, 4, device)

'''
r_all_data = []
    d_all_data = []

    for n in range(num_classes):
        or_idx = get_index_class(class_labels[n][1], class_labels)

        # for i, child in enumerate(raw_3_model.features):
        #    if type(child) == nn.Conv2d:
        #        child.register_forward_hook(save_output)

        # for i, child in enumerate(dist_3_model.features):
        #    if type(child) == nn.Conv2d:
        #        child.register_forward_hook(save_output)

        dims_t = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        #dims_s = [64, 64, 128, 128, 256, 256]
        dims_s = [64, 128, 256]

        d_data = []
        r_data = []
        for j in range(num_classes):
            # print(class_labels[j][1])
            lem_idx = get_index_class(class_labels[j][1], class_labels)

            t_or, t_lem, s_or, s_lem = get_shared_filers(raw_3_model, raw_3_model_name, dist_3_model, dist_3_model_name,
                                                         val_loader, or_idx, lem_idx, class_labels, dims_s, dims_s,
                                                         False, device)

            top = 32
            sum_t = 0
            sum_s = 0

            filters_teacher = []
            filters_student = []
            for i in range(len(t_or)):
                filters_t = []

                sorted_filters = np.argsort(t_or[i])[::-1]
                sorted_comp = np.argsort(t_lem[i])[::-1]
                counter = 0
                for y in range(top):
                    idx = sorted_filters[y]
                    if idx not in sorted_comp[:top]:
                        counter += 1
                        filters_t.append(idx)

                # print(f'Shared T in layer {i}: {counter}: {filters_t}')
                sum_t += counter
                filters_teacher.append(filters_t)

            for i in range(len(s_or)):
                filters_s = []

                sorted_filters = np.argsort(s_or[i])[::-1]
                sorted_comp = np.argsort(s_lem[i])[::-1]
                counter = 0
                for y in range(top):
                    idx = sorted_filters[y]

                    if idx not in sorted_comp[:top]:
                        counter += 1
                        filters_s.append(idx)

                # print(f'Shared S in layer {i}: {counter}: {filters_s}')
                sum_s += counter
                filters_student.append(filters_s)

            # res_r = sum_t / sum(dims_s)
            # res_d = sum_s / sum(dims_s)

            res_r = sum_t
            res_d = sum_s

            # print(res_r, res_d, '\n')
            # if class_labels[j][1] in ['ibex', 'great_white_shark']:
            r_data.append(res_r)
            d_data.append(res_d)

        r_all_data.append(sum(r_data) / num_classes)
        d_all_data.append(sum(d_data) / num_classes)

    print(np.mean(r_all_data), np.mean(d_all_data))
    plot_stats([i for i in range(num_classes)], r_all_data, d_all_data,
               "5 layer CNN non shared activations for all classes")

    
    
    
    val_transform1 = transforms.Compose([
        # torchvision.transforms.RandomAffine(30, translate=None, scale=(0.7, 1.3), shear=None, resample=0, fillcolor=0),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    root = '../data/imagenet_x64'
    dataset_val = Imagenet64(root=root, class_indices=indices, train=False, transform=val_transform1)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True)

 idx = get_index_class('gazelle', class_labels)

    img = Image.open("../generated/gazelle/fur_pattern.png")
    #img = Image.open("../generated/gazelle/test_image924.png")
    #img = Image.open("../generated/gazelle/test_image924_ed3.png")

    #img = Image.open("../generated/6_conv_distilled_50/fc_3_1_(12, 'gazelle').jpg")
    #ret = get_images_indices(idx, val_loader)
    ret = [img]
    rows = 2
    col = 4
    for i in range(0, len(ret)):
        image = ret[i]
        image = val_transform1(image).unsqueeze(0)
        image = image[:, :3, :, :]
        val_transform = transforms.Compose([

            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.4360, 0.4559, 0.4360], std=[0.2048, 0.1999, 0.2048]),
        ])
        img = val_transform(image[0]).unsqueeze(0)
        cam_img0, pred1 = grad_cam_image(vgg_model, image, img, idx, 35, class_labels, device)

        cam_img1, pred2 = grad_cam_image(raw_6_model, image, img, idx, -1, class_labels, device)

        #cam_img4, pred3 = grad_cam_image(raw_3_model, image, img, idx, -1, class_labels, device)

        cam_img2, pred4 = grad_cam_image(dist_6_model, image, img, idx, -1, class_labels, device)

        #cam_img3, pred5 = grad_cam_image(dist_3_model, image, img, idx, -1, class_labels, device)

        #if pred1 != idx and (pred4 == idx):
        fig = plt.figure(figsize=(10, 4))

        fig.add_subplot(rows, col, 1)
        plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))
        # plt.imshow(image)

        plt.xticks([])
        plt.yticks([])
        plt.title("Input image")

        fig.add_subplot(rows, col, 2)
        plt.imshow(cam_img0)
        plt.xticks([])
        plt.yticks([])
        plt.title("VGG19")

        fig.add_subplot(rows, col, 3)
        plt.imshow(cam_img1)
        plt.xticks([])
        plt.yticks([])
        plt.title("8 layer")

        #fig.add_subplot(rows, col, 4)
        #plt.imshow(cam_img4)
        #plt.xticks([])
        #plt.yticks([])
        #plt.title("5 layer")

        fig.add_subplot(rows, col, 4)
        plt.imshow(cam_img2)
        plt.xticks([])
        plt.yticks([])
        plt.title("8 layer dist")

        #fig.add_subplot(rows, col, 6)
        #plt.imshow(cam_img3)
        #plt.xticks([])
        #plt.yticks([])
        #plt.title("5 layer dist")

        plt.show()
        
        
###############################################################
    ret1, class_correct1, class_total1 = get_per_class_acc(vgg_model, val_loader, num_classes, class_labels,
                                                           device)
    ret2, class_correct2, class_total2 = get_per_class_acc(dist_6_model, val_loader, num_classes, class_labels,
                                                           device)
    ret3, class_correct3, class_total3 = get_per_class_acc(raw_6_model, val_loader, num_classes, class_labels,
                                                           device)
    ret4, class_correct4, class_total4 = get_per_class_acc(dist_3_model, val_loader, num_classes, class_labels,
                                                           device)
    ret5, class_correct5, class_total5 = get_per_class_acc(raw_3_model, val_loader, num_classes, class_labels,
                                                           device)

    for i in range(num_classes):
        print(class_labels[i][1], '&',
              round(class_correct1[i] / class_total1[i], 3), '&',
              round(class_correct2[i] / class_total2[i], 3), '&',
              round(class_correct3[i] / class_total3[i], 3), '&',
              round(class_correct4[i] / class_total4[i], 3), '&',
              round(class_correct5[i] / class_total5[i], 3), '\\\\ \\hline',)

    print(sum(ret1) / num_classes,
          sum(ret2) / num_classes,
          sum(ret3) / num_classes,
          sum(ret4) / num_classes,
          sum(ret5) / num_classes)
          ##############################################################
        

    student_model, student_model_name = get_6_conv_2_fc(num_classes, True, True)
    teacher_model, teacher_model_name = get_vgg19_bn(num_classes, True, False)

    dims_t = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    dims_s = [64, 64, 128, 128, 256, 256]

    t_or, t_lem, s_or, s_lem = get_shared_filers(teacher_model, teacher_model_name, student_model, student_model_name,
                                                 val_loader, 0, 1, class_labels, dims_s, dims_t, device)

    top = 10
    sum_t = 0
    sum_s = 0

    filters_teacher = []
    filters_student = []
    for i in range(len(t_or)):
        filters_t = []

        sorted_filters = np.argsort(t_or[i])[::-1]
        sorted_comp = np.argsort(t_lem[i])[::-1]
        counter = 0
        for y in range(top):
            idx = sorted_filters[y]
            if idx not in sorted_comp[:top]:
                counter += 1
                filters_t.append(idx)

        print(f'Shared T in layer {i}: {counter}: {filters_t}')
        sum_t += counter
        filters_teacher.append(filters_t)

    for i in range(len(s_or)):
        filters_s = []

        sorted_filters = np.argsort(s_or[i])[::-1]
        sorted_comp = np.argsort(s_lem[i])[::-1]
        counter = 0
        for y in range(top):
            idx = sorted_filters[y]

            if idx not in sorted_comp[:top]:
                counter += 1
                filters_s.append(idx)

        print(f'Shared S in layer {i}: {counter}: {filters_s}')
        sum_s += counter
        filters_student.append(filters_s)

    print(sum_t / sum(dims_t), sum_s / sum(dims_s))

    print(teacher_model)
    layers_t = []
    layers_s = []

    for i, layer in enumerate(teacher_model.features):
        if type(layer) == nn.Conv2d:
            layers_t.append(i)

    for i, layer in enumerate(student_model.features):
        if type(layer) == nn.Conv2d:
            layers_s.append(i)

    for i in range(len(layers_t)):
        path = f'../generated/{teacher_model_name}/layer_{i}/'
        visualize_filter(teacher_model, layers_t[i], filters_teacher[i], path, device)

    for i in range(len(layers_s)):
        path = f'../generated/{student_model_name}/layer_{i}/'
        visualize_filter(teacher_model, layers_s[i], filters_student[i], path, device)
        






    class_idx = get_index_class('ram', class_labels)
    ret = get_wrong_preds_class(get_confusion_matrix(teacher_model, val_loader, num_classes, device), num_classes, class_idx)

    print(class_labels[class_idx][1])
    print("Teacher:")
    for i in range(len(ret)):

        print(class_labels_50[ret[i][1]][1], ret[i][2])
    print("\n")

    ret = get_wrong_preds_class(get_confusion_matrix(student_model, val_loader, num_classes, device), num_classes, class_idx)

    print("Student:")
    for i in range(len(ret)):

        print(class_labels_50[ret[i][1]][1], ret[i][2])
        
        ##############################

    


    path = f"../generated/{student_model_name}/classifier"
    filters = [i for i in range(50)]
    visualize_filter_fc(student_model, 3, filters, path, class_labels, device)

    path = f"../generated/{teacher_model_name}/classifier"
    filters = [i for i in range(50)]
    visualize_filter_fc(teacher_model, 3, filters, path, class_labels, device)
    
    #######################################################################
        class_idx = get_index_class('fly', class_labels)
    ret = get_wrong_preds_class(get_confusion_matrix(teacher_model, val_loader, num_classes, device), num_classes, class_idx)

    print(class_labels[class_idx][1])
    print("Teacher:")
    for i in range(len(ret)):

        print(class_labels_50[ret[i][1]][1], ret[i][2])
    print("\n")

    ret = get_wrong_preds_class(get_confusion_matrix(student_model, val_loader, num_classes, device), num_classes, class_idx)

    print("Student:")
    for i in range(len(ret)):

        print(class_labels_50[ret[i][1]][1], ret[i][2])
        
##########################################################################################################################

        

    root = '../data/imagenet_x64'

    dataset_val = Imagenet64(root=root, class_indices=indices, train=False, transform=None)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    idx = 1
    ret = get_images_indices(idx, val_loader)

    rows = 2
    col = 4
    for i in range(len(ret)):
        fig = plt.figure(figsize=(10, 4))
        image = ret[i]
        fig.add_subplot(rows, col, 1)
        plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))

        val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.4360, 0.4559, 0.4360], std=[0.2048, 0.1999, 0.2048])
        ])

        img = val_transform(image[0]).unsqueeze(0)
        cam_img0 = grad_cam_image(vgg_model, image, img, idx, -3)
        fig.add_subplot(rows, col, 2)
        plt.imshow(cam_img0)
        cam_img1 = grad_cam_image(raw_6_model, image, img, idx, -1)
        fig.add_subplot(rows, col, 3)
        plt.imshow(cam_img1)
        cam_img4 = grad_cam_image(raw_3_model, image, img, idx, -1)
        fig.add_subplot(rows, col, 4)
        plt.imshow(cam_img4)
        cam_img2 = grad_cam_image(dist_6_model, image, img, idx, -1)
        fig.add_subplot(rows, col, 6)
        plt.imshow(cam_img2)
        cam_img3 = grad_cam_image(dist_3_model, image, img, idx, -1)
        fig.add_subplot(rows, col, 7)
        plt.imshow(cam_img3)

        plt.show()
        
#################################################################################################

    matrix1 = get_confusion_matrix(student_model, val_loader, num_classes, device).numpy()
    matrix2 = get_confusion_matrix(teacher_model, val_loader, num_classes, device).numpy()
    matrix3 = get_confusion_matrix(model, val_loader, num_classes, device).numpy()
    matrix4 = get_confusion_matrix(raw_model_6, val_loader, num_classes, device).numpy()
    matrix5 = get_confusion_matrix(raw_model_3, val_loader, num_classes, device).numpy()

    wrong_pred1 = get_wrong_pred(matrix1, num_classes)
    wrong_pred2 = get_wrong_pred(matrix2, num_classes)
    wrong_pred3 = get_wrong_pred(matrix3, num_classes)
    wrong_pred4 = get_wrong_pred(matrix4, num_classes)
    wrong_pred5 = get_wrong_pred(matrix5, num_classes)

    print_classnames(wrong_pred1, class_labels)
    print_classnames(wrong_pred2, class_labels)
    print_classnames(wrong_pred3, class_labels)
    print_classnames(wrong_pred4, class_labels)
    print_classnames(wrong_pred5, class_labels)

    plt.imshow(matrix1, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(matrix2, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(matrix3, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(matrix4, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(matrix5, cmap='hot', interpolation='nearest')
    plt.show()

####################################################################################################
    student_model, student_model_name = get_3_conv_2_fc(num_classes, False, False)
    teacher_model, teacher_model_name = get_6_conv_2_fc(num_classes, False, False)

    distiller = KdDistill(teacher_model,
                          student_model,
                          train_loader,
                          val_loader,
                          learning_rate=learning_rate,
                          decay_epoch=decay_epoch,
                          alpha=alpha,
                          temperature=temperature,
                          device=device,
                          teacher_model_name=teacher_model_name,
                          student_model_name=student_model_name,
                          mode=mode,
                          learning_rate_factor=learning_rate_factor,
                          default_factor_epoch=default_factor_epoch)

    distiller.run(epochs=epochs)
    distiller.eval(1)

    ############################################################################################
    
        class_indices_10 = [319, 320, 629, 630, 606, 614, 632, 230, 231, 280, 281, 236, 239]
    class_indices_15 = [319, 320, 629, 630, 606, 614, 632, 230, 231, 280, 281, 236, 239, 342, 343]
    class_indices_19 = [9, 12, 17, 18, 319, 320, 629, 630, 606, 614, 632, 230, 231, 280, 281, 236, 239, 342, 343]

    class_labels_10 = [(319, 'orange'), (320, 'lemon'),
                       (629, 'fly'), (630, 'bee'),
                       (606, 'garden_spider'),
                       (614, 'rock_crab'), (632, 'cricket'),
                       (230, 'airliner'), (231, 'warplane'),
                       (280, 'garbage_truck'), (281, 'pickup'),
                       (236, 'gondola'), (239, 'canoe')]

    class_labels_15 = [(319, 'orange'), (320, 'lemon'),
                       (629, 'fly'), (630, 'bee'),
                       (606, 'garden_spider'),
                       (614, 'rock_crab'), (632, 'cricket'),
                       (230, 'airliner'), (231, 'warplane'),
                       (280, 'garbage_truck'), (281, 'pickup'),
                       (236, 'gondola'), (239, 'canoe'),
                       (342, 'cello'), (343, 'violin')]

    class_labels_19 = [(9, 'ibex'), (12, 'gazelle'),
                       (17, 'great_dane'), (18, 'walker_hound'),
                       (319, 'orange'), (320, 'lemon'),
                       (629, 'fly'), (630, 'bee'),
                       (606, 'garden_spider'),
                       (614, 'rock_crab'), (632, 'cricket'),
                       (230, 'airliner'), (231, 'warplane'),
                       (280, 'garbage_truck'), (281, 'pickup'),
                       (236, 'gondola'), (239, 'canoe'),
                       (342, 'cello'), (343, 'violin')]

    cifar_labels = [(0, 'airplane'), (1, 'automobile'), (2, 'bird'), (3, 'cat'), (4, 'deer'),
                    (5, 'dog'), (6, 'frog'), (7, 'horse'), (8, 'ship'), (9, 'truck')]

    sim_classes = [(342, 'cello'), (343, 'violin'), (629, 'fly'), (632, 'cricket'), (319, 'orange'), (320, 'lemon')]
    sim_indices = [342, 343, 629, 632, 319, 320]
    
    
    ###############################################################################################
    
    
        ret1, class_correct1, class_total1 = get_per_class_acc(student_model, val_loader, num_classes, class_labels,
                                                           device)
    ret2, class_correct2, class_total2 = get_per_class_acc(teacher_model, val_loader, num_classes, class_labels,
                                                           device)
    ret3, class_correct3, class_total3 = get_per_class_acc(model, val_loader, num_classes, class_labels, device)
    ret4, class_correct4, class_total4 = get_per_class_acc(raw_model_6, val_loader, num_classes, class_labels,
                                                           device)
    ret5, class_correct5, class_total5 = get_per_class_acc(raw_model_3, val_loader, num_classes, class_labels,
                                                           device)

    for i in range(num_classes):
        print(class_labels[i][1])
        print(class_correct1[i] / class_total1[i], class_correct2[i] / class_total2[i],
              class_correct4[i] / class_total4[i],
              class_correct3[i] / class_total3[i], class_correct5[i] / class_total5[i])
    print(sum(ret1) / num_classes, sum(ret2) / num_classes, sum(ret4) / num_classes, sum(ret3) / num_classes,
          sum(ret5) / num_classes)
    
 
    
        model, modelname = get_6_conv_2_fc(num_classes, True, False)

    student_model.load_state_dict(
        torch.load(f'../models/6_conv_2_fc_15/best_weights_3_conv_2_fc_distilled_15'))
    teacher_model.load_state_dict(
        torch.load(f'../models/6_conv_2_fc_15/best_weights_6_conv_distilled_15'))

    ret1, class_correct1, class_total1 = get_per_class_acc(student_model, val_loader, num_classes, class_labels_15, device)
    ret2, class_correct2, class_total2 = get_per_class_acc(teacher_model, val_loader, num_classes, class_labels_15, device)
    ret3, class_correct3, class_total3 = get_per_class_acc(model, val_loader, num_classes, class_labels_15, device)

    for i in range(num_classes):
        print(class_labels_15[i][1])
        print(class_correct1[i] / class_total1[i], class_correct2[i] / class_total2[i], class_correct3[i] / class_total3[i])
    print((sum(ret1)/num_classes), sum(ret2)/num_classes, sum(ret3)/num_classes)
    
    

    student_model, student_model_name = get_3conv_2_fc_13(num_classes, False, False)
    teacher_model, teacher_model_name = get_vgg19_bn(num_classes, True, False)

    matrix_vgg = get_per_class_acc(teacher_model, val_loader, num_classes, class_labels_19, device)

    teacher_model, teacher_model_name = get_3conv_2_fc_13(num_classes, True, True)

    matrix_3_con = get_per_class_acc(teacher_model, val_loader, num_classes, class_labels_19, device)

    teacher_model, teacher_model_name = get_6_conv_2_fc(num_classes, True, True)

    matrix_6_con = get_per_class_acc(teacher_model, val_loader, num_classes, class_labels_19, device)

    for i in range(num_classes):
        print(class_labels_19[i][1])
        print('{:.2f} \t {:.2f} \t {:.2f}'.format(round(matrix_vgg[i], 2), round(matrix_6_con[i], 2),
                                                  round(matrix_3_con[i], 2)))

    print(sum(matrix_vgg) / num_classes, sum(matrix_6_con) / num_classes, sum(matrix_3_con) / num_classes)


    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = teacher_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    

    
    
    model = teacher_model
    model_name = teacher_model_name
    path = f"../generated/{model_name}/classifier/"
    filters = [i for i in range(40)]
    visualize_filter_fc(model, 6, filters, path, class_labels, device)


    
    model = teacher_model
    target_layer = model.features[-1]
    method = 'gradcam'  # Can be gradcam/gradcam++/scorecam

    images = get_images_indices(2, val_loader)
    print(len(images))
    for i in range(10):
        imshow(images[i])
    input_tensor = image_loader()# Create an input tensor image for your model..

    cam = CAM(model=model, target_layer=target_layer, use_cuda=args.use_cuda)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=1, method=method)
    visualization = show_cam_on_image(rgb_img, grayscale_cam)

    t_or, t_lem, s_or, s_lem = get_shared_filers(teacher_model, student_model, val_loader, 4, 5, class_labels_19, device)

    top = 20
    for i, filter_count in enumerate(t_or):

        sorted_filters = np.argsort(filter_count)[::-1]
        sorted_comp = np.argsort(t_lem[i])[::-1]
        counter = 0
        for y in range(top):
            idx = sorted_filters[y]
            for j in range(top):
                if idx == sorted_comp[j]:
                    counter += 1

        print(f'Shared T in layer {i}: {counter}')

        sorted_filters = np.argsort(s_or[i])[::-1]
        sorted_comp = np.argsort(s_lem[i])[::-1]
        counter = 0
        for y in range(top):
            idx = sorted_filters[y]
            for j in range(top):
                if idx == sorted_comp[j]:
                    counter += 1

        print(f'Shared S in layer {i}: {counter}')


  




    
     lays, filts = [], []
   
    lay, fil = get_layer_filters(f'{dir_path}/layer_filters.txt')
    lays.append(lay)
    filts.append(fil)

    lay, fil = get_layer_filters(f'{dir_path}/layer_filters1.txt')
    lays.append(lay)
    filts.append(fil)

    for j in lays[0]:
        idx1 = filts[0][j]
        idx2 = filts[1][j]
        counter = 0
        for i in range(len(idx1)):
            if idx1[i] in idx2:
                counter += 1

        print(j, counter)



    
        # Marine animals
    # 22 killer whale
    # 450 goldfish
    # 456 sturgeon

    # Land animals
    # 41 dalmatian
    # 44 skunk
    # 80 zebra
    # 81 ram

    # Birds
    # 399 vulture
    # 441 albatross
    # 417 toucan

    # Aircraft
    # 230 airliner
    # 231 warplane
    # 234 space_shuttle

    # Ground vehicles
    # 257 passenger_car
    # 281 pickup
    # 282 tow_truck
    # 545 crane

    # Insects
    # 629 fly
    # 630 bee
    # 632 cricket

    # 606 garden_spider
    # 614 rock_crab
    # 224 ant

    # Boats
    # 237 speedboat
    # 239 canoe
    # 243 container_ship

    # Buildings
    # 687 library
    # 690 church
    # 693 planetarium
    # 922 maze

    # 190 lion
    # 217 platypus

    # 232 airship
    # 233 balloon

    # 319 orange
    # 320 lemon
    # 322 pineapple

    # 415 hummingbird
    # 442 great_white_shark
    # 682 viaduct


   

    '''

main()
