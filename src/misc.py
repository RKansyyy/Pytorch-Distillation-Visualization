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


def get_training_data(learning_rate, mode):

    img = np.array(Image.open("dog_cat.png"), np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)
    epochs = 3
    #learning_rate = 0.01
    decay_epoch = 5
    alpha = 0.1
    temperature = 5
    batch_size = 32
    learning_rate_factor = 0.5
    default_factor_epoch = 15
    #mode = 'distill'
    class_labels = [(22, 'killer_whale'), (41, 'dalmatian'), (44, 'skunk'),
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

    class_indices = [22, 41, 44,
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

    num_classes = len(class_labels)
    indices = class_indices

    student_model, student_model_name = get_5conv_2_fc_13(num_classes, False, False)
    teacher_model, teacher_model_name = get_5conv_2_fc_13(num_classes, False, False)

    train_loader = get_imagenet64_dataloaders_train(batch_size, indices)
    val_loader = get_imagenet64_dataloaders_val(batch_size, indices)

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

    train_acc, train_loss = distiller.train(epoch=1, eval_flag=True)

    return train_acc


def generate_max_indices_img(model, image, dir_name, device, idx=0):

    if os.path.exists(f'../activations/model_activations/'):
        shutil.rmtree(f'../activations/model_activations/')

    generate_activations(model, image, device)
    act, means, names = get_max_activations(f'../activations/model_activations/', dir_name, idx=idx, top=5)
    shutil.rmtree(f'../activations/model_activations/')

    return names, act, means


def generate_max_indices_img_dir(model, dir_name, dir_path, device):

    if os.path.exists(f'../activations/model_activations/'):
        shutil.rmtree(f'../activations/model_activations/')

    for i, child in enumerate(model.features):
        if type(child) == nn.Conv2d:
            child.register_forward_hook(save_output)

    images = get_image_paths(dir_path)

    model.to(device).eval()

    for i, image in enumerate(images):
        generate_activations(model, image, device)
        name = image.__str__().split('\\')[-1].split('.')[0]
        print(name)
        get_max_activations(f'../activations/model_activations/', dir_name, idx=name, top=10)
        shutil.rmtree(f'../activations/model_activations/')


def generate_activations(model, img_path, device):

    image = image_loader(f"{img_path}", 64, device)
    model(image)


def save_output(self, input, output):

    if not os.path.exists(f'../activations/model_activations/'):
        os.makedirs(f'../activations/model_activations/')

    f = open(f'../activations/model_activations/{id(self)}_{self.__class__.__name__}.txt', '+w')

    data = output.data.detach().cpu().numpy()[0]

    for j in range(len(data)):
        res = 0
        for i in range(len(data[j])):
            res = sum(data[j][i])
        f.write(f'{res},')

    f.write('\n')
    f.close()


def get_max_activations(basepath, dir_name, idx, top=3):

    if not os.path.exists(f'../activations/{dir_name}/'):
        os.makedirs(f'../activations/{dir_name}/')

    time = []
    files = []
    for file in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, file)):
            fname = pathlib.Path(os.path.join(basepath, file))
            time.append(fname.stat().st_mtime)
            files.append(os.path.join(basepath, file))

    indices = np.argsort(time, axis=0)

    max_activations = []
    module_names = []
    means = []

    f_ind = open(f'../activations/{dir_name}/layer_filters_{idx}.txt', 'w+')

    for j, idx in enumerate(indices):

        f = open(files[idx])
        data = f.read()
        values = data.split(',')

        temp_min = 0
        topk_list = []
        mean = 0

        values.pop()

        for i, value in enumerate(values):

            temp = float(value)
            mean += temp

            if len(topk_list) < top:
                topk_list.append([temp, i])
                temp_min = get_min(topk_list)

            else:
                if temp > temp_min[0]:
                    topk_list.remove(temp_min)
                    topk_list.append([temp, i])
                    temp_min = get_min(topk_list)

        if len(topk_list) > 0:
            f_ind.write(f'{j}:')
            for value in topk_list:
                f_ind.write(f'{value[1]},')

            f_ind.write('/')

        f.close()

        mean /= len(values)
        max_activations.append(topk_list)
        module_names.append(files[idx].split('/')[-1].split('_')[-1].split('.')[0])
        means.append(round(mean, 0))

    f_ind.close()

    return max_activations, means, module_names


def get_min(values):

    temp_min = values[0][0]
    idx = values[0][1]
    for i in range(1, len(values)):
        if values[i][0] < temp_min:
            temp_min = values[i][0]
            idx = values[i][1]

    return [temp_min, idx]


def visualize(path, dir_name, model, device):

    model.to(device).eval()
    layers, filters = get_layer_filters(path)
    for i, layer in enumerate(layers):
        visualize_filter(model, layer, filters[i], f'../generated/3_conv_2_fc_40cl/{dir_name}/features_{layer}')


def get_image_paths(dir_path):

    images = []
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            name = os.path.join(dir_path, file)
            if '.jpg' in name:
                fname = pathlib.Path(os.path.join(dir_path, file))
                images.append(fname)

    return images


def get_txt_paths(dir_path):

    files = []
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            name = os.path.join(dir_path, file)
            if '.txt' in name:
                fname = pathlib.Path(os.path.join(dir_path, file))
                files.append(fname)

    return files


def prepare_activation_data(dir_path):

    files = []
    names = []
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            name = os.path.join(dir_path, file)
            fname = pathlib.Path(os.path.join(dir_path, file))
            files.append(fname)
            names.append(name)

    save_dir = '../activations/activation_compare/'

    class_names = []
    class_name_counters = []
    class_name_means = []

    for file in files:
        layers, filters = get_layer_filters(file)

        name = file.__str__().split('\'')[1]
        counters = []
        comp_names = []
        layers_shared = []

        mean = 0
        for j, comp_file in enumerate(files):
            layers_comp, filters_comp = get_layer_filters(comp_file)
            comp_name = comp_file.__str__().split('_')[-1].split('.')[0]
            counter = 0
            layer_shared = []
            for layer in layers:

                lay_counter = 0
                x = filters[layer]
                y = filters_comp[layer]

                for i in range(len(x)):
                    if x[i] in y:
                        lay_counter += 1

                counter += lay_counter
                layer_shared.append(lay_counter)

            class_mean = counter/len(layers)
            mean += counter
            counters.append(counter)
            comp_names.append(comp_name)
            layers_shared.append(layer_shared)

        mean /= len(files)
        class_names.append(name)
        class_name_means.append(mean)
        class_name_counters.append(counters)

    return class_names, class_name_means, class_name_counters


def generate_model_compare():

    dir_path1 = '../activations/cifar_3_conv/'
    dir_path2 = '../activations/cifar_3_conv_dist/'

    names, means, counters = prepare_activation_data(dir_path1)
    names_vg, means_vg, counters_vg = prepare_activation_data(dir_path2)

    labels = []
    teacher_means = []
    distilled_means = []
    for i in range(len(names)):
    #if names[i] == 'truck' or names[i] == 'pickup' or names[i] == 'viaduct':
        print('\n')
        print(names[i])
        print(names[i])
        print('\n')

        shared_teacher = []
        shared_distilled = []
        labels_inner = []

        for j in range(len(names)):
            #if names[j] == 'pickup' or names[j] == 'truck' or names[j] == 'viaduct':
            print(names[j])
            print('Teacher', 'Distilled')
            print(round(means[i], 0), '\t', round(means_vg[i], 0))
            print(counters[i][j], '\t\t', counters_vg[i][j])
            print('\n')
            shared_teacher.append(counters[i][j])
            shared_distilled.append(counters_vg[i][j])
            labels_inner.append(names[j])

        plot_stats(labels_inner, shared_teacher, shared_distilled, names[i])

        teacher_means.append(round(means[i], 0))
        distilled_means.append(round(means_vg[i], 0))
        labels.append(names[i])

    plot_stats(labels, teacher_means, distilled_means, 'Mean shared filters per class')


def plot_stats(labels, men_means, women_means, title):

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_means, width, label='Raw data')
    rects2 = ax.bar(x + width / 2, women_means, width, label='Distilled')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Non shared activations')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


def generate_activations_fc(teacher_model, teacher_model_name, class_labels_10, device):

    path = f'../generated/{teacher_model_name}/fc_3/'
    filters = [i for i in range(len(class_labels_10))]
    visualize_filter_fc(teacher_model, 6, filters, path, class_labels_10, device)
    generate_max_indices_img_dir(teacher_model, teacher_model_name, path, device)


def get_images_indices(idx, dataloader):

    images = []
    for i, (image, label) in enumerate(dataloader):

        if label == idx:
            images.append(image)

    return images


def generate_max_indices_images(model, dir_name, images, name, hook_flag, device):

    if os.path.exists(f'../activations/model_activations/'):
        shutil.rmtree(f'../activations/model_activations/')

    if hook_flag:
        for i, child in enumerate(model.features):
            if type(child) == nn.Conv2d:
                child.register_forward_hook(save_output)

    model.to(device).eval()

    for i, image in enumerate(images):
        model(image.to(device))
        print(i)
        get_max_activations(f'../activations/model_activations/', dir_name, idx=f'{name}_{i}', top=10)
        shutil.rmtree(f'../activations/model_activations/')


def prepare_activation_data_image_class(dir_path, dims):

    files = []
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            fname = pathlib.Path(os.path.join(dir_path, file))
            files.append(fname)

    layers_filter_counts = []
    for value in dims:
        filter_list = [0]*value
        layers_filter_counts.append(filter_list)

    for i in range(len(files)):
        file = files[i]
        layers, filters = get_layer_filters(file)

        for layer in layers:
            x = filters[layer]

            for filter in x:
                layers_filter_counts[layer][filter] += 1

    return layers_filter_counts


def generate_model_compare_images_class():

    dir_path1 = '../activations/vgg19_bn_imagenet_19/'
    dir_path2 = '../activations/vgg19_distilled_19/'

    means, counters = prepare_activation_data_image_class(dir_path1)
    means_vg, counters_vg = prepare_activation_data_image_class(dir_path2)

    labels = []
    teacher_means = []
    distilled_means = []
    for i in range(len(means)):

        shared_teacher = []
        shared_distilled = []
        labels_inner = []

        for j in range(len(means)):
            print('Teacher', 'Distilled')
            print(round(means[i], 0), '\t', round(means_vg[i], 0))
            print(counters[i][j], '\t\t', counters_vg[i][j])
            print('\n')
            shared_teacher.append(counters[i][j])
            shared_distilled.append(counters_vg[i][j])
            labels_inner.append(j)

        plot_stats(labels_inner, shared_teacher, shared_distilled, i)

        teacher_means.append(round(means[i], 0))
        distilled_means.append(round(means_vg[i], 0))
        labels.append(i)

    plot_stats(labels, teacher_means, distilled_means, 'Mean shared filters per class')


def get_shared_filers(teacher_model, teacher_model_name, student_model, student_model_name,
                      val_loader, idx, idx_comp, class_labels, dims_t, dims_s, hook_flag, device):

    dir_name = f'{teacher_model_name}_{class_labels[idx][1]}'
    path = f'../activations/{dir_name}/'

    images = None
    if not os.path.exists(path):
        images = get_images_indices(idx, val_loader)
        generate_max_indices_images(teacher_model, dir_name, images, class_labels[idx][1], hook_flag, device)

    shared = prepare_activation_data_image_class(path, dims_t)

    dir_name = f'{student_model_name}_{class_labels[idx][1]}'
    path = f'../activations/{dir_name}/'
    if not os.path.exists(path):
        if images is None:
            images = get_images_indices(idx, val_loader)
        generate_max_indices_images(student_model, dir_name, images, class_labels[idx][1], hook_flag, device)

    shared_dist = prepare_activation_data_image_class(path, dims_s)

    images = None
    dir_name = f'{teacher_model_name}_{class_labels[idx_comp][1]}'
    path = f'../activations/{dir_name}/'
    if not os.path.exists(path):
        images = get_images_indices(idx_comp, val_loader)
        generate_max_indices_images(teacher_model, dir_name, images, class_labels[idx_comp][1], hook_flag, device)

    shared_comp = prepare_activation_data_image_class(path, dims_t)

    dir_name = f'{student_model_name}_{class_labels[idx_comp][1]}'
    path = f'../activations/{dir_name}/'
    if not os.path.exists(path):
        if images is None:
            images = get_images_indices(idx_comp, val_loader)
        generate_max_indices_images(student_model, dir_name, images, class_labels[idx_comp][1], hook_flag, device)

    shared_dist_comp = prepare_activation_data_image_class(path, dims_s)

    return shared, shared_comp, shared_dist, shared_dist_comp


def draw_plot(y1, y2, mean1, mean2):

    x1 = [i for i in range(len(y1))]

    # plotting the line 1 points
    plt.plot(x1, y1, label="Network trained on data", marker='o')
    # line 2 points
    # plotting the line 2 points
    plt.plot(x1, y2, label="Distilled network", marker='o')

    x_mean1, y_mean1 = [0, len(y1)], [mean1, mean1]
    x_mean2, y_mean2 = [0, len(y1)], [mean2, mean2]
    plt.plot(x_mean1, y_mean1, x_mean2, y_mean2)

    plt.xlabel('Classes')
    # Set the y axis label of the current axis.
    plt.ylabel('Non shared channels')
    # Set a title of the current axes.
    plt.title('Non shared channels per class for orange in 5 layer CNN')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


def get_train_plot_from_file(model_name, submodel, mode):

    f = open(f'../models/{model_name}/{mode}_{submodel}_log.txt')
    s = f.read()

    s = s.split(';')
    s.pop()

    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []

    for i in range(len(s)):
        temp = s[i].split(':')[0]
        values = temp.split(',')

        if not (len(values) < 4):
            train_acc.append(float(values[0]))
            train_loss.append(float(values[1]))
            val_acc.append(float(values[2]))
            val_loss.append(float(values[3]))

    return train_acc, train_loss, val_acc, val_loss


def get_per_class_acc(teacher_model, val_loader, num_classes, class_labels, device):

    teacher_model.to(device)
    teacher_model.eval()
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = teacher_model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    ret = []
    for i in range(num_classes):
        ret.append(class_correct[i] / class_total[i])

    return ret, class_correct, class_total


def grad_cam_image(model, image, input_tensor, img_class, layer, class_labels, device):

    if not next(model.parameters()).is_cuda:
        model.to(device)

    model.eval()
    pred = model(input_tensor.to(device))
    preds = pred.cpu().detach().numpy()
    print(preds)
    sorted_ids = preds.argsort()
    print(sorted_ids)
    for i in range(1,10):
        print(sorted_ids[0][-i])
        print(class_labels[sorted_ids[0][-i]][1])
    class_pred = np.argmax(pred.cpu().detach().numpy())
    print(class_pred)
    print(class_labels[class_pred][1], '\n')

    target_layer = model.features[layer]
    method = 'gradcam++'  # Can be gradcam/gradcam++/scorecam

    cam = CAM(model=model, target_layer=target_layer, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, method=method)
    visualization = show_cam_on_image(np.transpose(image[0].numpy(), (1, 2, 0)), grayscale_cam)
    #visualization = show_cam_on_image(np.asarray(image), grayscale_cam)

    return visualization, class_pred
