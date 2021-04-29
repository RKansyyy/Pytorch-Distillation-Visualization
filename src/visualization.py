from lucent.optvis import render, param, transform, objectives
import torch
import os
import numpy as np


def visualize_filter(model, layer, filters, path, device):

    model.to(device).eval()

    if not os.path.exists(f'{path}/'):
        os.makedirs(f'{path}/')

    param_f = lambda: param.image(128, fft=False, decorrelate=False)

    for i in range(len(filters)):
        layer_name = f'features_{layer}:{filters[i]}'
        image_name = f"{path}/{layer_name}.jpg"
        _ = render.render_vis(model, layer_name, param_f, save_image=True,
                              image_name=image_name.replace(':', '_'), show_image=False)


def visualize_layer(model, layer, path, device):

    model.to(device).eval()

    if not os.path.exists(f'{path}/'):
        os.makedirs(f'{path}/')

    layer_name = f'features_{layer}'
    image_name = f"{path}/{layer_name}.jpg"
    _ = render.render_vis(model, layer_name, fixed_image_size=64, save_image=True,
                          image_name=image_name.replace(':', '_'), show_image=False)


def visualize_multiple_neurons(model, neuron_names, save_path, device):

    model.to(device).eval()

    param_f = lambda: param.image(512, batch=1)

    neuron1 = neuron_names[0]
    print(neuron1)

    obj = objectives.channel(neuron_names[0][0], neuron_names[0][1])
    for i in range(1, len(neuron_names)):
        obj += objectives.channel(neuron_names[i][0], neuron_names[i][1])

    _ = render.render_vis(model, obj, param_f, save_image=True,
                          image_name=f'{save_path}_placeholder.jpg', show_image=False)


def visualize_filter_fc(model, layer, filters, path, class_labels, device):

    model.to(device).eval()

    if not os.path.exists(f'{path}/'):
        os.makedirs(f'{path}/')

    for i in range(len(filters)):
        layer_name = f'classifier_{layer}:{filters[i]}'
        image_name = f"{path}/{layer_name}_{class_labels[filters[i]]}.jpg"
        _ = render.render_vis(model, layer_name, fixed_image_size=64, thresholds=(2048,), save_image=True,
                              image_name=image_name.replace(':', '_'), show_image=False)


def visualize_filter_fc2(model, layer, filters, path, class_labels, device):

    model.to(device).eval()

    if not os.path.exists(f'{path}/'):
        os.makedirs(f'{path}/')

    param_f = lambda: param.image(64, fft=False, decorrelate=False)

    for i in range(len(filters)):
        layer_name = f'classifier_{layer}:{filters[i]}'
        image_name = f"{path}/{layer_name}_{class_labels[filters[i]]}.jpg"
        _ = render.render_vis(model, layer_name, param_f, fixed_image_size=64, save_image=True,
                              image_name=image_name.replace(':', '_'), show_image=False)

def visualize_diversity(model, layer, filter, path, batch_size, device):

    model.to(device).eval()
    if not os.path.exists(f'{path}/'):
        os.makedirs(f'{path}/')

    batch_param_f = lambda: param.image(128, batch=batch_size)
    layer_name = f'features_{layer}:{filter}'

    obj = objectives.channel(f"features_{layer}", filter) - 1e2 * objectives.diversity(f"features_{layer}")

    image_name = f"{path}/{layer_name}_diversity.jpg"

    _ = render.render_vis(model, obj, batch_param_f,  save_image=True,
                          image_name=image_name.replace(':', '_'), show_image=False)


def visualize_diversity_fc(model, layer, filter, path, batch_size, device):

    model.to(device).eval()

    if not os.path.exists(f'{path}/'):
        os.makedirs(f'{path}/')

    batch_param_f = lambda: param.image(128, batch=batch_size)
    layer_name = f'fc_{layer}:{filter}'

    obj = objectives.channel(f"fc_{layer}", filter) - 1e2 * objectives.diversity(f"fc_{layer}")

    image_name = f"{path}/{layer_name}_diversity.jpg"

    _ = render.render_vis(model, obj, batch_param_f, save_image=True,
                          image_name=image_name.replace(':', '_'), show_image=False)


def visualize_jitter(model, obj, path, device):
    model.to(device).eval()
    jitter_only = [transform.jitter(8)]

    param_f = lambda: param.image(512, fft=False, decorrelate=True)

    _ = render.render_vis(model, obj, param_f, transforms=jitter_only, save_image=True,
                          image_name=path, show_image=False)


def visualize_grad_fourier(model, obj, path, device):
    model.to(device).eval()
    param_f = lambda: param.image(128, fft=True, decorrelate=False)
    _ = render.render_vis(model, obj, param_f, transforms=[], save_image=True,
                          image_name=path, show_image=False)


def visualize_decorrelate(model, obj, path, device):
    model.to(device).eval()

    param_f = lambda: param.image(128, fft=False, decorrelate=True)
    _ = render.render_vis(model, obj, param_f, transforms=[], save_image=True,
                          image_name=path, show_image=False)


def visualize_cppn(model, obj, path, device):
    model.to(device).eval()

    cppn_param_f = lambda: param.cppn(64)
    # We initialize an optimizer with lower learning rate for CPPN
    cppn_opt = lambda params: torch.optim.Adam(params, 5e-3)
    _ = render.render_vis(model, obj, cppn_param_f, cppn_opt, transforms=[], save_image=True,
                          image_name=path, show_image=False)


def feature_inversion(model, device, img, layer=None, n_steps=512, cossim_pow=0.0):
    # Convert image to torch.tensor and scale image
    img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
    upsample = torch.nn.Upsample(224)
    img = upsample(img)

    obj = objectives.Objective.sum([
        1.0 * dot_compare(layer, cossim_pow=cossim_pow),
        objectives.blur_input_each_step(),
    ])

    # Initialize parameterized input and stack with target image
    # to be accessed in the objective function
    params, image_f = param.image(224)

    def stacked_param_f():
        return params, lambda: torch.stack([image_f()[0], img])

    transforms = [
        transform.pad(8, mode='constant', constant_value=.5),
        transform.jitter(8),
        transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1] * 4),
        transform.random_rotate(list(range(-5, 5)) + [0] * 5),
        transform.jitter(2),
    ]

    _ = render.render_vis(model, obj, stacked_param_f, transforms=transforms, thresholds=(n_steps,), show_image=False,
                          progress=False)


@objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
    def inner(T):
        dot = (T(layer)[batch] * T(layer)[0]).sum()
        mag = torch.sqrt(torch.sum(T(layer)[0]**2))
        cossim = dot/(1e-6 + mag)
        return -dot * cossim ** cossim_pow
    return inner
