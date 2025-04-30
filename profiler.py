import argparse
from segmentation_models_pytorch import Unet
import torch
import torch.nn as nn

def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

# The nn.Identity layer performs no computations—it simply passes its input through unchanged. But for profiling consistency, 
# we can still implement a hook function for it that sets total_ops to zero and counts the number of elements it processes (if needed).
def count_identity(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = 0  # No computation in Identity
    m.total_ops += torch.Tensor([int(total_ops)])

def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        elif isinstance(m, nn.Identity):
            m.register_forward_hook(count_identity)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(1,*input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    return total_ops, total_params

def main(args):
    # Model setup (on CPU)
    device = torch.device("cpu")  # Force CPU
    model = Unet(encoder_name="densenet121", in_channels=2, classes=2).to(device)

    # Don't wrap with DataParallel on CPU
    # Load model parameters with map_location for CPU
    state_dict=torch.load("D:/ComputerEngineering/Fourth_Year/Second Term/Satellites/New folder/Cloud-Segmentation/new/unet_epochs30_lr0.0005_pretrained_densenet_lr_dice_loss_with_threshold/best_model (3).pth", map_location=device)
    # Strip "module." from keys if present (i.e., saved from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    total_ops, total_params = profile(model, args.input_size)
    print("#Ops: %f GOps"%(total_ops/1e9))
    print("#Parameters: %f M"%(total_params/1e6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pytorch model profiler")
    parser.add_argument("model", help="model to profile")
    parser.add_argument("input_size", nargs='+', type=int,
                        help="input size to the network")
    args = parser.parse_args()
    main(args)