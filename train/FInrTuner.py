import copy
import torch


def set_module(module, module_name, new_module):
    if isinstance(module_name, str):
        module_name = module_name.split('.')

    if len(module_name) == 1:
        return setattr(module, module_name[0], new_module)
    else:
        module = getattr(module, module_name[0])
        return set_module(module, module_name[1:], new_module)


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = True


class FineTunedModel(torch.nn.Module):
    def __init__(self,
                 model,
                 num_layers: int = 5,
                 ):
        super().__init__()
        self.model = model
        self.ft_modules = {}
        self.orig_modules = {}
        freeze(self.model)
        target_module = {f'layers.{i}.self_attn.' for i in range(num_layers)}
        print(target_module)
        for module_name, module in model.named_modules():
            if not any(module_name.startswith(target) for target in target_module):
                continue
            print(module_name)
            ft_module = copy.deepcopy(module)
            self.orig_modules[module_name] = module
            self.ft_modules[module_name] = ft_module
            unfreeze(ft_module)

        self.ft_modules_list = torch.nn.ModuleList(self.ft_modules.values())
        self.orig_modules_list = torch.nn.ModuleList(self.orig_modules.values())

    @classmethod
    def from_checkpoint(cls, model, checkpoint, train_method):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)
        modules = [f"{key}$" for key in list(checkpoint.keys())]
        ftm = FineTunedModel(model, train_method=train_method)
        ftm.load_state_dict(checkpoint)
        return ftm

    def __enter__(self):
        for key, ft_module in self.ft_modules.items():
            set_module(self.model, key, ft_module)

    def __exit__(self, exc_type, exc_value, tb):
        for key, module in self.orig_modules.items():
            set_module(self.model, key, module)

    def parameters(self):
        parameters = []
        for ft_module in self.ft_modules.values():
            parameters.extend(list(ft_module.parameters()))
        return parameters

    def state_dict(self):
        state_dict = {key: module.state_dict() for key, module in self.ft_modules.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        for key, sd in state_dict.items():
            if key in self.ft_modules:
                self.ft_modules[key].load_state_dict(sd)
            else:
                print(f"Key {key} not found in ft_modules. Skipping...")
