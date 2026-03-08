activations = {}

def make_hook(name):
    def hook_fn(module, input, output):
        activations[name] = output.detach()
    return hook_fn

