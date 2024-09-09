cfg = {
    "alpha": {"value": 0.1, "help": "error rate for conformal prediction"},
    "epsilon": {"value": 0.125, "help": "L2 bound on the adversarial noise"},
    "n_experiments": {"value": 50, "help": "number of experiments to estimate coverage"},
    "ratio": {"value": 2, "help": "ratio between adversarial noise bound to smoothed noise"},
    "n_smooth": {"value": 256, "help": "number of samples used for smoothing"},
    "N_steps": {"value": 20, "help": "number of gradiant steps for PGD attack"},
    "normalized": {"value": False},
    "model_type": {"value": "ResNet", "help": "type of model to use"},
    "n_test": {"value": 10000, "help": "number of test points"},
    "num_of_classes": {"value": 10, "help": "number of classes"},
    "min_pixel_value": {"value": 0.0, "help": "minimum pixel value"},
    "max_pixel_value": {"value": 1.0, "help": "maximum pixel value"},
    "directory": {"value": "examples/adversarial/adversarial_examples", "help": "directory to store adversarial examples and noises"},

}