import torch
import numpy as np

def get_clock_time():
    from time import gmtime, strftime
    result = strftime("%H:%M:%S", gmtime())
    return result

def get_datetime():
    from time import gmtime, strftime
    result = strftime("%Y-%m-%d - %H-%M-%S", gmtime())
    return result
    
def cprint(s: str, file=None):    
    clock = get_clock_time()
    print("{} | {}".format(clock, s))

    if file:
        file.write("{} | {}\n".format(clock, s))
        file.flush()
    
def create_logfile(filepath: str):
    return open(filepath, "w")

def constant_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def StatusString(pretext = None, data=None, file=None):
    if pretext == None:
        outstring = ""
    else: outstring = pretext + " \t| "
    for key in data.keys():
        outstring += "{}: {:.4f}, \t".format(key, np.mean(data[key]))
    return outstring

        #cprint("training | elbo: {:2f}, mse_loss: {:.4f}, kl: {:.2f}:".format(np.mean(training_epoch_data["elbo"]), np.mean(training_epoch_data["mse_loss"]), np.mean(training_epoch_data["kl"])), logfile)
def DiscStatusString(pretext = None, data=None, file=None):
    if pretext == None:
        outstring = ""
    else: outstring = pretext + " \t| "
    for key in data.keys():
        outstring += "{}: {:.4f}, \t".format(key, np.sum(data[key]))
    return outstring
