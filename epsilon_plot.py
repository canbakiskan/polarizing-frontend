
import subprocess
import numpy as np

arglist=["python",
"attack.py",
"--classifier_name",
"fashion_polarized_model_epoch59",
"--model",
"Polarization_quantization_model",
"--num_restarts",
"20",
"--num_steps",
"100",
"--batch_size",
"5000",
"--eps",
"0.1",
"--step_size",
"0.01",
"--attack_method",
"pgd",
"--jump",
"0.2",
"--dataset",
"fashion"]


with open("./epsilon_fashion_bim_plain.txt", "a") as output:
    for epsilon in np.arange(0,0.4,0.02):
        output.write("\nepsilon = "+str(epsilon)+'\n')
        output.flush()
        arglist[13]=str(epsilon)
        arglist[15]=str(epsilon/10)
        subprocess.call(arglist, stdout=output);

arglist[7]=str(20)
arglist[9]=str(100)


with open("./epsilon_fashion_pgd_plain.txt", "a") as output:
    for epsilon in np.arange(0,0.4,0.02):
        output.write("\nepsilon = "+str(epsilon)+'\n')
        output.flush()
        arglist[13]=str(epsilon)
        arglist[15]=str(epsilon/10)
        subprocess.call(arglist, stdout=output);



arglist = [
    "python",
    "attack.py",
    "--classifier_name",
    "mnist_polarized_model_epoch59",
    "--model",
    "Polarization_quantization_model",
    "--num_restarts",
    "20",
    "--num_steps",
    "100",
    "--batch_size",
    "5000",
    "--eps",
    "0.3",
    "--step_size",
    "0.03",
    "--attack_method",
    "pgd",
    "--jump",
    "0.5",
    "--dataset",
    "mnist",
]


with open("./epsilon_mnist_bim_plain.txt", "a") as output:
    for epsilon in np.arange(0, 0.4, 0.02):
        output.write("\nepsilon = " + str(epsilon) + "\n")
        output.flush()
        arglist[13] = str(epsilon)
        arglist[15] = str(epsilon / 10)
        subprocess.call(arglist, stdout=output)

arglist[7] = str(20)
arglist[9] = str(100)

with open("./epsilon_mnist_pgd_plain.txt", "a") as output:
    for epsilon in np.arange(0, 0.4, 0.02):
        output.write("\nepsilon = " + str(epsilon) + "\n")
        output.flush()
        arglist[13] = str(epsilon)
        arglist[15] = str(epsilon / 10)
        subprocess.call(arglist, stdout=output)
